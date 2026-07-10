import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.nocv.win_env import WINRobot


class WINCloseClimbRobot(WINRobot):
    """WIN locomotion plus a command-gated, real-geometry thin-wall task."""

    def _init_buffers(self):
        super()._init_buffers()
        self.climb_command_idx = self.cfg.commands.climb_command_idx
        self.wall_height_command_idx = self.cfg.commands.wall_height_command_idx
        self.commands_scale[-1] = self.cfg.commands.wall_height_command_obs_scale

        n = self.num_envs
        self.climb_triggered = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.climb_trigger_distance = torch.zeros(n, device=self.device)
        self.climb_success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.climb_cleared = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.climb_new_clearance = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.climb_new_success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.climb_stable_timer = torch.zeros(n, device=self.device)
        self.climb_stuck_timer = torch.zeros(n, device=self.device)
        self.climb_tail_timer = torch.zeros(n, device=self.device)
        self.climb_disengaged = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.climb_tail_complete = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.climb_fall = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.climb_stuck = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.climb_out_of_track = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.climb_last_progress = torch.zeros(n, device=self.device)
        self.climb_progress_delta = torch.zeros(n, device=self.device)
        self.climb_last_rear_progress = torch.zeros(n, device=self.device)
        self.climb_rear_progress_delta = torch.zeros(n, device=self.device)
        self.climb_hip_peak_force = torch.zeros(n, device=self.device)
        self.climb_hip_excess_integral = torch.zeros(n, device=self.device)
        self.climb_success_time = torch.zeros(n, device=self.device)

        body_names = list(self.body_names)
        self.calf_body_indices = self._body_indices(body_names, ("calf",))
        self.front_hip_body_indices = self._leg_body_indices(body_names, ("FL", "FR"), "hip")
        self.rear_hip_body_indices = self._leg_body_indices(body_names, ("RL", "RR"), "hip")
        self.hip_body_indices = torch.cat((self.front_hip_body_indices, self.rear_hip_body_indices))
        rear_feet = [i for i, name in enumerate(body_names) if ("RL" in name or "RR" in name) and self.cfg.asset.foot_name in name]
        self.rear_feet_indices = torch.tensor(rear_feet, dtype=torch.long, device=self.device)
        if self.rear_feet_indices.numel() != 2:
            self.rear_feet_indices = self.feet_indices[1::2]

    def _body_indices(self, body_names, fragments):
        indices = [i for i, name in enumerate(body_names) if any(fragment in name.lower() for fragment in fragments)]
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _leg_body_indices(self, body_names, leg_prefixes, fragment):
        indices = [
            i
            for i, name in enumerate(body_names)
            if fragment in name.lower() and any(name.upper().startswith(prefix) for prefix in leg_prefixes)
        ]
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _wall_mask(self, env_ids=None):
        terrain_ids = self.terrain_ids if env_ids is None else self.terrain_ids[env_ids]
        return terrain_ids == self.cfg.commands.wall_terrain_id

    def _active_climb_mask(self):
        return self._wall_mask() & (self.commands[:, self.climb_command_idx] > 0.5)

    def _wall_height(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        fixed_height = self.cfg.terrain.thin_wall_fixed_height
        if fixed_height is not None:
            return torch.full((len(ids),), fixed_height, device=self.device)
        heights = torch.tensor(self.cfg.terrain.thin_wall_heights, device=self.device)
        return heights[self.terrain_levels[ids].long()]

    def _wall_front_x(self, env_ids=None):
        origins = self.env_origins if env_ids is None else self.env_origins[env_ids]
        return origins[:, 0] + self.cfg.terrain.thin_wall_front_offset

    def _wall_rear_x(self, env_ids=None):
        return self._wall_front_x(env_ids) + self.cfg.terrain.thin_wall_thickness

    def _reset_climb_buffers(self, env_ids):
        for buffer in (
            self.climb_triggered,
            self.climb_success,
            self.climb_cleared,
            self.climb_new_clearance,
            self.climb_new_success,
            self.climb_disengaged,
            self.climb_tail_complete,
            self.climb_fall,
            self.climb_stuck,
            self.climb_out_of_track,
        ):
            buffer[env_ids] = False
        for buffer in (
            self.climb_stable_timer,
            self.climb_stuck_timer,
            self.climb_tail_timer,
            self.climb_last_progress,
            self.climb_progress_delta,
            self.climb_last_rear_progress,
            self.climb_rear_progress_delta,
            self.climb_hip_peak_force,
            self.climb_hip_excess_integral,
            self.climb_success_time,
        ):
            buffer[env_ids] = 0.0
        low, high = self.cfg.commands.wall_trigger_distance_range
        self.climb_trigger_distance[env_ids] = torch_rand_float(low, high, (len(env_ids), 1), device=self.device).squeeze(1)

    def _update_terrain_curriculum(self, env_ids):
        """Use terminal climb outcomes, not locomotion distance, as wall curriculum."""
        if not self.init_done:
            return
        wall = self._wall_mask(env_ids)
        wall_ids = env_ids[wall]
        if len(wall_ids) > 0 and self.cfg.close_climb_phase != "eval":
            succeeded = self.climb_success[wall_ids]
            failed = (
                self.climb_fall[wall_ids]
                | self.climb_stuck[wall_ids]
                | self.climb_out_of_track[wall_ids]
                | self.time_out_buf[wall_ids]
            ) & ~succeeded
            levels = self.terrain_levels[wall_ids] + succeeded.long() - failed.long()
            self.terrain_levels[wall_ids] = torch.clamp(levels, 0, len(self.cfg.terrain.thin_wall_heights) - 1)

        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids].long(), self.terrain_types[env_ids]
        ]
        self.max_move_distance[env_ids] = 0.0
        self._reset_climb_buffers(env_ids)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        previous_wall = self._wall_mask(env_ids)
        previous_success = self.climb_success[env_ids].clone()
        previous_fall = self.climb_fall[env_ids].clone()
        previous_stuck = self.climb_stuck[env_ids].clone()
        previous_out = self.climb_out_of_track[env_ids].clone()
        previous_timeout = self.time_out_buf[env_ids].clone()
        previous_hip_peak = self.climb_hip_peak_force[env_ids].clone()
        previous_hip_excess = self.climb_hip_excess_integral[env_ids].clone()
        previous_success_time = self.climb_success_time[env_ids].clone()
        super().reset_idx(env_ids)
        wall_count = previous_wall.sum().clamp(min=1)
        self.extras["episode"].update({
            "climb_success": (previous_success & previous_wall).sum().float() / wall_count,
            "climb_fall": (previous_fall & previous_wall).sum().float() / wall_count,
            "climb_stuck": (previous_stuck & previous_wall).sum().float() / wall_count,
            "climb_out_of_track": (previous_out & previous_wall).sum().float() / wall_count,
            "climb_timeout": (previous_timeout & previous_wall & ~previous_success).sum().float() / wall_count,
            "climb_hip_peak_force": previous_hip_peak[previous_wall].mean() if previous_wall.any() else torch.tensor(0.0, device=self.device),
            "climb_hip_excess": previous_hip_excess[previous_wall].mean() if previous_wall.any() else torch.tensor(0.0, device=self.device),
            "climb_success_time": previous_success_time[previous_success & previous_wall].mean() if (previous_success & previous_wall).any() else torch.tensor(0.0, device=self.device),
            "climb_wall_episodes": previous_wall.sum().float(),
        })

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        wall = self._wall_mask(env_ids)
        wall_ids = env_ids[wall]
        flat_ids = env_ids[~wall]

        if len(flat_ids) > 0:
            self.root_states[flat_ids, 0] += torch_rand_float(-1.0, 1.0, (len(flat_ids), 1), device=self.device).squeeze(1)
            self.root_states[flat_ids, 1] += torch_rand_float(-0.5, 0.5, (len(flat_ids), 1), device=self.device).squeeze(1)
        if len(wall_ids) > 0:
            if self.cfg.close_climb_phase == "acquire":
                low, high = self.cfg.commands.acquisition_reset_distance_range
                lateral = 0.03
                velocity_noise = 0.05
            else:
                low, high = self.cfg.commands.transition_reset_distance_range
                lateral = 0.08
                velocity_noise = 0.15
            distance = torch_rand_float(low, high, (len(wall_ids), 1), device=self.device).squeeze(1)
            self.root_states[wall_ids, 0] = self._wall_front_x(wall_ids) - distance
            self.root_states[wall_ids, 1] += torch_rand_float(-lateral, lateral, (len(wall_ids), 1), device=self.device).squeeze(1)
            self.root_states[wall_ids, 7:13] = torch_rand_float(
                -velocity_noise, velocity_noise, (len(wall_ids), 6), device=self.device
            )
        if len(flat_ids) > 0:
            self.root_states[flat_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(flat_ids), 6), device=self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _set_wall_command(self, env_ids, climb_enabled):
        if len(env_ids) == 0:
            return
        self.commands[env_ids] = 0.0
        self.commands[env_ids, 0] = self.cfg.commands.wall_approach_velocity
        self.commands[env_ids, self.climb_command_idx] = float(climb_enabled)
        if climb_enabled:
            self.commands[env_ids, self.wall_height_command_idx] = self._wall_height(env_ids)
        self.commands_resampling_step[env_ids] = self.max_episode_length + 1

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        wall = self._wall_mask(env_ids)
        flat_ids = env_ids[~wall]
        wall_ids = env_ids[wall]
        if len(flat_ids) > 0:
            super()._resample_commands(flat_ids)
        if len(wall_ids) > 0:
            new_episode = self.episode_length_buf[wall_ids] == 0
            reset_ids = wall_ids[new_episode]
            if len(reset_ids) > 0:
                engage = self.cfg.close_climb_phase == "acquire"
                self._set_wall_command(reset_ids, engage)
                self.climb_triggered[reset_ids] = engage
            # Periodic resampling must never overwrite an in-progress wall command.
            periodic_ids = wall_ids[~new_episode]
            if len(periodic_ids) > 0:
                self.commands_resampling_step[periodic_ids] = self.max_episode_length + 1

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_climb_state()

    def _update_climb_state(self):
        wall = self._wall_mask()
        pending = wall & ~self.climb_triggered & ~self.climb_success
        distance_to_wall = self._wall_front_x() - self.root_states[:, 0]
        trigger = pending & (distance_to_wall <= self.climb_trigger_distance)
        trigger_ids = trigger.nonzero(as_tuple=False).flatten()
        if len(trigger_ids) > 0:
            self._set_wall_command(trigger_ids, True)
            self.climb_triggered[trigger_ids] = True

        active = self._active_climb_mask() & ~self.climb_success
        body_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        feet_x = body_states[:, self.feet_indices, 0]
        rear_feet_x = body_states[:, self.rear_feet_indices, 0]
        wall_front = self._wall_front_x()
        wall_rear = self._wall_rear_x()
        wall_height = self._wall_height()

        base_progress = self.root_states[:, 0] - wall_front
        rear_progress = rear_feet_x.min(dim=1).values - wall_front
        progress_metric = torch.maximum(base_progress, rear_progress)
        raw_delta = progress_metric - self.climb_last_progress
        rear_delta = rear_progress - self.climb_last_rear_progress
        self.climb_progress_delta[:] = torch.where(active, torch.clamp(raw_delta / wall_height, 0.0, 0.1), 0.0)
        self.climb_rear_progress_delta[:] = torch.where(active, torch.clamp(rear_delta / wall_height, 0.0, 0.1), 0.0)
        self.climb_last_progress[:] = torch.where(active, progress_metric, self.climb_last_progress)
        self.climb_last_rear_progress[:] = torch.where(active, rear_progress, self.climb_last_rear_progress)

        made_progress = (raw_delta > self.cfg.rewards.close_climb_no_progress_delta) | (
            rear_delta > self.cfg.rewards.close_climb_no_progress_delta
        )
        stuck_region = active & (distance_to_wall < 0.50)
        self.climb_stuck_timer[:] = torch.where(
            stuck_region & ~made_progress,
            self.climb_stuck_timer + self.dt,
            torch.zeros_like(self.climb_stuck_timer),
        )
        self.climb_stuck |= self.climb_stuck_timer >= self.cfg.rewards.close_climb_stuck_duration_s

        fully_clear = wall & torch.all(
            feet_x > (wall_rear + self.cfg.rewards.close_climb_success_feet_margin).unsqueeze(1), dim=1
        )
        self.climb_new_clearance[:] = fully_clear & ~self.climb_cleared
        self.climb_cleared |= fully_clear

        base_height = self.root_states[:, 2] - self.env_origins[:, 2]
        stable = (
            fully_clear
            & (self.root_states[:, 0] > wall_rear + self.cfg.rewards.close_climb_success_base_margin)
            & (base_height >= self.cfg.rewards.close_climb_stable_base_height[0])
            & (base_height <= self.cfg.rewards.close_climb_stable_base_height[1])
            & (self.rpy[:, 0].abs() < self.cfg.rewards.close_climb_stable_roll)
            & (self.rpy[:, 1].abs() < self.cfg.rewards.close_climb_stable_pitch)
            & (self.base_ang_vel[:, 2].abs() < self.cfg.rewards.close_climb_stable_yaw_rate)
            & ((self.contact_forces[:, self.feet_indices, 2] > 1.0).sum(dim=1) >= 2)
        )
        self.climb_stable_timer[:] = torch.where(stable, self.climb_stable_timer + self.dt, 0.0)
        self.climb_new_success[:] = (
            self.climb_stable_timer >= self.cfg.rewards.close_climb_stable_duration_s
        ) & ~self.climb_success
        self.climb_success |= self.climb_new_success
        self.climb_success_time[:] = torch.where(
            self.climb_new_success,
            self.episode_length_buf.float() * self.dt,
            self.climb_success_time,
        )

        successful = self.climb_success & wall
        self.climb_tail_timer[:] = torch.where(successful, self.climb_tail_timer + self.dt, self.climb_tail_timer)
        transition_phase = self.cfg.close_climb_phase in ("transition", "eval")
        if transition_phase:
            disengage = successful & ~self.climb_disengaged & (
                self.climb_tail_timer >= self.cfg.rewards.close_climb_post_success_s
            )
            disengage_ids = disengage.nonzero(as_tuple=False).flatten()
            if len(disengage_ids) > 0:
                self._set_wall_command(disengage_ids, False)
                self.climb_disengaged[disengage_ids] = True
            self.climb_tail_complete |= successful & (
                self.climb_tail_timer
                >= self.cfg.rewards.close_climb_post_success_s + self.cfg.rewards.close_climb_post_disengage_s
            )
        else:
            self.climb_tail_complete |= successful & (
                self.climb_tail_timer >= self.cfg.rewards.close_climb_post_success_s
            )

        lateral_error = (self.root_states[:, 1] - self.env_origins[:, 1]).abs()
        self.climb_out_of_track |= wall & (lateral_error > self.cfg.rewards.close_climb_lateral_limit)
        self.climb_fall |= wall & (
            (base_height < self.cfg.rewards.close_climb_fall_base_height)
            | (self.rpy[:, 0].abs() > self.cfg.rewards.close_climb_fall_roll)
            | (self.rpy[:, 1].abs() > self.cfg.rewards.close_climb_fall_pitch)
        )

        if self.hip_body_indices.numel() > 0:
            hip_force = torch.norm(self.contact_forces[:, self.hip_body_indices, :], dim=-1).max(dim=1).values
            self.climb_hip_peak_force[:] = torch.maximum(self.climb_hip_peak_force, torch.where(wall, hip_force, 0.0))
            front_excess = self._hip_force_excess(
                self.front_hip_body_indices,
                self.cfg.rewards.close_climb_front_hip_force_threshold,
            )
            rear_excess = self._hip_force_excess(
                self.rear_hip_body_indices,
                self.cfg.rewards.close_climb_rear_hip_force_threshold,
            )
            self.climb_hip_excess_integral += torch.where(
                active,
                (front_excess + rear_excess) * self.dt,
                0.0,
            )

    def check_termination(self):
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        wall = self._wall_mask()
        flat = ~wall
        if self.termination_contact_indices.numel() > 0:
            contact_termination = torch.any(
                torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
                dim=1,
            )
            self.reset_buf |= flat & contact_termination
        self.reset_buf |= wall & (
            self.climb_fall | self.climb_stuck | self.climb_out_of_track | self.climb_tail_complete
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def _flat_only(self, value):
        return value * (~self._wall_mask())

    def _reward_collision(self):
        return self._flat_only(super()._reward_collision())

    def _reward_orientation(self):
        return self._flat_only(super()._reward_orientation())

    def _reward_ang_vel_xy(self):
        return self._flat_only(super()._reward_ang_vel_xy())

    def _reward_correct_base_height(self):
        return self._flat_only(super()._reward_correct_base_height())

    def _reward_low_height_correct_base_height(self):
        return self._flat_only(super()._reward_low_height_correct_base_height())

    def _reward_base_height(self):
        return self._flat_only(super()._reward_base_height())

    def _reward_hip_to_default(self):
        return self._flat_only(super()._reward_hip_to_default())

    def _reward_hip_to_zero(self):
        return self._flat_only(super()._reward_hip_to_zero())

    def _reward_similar_to_default(self):
        return self._flat_only(super()._reward_similar_to_default())

    def _reward_lateral_yaw_tracking_error(self):
        return self._flat_only(super()._reward_lateral_yaw_tracking_error())

    def _reward_feet_regulation(self):
        return self._flat_only(super()._reward_feet_regulation())

    def _reward_x_command_hip_regular(self):
        return self._flat_only(super()._reward_x_command_hip_regular())

    def _reward_climb_progress(self):
        return self.climb_progress_delta

    def _reward_climb_rear_feet_progress(self):
        return self.climb_rear_progress_delta

    def _reward_climb_clearance(self):
        return self.climb_new_clearance.float()

    def _reward_climb_stable_success(self):
        return self.climb_new_success.float()

    def _reward_climb_wall_contact(self):
        active = self._active_climb_mask()
        body_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        indices = torch.cat((self.feet_indices, self.calf_body_indices))
        if indices.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        positions = body_states[:, indices, :3]
        forces = torch.norm(self.contact_forces[:, indices, :], dim=-1)
        near_wall = (
            (positions[:, :, 0] > (self._wall_front_x() - 0.06).unsqueeze(1))
            & (positions[:, :, 0] < (self._wall_rear_x() + 0.06).unsqueeze(1))
            & (positions[:, :, 2] > 0.02)
            & (positions[:, :, 2] < (self._wall_height() + 0.06).unsqueeze(1))
        )
        effective = torch.clamp(forces / 100.0, max=1.0) * near_wall
        return effective.sum(dim=1) * active

    def _reward_climb_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self._active_climb_mask()

    def _reward_climb_centerline(self):
        error = self.root_states[:, 1] - self.env_origins[:, 1]
        return torch.square(error) * self._wall_mask()

    def _reward_climb_yaw(self):
        return torch.square(self.rpy[:, 2]) * self._wall_mask()

    def _reward_climb_hip_force(self):
        front_excess = self._hip_force_excess(
            self.front_hip_body_indices,
            self.cfg.rewards.close_climb_front_hip_force_threshold,
        )
        rear_excess = self._hip_force_excess(
            self.rear_hip_body_indices,
            self.cfg.rewards.close_climb_rear_hip_force_threshold,
        )
        weighted_excess = (
            front_excess
            + self.cfg.rewards.close_climb_rear_hip_force_multiplier * rear_excess
        )
        return weighted_excess * self._active_climb_mask()

    def _hip_force_excess(self, body_indices, threshold):
        if body_indices.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        forces = torch.norm(self.contact_forces[:, body_indices, :], dim=-1)
        return torch.clamp(forces - threshold, min=0.0).sum(dim=1)

    def _reward_climb_stuck(self):
        return self.climb_stuck.float()
