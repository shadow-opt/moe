import math

import torch

from legged_gym.envs.nocv.win_env import WINRobot
from legged_gym.envs.nocv.win_jump_logic import (
    classify_mode_draws,
    jump_stance_mask,
)


MODE_WALK = 0
MODE_JUMP = 1
MODE_STAND = 2
MODE_PREP = 3


class WINJumpRobot(WINRobot):
    """WIN locomotion task with a latched, command-gated jump mode."""

    _COMMON_REWARDS = {
        "collision",
        "dof_pos_limits",
        "dof_vel_limits",
        "action_smoothness",
    }

    def _init_buffers(self):
        super()._init_buffers()
        n = self.num_envs
        device = self.device
        self.requested_mode = torch.full((n,), MODE_STAND, dtype=torch.long, device=device)
        self.active_mode = torch.full((n,), MODE_STAND, dtype=torch.long, device=device)
        self.target_commands = torch.zeros_like(self.commands)
        self.jump_phase_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.jump_motion_enabled = torch.zeros(n, dtype=torch.bool, device=device)
        self.jump_prep_remaining = torch.zeros(n, dtype=torch.long, device=device)
        self.jump_prep_just_started = torch.zeros(n, dtype=torch.bool, device=device)
        self.jump_exit_pending = torch.zeros(n, dtype=torch.bool, device=device)
        self.jump_seen_airborne = torch.zeros(n, dtype=torch.bool, device=device)
        self.jump_landing_streak = torch.zeros(n, dtype=torch.long, device=device)
        self.jump_exit_elapsed = torch.zeros(n, device=device)
        self.jump_transition_timeout = torch.zeros(n, dtype=torch.bool, device=device)
        self.jump_sanitized_count = torch.zeros((), dtype=torch.long, device=device)
        self.jump_requested_counts = torch.zeros(3, dtype=torch.long, device=device)
        self.jump_active_counts = torch.zeros(4, dtype=torch.long, device=device)

        self.jump_feet_air_time = torch.zeros_like(self.feet_air_time)
        self.jump_last_contacts = torch.zeros_like(self.last_contacts)
        self.jump_speed_unlocked = False
        self.jump_tracking_streak = 0
        self.jump_tracking_sum = torch.zeros(2, device=device)
        self.jump_tracking_count = torch.zeros(2, dtype=torch.long, device=device)
        self.jump_last_tracking_mean = torch.zeros(2, device=device)

    def _reset_mode_reward_state(self, env_ids):
        if len(env_ids) == 0:
            return
        self.feet_air_time[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.feet_current_air_time[env_ids] = 0.0
        self.feet_current_contact_time[env_ids] = 0.0
        self.feet_last_air_time[env_ids] = 0.0
        self.feet_last_contact_time[env_ids] = 0.0
        self.feet_timing_contact[env_ids] = False
        self.feet_timing_last_contacts[env_ids] = False
        self.feet_first_contact[env_ids] = False
        self.jump_feet_air_time[env_ids] = 0.0
        self.jump_last_contacts[env_ids] = False
        if hasattr(self, "low_speed_feet_air_time"):
            self.low_speed_feet_air_time[env_ids] = 0.0
            self.low_speed_last_contacts[env_ids] = False
        if hasattr(self, "foot_slip_last_contacts"):
            self.foot_slip_last_contacts[env_ids] = False

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if hasattr(self, "requested_mode"):
            self.requested_mode[env_ids] = MODE_STAND
            self.active_mode[env_ids] = MODE_STAND
            self.target_commands[env_ids] = 0.0
            self.jump_phase_steps[env_ids] = 0
            self.jump_motion_enabled[env_ids] = False
            self.jump_prep_remaining[env_ids] = 0
            self.jump_prep_just_started[env_ids] = False
            self.jump_exit_pending[env_ids] = False
            self.jump_seen_airborne[env_ids] = False
            self.jump_landing_streak[env_ids] = 0
            self.jump_exit_elapsed[env_ids] = 0.0
            self.jump_transition_timeout[env_ids] = False
            self._reset_mode_reward_state(env_ids)
        super().reset_idx(env_ids)

    def _sample_jump_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        max_abs = (
            self.cfg.commands.jump_unlocked_max_abs_vx
            if self.jump_speed_unlocked
            else self.cfg.commands.jump_initial_max_abs_vx
        )
        magnitudes = torch.rand(len(env_ids), device=self.device)
        magnitudes = self.cfg.commands.jump_min_abs_vx + magnitudes * (
            max_abs - self.cfg.commands.jump_min_abs_vx
        )
        signs = torch.where(
            torch.rand(len(env_ids), device=self.device) < 0.5,
            -torch.ones(len(env_ids), device=self.device),
            torch.ones(len(env_ids), device=self.device),
        )
        self.target_commands[env_ids] = 0.0
        self.target_commands[env_ids, 0] = signs * magnitudes
        self.target_commands[env_ids, self.body_height_command_idx] = self.cfg.commands.jump_mode_command

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        draw = torch.rand(len(env_ids), device=self.device)
        sampled_modes = classify_mode_draws(
            draw,
            self.cfg.commands.jump_sample_probability,
            self.cfg.commands.walk_sample_probability,
        )
        jump_ids = env_ids[sampled_modes == MODE_JUMP]
        walk_ids = env_ids[sampled_modes == MODE_WALK]
        stand_ids = env_ids[sampled_modes == MODE_STAND]

        if len(walk_ids) > 0:
            super()._resample_commands(walk_ids)
            self.target_commands[walk_ids] = self.commands[walk_ids]
            self.target_commands[walk_ids, 5:7] = 0.0
            self.requested_mode[walk_ids] = MODE_WALK
        if len(stand_ids) > 0:
            self.target_commands[stand_ids] = 0.0
            self.requested_mode[stand_ids] = MODE_STAND
            self.commands_resampling_step[stand_ids] = self.cfg.commands.resampling_time / self.dt
        if len(jump_ids) > 0:
            self._sample_jump_commands(jump_ids)
            self.requested_mode[jump_ids] = MODE_JUMP
            self.commands_resampling_step[jump_ids] = self.cfg.commands.resampling_time / self.dt

        self.jump_requested_counts[MODE_JUMP] += len(jump_ids)
        self.jump_requested_counts[MODE_WALK] += len(walk_ids)
        self.jump_requested_counts[MODE_STAND] += len(stand_ids)

        non_jump_ids = env_ids[self.requested_mode[env_ids] != MODE_JUMP]
        if len(non_jump_ids) > 0:
            currently_jumping = self.active_mode[non_jump_ids] == MODE_JUMP
            latched_ids = non_jump_ids[currently_jumping]
            immediate_ids = non_jump_ids[~currently_jumping]
            if len(latched_ids) > 0:
                self.jump_exit_pending[latched_ids] = True
                self.jump_exit_elapsed[latched_ids] = 0.0
                self.jump_landing_streak[latched_ids] = 0
            if len(immediate_ids) > 0:
                self.active_mode[immediate_ids] = self.requested_mode[immediate_ids]
                self.commands[immediate_ids] = self.target_commands[immediate_ids]
                self.jump_prep_remaining[immediate_ids] = 0
                self.jump_prep_just_started[immediate_ids] = False
                self._reset_mode_reward_state(immediate_ids)

        if len(jump_ids) > 0:
            already_jumping = self.active_mode[jump_ids] == MODE_JUMP
            continuing_ids = jump_ids[already_jumping]
            entering_ids = jump_ids[~already_jumping]
            if len(continuing_ids) > 0:
                self.commands[continuing_ids, 0] = self.target_commands[continuing_ids, 0]
                self.jump_exit_pending[continuing_ids] = False
            if len(entering_ids) > 0:
                self.active_mode[entering_ids] = MODE_PREP
                self.commands[entering_ids] = 0.0
                self.jump_prep_remaining[entering_ids] = self.cfg.commands.jump_prep_steps
                self.jump_prep_just_started[entering_ids] = True
                self.jump_phase_steps[entering_ids] = 0
                self.jump_motion_enabled[entering_ids] = False
                self._reset_mode_reward_state(entering_ids)

    def set_requested_commands(self, env_ids, command_values):
        """Set external internal-buffer commands for deterministic evaluation."""
        if command_values.shape != (len(env_ids), self.commands.shape[1]):
            raise ValueError(
                f"command_values shape {command_values.shape} does not match "
                f"({len(env_ids)}, {self.commands.shape[1]})"
            )
        self.target_commands[env_ids] = command_values
        body = command_values[:, self.body_height_command_idx]
        jump = body < -0.5
        stand = (~jump) & (torch.norm(command_values[:, :3], dim=1) < 0.1) & (body.abs() < 0.5)
        self.requested_mode[env_ids] = MODE_WALK
        self.requested_mode[env_ids[jump]] = MODE_JUMP
        self.requested_mode[env_ids[stand]] = MODE_STAND
        invalid = jump & ((command_values[:, 1].abs() > 0.0) | (command_values[:, 2].abs() > 0.0))
        if invalid.any():
            invalid_ids = env_ids[invalid]
            self.target_commands[invalid_ids, 1:3] = 0.0
            self.jump_sanitized_count += invalid.sum()
        self._begin_requested_transitions(env_ids)

    def _begin_requested_transitions(self, env_ids):
        """Apply externally requested modes using the same latch rules as sampling."""
        jump_ids = env_ids[self.requested_mode[env_ids] == MODE_JUMP]
        non_jump_ids = env_ids[self.requested_mode[env_ids] != MODE_JUMP]
        if len(non_jump_ids) > 0:
            latched = non_jump_ids[self.active_mode[non_jump_ids] == MODE_JUMP]
            immediate = non_jump_ids[self.active_mode[non_jump_ids] != MODE_JUMP]
            self.jump_exit_pending[latched] = True
            self.jump_exit_elapsed[latched] = 0.0
            self.active_mode[immediate] = self.requested_mode[immediate]
            self.commands[immediate] = self.target_commands[immediate]
            self._reset_mode_reward_state(immediate)
        if len(jump_ids) > 0:
            entering = jump_ids[self.active_mode[jump_ids] != MODE_JUMP]
            continuing = jump_ids[self.active_mode[jump_ids] == MODE_JUMP]
            self.active_mode[entering] = MODE_PREP
            self.commands[entering] = 0.0
            self.jump_prep_remaining[entering] = self.cfg.commands.jump_prep_steps
            self.jump_prep_just_started[entering] = True
            self._reset_mode_reward_state(entering)
            self.commands[continuing, 0] = self.target_commands[continuing, 0]
            self.jump_exit_pending[continuing] = False

    def _enter_jump(self, env_ids):
        if len(env_ids) == 0:
            return
        self.active_mode[env_ids] = MODE_JUMP
        self.commands[env_ids] = self.target_commands[env_ids]
        self.commands[env_ids, 1:3] = 0.0
        self.commands[env_ids, self.body_height_command_idx] = self.cfg.commands.jump_mode_command
        # The phase updater increments enabled jumps once later in this callback.
        self.jump_phase_steps[env_ids] = -1
        self.jump_motion_enabled[env_ids] = torch.abs(self.commands[env_ids, 0]) >= self.cfg.commands.jump_enable_threshold
        self.jump_exit_pending[env_ids] = False
        self.jump_seen_airborne[env_ids] = False
        self.jump_landing_streak[env_ids] = 0
        self.jump_exit_elapsed[env_ids] = 0.0
        self._reset_mode_reward_state(env_ids)

    def _leave_jump(self, env_ids):
        if len(env_ids) == 0:
            return
        self.active_mode[env_ids] = self.requested_mode[env_ids]
        self.commands[env_ids] = self.target_commands[env_ids]
        self.commands[env_ids, 5:7] = 0.0
        self.jump_phase_steps[env_ids] = 0
        self.jump_motion_enabled[env_ids] = False
        self.jump_exit_pending[env_ids] = False
        self.jump_seen_airborne[env_ids] = False
        self.jump_landing_streak[env_ids] = 0
        self.jump_exit_elapsed[env_ids] = 0.0
        self._reset_mode_reward_state(env_ids)

    def _update_jump_phase_commands(self):
        active = self.active_mode == MODE_JUMP
        if not active.any():
            return
        ids = active.nonzero(as_tuple=False).flatten()
        abs_vx = torch.abs(self.commands[ids, 0])
        enable = abs_vx >= self.cfg.commands.jump_enable_threshold
        disable = abs_vx < self.cfg.commands.jump_disable_threshold
        self.jump_motion_enabled[ids[enable]] = True
        self.jump_motion_enabled[ids[disable]] = False
        disabled_ids = ids[~self.jump_motion_enabled[ids]]
        self.jump_phase_steps[disabled_ids] = 0
        enabled_ids = ids[self.jump_motion_enabled[ids]]
        self.jump_phase_steps[enabled_ids] += 1
        phase = self.jump_phase_steps[ids].float() * self.dt / self.cfg.commands.jump_cycle_time
        self.commands[ids, 1:3] = 0.0
        self.commands[ids, 5] = torch.sin(2.0 * math.pi * phase)
        self.commands[ids, 6] = torch.cos(2.0 * math.pi * phase)
        self.commands[disabled_ids, 5:7] = 0.0

    def _update_jump_fsm(self):
        prep_ids = (self.active_mode == MODE_PREP).nonzero(as_tuple=False).flatten()
        if len(prep_ids) > 0:
            self.commands[prep_ids] = 0.0
            fresh = self.jump_prep_just_started[prep_ids]
            self.jump_prep_just_started[prep_ids] = False
            ticking_ids = prep_ids[~fresh]
            self.jump_prep_remaining[ticking_ids] -= 1
            ready_ids = prep_ids[self.jump_prep_remaining[prep_ids] <= 0]
            self._enter_jump(ready_ids)

        jump_ids = (self.active_mode == MODE_JUMP).nonzero(as_tuple=False).flatten()
        if len(jump_ids) == 0:
            return
        contact = self.contact_forces[jump_ids][:, self.feet_indices, 2] > self.cfg.commands.jump_contact_threshold
        all_airborne = ~torch.any(contact, dim=1)
        self.jump_seen_airborne[jump_ids] |= all_airborne & self.jump_motion_enabled[jump_ids]

        pending = self.jump_exit_pending[jump_ids]
        pending_ids = jump_ids[pending]
        if len(pending_ids) > 0:
            pending_contact = self.contact_forces[pending_ids][:, self.feet_indices, 2] > self.cfg.commands.jump_contact_threshold
            landed = self.jump_seen_airborne[pending_ids] & torch.all(pending_contact, dim=1)
            self.jump_landing_streak[pending_ids] = torch.where(
                landed,
                self.jump_landing_streak[pending_ids] + 1,
                torch.zeros_like(self.jump_landing_streak[pending_ids]),
            )
            self.jump_exit_elapsed[pending_ids] += self.dt
            complete = self.jump_landing_streak[pending_ids] >= self.cfg.commands.jump_landing_contact_steps
            self._leave_jump(pending_ids[complete])
            timeout = self.jump_exit_elapsed[pending_ids] >= self.cfg.commands.jump_exit_timeout_s
            self.jump_transition_timeout[pending_ids[timeout]] = True

        self._update_jump_phase_commands()

    def _update_jump_speed_curriculum(self):
        active = (self.active_mode == MODE_JUMP) & self.jump_motion_enabled
        if active.any():
            tracking = torch.exp(
                -torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
                / self.cfg.rewards.tracking_sigma
            )
            positive = active & (self.commands[:, 0] > 0.0)
            negative = active & (self.commands[:, 0] < 0.0)
            for index, mask in enumerate((positive, negative)):
                self.jump_tracking_sum[index] += tracking[mask].sum()
                self.jump_tracking_count[index] += mask.sum()

        if self.common_step_counter % self.num_steps_per_env != 0:
            return
        counts = self.jump_tracking_count.clone()
        means = self.jump_tracking_sum / counts.clamp(min=1).float()
        self.jump_last_tracking_mean[:] = means
        enough = torch.all(counts >= self.cfg.commands.jump_curriculum_min_samples)
        passed = enough & torch.all(means >= self.cfg.commands.jump_curriculum_threshold)
        self.jump_tracking_streak = self.jump_tracking_streak + 1 if bool(passed) else 0
        if self.jump_tracking_streak >= self.cfg.commands.jump_curriculum_windows:
            self.jump_speed_unlocked = True
        self.jump_tracking_sum.zero_()
        self.jump_tracking_count.zero_()

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_jump_fsm()
        self._update_jump_speed_curriculum()
        for mode in (MODE_WALK, MODE_JUMP, MODE_STAND, MODE_PREP):
            self.jump_active_counts[mode] += torch.sum(self.active_mode == mode)
        self.extras["jump_metrics"] = {
            "requested_counts": self.jump_requested_counts.clone(),
            "active_counts": self.jump_active_counts.clone(),
            "sanitized_count": self.jump_sanitized_count.clone(),
            "tracking_forward": self.jump_last_tracking_mean[0].clone(),
            "tracking_backward": self.jump_last_tracking_mean[1].clone(),
            "speed_unlocked": float(self.jump_speed_unlocked),
        }

    def consume_jump_metrics(self):
        requested_total = self.jump_requested_counts.sum().clamp(min=1).float()
        active_total = self.jump_active_counts.sum().clamp(min=1).float()
        metrics = {
            "requested_walk_ratio": (self.jump_requested_counts[MODE_WALK] / requested_total).item(),
            "requested_jump_ratio": (self.jump_requested_counts[MODE_JUMP] / requested_total).item(),
            "requested_stand_ratio": (self.jump_requested_counts[MODE_STAND] / requested_total).item(),
            "active_walk_ratio": (self.jump_active_counts[MODE_WALK] / active_total).item(),
            "active_jump_ratio": (self.jump_active_counts[MODE_JUMP] / active_total).item(),
            "active_stand_ratio": (self.jump_active_counts[MODE_STAND] / active_total).item(),
            "active_prep_ratio": (self.jump_active_counts[MODE_PREP] / active_total).item(),
            "sanitized_count": self.jump_sanitized_count.item(),
            "tracking_forward": self.jump_last_tracking_mean[0].item(),
            "tracking_backward": self.jump_last_tracking_mean[1].item(),
            "speed_unlocked": float(self.jump_speed_unlocked),
        }
        self.jump_requested_counts.zero_()
        self.jump_active_counts.zero_()
        self.jump_sanitized_count.zero_()
        return metrics

    def get_training_state(self):
        return {
            "jump_speed_unlocked": bool(self.jump_speed_unlocked),
            "jump_tracking_streak": int(self.jump_tracking_streak),
        }

    def load_training_state(self, state):
        state = state or {}
        self.jump_speed_unlocked = bool(state.get("jump_speed_unlocked", False))
        self.jump_tracking_streak = int(state.get("jump_tracking_streak", 0))

    def check_termination(self):
        super().check_termination()
        self.reset_buf |= self.jump_transition_timeout

    def compute_reward(self):
        self.rew_buf[:] = 0.0
        jump_mask = (self.active_mode == MODE_JUMP).float()
        walk_mask = 1.0 - jump_mask
        for name, reward_function in zip(self.reward_names, self.reward_functions):
            raw_reward = reward_function()
            if name.startswith("jump_"):
                raw_reward *= jump_mask
            elif name not in self._COMMON_REWARDS:
                raw_reward *= walk_mask
            reward = raw_reward * self.reward_scales.get(name, 0.0)
            self.rew_buf += reward
            self.episode_sums[name] += reward
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf, min=0.0)
        if "termination" in self.reward_scales:
            reward = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += reward
            self.episode_sums["termination"] += reward

    def _jump_phase(self):
        return self.jump_phase_steps.float() * self.dt / self.cfg.commands.jump_cycle_time

    def _jump_stance_mask(self):
        return jump_stance_mask(
            self._jump_phase(),
            self.cfg.commands.cyclic_jump_phase,
        )

    def _reward_jump_contact_pattern(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.commands.jump_contact_threshold
        synchronous = torch.all(contact == contact[:, :1], dim=1)
        expected = contact[:, 0] == self._jump_stance_mask()
        return (synchronous & expected & (torch.abs(self.commands[:, 0]) > 0.2)).float()

    def _reward_jump_tracking_lin_vel(self):
        error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        moving = torch.norm(self.commands[:, :2], dim=1) > 0.2
        return torch.where(
            moving,
            torch.exp(-error / self.cfg.rewards.tracking_sigma),
            torch.exp(-torch.norm(self.base_lin_vel[:, :2], dim=1) / self.cfg.rewards.tracking_sigma),
        )

    def _reward_jump_tracking_ang_vel(self):
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        moving = torch.abs(self.commands[:, 2]) > 0.2
        return torch.where(
            moving,
            torch.exp(-error / self.cfg.rewards.tracking_sigma),
            torch.exp(-torch.abs(self.base_ang_vel[:, 2]) / self.cfg.rewards.tracking_sigma),
        )

    def _reward_jump_lin_vel_z(self):
        return torch.exp(-torch.abs(self.base_lin_vel[:, 2]))

    def _reward_jump_ang_vel_xy(self):
        return torch.exp(-torch.norm(torch.abs(self.base_ang_vel[:, :2]), dim=1))

    def _reward_jump_orientation(self):
        return torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 10.0)

    def _reward_jump_base_height(self):
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        stationary = torch.norm(self.commands[:, :2], dim=1) < 0.2
        return torch.exp(
            -torch.abs(base_height - self.cfg.rewards.jump_base_height_target) * 10.0
        ) * stationary.float()

    def _reward_jump_torques(self):
        return torch.sum(torch.abs(self.torques), dim=1)

    def _reward_jump_dof_acc(self):
        return torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1)

    def _reward_jump_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_jump_default_pos(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_jump_default_hip_pos(self):
        hip_ids = [index for index, name in enumerate(self.dof_names) if "hip" in name]
        return torch.exp(-4.0 * torch.sum(torch.abs(self.dof_pos[:, hip_ids]), dim=1))

    def _reward_jump_feet_contact_forces(self):
        force = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        return torch.sum((force - self.cfg.rewards.max_contact_force).clip(min=0.0), dim=1)

    def _reward_jump_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filtered = torch.logical_or(contact, self.jump_last_contacts)
        self.jump_last_contacts = contact
        first_contact = (self.jump_feet_air_time > 0.0) & contact_filtered
        self.jump_feet_air_time += self.dt
        reward = torch.sum((self.jump_feet_air_time - 0.5) * first_contact, dim=1)
        reward *= (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()
        self.jump_feet_air_time *= ~contact_filtered
        return reward

    def _reward_jump_feet_clearance(self):
        body_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        feet_height = body_states[:, self.feet_indices, 2] - 0.02
        swing = (~self._jump_stance_mask()).float().unsqueeze(1)
        clearance = torch.clip(feet_height, min=0.0, max=self.cfg.rewards.jump_target_feet_height)
        reward = torch.sum(clearance * swing, dim=1)
        return reward * (torch.norm(self.commands[:, :2], dim=1) > 0.2).float()
