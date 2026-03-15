import torch

from legged_gym.envs.nocv.jump_env import JumpRobot
from legged_gym.envs.nocv.nocv_env import NoCVRobot


class CurriculumJumpRobot(JumpRobot):
    def _init_buffers(self):
        super()._init_buffers()

        cfg_cmd = self.cfg.commands
        levels = cfg_cmd.curriculum_levels
        self.curriculum_num_levels = len(levels)
        self.curriculum_max_level = self.curriculum_num_levels - 1

        self.curriculum_dx_l = torch.tensor([lvl["jump_dx"][0] for lvl in levels], device=self.device, dtype=torch.float)
        self.curriculum_dx_h = torch.tensor([lvl["jump_dx"][1] for lvl in levels], device=self.device, dtype=torch.float)
        self.curriculum_dy_l = torch.tensor([lvl["jump_dy"][0] for lvl in levels], device=self.device, dtype=torch.float)
        self.curriculum_dy_h = torch.tensor([lvl["jump_dy"][1] for lvl in levels], device=self.device, dtype=torch.float)
        self.curriculum_dz_l = torch.tensor([lvl["jump_dz"][0] for lvl in levels], device=self.device, dtype=torch.float)
        self.curriculum_dz_h = torch.tensor([lvl["jump_dz"][1] for lvl in levels], device=self.device, dtype=torch.float)

        start_level = int(getattr(cfg_cmd, "curriculum_start_level", 0))
        start_level = max(0, min(start_level, self.curriculum_max_level))
        self.command_dist_levels = torch.full(
            (self.num_envs,),
            start_level,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

        self.curriculum_window = int(cfg_cmd.curriculum_success_window)
        self.curriculum_min_valid_episodes = int(cfg_cmd.curriculum_min_valid_episodes)
        self.curriculum_success_high = float(cfg_cmd.curriculum_success_high)
        self.curriculum_success_low = float(cfg_cmd.curriculum_success_low)

        self.success_hist = torch.zeros(
            self.num_envs,
            self.curriculum_window,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.success_hist_ptr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.success_hist_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._apply_jump_velocity_command_mux()

    def _apply_jump_velocity_command_mux(self):
        active = self._get_active_jump_mask() & self.jump_initialized & ~self.has_jumped
        if not active.any():
            return

        jump_dz = torch.clamp(self.commands[:, self.jump_dz_command_idx], min=0.05)
        vz = torch.sqrt(2.0 * 9.81 * jump_dz)
        flight_time = torch.clamp(2.0 * vz / 9.81, min=0.1)

        target_vx = self.commands[:, self.jump_dx_command_idx] / flight_time
        target_vy = self.commands[:, self.jump_dy_command_idx] / flight_time

        self.commands[active, 0] = target_vx[active]
        self.commands[active, 1] = target_vy[active]
        self.commands[active, 2] = 0.0
        self.commands[active, 3] = 0.0

    def _resample_commands(self, env_ids):
        NoCVRobot._resample_commands(self, env_ids)
        if len(env_ids) == 0:
            return

        jump_terrain_mask = self._get_jump_terrain_mask(env_ids)
        eligible_env_ids = env_ids[jump_terrain_mask]
        if len(eligible_env_ids) == 0:
            return

        jump_prob = float(self.cfg.commands.jump_command_prob)
        jump_mask = torch.rand(len(eligible_env_ids), device=self.device) < jump_prob
        jump_env_ids = eligible_env_ids[jump_mask]
        if len(jump_env_ids) == 0:
            return

        levels = self.command_dist_levels[jump_env_ids]
        rand_u = torch.rand((len(jump_env_ids), 3), device=self.device)

        dx = self.curriculum_dx_l[levels] + rand_u[:, 0] * (self.curriculum_dx_h[levels] - self.curriculum_dx_l[levels])
        dy = self.curriculum_dy_l[levels] + rand_u[:, 1] * (self.curriculum_dy_h[levels] - self.curriculum_dy_l[levels])
        dz = self.curriculum_dz_l[levels] + rand_u[:, 2] * (self.curriculum_dz_h[levels] - self.curriculum_dz_l[levels])

        self.commands[jump_env_ids, self.jump_dx_command_idx] = dx
        self.commands[jump_env_ids, self.jump_dy_command_idx] = dy
        self.commands[jump_env_ids, self.jump_dz_command_idx] = dz
        self.commands[jump_env_ids, self.jump_trigger_command_idx] = self.cfg.commands.active_jump_command
        self.commands[jump_env_ids, self.body_height_command_idx] = self.cfg.commands.low_body_height_command

    def reset_idx(self, env_ids):
        if len(env_ids) > 0:
            self._update_performance_curriculum(env_ids)
        super().reset_idx(env_ids)

    def _update_performance_curriculum(self, env_ids):
        cfg_rew = self.cfg.rewards
        if len(env_ids) == 0:
            return

        landed = self.has_jumped[env_ids]
        if not landed.any():
            return

        target_xy = self._get_jump_target_xy()[env_ids]
        landing = self.landing_poses[env_ids]
        land_err = torch.norm(landing - target_xy, dim=1)

        max_h = self.max_height[env_ids] - self.jump_start_height[env_ids]
        valid_height = max_h > cfg_rew.curriculum_success_min_apex_height
        valid_att = (torch.abs(self.rpy[env_ids, 0]) < self.cfg.env.jump_land_roll_threshold) & (
            torch.abs(self.rpy[env_ids, 1]) < self.cfg.env.jump_land_pitch_threshold
        )
        valid_err = land_err < cfg_rew.curriculum_success_landing_sigma
        success = (landed & valid_height & valid_att & valid_err).float()

        ptr = self.success_hist_ptr[env_ids]
        self.success_hist[env_ids, ptr] = success
        self.success_hist_ptr[env_ids] = (ptr + 1) % self.curriculum_window
        self.success_hist_count[env_ids] = torch.clamp(self.success_hist_count[env_ids] + 1, max=self.curriculum_window)

        ready = self.success_hist_count[env_ids] >= self.curriculum_min_valid_episodes
        if not ready.any():
            return

        ready_env_ids = env_ids[ready]
        counts = self.success_hist_count[ready_env_ids].float().clamp(min=1.0)
        rates = self.success_hist[ready_env_ids].sum(dim=1) / counts

        up_mask = rates > self.curriculum_success_high
        down_mask = rates < self.curriculum_success_low

        if up_mask.any():
            ids = ready_env_ids[up_mask]
            self.command_dist_levels[ids] = torch.clamp(self.command_dist_levels[ids] + 1, max=self.curriculum_max_level)
        if down_mask.any():
            ids = ready_env_ids[down_mask]
            self.command_dist_levels[ids] = torch.clamp(self.command_dist_levels[ids] - 1, min=0)

    def compute_reward(self):
        if not getattr(self.cfg.rewards, "only_positive_rewards_ji22_style", False):
            return super().compute_reward()

        sigma = float(self.cfg.rewards.ji22_neg_reward_sigma)
        sigma = max(sigma, 1e-5)

        self.rew_buf[:] = 0.0
        rew_pos = torch.zeros_like(self.rew_buf)
        rew_neg = torch.zeros_like(self.rew_buf)

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            raw_rew = self.reward_functions[i]()
            rew = raw_rew * self.reward_scales.get(name, 0.0)
            if name in self.reward_curriculum_scales:
                rew *= self.reward_curriculum_scales[name]

            rew_pos += torch.clamp(rew, min=0.0)
            rew_neg += torch.clamp(rew, max=0.0)
            self.episode_sums[name] += rew

        self.rew_buf[:] = rew_pos * torch.exp(rew_neg / sigma)

        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _reward_task_pos(self):
        landed_now = self.jump_landed_this_step
        if not landed_now.any():
            return torch.zeros(self.num_envs, device=self.device)

        sigma = self.cfg.rewards.task_pos_sigma
        target_xy = self._get_jump_target_xy()
        err_sq = torch.sum(torch.square(self.landing_poses - target_xy), dim=1)
        rew = torch.exp(-err_sq / sigma)
        return rew * landed_now.float()

    def _reward_task_ori(self):
        landed_now = self.jump_landed_this_step
        if not landed_now.any():
            return torch.zeros(self.num_envs, device=self.device)

        sigma = self.cfg.rewards.task_ori_sigma
        ori_err = torch.square(self.rpy[:, 0]) + torch.square(self.rpy[:, 1])
        rew = torch.exp(-ori_err / sigma)
        return rew * landed_now.float()

    def _reward_task_max_height(self):
        landed_now = self.jump_landed_this_step
        if not landed_now.any():
            return torch.zeros(self.num_envs, device=self.device)

        sigma = self.cfg.rewards.task_max_height_sigma
        target_h = self.commands[:, self.jump_dz_command_idx]
        jump_h = self.max_height - self.jump_start_height
        err_sq = torch.square(jump_h - target_h)
        rew = torch.exp(-err_sq / sigma)
        return rew * landed_now.float()
