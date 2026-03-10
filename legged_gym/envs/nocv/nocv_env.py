import torch

from legged_gym.envs.go2.go2_env import Go2Robot


class NoCVRobot(Go2Robot):
    def _init_buffers(self):
        super()._init_buffers()

        self.body_height_command_idx = self.cfg.commands.body_height_command_idx
        if self.body_height_command_idx >= self.cfg.commands.num_commands:
            raise ValueError(
                f"body_height_command_idx={self.body_height_command_idx} is out of range for "
                f"num_commands={self.cfg.commands.num_commands}"
            )

        self.body_height_command_threshold = self.cfg.commands.body_height_command_threshold
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.cfg.commands.body_height_command_obs_scale,
            ],
            device=self.device,
            requires_grad=False,
        )
        self.low_height_terrain_ids = torch.tensor(
            self.cfg.commands.low_height_terrain_ids,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:10] = 0.0
        noise_vec[10:10+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[10+self.num_actions:10+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[10+2*self.num_actions:10+3*self.num_actions] = 0.0

        return noise_vec

    def _get_body_height_command(self):
        return self.commands[:, self.body_height_command_idx:self.body_height_command_idx+1]

    def _get_low_height_terrain_mask(self, env_ids=None):
        if env_ids is None:
            terrain_ids = getattr(self, "terrain_ids", None)
        else:
            terrain_ids = getattr(self, "terrain_ids", None)
            if terrain_ids is not None:
                terrain_ids = terrain_ids[env_ids]

        if terrain_ids is None or self.low_height_terrain_ids.numel() == 0:
            if env_ids is None:
                return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        return (terrain_ids.unsqueeze(1) == self.low_height_terrain_ids.unsqueeze(0)).any(dim=1)

    def _get_commanded_base_height_target(self):
        targets = torch.full(
            (self.num_envs,),
            self.cfg.rewards.base_height_target,
            dtype=torch.float,
            device=self.device,
        )
        low_height_mask = self._get_low_height_terrain_mask() & (
            self._get_body_height_command().squeeze(1) > self.body_height_command_threshold
        )
        targets[low_height_mask] = self.cfg.rewards.low_base_height_target
        return targets

    def compute_observations(self):
        command_obs = torch.cat(
            (
                self.commands[:, :3] * self.commands_scale[:3],
                self._get_body_height_command() * self.commands_scale[3:4],
            ),
            dim=-1,
        )
        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            command_obs,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)

        heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
            -1,
            1.0,
        ) * self.obs_scales.height_measurements

        self.privileged_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            command_obs,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) * 1e-3,
            self.torques / self.torque_limits,
            (self.last_dof_vel - self.dof_vel) / self.dt * 1e-4,
            heights,
        ), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return

        self.commands[env_ids, self.body_height_command_idx] = self.cfg.commands.normal_body_height_command
        eligible_mask = self._get_low_height_terrain_mask(env_ids)
        eligible_env_ids = env_ids[eligible_mask]
        if len(eligible_env_ids) == 0:
            return

        low_height_mask = torch.rand(len(eligible_env_ids), device=self.device) < self.cfg.commands.low_height_command_prob
        low_height_env_ids = eligible_env_ids[low_height_mask]
        if len(low_height_env_ids) > 0:
            self.commands[low_height_env_ids, self.body_height_command_idx] = self.cfg.commands.low_body_height_command

    def _reward_correct_base_height(self):
        base_height = self._get_base_height()
        target_height = self._get_commanded_base_height_target()
        return torch.square(base_height - target_height)