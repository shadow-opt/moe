import torch 

from legged_gym.envs.go2.go2_env import Go2Robot
from legged_gym.utils.isaacgym_utils import sample_disjoint_intervals, sample_single_interval

class WINRobot(Go2Robot):
    """
    当前约定的 command 语义为：
    - `commands[:, 0]`: `lin_vel_x`
    - `commands[:, 1]`: `lin_vel_y`
    - `commands[:, 2]`: `ang_vel_yaw`
    - `commands[:, 3]`: `heading`（保持与基类兼容）
    - `commands[:, 4]`: `body_height_mode`
    - `commands[:, 5]`: `stairs_mode`
    - `commands[:, 6]`: `gap_mode`
    """
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.num_one_step_obs = self.num_obs
        self.num_one_step_privileged_obs = self.num_privileged_obs

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, "low_speed_feet_air_time"):
            self.low_speed_feet_air_time[env_ids] = 0.0
            self.low_speed_last_contacts[env_ids] = False
        if hasattr(self, "foot_slip_last_contacts"):
            self.foot_slip_last_contacts[env_ids] = False
        



    def _init_buffers(self):
        super()._init_buffers()
        
        if self.commands.shape[1] < 7:
            raise RuntimeError(
                f"WIN requires at least 7 command dims, got {self.commands.shape[1]}"
            )
        
        self.body_height_command_idx = self.cfg.commands.body_height_command_idx
        if not 0 <= self.body_height_command_idx < self.commands.shape[1]:
            raise RuntimeError(
                f"body_height_command_idx={self.body_height_command_idx} is out of range for "
                f"commands with shape {self.commands.shape}"
            )
        self.body_height_command_threshold = self.cfg.commands.body_height_command_threshold
        


        self.special_terrain_options_5 = self.cfg.commands.special_terrain_options_5
        self.special_terrain_options_6 = self.cfg.commands.special_terrain_options_6
        self.special_terrain_command_threshold = (
            self.cfg.commands.normal_terrain_command
            + 0.5 * (self.cfg.commands.special_terrain_command - self.cfg.commands.normal_terrain_command)
        )
        self.has_distinct_special_terrain_command = (
            self.cfg.commands.special_terrain_command != self.cfg.commands.normal_terrain_command
        )

        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.cfg.commands.body_height_command_obs_scale,
                self.cfg.commands.special_terrain_command_obs_scale,
                self.cfg.commands.special_terrain_command_obs_scale,
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
        self.special_terrain_5_ids = torch.tensor(
            self.cfg.commands.special_terrain_options_5,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.special_terrain_6_ids = torch.tensor(
            self.cfg.commands.special_terrain_options_6,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

        self.low_height_command_velocity_scale = torch.tensor(
            self.cfg.commands.low_height_command_velocity_scale,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        if self.low_height_command_velocity_scale.numel() != 3:
            raise RuntimeError(
                "low_height_command_velocity_scale must contain exactly "
                f"3 values for x/y/yaw, got {self.low_height_command_velocity_scale}"
            )

        self.foot_slip_deadzone = float(getattr(self.cfg.rewards, "foot_slip_deadzone", 0.0))
        if self.foot_slip_deadzone < 0.0:
            raise RuntimeError(f"foot_slip_deadzone must be non-negative, got {self.foot_slip_deadzone}")
        self.foot_slip_excluded_terrain_ids = torch.tensor(
            getattr(self.cfg.rewards, "foot_slip_excluded_terrain_ids", []),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        
    def _get_noise_scale_vec(self, cfg):
        """构造 WIN 的 actor observation 噪声向量。

        与 `Go2Robot` 的差别仅在于 command 段从 3 维变成 6 维：
        - `[6:12]` 对应显式 command observation
        - 其中新增的高度档位 / terrain mode command 与原有 command 一样，不加噪声

        这样做的原因是：
        command 属于任务给定目标，不是传感器测量值；
        对它加噪通常只会增加学习难度，而不会提升鲁棒性。
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:12] = 0.0
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0.0

        return noise_vec
    
    
    def _get_body_height_command(self):
        """取出新增的 body-height command，shape = `[N, 1]`。

        保持 `[N, 1]` 而不是压成 `[N]`，是为了方便直接参与
        observation 拼接与广播运算。
        """
        return self.commands[:, self.body_height_command_idx:self.body_height_command_idx+1]


    def _get_low_height_terrain_mask(self, env_ids=None):
        """判断哪些 env 允许采样低高度档位。

        逻辑基于 `terrain_ids`，而不是 terrain column 或 level：
        - `1`: slope
        - `2`: rough_slope
        - `8`: flat

        如果当前环境没有 rough-terrain 的 `terrain_ids` 语义，
        则直接返回全 False，表示不启用低高度档位。
        """
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



    def _get_special_terrain_5_mask(self, env_ids=None):
        """判断哪些 env 处于 stairs up/down/obstacles 5 上。"""
        if env_ids is None:
            terrain_ids = getattr(self, "terrain_ids", None)
        else:
            terrain_ids = getattr(self, "terrain_ids", None)
            if terrain_ids is not None:
                terrain_ids = terrain_ids[env_ids]

        if terrain_ids is None or self.special_terrain_5_ids.numel() == 0:
            if env_ids is None:
                return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        return (terrain_ids.unsqueeze(1) == self.special_terrain_5_ids.unsqueeze(0)).any(dim=1)

    def _get_special_terrain_6_mask(self, env_ids=None):
        """判断哪些 env 处于 stones 6 上。"""
        if env_ids is None:
            terrain_ids = getattr(self, "terrain_ids", None)
        else:
            terrain_ids = getattr(self, "terrain_ids", None)
            if terrain_ids is not None:
                terrain_ids = terrain_ids[env_ids]

        if terrain_ids is None or self.special_terrain_6_ids.numel() == 0:
            if env_ids is None:
                return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        return (terrain_ids.unsqueeze(1) == self.special_terrain_6_ids.unsqueeze(0)).any(dim=1)

    def _get_commanded_base_height_target(self):
        """根据当前 command 档位，生成每个 env 的目标机身高度。

        默认 target 是 `cfg.rewards.base_height_target`；
        只有同时满足以下两个条件时，才切到更低的 target：
        1. 当前 env 位于允许低高度的 terrain；
        2. 第 5 维 command 已被采样为低高度档位。

        这样可以避免其他地形被错误地套用低高度奖励目标。
        """
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

    def _get_is_low_height_command_mask(self):
        """返回当前哪些 env 正在执行低高度特殊指令。

        这个 mask 同时要求：
        1. 地形属于允许低高度的类型；
        2. 第 5 维 command 已切到低高度档位。

        它用于把“常规 reward 路径”和“特殊指令 reward 路径”清晰分开：
        常规状态完全保持主线不变；特殊指令状态则改用新的低高度目标。
        """
        return self._get_low_height_terrain_mask() & (
            self._get_body_height_command().squeeze(1) > self.body_height_command_threshold
        )


    # useless but reserved for future use
    def _is_special_terrain_command(self, command_values):
        """按 normal/special 的相对大小判断特殊 terrain command 是否激活。"""
        if not self.has_distinct_special_terrain_command:
            return torch.zeros_like(command_values, dtype=torch.bool)
        if self.cfg.commands.special_terrain_command > self.cfg.commands.normal_terrain_command:
            return command_values > self.special_terrain_command_threshold
        return command_values < self.special_terrain_command_threshold

    def _get_is_special_terrain_5_command_mask(self):
        """返回当前哪些 env 正在执行 stairs up/down/obstacles 5 的特殊指令。"""
        return self._get_special_terrain_5_mask() & self._is_special_terrain_command(self.commands[:, 5])
    # useless but reserved for future use
    def _get_is_special_terrain_6_command_mask(self):
        """返回当前哪些 env 正在执行 stones 6 的特殊指令。"""
        return self._get_special_terrain_6_mask() & self._is_special_terrain_command(self.commands[:, 6])
    
    

    def _apply_low_height_command_speed_limit(self, env_ids):
        """在低高度档位下适度压低速度命令。

        这里不是重新采样 command range，而是对已经采样出的
        `x/y/yaw` 命令做一次比例缩放。

        这么做的好处是：
        - 实现简单，且不破坏基类已有的 command curriculum；
        - 能保留原始命令方向与相对比例；
        - 可以降低“低姿态 + 高速”组合带来的学习难度。
        """
        if len(env_ids) == 0:
            return

        velocity_dims = 2 if self.cfg.commands.heading_command else 3
        self.commands[env_ids, :velocity_dims] *= self.low_height_command_velocity_scale[:velocity_dims].unsqueeze(0)

    def _apply_low_height_heading_yaw_limit(self):
        """heading mode 下，父类每步重算 yaw 后再补一次低高度 yaw 限速。"""
        if not self.cfg.commands.heading_command:
            return

        low_height_mask = self._get_is_low_height_command_mask() & ~self.stop_heading
        if low_height_mask.any():
            self.commands[low_height_mask, 2] *= self.low_height_command_velocity_scale[2]

    def _apply_monotonic_command_sampling(self, env_ids):
        """Project some sampled commands to pure x, pure y, or pure yaw commands."""
        prob = getattr(self.cfg.commands, "monotonic_command_prob", 0.0)
        if prob <= 0.0 or len(env_ids) == 0:
            return

        monotonic_mask = torch.rand(len(env_ids), device=self.device) < prob
        monotonic_env_ids = env_ids[monotonic_mask]
        if len(monotonic_env_ids) == 0:
            return

        type_probs = torch.tensor(
            getattr(self.cfg.commands, "monotonic_command_type_probs", [1.0, 0.0, 0.0]),
            dtype=torch.float,
            device=self.device,
        )
        if type_probs.numel() != 3 or torch.sum(type_probs) <= 0.0:
            raise RuntimeError(
                "monotonic_command_type_probs must contain three positive-sum values "
                "for x/y/yaw command sampling"
            )
        type_probs = type_probs / torch.sum(type_probs)
        type_ids = torch.multinomial(type_probs, len(monotonic_env_ids), replacement=True)

        old_commands = self.commands[monotonic_env_ids, :3].clone()
        self.commands[monotonic_env_ids, :3] = 0.0

        x_mask = type_ids == 0
        y_mask = type_ids == 1
        yaw_mask = type_ids == 2
        if x_mask.any():
            ids = monotonic_env_ids[x_mask]
            self.commands[ids, 0] = old_commands[x_mask, 0]
        if y_mask.any():
            ids = monotonic_env_ids[y_mask]
            self.commands[ids, 1] = old_commands[y_mask, 1]
        if yaw_mask.any():
            ids = monotonic_env_ids[yaw_mask]
            self.commands[ids, 2] = old_commands[yaw_mask, 2]

    def _apply_min_abs_lin_vel_x_by_terrain(self, env_ids):
        min_abs_by_terrain = getattr(self.cfg.commands, "min_abs_lin_vel_x_by_terrain", {})
        if not min_abs_by_terrain or len(env_ids) == 0 or not hasattr(self, "terrain_ids"):
            return

        for terrain_id, min_abs_lin_vel_x in min_abs_by_terrain.items():
            min_abs_lin_vel_x = float(min_abs_lin_vel_x)
            if min_abs_lin_vel_x <= 0.0:
                continue

            terrain_env_ids = env_ids[self.terrain_ids[env_ids] == int(terrain_id)]
            if len(terrain_env_ids) == 0:
                continue

            abs_lin_vel_x = torch.abs(self.commands[terrain_env_ids, 0])
            small_nonzero_mask = (abs_lin_vel_x > 0.0) & (abs_lin_vel_x < min_abs_lin_vel_x)
            if not small_nonzero_mask.any():
                continue

            small_nonzero_env_ids = terrain_env_ids[small_nonzero_mask]
            terrain_command_ranges = self.cfg.commands.terrain_max_command_ranges[int(terrain_id)]
            min_abs_tensor = torch.full((len(small_nonzero_env_ids),), min_abs_lin_vel_x, device=self.device)
            cap_lower = torch.full_like(min_abs_tensor, terrain_command_ranges["lin_vel_x"][0])
            cap_upper = torch.full_like(min_abs_tensor, terrain_command_ranges["lin_vel_x"][1])
            current_lower = self.env_command_ranges["lin_vel_x"][small_nonzero_env_ids, 0]
            current_upper = self.env_command_ranges["lin_vel_x"][small_nonzero_env_ids, 1]

            lower = torch.maximum(torch.minimum(current_lower, -min_abs_tensor), cap_lower)
            upper = torch.minimum(torch.maximum(current_upper, min_abs_tensor), cap_upper)
            neg_available = lower <= -min_abs_lin_vel_x
            pos_available = upper >= min_abs_lin_vel_x
            valid_nonzero_mask = neg_available | pos_available
            if valid_nonzero_mask.any():
                valid_env_ids = small_nonzero_env_ids[valid_nonzero_mask]
                valid_lower = lower[valid_nonzero_mask]
                valid_upper = upper[valid_nonzero_mask]
                valid_min_abs = min_abs_tensor[valid_nonzero_mask]
                width_neg = torch.nn.functional.relu(-valid_min_abs - valid_lower)
                width_pos = torch.nn.functional.relu(valid_upper - valid_min_abs)
                interval_mask = width_neg + width_pos > 1e-6
                if interval_mask.any():
                    interval_env_ids = valid_env_ids[interval_mask]
                    self.commands[interval_env_ids, 0] = sample_disjoint_intervals(
                        interval_env_ids,
                        valid_min_abs[interval_mask],
                        valid_lower[interval_mask],
                        valid_upper[interval_mask],
                        self.device,
                    )
                point_mask = ~interval_mask
                if point_mask.any():
                    point_env_ids = valid_env_ids[point_mask]
                    point_neg_available = valid_lower[point_mask] <= -valid_min_abs[point_mask]
                    point_pos_available = valid_upper[point_mask] >= valid_min_abs[point_mask]
                    choose_pos = torch.rand(len(point_env_ids), device=self.device) < 0.5
                    choose_pos = torch.where(point_neg_available & point_pos_available, choose_pos, point_pos_available)
                    self.commands[point_env_ids, 0] = torch.where(
                        choose_pos,
                        valid_min_abs[point_mask],
                        -valid_min_abs[point_mask],
                    )
            if (~valid_nonzero_mask).any():
                self.commands[small_nonzero_env_ids[~valid_nonzero_mask], 0] = 0.0

    def _apply_flat_low_speed_command_sampling(self, env_ids):
        prob = getattr(self.cfg.commands, "flat_low_speed_command_prob", 0.0)
        terrain_ids_cfg = getattr(self.cfg.commands, "flat_low_speed_terrain_ids", [])
        if prob <= 0.0 or len(env_ids) == 0 or not terrain_ids_cfg or not hasattr(self, "terrain_ids"):
            return

        target_terrain_ids = torch.tensor(terrain_ids_cfg, dtype=torch.long, device=self.device)
        target_mask = (self.terrain_ids[env_ids].unsqueeze(1) == target_terrain_ids.unsqueeze(0)).any(dim=1)
        if not target_mask.any():
            return

        candidate_env_ids = env_ids[target_mask]
        low_speed_mask = torch.rand(len(candidate_env_ids), device=self.device) < prob
        low_speed_env_ids = candidate_env_ids[low_speed_mask]
        if len(low_speed_env_ids) == 0:
            return

        ranges = getattr(self.cfg.commands, "flat_low_speed_command_ranges", {})
        for dim, command_idx in (("lin_vel_x", 0), ("lin_vel_y", 1), ("ang_vel_yaw", 2)):
            if dim not in ranges:
                continue
            lower = torch.maximum(
                torch.full((len(low_speed_env_ids),), ranges[dim][0], device=self.device),
                self.env_command_ranges[dim][low_speed_env_ids, 0],
            )
            upper = torch.minimum(
                torch.full((len(low_speed_env_ids),), ranges[dim][1], device=self.device),
                self.env_command_ranges[dim][low_speed_env_ids, 1],
            )
            self.commands[low_speed_env_ids, command_idx] = sample_single_interval(
                low_speed_env_ids,
                lower,
                upper,
                self.device,
            )

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._apply_low_height_heading_yaw_limit()


    def compute_observations(self):
        """构造 WIN 的 actor obs 与 privileged obs。

        相比主线 `Go2Robot`，唯一结构性变化是把显式 command 段从 3 维改成 6 维：
        - 原有 3 维 locomotion command：`x / y / yaw`
        - 新增 1 维 body-height mode
        - 新增 2 维 terrain-specific modes（stairs / stones）

        因此：
        - actor obs: 45 -> 48
        - privileged obs: 263 -> 266

        这里刻意没有把第 4 维 `heading` 拼进 obs，
        以保持与主线 Go2 的观测风格一致：
        actor/critic 显式关注的是可直接用于控制的速度命令和高度档位，
        而不是内部 heading 目标本身。
        """
        # command_obs = self._get_command_obs()

        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale[:3],
            self.commands[:, 4:] * self.commands_scale[3:], # heading 不进入 obs
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
            self.commands[:, :3] * self.commands_scale[:3],
            self.commands[:, 4:] * self.commands_scale[3:], # heading 不进入 obs
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) * 1e-3,
            self.torques / self.torque_limits,
            (self.last_dof_vel - self.dof_vel) / self.dt * 1e-4,
            heights,
            # WIN 不使用 jump state 观测占位
        ), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        self.stop_heading[env_ids] = False

        if len(self.cfg.commands.command_range_curriculum):
            current_iter = self.common_step_counter // self.num_steps_per_env
            for i in range(len(self.cfg.commands.command_range_curriculum) - 1, -1, -1):
                cfg = self.cfg.commands.command_range_curriculum[i]
                if current_iter >= cfg["iter"]:
                    self.command_ranges["lin_vel_x"] = cfg["lin_vel_x"]
                    self.command_ranges["lin_vel_y"] = cfg["lin_vel_y"]
                    self.command_ranges["ang_vel_yaw"] = cfg["ang_vel_yaw"]
                    self.command_ranges["heading"] = cfg["heading"]
                    self.max_lin_vel = max(
                        abs(self.command_ranges["lin_vel_x"][0]),
                        abs(self.command_ranges["lin_vel_x"][1]),
                        abs(self.command_ranges["lin_vel_y"][0]),
                        abs(self.command_ranges["lin_vel_y"][1]),
                    )
                    self.cfg.commands.command_range_curriculum.pop(i)
                    self._update_env_command_ranges()
                    print(f"Command range updated at iter {current_iter}: {self.command_ranges}")

        remaining_dist = torch.clip(
            0.625 * self.cfg.terrain.terrain_length
            - torch.norm(self.commands_xy_accumulation[env_ids], dim=1) * self.cfg.commands.resampling_time,
            0.0,
        )
        self.commands_resampling_step[env_ids] = self.cfg.commands.resampling_time / self.dt

        if self.cfg.commands.dynamic_resample_commands:
            if ((self.max_episode_length - self.episode_length_buf[env_ids]) == 0).any():
                raise ValueError("Some envs have zero remaining episode length during command resampling")

            vel_low_bound = torch.clip(
                remaining_dist
                / ((self.max_episode_length - self.episode_length_buf[env_ids] + 1e-9) * self.dt),
                0.0,
            )
            self.commands[env_ids, 0] = sample_disjoint_intervals(
                env_ids,
                vel_low_bound,
                self.env_command_ranges["lin_vel_x"][env_ids, 0],
                self.env_command_ranges["lin_vel_x"][env_ids, 1],
                self.device,
            )
            self.commands[env_ids, 1] = sample_disjoint_intervals(
                env_ids,
                vel_low_bound,
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                self.device,
            )
            if self.cfg.commands.heading_command:
                r = torch.rand(len(env_ids), device=self.device)
                lower = self.env_command_ranges["heading"][env_ids, 0]
                upper = self.env_command_ranges["heading"][env_ids, 1]
                self.commands[env_ids, 3] = (upper - lower) * r + lower
            else:
                r = torch.rand(len(env_ids), device=self.device)
                lower = self.env_command_ranges["ang_vel_yaw"][env_ids, 0]
                upper = self.env_command_ranges["ang_vel_yaw"][env_ids, 1]
                self.commands[env_ids, 2] = (upper - lower) * r + lower
        else:
            self.commands[env_ids, 0] = sample_single_interval(
                env_ids,
                self.env_command_ranges["lin_vel_x"][env_ids, 0],
                self.env_command_ranges["lin_vel_x"][env_ids, 1],
                self.device,
            )
            self.commands[env_ids, 1] = sample_single_interval(
                env_ids,
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                self.device,
            )
            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = sample_single_interval(
                    env_ids,
                    self.env_command_ranges["heading"][env_ids, 0],
                    self.env_command_ranges["heading"][env_ids, 1],
                    self.device,
                )
            else:
                self.commands[env_ids, 2] = sample_single_interval(
                    env_ids,
                    self.env_command_ranges["ang_vel_yaw"][env_ids, 0],
                    self.env_command_ranges["ang_vel_yaw"][env_ids, 1],
                    self.device,
                )

            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        rand_prob = torch.rand(len(env_ids), device=self.device)
        min_prob, max_prob = 0.0, 0.0

        if self.limit_vel_prob > 0.0:
            max_prob += self.limit_vel_prob
            lim_mask = (rand_prob >= min_prob) * (rand_prob < max_prob)
            lim_env_ids = env_ids[lim_mask]
            if len(lim_env_ids) > 0:
                change_lim_env_ids = lim_env_ids
                if self.cfg.commands.limit_vel_invert_when_continuous:
                    was_limited = self.last_is_limit_vel[lim_env_ids]
                    invert_env_ids = lim_env_ids[was_limited]
                    self.commands[invert_env_ids, 0] *= -1.0
                    self.commands[invert_env_ids, 1] *= -1.0
                    self.commands[invert_env_ids, 2] *= -1.0
                    change_lim_env_ids = lim_env_ids[~was_limited]

                vel_idx = torch.randint(0, self.limit_vel_comb.shape[0], (len(change_lim_env_ids),), device=self.device)
                lin_vel_x_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 0] == -1,
                    self.env_command_ranges["lin_vel_x"][change_lim_env_ids, 0],
                    self.env_command_ranges["lin_vel_x"][change_lim_env_ids, 1],
                )
                lin_vel_x_lim[self.limit_vel_comb[vel_idx, 0] == 0] = 0.0
                lin_vel_y_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 1] == -1,
                    self.env_command_ranges["lin_vel_y"][change_lim_env_ids, 0],
                    self.env_command_ranges["lin_vel_y"][change_lim_env_ids, 1],
                )
                lin_vel_y_lim[self.limit_vel_comb[vel_idx, 1] == 0] = 0.0
                ang_vel_z_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 2] == -1,
                    self.env_command_ranges["ang_vel_yaw"][change_lim_env_ids, 0],
                    self.env_command_ranges["ang_vel_yaw"][change_lim_env_ids, 1],
                )
                ang_vel_z_lim[self.limit_vel_comb[vel_idx, 2] == 0] = 0.0
                self.commands[change_lim_env_ids, 0] = lin_vel_x_lim
                self.commands[change_lim_env_ids, 1] = lin_vel_y_lim
                self.commands[change_lim_env_ids, 2] = ang_vel_z_lim
                if self.cfg.commands.heading_command and self.cfg.commands.stop_heading_at_limit:
                    self.stop_heading[lim_env_ids] = True
                self.last_is_limit_vel[env_ids] = False
                self.last_is_limit_vel[lim_env_ids] = True
            else:
                self.last_is_limit_vel[env_ids] = False
            min_prob += self.limit_vel_prob

        zero_command_env_ids = env_ids[:0]
        if self.cfg.commands.zero_command_curriculum is not None:
            self.zero_command_proba = self.get_current_scale(self.cfg.commands.zero_command_curriculum)
        if self.zero_command_proba > 0.0:
            max_prob += self.zero_command_proba
            next_resampling_step = torch.clip(
                self.max_episode_length
                - self.episode_length_buf[env_ids]
                - (remaining_dist / (0.8 * self.max_lin_vel * self.dt + 1e-9)),
                min=0.0,
                max=self.cfg.commands.resampling_time / self.dt,
            )
            zero_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) * (next_resampling_step > 0.0)
            zero_command_env_ids = env_ids[zero_mask]
            if len(zero_command_env_ids) > 0:
                self.commands[zero_command_env_ids, :2] = 0.0
                self.commands_resampling_step[zero_command_env_ids] = next_resampling_step[zero_mask]
                if self.cfg.commands.limit_ang_vel_at_zero_command_prob > 0.0:
                    ang_vel_rand = torch.rand(len(zero_command_env_ids), device=self.device)
                    add_ang_mask = ang_vel_rand < self.cfg.commands.limit_ang_vel_at_zero_command_prob
                    add_ang_env_ids = zero_command_env_ids[add_ang_mask]
                    if len(add_ang_env_ids) > 0:
                        direction_rand = torch.rand(len(add_ang_env_ids), device=self.device)
                        self.commands[add_ang_env_ids, 2] = torch.where(
                            direction_rand < 0.5,
                            self.env_command_ranges["ang_vel_yaw"][add_ang_env_ids, 0],
                            self.env_command_ranges["ang_vel_yaw"][add_ang_env_ids, 1],
                        )
                        if self.cfg.commands.heading_command:
                            self.stop_heading[add_ang_env_ids] = True
            min_prob += self.zero_command_proba

        if self.cfg.init_state.turn_over and (self.turn_over_timer[env_ids] > 0).any():
            zero_mask = self.turn_over_timer[env_ids] > 0
            zero_env_ids = env_ids[zero_mask]
            self.commands[zero_env_ids, :3] = 0.0
            self.stop_heading[zero_env_ids] = True

        # 每次重采样先回到默认档位，避免上一轮的特殊状态残留。
        self.commands[env_ids, self.body_height_command_idx] = self.cfg.commands.normal_body_height_command
        self.commands[env_ids, 5] = self.cfg.commands.normal_terrain_command
        self.commands[env_ids, 6] = self.cfg.commands.normal_terrain_command

        low_height_env_ids = env_ids[:0]
        low_height_eligible_env_ids = env_ids[self._get_low_height_terrain_mask(env_ids)]
        if len(low_height_eligible_env_ids) > 0:
            # 只对允许的 terrain 按概率采样低高度档位。
            low_height_mask = torch.rand(len(low_height_eligible_env_ids), device=self.device) < self.cfg.commands.low_height_command_prob
            low_height_env_ids = low_height_eligible_env_ids[low_height_mask]
            if len(low_height_env_ids) > 0:
                self.commands[low_height_env_ids, self.body_height_command_idx] = self.cfg.commands.low_body_height_command
                self._apply_low_height_command_speed_limit(low_height_env_ids)
                stopped_heading_env_ids = low_height_env_ids[self.stop_heading[low_height_env_ids]]
                if self.cfg.commands.heading_command and len(stopped_heading_env_ids) > 0:
                    self.commands[stopped_heading_env_ids, 2] *= self.low_height_command_velocity_scale[2]

        flat_low_speed_candidate_mask = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        if len(low_height_env_ids) > 0:
            flat_low_speed_candidate_mask &= ~(
                env_ids.unsqueeze(1) == low_height_env_ids.unsqueeze(0)
            ).any(dim=1)
        if len(zero_command_env_ids) > 0:
            flat_low_speed_candidate_mask &= ~(
                env_ids.unsqueeze(1) == zero_command_env_ids.unsqueeze(0)
            ).any(dim=1)
        self._apply_flat_low_speed_command_sampling(env_ids[flat_low_speed_candidate_mask])

        monotonic_candidate_mask = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        if len(zero_command_env_ids) > 0:
            monotonic_candidate_mask &= ~(
                env_ids.unsqueeze(1) == zero_command_env_ids.unsqueeze(0)
            ).any(dim=1)
        self._apply_monotonic_command_sampling(env_ids[monotonic_candidate_mask])

        special_terrain_5_env_ids = env_ids[self._get_special_terrain_5_mask(env_ids)]
        if special_terrain_5_env_ids.numel() > 0:
            special_terrain_5_mask = torch.rand(len(special_terrain_5_env_ids), device=self.device) < self.cfg.commands.special_terrain_probs
            self.commands[special_terrain_5_env_ids[special_terrain_5_mask], 5] = self.cfg.commands.special_terrain_command

        special_terrain_6_env_ids = env_ids[self._get_special_terrain_6_mask(env_ids)]
        if special_terrain_6_env_ids.numel() > 0:
            special_terrain_6_mask = torch.rand(len(special_terrain_6_env_ids), device=self.device) < self.cfg.commands.special_terrain_probs
            self.commands[special_terrain_6_env_ids[special_terrain_6_mask], 6] = self.cfg.commands.special_terrain_command

        self._apply_min_abs_lin_vel_x_by_terrain(env_ids)
        self.commands_xy_accumulation[env_ids] += self.commands[env_ids, :2]




    def _reward_similar_to_default(self):
        """低高度特殊指令下关闭“全关节接近默认位姿”惩罚。"""
        penalty = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        low_height_mask = self._get_is_low_height_command_mask()
        penalty[low_height_mask] = 0.0
        return penalty

    def _reward_hip_to_default(self):
        """低高度特殊指令下关闭“hip 接近默认位姿”惩罚。"""
        hip_dof_indices = [0, 3, 6, 9]
        hip_pos = self.dof_pos[:, hip_dof_indices]
        default_hip_pos = self.default_dof_pos[:, hip_dof_indices]
        penalty = torch.sum(torch.abs(hip_pos - default_hip_pos), dim=1)
        low_height_mask = self._get_is_low_height_command_mask()
        penalty[low_height_mask] = 0.0
        return penalty

    def _reward_lateral_yaw_tracking_error(self):
        """低高度特殊指令下，对 y 速度和 yaw 角速度的命令偏差施加强惩罚。"""
        lateral_vel_error = self.base_lin_vel[:, 1] - self.commands[:, 1]
        yaw_vel_error = self.base_ang_vel[:, 2] - self.commands[:, 2]
        low_height_mask = self._get_is_low_height_command_mask()
        return (torch.square(lateral_vel_error) + torch.square(yaw_vel_error)) * low_height_mask.float()

    def _reward_hip_to_zero(self):
        """低高度纯 x 速度指令下，鼓励 4 个 hip 关节角靠近 0，减少横向张腿。"""
        hip_dof_indices = [0, 3, 6, 9]
        hip_pos = self.dof_pos[:, hip_dof_indices]
        low_height_mask = self._get_is_low_height_command_mask()
        pure_x_mask = (
            (torch.abs(self.commands[:, 0]) > 1e-3)
            & (torch.abs(self.commands[:, 1]) < 1e-3)
            & (torch.abs(self.commands[:, 2]) < 1e-3)
        )
        return torch.sum(torch.square(hip_pos), dim=1) * (low_height_mask & pure_x_mask).float()

    def _reward_correct_base_height(self):
        """按 command 档位切换目标高度的 base-height reward。

        这里沿用基类 `_get_base_height()` 的高度估计方式，
        但把固定 target 改成“条件 target”：
        - 正常档位：追踪常规 `base_height_target`
        - 低高度档位：追踪 `low_base_height_target`

        返回值保持为平方误差，便于继续复用主线配置中的
        `correct_base_height` reward scale。
        """
        base_height = self._get_base_height()
        target_height = self._get_commanded_base_height_target()
        return torch.square(base_height - target_height)

    def _reward_base_height(self):
        """兼容版 base-height reward。

        虽然当前主线配置默认主要使用 `_reward_correct_base_height()`，
        但这里仍同步覆写 `_reward_base_height()`，避免后续若打开该 reward 时：
        - 常规状态与主线一致；
        - 低高度特殊指令下不会继续错误追踪原始 `base_height_target`。
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        if not hasattr(self, 'last_contacts2'):
            self.last_contacts2 = torch.zeros_like(contact)
        contact_filt = torch.logical_or(contact, self.last_contacts2)
        self.last_contacts2 = contact
        feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        num_feet_contact = torch.sum(contact_filt, dim=1, keepdim=True).clamp(min=1.0)
        feet_contact_pos = (feet_pos * contact_filt.unsqueeze(-1)).sum(dim=1) / num_feet_contact
        base_pos = self.root_states[:, 0:3]
        delta_pos = feet_contact_pos - base_pos
        base_height = (delta_pos * self.projected_gravity).sum(1)
        target_height = self._get_commanded_base_height_target()
        rew = torch.square(base_height - target_height) * (contact_filt.sum(1) > 0)
        return rew

    def _reward_feet_regulation(self):
        """低高度特殊指令兼容的 feet-regulation reward。

        主线版本中，这一项使用固定的 `base_height_target` 来归一化脚抬高惩罚。
        如果直接沿用主线实现，那么在低高度特殊指令下，
        这里仍会隐式鼓励机器人维持原来的较高机身，和特殊指令目标冲突。

        本实现做法是：
        - 常规状态：完全保持主线公式不变；
        - 低高度特殊指令：只把归一化里的高度 target 切到更低值。

        这样可以保证“正常状态奖励不变”，同时“原高度目标不会在特殊指令下继续生效”。
        """
        base_height = self._get_base_height()
        feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        feet_xy_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:9]
        base_pos = self.root_states[:, 0:3].unsqueeze(1)
        delta_feet = feet_pos - base_pos
        feet2base_height = (delta_feet * self.projected_gravity.unsqueeze(1)).sum(-1)
        feet_height = torch.clamp(base_height.unsqueeze(1) - feet2base_height, min=0.0)

        target_height = torch.full(
            (self.num_envs,),
            self.cfg.rewards.base_height_target,
            dtype=torch.float,
            device=self.device,
        )
        low_height_mask = self._get_is_low_height_command_mask()
        target_height[low_height_mask] = self.cfg.rewards.low_base_height_target

        rew = (
            feet_xy_vel.pow(2).sum(-1)
            * torch.exp(-feet_height / (0.025 * target_height.unsqueeze(1)))
        ).sum(-1)
        return rew
    

    def _reward_straight_path(self):
        # 奖励直线路径
        walking_straight_envs = (self.commands[:, 0]> 0.7) & (torch.abs(self.commands[:, 1]) < 1e-3)
        # 实际上：命令小于0.2就会被裁剪为0
        lat_error = torch.abs(self.commands[:, 1] - self.base_lin_vel[:, 1])

        reward = torch.exp(-lat_error / self.cfg.rewards.tracking_sigma * 2.0)

        return reward * walking_straight_envs.float()

    def _reward_straight_path_deviation(self):
        # 惩罚直走命令下的侧向速度与偏航角速度
        walking_straight_envs = (self.commands[:, 0] > 0.5) & (torch.abs(self.commands[:, 1]) < 0.01) & (torch.abs(self.commands[:,2]) < 0.01)
        # V_y 与 omega_z 偏差越大惩罚越大
        lat_vel_error = torch.abs(self.base_lin_vel[:, 1])
        yaw_vel_error = torch.abs(self.base_ang_vel[:, 2])
        penalty = lat_vel_error + yaw_vel_error

        return penalty * walking_straight_envs.float()
    
        
    def _reward_foot_slip(self):
        """
        [脚底打滑惩罚]
        触地时如果脚有水平速度则惩罚
        """
        feet_xy_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:9]
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        if not hasattr(self, "foot_slip_last_contacts"):
            self.foot_slip_last_contacts = torch.zeros_like(contact)
        contact_filt = torch.logical_or(contact, self.foot_slip_last_contacts)
        self.foot_slip_last_contacts = contact

        foot_speed_norm = torch.relu(torch.norm(feet_xy_vel, dim=2) - self.foot_slip_deadzone)
        terrain_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if hasattr(self, "terrain_ids") and self.foot_slip_excluded_terrain_ids.numel() > 0:
            terrain_mask = ~(
                self.terrain_ids.unsqueeze(1) == self.foot_slip_excluded_terrain_ids.unsqueeze(0)
            ).any(dim=1)

        rew = foot_speed_norm * contact_filt * terrain_mask.unsqueeze(1)
        return torch.sum(rew, dim=1)

    def _reward_low_speed_feet_air_time(self):
        """Encourage clear stepping for low-speed translational commands."""
        xy_command_norm = torch.norm(self.commands[:, :2], dim=1)
        low_speed_mask = (xy_command_norm > 0.1) & (xy_command_norm < 0.5)
        low_speed_mask = low_speed_mask & (~self._get_is_low_height_command_mask())

        if not hasattr(self, "low_speed_feet_air_time"):
            self.low_speed_feet_air_time = torch.zeros_like(self.feet_air_time)
            self.low_speed_last_contacts = torch.zeros_like(self.last_contacts)

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.low_speed_last_contacts)
        self.low_speed_last_contacts = contact
        first_contact = (self.low_speed_feet_air_time > 0.0) * contact_filt
        self.low_speed_feet_air_time += self.dt
        air_time = torch.clamp(self.low_speed_feet_air_time, max=0.9)
        rew_air_time = torch.sum((air_time - 0.3) * first_contact, dim=1)
        self.low_speed_feet_air_time *= ~contact_filt
        return rew_air_time * low_speed_mask.float()

    def _reward_low_foot(self):
        """在 special_terrain_6 命令激活时，按悬空脚下探深度惩罚。"""
        feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        special_command_mask = self._get_is_special_terrain_6_command_mask().unsqueeze(1)
        margin = getattr(self.cfg.rewards, "low_foot_margin", 0.02)
        depth_scale = max(getattr(self.cfg.rewards, "low_foot_depth_scale", 0.05), 1e-6)
        depth = torch.relu(0.0 - feet_pos[:, :, 2] - margin)
        penalty = torch.square(depth / depth_scale)
        return torch.sum(penalty * (~contact).float() * special_command_mask.float(), dim=1)

    # def _reward_progress(self):
    #     """
    #     轻量的命令方向进展奖励。
    #     只鼓励沿当前平移指令方向的正向速度，避免台阶前“停住保平衡”。
    #     """
    #     cmd_xy = self.commands[:, :2]
    #     cmd_norm = torch.norm(cmd_xy, dim=1)
    #     move_cmd = cmd_norm > 0.1
    #     cmd_dir = cmd_xy / torch.clamp(cmd_norm.unsqueeze(1), min=1e-6)
    #     progress_speed = torch.sum(self.base_lin_vel[:, :2] * cmd_dir, dim=1)
    #     return torch.relu(progress_speed) * move_cmd.float()
