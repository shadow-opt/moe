import torch

from legged_gym.envs.go2.go2_env import Go2Robot


class NoCVRobot(Go2Robot):
    """NoCV 实验环境。

    该环境继承自 `Go2Robot`，只在少量关键点上覆写主线行为，目标是：
    1. 在内部 command buffer 中新增第 5 维 `body_height_mode`；
    2. 继续预留 jump 任务使用的 `jump_dx / jump_dy / jump_dz / jump_trigger`；
    3. 让这些维度与 jump 任务保持一致地进入 actor / privileged observation；
    4. 对于 NoCV 任务本身，jump 相关输入恒为 0；
    5. 奖励中的 base height target 随 command 档位切换。

    当前约定的 command 语义为：
    - `commands[:, 0]`: `lin_vel_x`
    - `commands[:, 1]`: `lin_vel_y`
    - `commands[:, 2]`: `ang_vel_yaw`
    - `commands[:, 3]`: `heading`（保持与基类兼容）
    - `commands[:, 4]`: `body_height_mode`
    - `commands[:, 5:9]`: `jump_dx / jump_dy / jump_dz / jump_trigger`（NoCV 中恒为 0）
    """

    def _init_buffers(self):
        """在基类 buffer 初始化后，补充 NoCV 自己的 command 相关缓存。

        这里最重要的事情有两件：
        1. 校验新增高度命令所在的槽位索引是否合法；
          2. 把 `commands_scale` 从主线 Go2 的 3 维扩成 8 维，
              使 observation 中显式使用的 command 段变为：
              `[x_vel, y_vel, yaw_vel, body_height_mode, jump_dx, jump_dy, jump_dz, jump_trigger]`。

        注意这里没有把 `heading` 送进 actor/critic obs；
        因为主线 Go2 本身就只显式喂前 3 个 locomotion command。
        NoCV/jump 共用同一份显式 command 布局，但 NoCV 默认把 jump 部分置零。
        """
        super()._init_buffers()

        self.body_height_command_idx = self.cfg.commands.body_height_command_idx
        self.jump_dx_command_idx = self.cfg.commands.jump_dx_command_idx
        self.jump_dy_command_idx = self.cfg.commands.jump_dy_command_idx
        self.jump_dz_command_idx = self.cfg.commands.jump_dz_command_idx
        self.jump_trigger_command_idx = self.cfg.commands.jump_trigger_command_idx
        if self.body_height_command_idx >= self.cfg.commands.num_commands:
            raise ValueError(
                f"body_height_command_idx={self.body_height_command_idx} is out of range for "
                f"num_commands={self.cfg.commands.num_commands}"
            )

        command_indices = {
            "body_height_command_idx": self.body_height_command_idx,
            "jump_dx_command_idx": self.jump_dx_command_idx,
            "jump_dy_command_idx": self.jump_dy_command_idx,
            "jump_dz_command_idx": self.jump_dz_command_idx,
            "jump_trigger_command_idx": self.jump_trigger_command_idx,
        }
        for name, idx in command_indices.items():
            if idx < 0 or idx >= self.cfg.commands.num_commands:
                raise ValueError(
                    f"{name}={idx} is out of range for num_commands={self.cfg.commands.num_commands}"
                )

        if not (
            self.body_height_command_idx
            < self.jump_dx_command_idx
            < self.jump_dy_command_idx
            < self.jump_dz_command_idx
            < self.jump_trigger_command_idx
        ):
            raise ValueError(
                "Expected command layout to be ordered as "
                "body_height < jump_dx < jump_dy < jump_dz < jump_trigger"
            )

        if len(self.cfg.commands.jump_command_obs_scale) != 4:
            raise ValueError(
                "jump_command_obs_scale must contain 4 values: "
                "[jump_dx, jump_dy, jump_dz, jump_trigger]"
            )

        if len(self.cfg.commands.low_height_command_velocity_scale) != 3:
            raise ValueError(
                "low_height_command_velocity_scale must contain 3 values: "
                "[lin_vel_x, lin_vel_y, ang_vel_yaw]"
            )

        self.body_height_command_threshold = self.cfg.commands.body_height_command_threshold
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.cfg.commands.body_height_command_obs_scale,
                self.cfg.commands.jump_command_obs_scale[0],
                self.cfg.commands.jump_command_obs_scale[1],
                self.cfg.commands.jump_command_obs_scale[2],
                self.cfg.commands.jump_command_obs_scale[3],
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
        self.low_height_command_velocity_scale = torch.tensor(
            self.cfg.commands.low_height_command_velocity_scale,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    def _get_noise_scale_vec(self, cfg):
        """构造 NoCV 的 actor observation 噪声向量。

        与 `Go2Robot` 的差别仅在于 command 段从 3 维变成 8 维：
        - `[6:14]` 对应显式 command observation
        - 其中新增的高度档位 / jump command 与原有 command 一样，不加噪声

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
        noise_vec[6:14] = 0.0
        noise_vec[14:14+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[14+self.num_actions:14+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[14+2*self.num_actions:14+3*self.num_actions] = 0.0

        return noise_vec

    def _get_body_height_command(self):
        """取出新增的 body-height command，shape = `[N, 1]`。

        保持 `[N, 1]` 而不是压成 `[N]`，是为了方便直接参与
        observation 拼接与广播运算。
        """
        return self.commands[:, self.body_height_command_idx:self.body_height_command_idx+1]

    def _get_jump_command_obs(self):
        return torch.cat(
            (
                self._get_body_height_command() * self.commands_scale[3:4],
                self.commands[:, self.jump_dx_command_idx:self.jump_dx_command_idx+1] * self.commands_scale[4:5],
                self.commands[:, self.jump_dy_command_idx:self.jump_dy_command_idx+1] * self.commands_scale[5:6],
                self.commands[:, self.jump_dz_command_idx:self.jump_dz_command_idx+1] * self.commands_scale[6:7],
                self.commands[:, self.jump_trigger_command_idx:self.jump_trigger_command_idx+1] * self.commands_scale[7:8],
            ),
            dim=-1,
        )

    def _get_command_obs(self):
        return torch.cat(
            (
                self.commands[:, :3] * self.commands_scale[:3],
                self._get_jump_command_obs(),
            ),
            dim=-1,
        )

    def _get_jump_state_obs(self):
        return torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

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

        self.commands[env_ids, :3] *= self.low_height_command_velocity_scale.unsqueeze(0)

    def compute_observations(self):
        """构造 NoCV 的 actor obs 与 privileged obs。

        相比主线 `Go2Robot`，唯一结构性变化是把显式 command 段从 3 维改成 8 维：
        - 原有 3 维 locomotion command：`x / y / yaw`
        - 新增 1 维 body-height mode
        - 额外预留 4 维 jump command（NoCV 中恒为 0）

        因此：
        - actor obs: 45 -> 50
        - privileged obs: 263 -> 271

        这里刻意没有把第 4 维 `heading` 拼进 obs，
        以保持与主线 Go2 的观测风格一致：
        actor/critic 显式关注的是可直接用于控制的速度命令和高度档位，
        而不是内部 heading 目标本身。
        """
        command_obs = self._get_command_obs()
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
            self._get_jump_state_obs(),
        ), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _resample_commands(self, env_ids):
        """在基类 locomotion command 重采样后，追加 NoCV 的高度档位采样。

        处理顺序是：
        1. 先调用基类，完成 `x / y / yaw / heading` 的常规采样；
        2. 把新增的高度 command 默认重置为正常档位；
        3. 仅在允许的 terrain 上，按概率切换到低高度档位；
        4. 对被切到低高度档位的 env，再额外降低速度命令。

        这种写法的好处是：
        - 不需要侵入基类通用 command range schema；
        - `heading` 逻辑完全保持原样；
        - 新增逻辑只影响 NoCV 分支。
        """
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return

        # 每次重采样先回到默认档位，避免上一轮的特殊状态残留。
        self.commands[env_ids, self.body_height_command_idx] = self.cfg.commands.normal_body_height_command
        self.commands[env_ids, self.jump_dx_command_idx:self.jump_trigger_command_idx+1] = self.cfg.commands.normal_jump_command
        eligible_mask = self._get_low_height_terrain_mask(env_ids)
        eligible_env_ids = env_ids[eligible_mask]
        if len(eligible_env_ids) == 0:
            return

        # 只对允许的 terrain 按概率采样低高度档位。
        low_height_mask = torch.rand(len(eligible_env_ids), device=self.device) < self.cfg.commands.low_height_command_prob
        low_height_env_ids = eligible_env_ids[low_height_mask]
        if len(low_height_env_ids) > 0:
            self.commands[low_height_env_ids, self.body_height_command_idx] = self.cfg.commands.low_body_height_command
            self._apply_low_height_command_speed_limit(low_height_env_ids)

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