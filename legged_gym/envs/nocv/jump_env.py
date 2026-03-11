import torch
from isaacgym.torch_utils import torch_rand_float
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs.nocv.nocv_env import NoCVRobot


class JumpRobot(NoCVRobot):
    """基于 `NoCVRobot` 的可控跳跃任务。

    command 布局与 NoCVRobot 完全相同（9 维），
    但 jump 相关维度会被真正采样（而非恒 0）。

    跳跃状态机：idle → initialized → in_flight → landed
    - idle: jump_trigger ≤ threshold 或不在允许地形上
    - initialized: 首次检测到 active_jump_mask，记录起跳原点
    - in_flight: 四脚全部离地
    - landed: 飞行后再次检测到脚接触

    终止条件（在基类基础上追加）：
    1. 机身 z 低于 `reset_height`（坠落）
    2. 落地后 pitch 超限（翻滚）
    3. 落地后 roll 超限（侧翻）
    4. 落地后角速度过大（失稳旋转）
    """

    def _init_buffers(self):
        """初始化跳跃专用 buffer。

        在 NoCVRobot 已完成的 command/obs 基础上，额外创建：
        - 跳跃状态位图（initialized / was_in_flight / has_jumped 等）
        - 空中最高点、起跳高度、落地坐标等追踪量
        - 着地脚接触滤波器
        - 缓存的 terrain id tensor（避免 `_get_jump_terrain_mask` 每帧重建）
        """
        super()._init_buffers()

        cfg_commands = self.cfg.commands
        self.jump_command_threshold = cfg_commands.jump_command_threshold
        self.jump_locomotion_command_scale = torch.tensor(
            cfg_commands.jump_locomotion_command_scale,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # 缓存 jump_terrain_ids，避免 _get_jump_terrain_mask 每步重建
        self._jump_terrain_ids = torch.tensor(
            cfg_commands.jump_terrain_ids,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

        # ---- 跳跃状态 buffer ----
        self.jump_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.jump_last_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices),
            dtype=torch.bool, device=self.device, requires_grad=False,
        )
        self.jump_contact_filt = torch.zeros_like(self.jump_last_contacts)
        self.was_in_flight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.has_jumped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.jump_origin_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.jump_start_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.landing_poses = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.max_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.jump_landed_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

    # ------------------------------------------------------------------
    # jump trigger / terrain mask
    # ------------------------------------------------------------------

    def _get_jump_trigger_command(self):
        return self.commands[:, self.jump_trigger_command_idx:self.jump_trigger_command_idx+1]

    def _get_jump_terrain_mask(self, env_ids=None):
        """判断哪些 env 位于允许跳跃的 terrain 上。

        使用缓存的 `_jump_terrain_ids` tensor，避免每帧重建。
        """
        terrain_ids = getattr(self, "terrain_ids", None)
        if terrain_ids is None:
            if env_ids is None:
                return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            return torch.ones(len(env_ids), dtype=torch.bool, device=self.device)

        if env_ids is not None:
            terrain_ids = terrain_ids[env_ids]
        return (terrain_ids.unsqueeze(1) == self._jump_terrain_ids.unsqueeze(0)).any(dim=1)

    def _get_active_jump_mask(self):
        return self._get_jump_terrain_mask() & (self._get_jump_trigger_command().squeeze(1) > self.jump_command_threshold)

    def _clear_jump_trigger_after_landing(self, env_ids):
        """落地后关闭 jump trigger，让同一 episode 后半段进入恢复阶段。

        这样可以避免：
        1. tracking reward 在落地后一直被 jump mask 屏蔽；
        2. 训练与部署出现“部署会回到 IDLE，但训练一直保持 jump mode”的语义偏差。

        注意这里只清 `jump_trigger`，不立即清空 `jump_dx/jump_dy/jump_dz`，
        因为本步还需要用它们结算一次性的 apex / landing reward。
        """
        if len(env_ids) == 0:
            return
        self.commands[env_ids, self.jump_trigger_command_idx] = self.cfg.commands.normal_jump_command

    def _get_jump_target_xy(self):
        return self.jump_origin_xy + self.commands[:, self.jump_dx_command_idx:self.jump_dy_command_idx+1]

    # ------------------------------------------------------------------
    # observation（仅覆写 _get_jump_state_obs，其余继承 NoCVRobot）
    # ------------------------------------------------------------------

    def _get_jump_state_obs(self):
        """返回 3 维 jump 状态向量，用于 privileged observation。

        NoCVRobot 的默认实现返回全零；JumpRobot 返回真实状态机信息：
        - dim 0: active_jump_mask（是否在跳跃模式）
        - dim 1: was_in_flight（是否已经起飞过）
        - dim 2: has_jumped（是否已经落地）
        """
        return torch.stack(
            (
                self._get_active_jump_mask().float(),
                self.was_in_flight.float(),
                self.has_jumped.float(),
            ),
            dim=1,
        )

    # ------------------------------------------------------------------
    # command resample
    # ------------------------------------------------------------------

    def _resample_commands(self, env_ids):
        """在 NoCVRobot 采样后，为 eligible env 注入跳跃命令。

        NoCVRobot._resample_commands 已经把 jump 相关维度归零，
        这里只需要对通过 terrain 门控且命中 jump 概率的 env 重新采样。
        """
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return

        jump_terrain_mask = self._get_jump_terrain_mask(env_ids)
        eligible_env_ids = env_ids[jump_terrain_mask]
        if len(eligible_env_ids) == 0:
            return

        jump_mask = torch.rand(len(eligible_env_ids), device=self.device) < self.cfg.commands.jump_command_prob
        jump_env_ids = eligible_env_ids[jump_mask]
        if len(jump_env_ids) == 0:
            return

        ranges = self.cfg.commands.ranges
        # 跳跃模式下压低速度命令——纯跳跃训练时 scale=[0,0,0] 即清零
        self.commands[jump_env_ids, :3] *= self.jump_locomotion_command_scale.unsqueeze(0)
        self.commands[jump_env_ids, 3] = 0.0  # heading
        self.commands[jump_env_ids, self.body_height_command_idx] = self.cfg.commands.low_body_height_command
        self.commands[jump_env_ids, self.jump_dx_command_idx] = torch_rand_float(
            ranges.jump_dx[0], ranges.jump_dx[1], (len(jump_env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[jump_env_ids, self.jump_dy_command_idx] = torch_rand_float(
            ranges.jump_dy[0], ranges.jump_dy[1], (len(jump_env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[jump_env_ids, self.jump_dz_command_idx] = torch_rand_float(
            ranges.jump_dz[0], ranges.jump_dz[1], (len(jump_env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[jump_env_ids, self.jump_trigger_command_idx] = self.cfg.commands.active_jump_command

    # ------------------------------------------------------------------
    # physics step callback（跳跃状态机核心）
    # ------------------------------------------------------------------

    def _post_physics_step_callback(self):
        """在基类回调后，驱动跳跃状态机。

        状态转移：
        1. 新 jump_trigger 激活 → 记录起跳原点与高度（idle → initialized）
        2. 四脚全离地 → 标记 was_in_flight，持续追踪 max_height
        3. 飞行后脚接触恢复 → 标记 has_jumped，记录落地坐标（一次性结算 apex/land 奖励）
        """
        super()._post_physics_step_callback()
        self.jump_landed_this_step[:] = False

        active_jump_mask = self._get_active_jump_mask()

        # ---- idle → initialized ----
        new_jump_mask = active_jump_mask & ~self.jump_initialized
        if new_jump_mask.any():
            self.jump_origin_xy[new_jump_mask] = self.root_states[new_jump_mask, :2]
            self.landing_poses[new_jump_mask] = self.root_states[new_jump_mask, :2]
            self.jump_start_height[new_jump_mask] = self.root_states[new_jump_mask, 2]
            self.max_height[new_jump_mask] = self.root_states[new_jump_mask, 2]
            self.jump_initialized[new_jump_mask] = True
            self.commands[new_jump_mask, self.body_height_command_idx] = self.cfg.commands.low_body_height_command

        # ---- 脚接触检测（含单步滤波） ----
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        self.jump_contact_filt = torch.logical_or(contact, self.jump_last_contacts)
        self.jump_last_contacts = contact.clone()

        all_feet_off = torch.all(~self.jump_contact_filt, dim=1)

        # ---- initialized → in_flight ----
        self.was_in_flight[active_jump_mask & all_feet_off] = True

        # ---- 空中持续追踪最高点 ----
        airborne_mask = active_jump_mask & self.was_in_flight & ~self.has_jumped
        if airborne_mask.any():
            self.max_height[airborne_mask] = torch.maximum(
                self.max_height[airborne_mask],
                self.root_states[airborne_mask, 2],
            )

        # ---- in_flight → landed（一次性） ----
        any_foot_on = torch.any(self.jump_contact_filt, dim=1)
        newly_landed_mask = active_jump_mask & self.was_in_flight & any_foot_on & ~self.has_jumped
        if newly_landed_mask.any():
            self.landing_poses[newly_landed_mask] = self.root_states[newly_landed_mask, :2]
            self.has_jumped[newly_landed_mask] = True
            self.jump_landed_this_step[newly_landed_mask] = True
            self.commands[newly_landed_mask, self.body_height_command_idx] = self.cfg.commands.normal_body_height_command
            self._clear_jump_trigger_after_landing(newly_landed_mask.nonzero(as_tuple=False).flatten())

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        self.jump_initialized[env_ids] = False
        self.jump_last_contacts[env_ids] = False
        self.jump_contact_filt[env_ids] = False
        self.was_in_flight[env_ids] = False
        self.has_jumped[env_ids] = False
        self.jump_origin_xy[env_ids] = self.root_states[env_ids, :2]
        self.jump_start_height[env_ids] = self.root_states[env_ids, 2]
        self.landing_poses[env_ids] = self.root_states[env_ids, :2]
        self.max_height[env_ids] = self.root_states[env_ids, 2]
        self.jump_landed_this_step[env_ids] = False

    # ------------------------------------------------------------------
    # termination
    # ------------------------------------------------------------------

    def check_termination(self):
        """在基类终止条件上追加跳跃专属终止。

        基类已处理：body 碰撞、timeout。
        JumpRobot 追加：
        1. 机身 z 低于 `reset_height`（坠落 / 压趴）
        2. 落地后 pitch 超限（前后翻滚）
        3. 落地后 roll 超限（侧翻）
        4. 落地后角速度幅值过大（旋转失稳）

        姿态终止仅在 `has_jumped=True` 后生效，
        避免飞行阶段的自然俯仰被误判为失败。
        """
        super().check_termination()

        # 坠落 / 压趴
        self.reset_buf |= self.root_states[:, 2] <= self.cfg.env.reset_height

        # 落地姿态终止
        landed = self.has_jumped
        if landed.any():
            cfg_env = self.cfg.env
            pitch_fail = torch.abs(self.rpy[:, 1]) > cfg_env.jump_land_pitch_threshold
            roll_fail = torch.abs(self.rpy[:, 0]) > cfg_env.jump_land_roll_threshold
            ang_vel_fail = torch.norm(self.base_ang_vel, dim=1) > cfg_env.jump_land_ang_vel_threshold
            self.reset_buf |= landed & (pitch_fail | roll_fail | ang_vel_fail)

    # ------------------------------------------------------------------
    # reward（tracking 类在跳跃模式下置零）
    # ------------------------------------------------------------------

    def _reward_tracking_lin_vel(self):
        rew = super()._reward_tracking_lin_vel()
        rew[self._get_active_jump_mask()] = 0.0
        return rew

    def _reward_tracking_ang_vel(self):
        rew = super()._reward_tracking_ang_vel()
        rew[self._get_active_jump_mask()] = 0.0
        return rew

    # ---- jump 专属奖励 ----

    def _reward_jump_takeoff_vel(self):
        """起飞前鼓励向上的 base 线速度。"""
        mask = self._get_active_jump_mask() & ~self.was_in_flight
        return torch.clamp(self.base_lin_vel[:, 2], min=0.0) * mask.float()

    def _reward_jump_apex_height(self):
        """落地一次性结算：max_height 与目标 apex 的高斯匹配。

        目标 apex = jump_start_height + jump_dz。
        """
        mask = self.jump_landed_this_step
        target_height = self.jump_start_height + self.commands[:, self.jump_dz_command_idx]
        height_error = torch.square(self.max_height - target_height)
        return torch.exp(-height_error / self.cfg.rewards.jump_apex_sigma) * mask.float()

    def _reward_jump_land_target(self):
        """落地一次性结算：着地点与目标 xy 的高斯匹配。"""
        mask = self.jump_landed_this_step
        landing_error = torch.norm(self.landing_poses - self._get_jump_target_xy(), dim=1)
        return torch.exp(-landing_error / self.cfg.rewards.jump_land_sigma) * mask.float()

    def _reward_jump_land_compact(self):
        """落地一次性结算：鼓励前后足在 x 方向更紧凑。

        计算方式：
        1. 将四脚位置转换到 base 坐标系；
        2. 取前足中心与后足中心的 x 向距离作为 stance length；
        3. 仅当该距离超过阈值时才衰减奖励，避免过度干预正常落地稳定性。
        """
        mask = self.jump_landed_this_step
        if not mask.any():
            return mask.float()

        feet_pos_world = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        feet_pos_relative_world = feet_pos_world - self.root_states[:, 0:3].unsqueeze(1)
        feet_pos_local = quat_rotate_inverse(
            self.base_quat.repeat_interleave(len(self.feet_indices), dim=0),
            feet_pos_relative_world.reshape(-1, 3),
        ).reshape(self.num_envs, len(self.feet_indices), 3)

        # feet_indices 顺序约定为 [FL, FR, RL, RR]
        front_center_x = feet_pos_local[:, 0:2, 0].mean(dim=1)
        rear_center_x = feet_pos_local[:, 2:4, 0].mean(dim=1)
        stance_length = torch.abs(front_center_x - rear_center_x)

        max_length = self.cfg.rewards.jump_land_stance_length_max
        sigma = self.cfg.rewards.jump_land_stance_length_sigma
        excess = torch.clamp(stance_length - max_length, min=0.0)
        return torch.exp(-torch.square(excess) / sigma) * mask.float()

    def _reward_jump_flight(self):
        """空中每步给予常量奖励，鼓励维持飞行时间。"""
        mask = self._get_active_jump_mask() & self.was_in_flight & ~self.has_jumped
        return mask.float()