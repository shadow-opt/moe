from legged_gym.envs.nocv.nocv_env import NoCVRobot
from legged_gym.envs.nocv.nocv_cfg import NOCVCfg, NOCVCfgMoECTS

# [实验分支提示]
# 这个文件目前更像是一个半成品/实验草稿，而不是主线可直接复用的教学样例：
# 1. 文件里直接使用了 `torch.where` / `torch.clip` / `torch.cat`，但顶部没有 `import torch`；
# 2. 当前内容只覆盖了 `compute_reward()` / `compute_observations()`，没有给出完整任务注册与配置链路；
# 3. 因此阅读时应把它当作“局部试验修改”，不要默认其与主线 Go2 一样完整可靠。

class JumpRobot(NoCVRobot):

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            # `raw_rew` 是某个 reward 项的原始值，还没有乘权重。
            raw_rew = self.reward_functions[i]()
            rew = raw_rew * self.reward_scales.get(name, 0.0)
            if self.cfg.init_state.turn_over:
                # turn-over 训练会为同一个 reward 维护另一套权重，
                # 用于“翻身阶段”和“正常行走阶段”切换不同目标。
                turn_over_rew = raw_rew * self.reward_turn_over_scales.get(name, 0.0)
            if name in self.reward_curriculum_scales:
                # curriculum scale 是在 reward weight 之外再乘一层随训练变化的系数。
                rew *= self.reward_curriculum_scales[name]
                if self.cfg.init_state.turn_over:
                    turn_over_rew *= self.reward_curriculum_scales[name]
            if self.cfg.init_state.turn_over:
                need_turn_over = self.rpy[:, 0].abs() > self.cfg.rewards.turn_over_roll_threshold
                # 如果当前 roll 仍大于阈值，说明机器人还处在“翻倒/起身”阶段，
                # 就优先使用 turn-over 专用奖励。
                rew = torch.where(need_turn_over, turn_over_rew, rew)
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            # 有些任务会把总 reward 裁成非负，避免前期大量负回报导致学习不稳定。
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        # 基类 observation 默认包含 base linear velocity，
        # 某些具体机器人（如 Go2）会覆写这个函数，改成更适合部署的观测布局。
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    # 线速度通常是 locomotion tracking 的核心反馈之一。
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    # 使用相对默认姿态的关节角，可减轻不同机器人零点差异。
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    # 当前动作作为短时历史，有助于策略隐式感知控制惯性与动作连续性。
                                    self.actions
                                    ),dim=-1)