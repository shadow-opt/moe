
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class Go2Robot(LeggedRobot):
    """Go2 对通用 `LeggedRobot` 的轻量定制层。

    本类没有改 `step()` / `reset_idx()` / `create_sim()` 等主流程，
    说明 Go2 的核心差异主要集中在：
    1. 观测定义（observation layout）；
    2. 噪声注入方式（noise layout）；
    3. 少量 Go2 专属奖励。
    """

    def _get_noise_scale_vec(self, cfg):
        """构造 Go2 观测噪声向量。

        与基类相比，Go2 的 `obs_buf` 去掉了 base linear velocity，
        因此这里的槽位布局也必须同步改成：
        [0:3]   base_ang_vel
        [3:6]   projected_gravity
        [6:9]   commands
        [9:21]  dof_pos
        [21:33] dof_vel
        [33:45] previous/current actions

        一旦 `compute_observations()` 改了拼接顺序，这里也必须一起改，
        否则噪声会落到错误的观测维度上。

        对“扩展更多命令”尤其要小心：
        如果新增命令显式进入 observation，除了改 obs 拼接外，这里的命令槽位长度也要同步考虑。
        """

        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # Go2 actor obs 的前 3 维就是角速度，因此第一段噪声直接对应 `base_ang_vel`。
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # 接下来 3 维对应 `projected_gravity`，它本身已经是归一化方向量，因此不再乘额外 obs scale。
        noise_vec[3:6] = noise_scales.gravity * noise_level
        # commands 来自 task sampler，而不是传感器观测，通常不对它加噪。
        noise_vec[6:9] = 0. # commands
        # 12 个 dof position 槽位。
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # 12 个 dof velocity 槽位。
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # 最后 12 维是动作历史，同样不加噪，避免人为破坏控制连续性信息。
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions

        return noise_vec
    
    def compute_observations(self):
        """构造 actor obs 与 privileged critic obs。

        Go2 的普通 observation 一共 45 维：
        - 3 维 `base_ang_vel`
        - 3 维 `projected_gravity`
        - 3 维 command (`x/y/yaw`)
        - 12 维关节位置误差
        - 12 维关节速度
        - 12 维上一时刻/当前动作

        这里故意不把 `base_lin_vel` 放进 actor observation，
        但会把它放进 `privileged_obs_buf` 供 critic 使用，
        属于典型的 asymmetric training 设计。

        对后续二次开发最关键的提醒：
        当前 command 只取 `self.commands[:, :3]`，也就是策略显式看到的只有 `x/y/yaw`。
        如果以后想让策略直接感知“高度命令/跳跃命令”，这里是第一批必须同步修改的地方。
        """
        # Actor observation：尽量只保留部署时也容易获得的本体状态。
        # 其中 `projected_gravity` 可以理解为“机身当前朝向相对重力方向的投影”，
        # 它比直接喂欧拉角更连续，也更适合神经网络学习姿态信息。
        # 这里和基类最大的不同是：
        # - actor 不再显式看到 `base_lin_vel`
        # - critic 通过 `privileged_obs_buf` 才额外拿到更完整的真值信息
        # 因此如果你在调试时发现“同名环境里 obs 维度和基类不一致”，往往就是从这里开始分叉的。
        self.obs_buf = torch.cat((self.base_ang_vel  * self.obs_scales.ang_vel,
                                  self.projected_gravity,
                      # 这里切的是 `:3`，不是 `:self.cfg.commands.num_commands`。
                      # 这是当前框架里“内部 command 维度”和“显式喂给 actor 的 command 维度”不完全一致的核心体现。
                                  self.commands[:, :3] * self.commands_scale,
                                  # 用相对默认站姿的关节偏移，而不是绝对关节角，
                                  # 能让网络更容易学到“偏离默认步态多少”。
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  # 这里直接拼当前 actions，主要是给策略提供短时动作历史，
                                  # 便于学习平滑控制，而不是每一步都突然跳变。
                                  self.actions,
                                  ),dim=-1)
        
        # 187 个高度点来自 base 周围的规则网格扫描。
        # `root_z - 0.5 - measured_heights` 可以理解为“机器人底盘下方局部地形轮廓”，
        # 其中 0.5 是一个经验偏置，让高度特征大致围绕机身下方区域展开。
        # 再通过 clip 压到稳定范围，避免极端地形把观测数值拉爆。
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0) * self.obs_scales.height_measurements
        
        # Privileged observation：在 actor obs 基础上补充 simulator 才容易拿到的信息，
        # 主要给 critic 或 teacher 使用，帮助训练更稳定。
        # 拼接顺序与 `GO2Cfg.env.num_privileged_obs` 强耦合。
        # 可以把它理解成“训练时才开的上帝视角 buffer”：
        # actor 不依赖它做部署，critic/teacher 则利用它让价值估计或蒸馏更准确。
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    # 先放线速度，补上 actor 看不到但 critic 很有帮助的速度真值。
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                        # critic 当前也只显式接收 3 维 command。
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    # 4 个脚端合力，帮助 critic 更准确判断当前支撑/摆动相位。
                                    torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) * 1e-3,  # foot contact forces (4,)
                                    # 力矩除以上限，变成大致 [-1, 1] 量级，便于与其他输入共存。
                                    self.torques / self.torque_limits,  # motor torques (12,)
                                    # 用有限差分近似电机加速度，同样做一个经验缩放，避免量级过大。
                                    (self.last_dof_vel - self.dof_vel) / self.dt * 1e-4,  # motor accelerations (12,)
                                    heights,  # height measurements (187,)
                                    ),dim=-1)
        # print(f"foot contact: {self.privileged_obs_buf[:,48:48+4].min(), self.privileged_obs_buf[:,48:48+4].max()}")
        # print(f"torques: {self.privileged_obs_buf[:,48+4:48+4+12].min(), self.privileged_obs_buf[:,48+4:48+4+12].max()}")
        # print(f"acc: {self.privileged_obs_buf[:,48+4+12:48+4+12+12].min(), self.privileged_obs_buf[:,48+4+12:48+4+12+12].max()}")
        
        if self.add_noise:
            # 只对 actor obs 加噪；privileged obs 保持“更真实的 teacher/critic 视角”。
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_hip_to_default(self):
        """惩罚髋关节偏离默认站姿过多。

        这个奖励通常用于抑制过度外八/内八，
        让步态在复杂地形上更接近一套可迁移到真机的默认姿态。
        """

        # 这里直接写死 4 个 hip 的索引，是因为 Go2 的 DOF 顺序固定且已知。
        # 如果以后换机器人模型，这里的索引通常也要一起改。
        # 变量 `hip_dof_names` 主要是给阅读者看，真正参与切片的是下面的索引列表。
        hip_dof_names = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint']
        hip_dof_indices = [0, 3, 6, 9]
        # `dof_pos[:, hip_dof_indices]` 取出 4 条腿的 hip 关节角，shape = [N, 4]。
        hip_pos = self.dof_pos[:, hip_dof_indices]
        default_hip_pos = self.default_dof_pos[:, hip_dof_indices]
        # 返回的是 L1 距离和：越大表示髋关节越偏离默认站姿，惩罚就越重。
        return torch.sum(torch.abs(hip_pos - default_hip_pos), dim=1)

    def _reward_x_command_hip_regular(self):
        """在主要执行 x 向前进命令时，鼓励左右髋关节保持更对称。

        设计动机：
        - 该奖励主要服务于高速直行场景；
        - 当前进指令占主导时，希望左右前腿、左右后腿的髋关节姿态不要明显相互打架；
        - 当命令更偏侧移/转向时，该项权重会自动变小。
        """

        # 与 `_reward_hip_to_default()` 相同，这里同样只关心 4 个 hip 关节。
        hip_dof_names = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint']
        hip_dof_indices = [0, 3, 6, 9]
        hip_pos = self.dof_pos[:, hip_dof_indices]
        # 用 x command 在总 command 中的占比，近似衡量“当前任务有多像直线前进”。
        # 分母是 3 维 command 的范数，因此当 command 几乎全为 0 时，这里会很敏感；
        # 当前配置里该奖励主要给高速平地专项任务使用，通常不会长期停在纯零命令上。
        x_command_ratio = torch.abs(self.commands[:,0]) / torch.norm(self.commands[:,:3], dim=1)
        # 当前后两组髋关节和接近 0 时，更像左右对称摆动。
        # 这里的配对方式是：(FL + FR) 和 (RL + RR)，
        # 也就是前腿一组、后腿一组。
        rew = torch.abs(hip_pos[:,0]+hip_pos[:,1]) + torch.abs(hip_pos[:,2]+hip_pos[:,3])
        # x command 占比越高，该正则越重要；侧移/转向任务下它会自然弱化。
        return rew * x_command_ratio
