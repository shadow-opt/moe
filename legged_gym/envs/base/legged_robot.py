from itertools import product
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import wrap_to_pi, quat_apply_yaw
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.isaacgym_utils import sample_disjoint_intervals, sample_single_interval
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from legged_gym.utils.terrain import Terrain

"""LeggedRobot 通用环境实现。

理解本文件时，推荐把它拆成 5 条主线：
1. `__init__()` / `_parse_cfg()`：把配置转成运行时参数；
2. `step()` / `post_physics_step()`：每个 RL step 真正做了什么；
3. `_init_buffers()`：有哪些关键张量，它们各自代表什么；
4. `_resample_commands()` / `_update_terrain_curriculum()`：任务难度如何变化；
5. `_prepare_reward_function()` + `_reward_*()`：reward 如何由配置驱动。

如果后续准备扩展“更多命令”（例如高度命令、跳跃命令），建议优先盯住以下位置：
- `self.commands` / `self.commands_scale` 的初始化；
- `_resample_commands()` 的采样与特殊模板逻辑；
- `compute_observations()` / 子类 `compute_observations()` 的命令拼接；
- `_reward_tracking_*()` 是否仍然覆盖所有需要跟踪的新目标。
"""

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        # `cfg` 在这里还是一棵配置树；真正会被频繁使用的字段会在 `_parse_cfg()` 中
        # 被整理成 reward_scales / command_ranges / dt 等运行时成员。
        # 对新手来说，可以把本类初始化分成 4 步：
        # 1. `_parse_cfg()`：把配置树压平成 rollout 期间高频使用的运行时参数；
        # 2. `BaseTask.__init__()`：创建通用 RL 接口 buffer 与 simulator；
        # 3. `_init_buffers()`：把 simulator 状态包装成大量 torch buffer；
        # 4. `_prepare_reward_function()`：根据配置把 reward 名绑定到 `_reward_*()` 实现。
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        # [NEW] reward curriculum
        self.reward_curriculum_scales = {}
        self.reward_curriculum_configs = []
        if hasattr(self.cfg.rewards, "curriculum_rewards") and self.cfg.rewards.curriculum_rewards is not None:
            self.reward_curriculum_configs = self.cfg.rewards.curriculum_rewards
            for config in self.reward_curriculum_configs:
                self.reward_curriculum_scales[config['reward_name']] = config['start_value']
        self.num_steps_per_env = 24  # PPO default num_steps_per_env

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # 这里的一个 RL step，不等于一次 physics step。
        # `decimation` 表示 policy 每输出一次 action，底层物理会连续推进多次 sim step。
        # 从 buffer 生命周期看，这个函数主要负责两件事：
        # 1. 把当前 `actions` 写入动作相关 buffer；
        # 2. 驱动 simulator 前进，等到 `post_physics_step()` 再统一刷新状态/奖励/观测。
        # 因此在本函数内看到的 `dof_pos` / `base_lin_vel` 等，多数仍是“上一轮已同步”的缓存，
        # 真正的最新物理状态要等 post-step 刷新后才可靠。
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        # [new] action delay domain randomization
        if self.cfg.domain_rand.randomize_action_delay:
            # action delay 的实现方式：对每个 env 随机挑一个 decimation 起始点，
            # 在此之前继续沿用 last_actions，模拟控制链路延迟。
            actions_start_decimation = torch.randint(0, self.cfg.control.decimation+1, (self.num_envs, 1), device=self.device)
        for i in range(self.cfg.control.decimation):
            if self.cfg.domain_rand.randomize_action_delay:
                use_actions = (i >= actions_start_decimation).float()
                input_actions = (1 - use_actions) * self.last_actions + use_actions * self.actions
            else:
                input_actions = self.actions
            # action -> torque 是环境最核心的接口：
            # policy 并不直接接触真实电机力矩，而是先经 `_compute_torques()` 解释。
            self.torques = self._compute_torques(input_actions).view(self.torques.shape)
            if self.cfg.domain_rand.randomize_motor_strength:
                self.torques *= self.motor_strengths
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        # 先把 simulator 里的最新状态同步回 torch tensor，
        # 后续所有 observation / reward / reset 判断都基于这些缓存张量。
        # 如果把整个环境看成“状态机”，这里就是最核心的同步点：
        # - simulator -> torch buffer
        # - torch buffer -> reward / reset / obs
        # 因而很多 buffer 名字虽然像“普通变量”，实际上都承担着状态缓存层的角色。
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # [new] 不太理解为什么额外刷新
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        # [new] commands 的重采样时机由一个计数器控制，而不是单纯的 episode 步数间隔，
        # [new] 这样能更灵活地适配 dynamic resampling 和 curriculum 需求。
        # [new] turn over 训练下，命令重采样会被强制推迟，直到翻身阶段结束，以免过早出现新命令导致训练信号混乱。
        self.commands_resampling_step -= 1
        if self.cfg.init_state.turn_over:
            self.turn_over_timer = (self.turn_over_timer - self.dt).clip(min=0.0)

        self.update_reward_curriculum()
        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # `max_move_distance` 记录本 episode 到目前为止离出生点最远走了多远，
        # terrain curriculum 会用它决定该 env 下次该升难度还是降难度。
        self.max_move_distance = self.max_move_distance.maximum(torch.norm(self.root_states[:, :2] - self.env_origins[:, :2], dim=1))

        # `_post_physics_step_callback()` 是“状态更新”和“计算 obs/reward/reset”之间的桥梁：
        # 常见的命令重采样、heading 转 yaw-rate、高度扫描，都在这里完成。
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_ids = env_ids.clone()
        termination_privileged_obs = None
        if len(env_ids) > 0 and self.privileged_obs_buf is not None:
            self.compute_observations()
            termination_privileged_obs = self.privileged_obs_buf[env_ids].clone()
        self.reset_idx(env_ids)
        # [new] push
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.extras["termination_ids"] = termination_ids
        if termination_privileged_obs is None:
            privileged_dim = self.num_privileged_obs if self.num_privileged_obs is not None else 0
            termination_privileged_obs = torch.empty((0, privileged_dim), dtype=torch.float, device=self.device)
        else:
            clip_obs = self.cfg.normalization.clip_observations
            termination_privileged_obs = torch.clip(termination_privileged_obs, -clip_obs, clip_obs)
        self.extras["termination_privileged_obs"] = termination_privileged_obs

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
    # [new] reward curriculum scale 的更新时机是每个训练迭代开始时（即每隔 num_steps_per_env 个 physics step）
    def update_reward_curriculum(self, force_update: bool = False):
        # update reward curriculum
        if self.reward_curriculum_configs:
            if self.common_step_counter % self.num_steps_per_env == 0 or force_update:
                # reward curriculum 按“训练迭代”更新，而不是按单个 physics step 更新；
                # 这样它和 PPO/CTS 的 rollout-optimization 节奏保持一致。
                for config in self.reward_curriculum_configs:
                    current_scale = self.get_current_scale(config)
                    reward_name = config['reward_name']
                    self.reward_curriculum_scales[reward_name] = current_scale
    
    def get_current_scale(self, config):
        """ config: Dict
            {'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0}
        """
        current_iter = self.common_step_counter // self.num_steps_per_env
        cfg_start_iter = config['start_iter']
        cfg_end_iter = config['end_iter']
        cfg_start_val = config['start_value']
        cfg_end_val = config['end_value']

        # 用线性插值把当前训练迭代映射到 [start_value, end_value]。
        percentage = (current_iter - cfg_start_iter) / (cfg_end_iter - cfg_start_iter)
        percentage = max(min(percentage, 1.0), 0.0)
        
        current_scale = (1.0 - percentage) * cfg_start_val + percentage * cfg_end_val
        return current_scale

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not self.cfg.init_state.turn_over:
            # 对常规 locomotion 任务，只要 base 等终止 body 发生明显接触，就判定 episode 结束。
            self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        # timeout 和碰撞终止共用一张 reset mask。
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # reset 并不是简单清零，而是一个“小型再初始化流水线”：
        # 1. 先重采样 domain randomization；
        # 2. 再更新 terrain curriculum；
        # 3. 重置 root/dof 状态；
        # 4. 最后重采样 command 并统计 episode 信息。
        # 这里清掉的 buffer 基本都带有“上一局记忆”属性；
        # 如果漏清，最常见的问题是上一局的历史/统计泄漏到下一局，导致 reward 或 command curriculum 判断失真。
        ### Domain randomizations ###
        # randomization of the motor strength
        if self.cfg.domain_rand.randomize_motor_strength:
            rng = self.cfg.domain_rand.motor_strength_range
            # 每个 env、每个关节都可采到不同 strength，
            # 相当于模拟电机常数误差或不同关节负载差异。
            self.motor_strengths[env_ids] = torch_rand_float(
                rng[0], rng[1], (len(env_ids), self.num_actions), device=self.device
            )
        # randomization of the motor zero calibration for real machine
        if self.cfg.domain_rand.randomize_motor_zero_offset:
            # `motor_zero_offset` 更偏向真机校准误差：
            # 编码器零点不准时，策略需要学会容忍一个固定偏移。
            self.motor_zero_offsets[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_zero_offset_range[0], self.cfg.domain_rand.motor_zero_offset_range[1], (len(env_ids), self.num_actions), device=self.device)
        # randomization of the motor pd gains
        if self.cfg.domain_rand.randomize_pd_gains:
            # 这里随机的是 gain multiplier，而不是直接覆盖原始 PD 值，
            # 这样能保留关节之间原本的刚度/阻尼相对比例。
            self.p_gains_multiplier[env_ids] = torch_rand_float(self.cfg.domain_rand.stiffness_multiplier_range[0], self.cfg.domain_rand.stiffness_multiplier_range[1], (len(env_ids), self.num_actions), device=self.device)
            self.d_gains_multiplier[env_ids] =  torch_rand_float(self.cfg.domain_rand.damping_multiplier_range[0], self.cfg.domain_rand.damping_multiplier_range[1], (len(env_ids), self.num_actions), device=self.device)

        # update terrain curriculum before reset root states
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # reset buffers
        # 这些 buffer 都是“和上一个 episode 强相关”的状态，
        # reset 后若不清零，会把上一局的信息泄漏到下一局。
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.commands_resampling_step[env_ids] = self.cfg.commands.resampling_time / self.dt
        # `commands_xy_accumulation` 记录本 episode 累计分配过多少 xy 命令，
        # reset 时必须清零，否则 terrain curriculum 会误判该机器人已经“被要求走了很远”。
        self.commands_xy_accumulation[env_ids] = 0.0
        if self.cfg.commands.curriculum:
            self.update_command_curriculum(env_ids)
        self._resample_commands(env_ids)
        # fill extras
        # `extras["episode"]` 会被训练器拿去做 logging，因此这里也是观察训练状态的重要出口。
        self.extras["episode"] = {}
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.extras["episode"]['terrain_level_all'] = torch.mean(self.terrain_levels.float())
            for name, cols in self.terrain.name2cols.items():
                if isinstance(cols, set):
                    # 某些 terrain 列集合是动态 set，这里转成 tensor 以便后续用 `torch.isin` 做筛选。
                    cols = self.terrain.name2cols[name] = torch.tensor(list(cols), device=self.device)
                self.extras["episode"]['terrain_level_' + name] = torch.mean(self.terrain_levels[torch.isin(self.terrain_types, cols)].float())
        else:
            self.extras["episode"]['terrain_level_all'] = 0.0
        for key in self.episode_sums.keys():
            # 按 episode 时长做平均，更方便不同长度任务之间横向比较 reward 曲线。
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        # `rew_buf` 是“本步总奖励”，每一帧都会整块重算，不会跨步累积。
        # 跨 episode 的统计则写到 `episode_sums[name]`。
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
        # 这里同样只把前 3 维 command 显式送入 actor obs。
        # 因此 `cfg.commands.num_commands` 变大，不代表 observation 会自动跟着变大。
        # 新增命令若希望被策略直接感知，必须显式修改这里的拼接逻辑。
        # 这里的拼接顺序就是 `obs_buf` 的语义定义。
        # 训练器只看到一个扁平向量，但对开发者来说必须记住：
        # 每一段槽位都来自某个具体 buffer，且改动顺序会连带影响噪声注入、模型输入维度、导出部署对齐。
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
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            # rough terrain 会先离线生成整张地形图/网格，再把不同 env 摆到各自的平台上。
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                # 先离散成一组 friction bucket，再给每个 env 随机分桶。
                # 这样同一个 batch 内会稳定出现多种摩擦条件，而不是完全连续均匀噪声。
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                # 同一 env 里的所有 shape 共用一个 friction，
                # 否则单个机器人不同 link 摩擦差异过大，可能偏离真实设定。
                props[s].friction = self.friction_coeffs[env_id]
        # [new] restitution randomization : 虽然大多数 locomotion 任务对 restitution 不太敏感，
        # 但在一些需要频繁跳跃/落地的任务里，过高的 restitution 可能导致训练不稳定，因此也加了随机化选项。
        if self.cfg.domain_rand.randomize_restitution:
            rand_restitution = np.random.uniform(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1])
            for s in range(len(props)):
                props[s].restitution = rand_restitution
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            # 这些 limit 只需要从 asset 里读一次，之后会广播给所有 env 使用。
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                # 这里把 URDF 的硬限位收紧成 soft limit，
                # reward 会更早开始惩罚，避免训练过程中频繁撞到硬边界。
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            # 默认只改 base mass，因为它对整体惯性、落地冲击和稳定性影响最大。
            props[0].mass += np.random.uniform(rng[0], rng[1])

        # randomize link masses
        if self.cfg.domain_rand.randomize_link_mass:
            self.multiplied_link_masses_ratio = torch_rand_float(self.cfg.domain_rand.multiplied_link_mass_range[0], self.cfg.domain_rand.multiplied_link_mass_range[1], (1, self.num_bodies-1), device=self.device)
            for i in range(1, len(props)):
                # link mass 用乘法缩放，含义更像“比例误差”；
                # base mass 用加法，更像“额外挂载/载荷变化”。
                props[i].mass *= self.multiplied_link_masses_ratio[0,i-1]

        # randomize base com
        if self.cfg.domain_rand.randomize_base_com:
            self.added_base_com = torch_rand_float(self.cfg.domain_rand.added_base_com_range[0], self.cfg.domain_rand.added_base_com_range[1], (1, 3), device=self.device)
            props[0].com += gymapi.Vec3(self.added_base_com[0, 0], self.added_base_com[0, 1],
                                    self.added_base_com[0, 2])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # command 的重采样是在 post-step 里做的，
        # 这意味着 command 可以依赖刚走完这一小段 rollout 的最新状态。
        # env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        resampling_env_ids = ((self.commands_resampling_step <= 0.0) * (self.episode_length_buf < self.max_episode_length - 1)).nonzero(as_tuple=False).flatten()
        # 命令采样被放在这里，而不是 reset 内独占处理。
        # 这样一个 episode 中可以发生多次 command 切换，形成速度跟踪任务。
        self._resample_commands(resampling_env_ids)
        if self.cfg.commands.heading_command:
            # heading mode 下，policy 不直接跟踪 yaw velocity command，
            # 而是先由 heading error 转成一个期望 yaw rate。
            mask = (self.stop_heading == 0.0)
            forward = quat_apply(self.base_quat[mask], self.forward_vec[mask])
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            # 第 4 维 `heading` 不会直接送给策略动作头，
            # 而是在这里被转换成第 3 维 `ang_vel_yaw` 的期望值。
            self.commands[mask, 2] = torch.clip(
                0.5*wrap_to_pi(self.commands[mask, 3] - heading),
                self.env_command_ranges["ang_vel_yaw"][:, 0],
                self.env_command_ranges["ang_vel_yaw"][:, 1]
            )
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            # `measured_heights` 会在 observation / reward 中复用，
            # 因此这里只算一次，避免后续重复采样。

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return
        # 这是 command 逻辑最密集的函数。
        # 可以按以下顺序理解：
        # 1. 训练迭代驱动的全局 command curriculum；
        # 2. 基于剩余时间/剩余距离的 command 采样；
        # 3. limit velocity / zero command 等特殊命令模板；
        # 4. turn-over 训练下的强制静止保护期。
        # 如果未来加入“高度命令/跳跃命令”，通常也要在这个函数里决定：
        # - 新命令是每次重采样都随机给，还是按特殊事件触发；
        # - 新命令是否参与 zero/limit/dynamic-resample 等模板；
        # - 新命令是否也要受 terrain range 裁剪。
        self.stop_heading[env_ids] = False
        # update command curriculum with train steps
        if len(self.cfg.commands.command_range_curriculum):
            current_iter = self.common_step_counter // self.num_steps_per_env 
            for i in range(len(self.cfg.commands.command_range_curriculum)-1, -1, -1):  # iterate backwards to be able to pop entries
                cfg = self.cfg.commands.command_range_curriculum[i]
                if current_iter >= cfg["iter"]:
                    self.command_ranges["lin_vel_x"] = cfg["lin_vel_x"]
                    self.command_ranges["lin_vel_y"] = cfg["lin_vel_y"]
                    self.command_ranges["ang_vel_yaw"] = cfg["ang_vel_yaw"]
                    self.command_ranges["heading"] = cfg["heading"]
                    self.max_lin_vel = max(abs(self.command_ranges["lin_vel_x"][0]), abs(self.command_ranges["lin_vel_x"][1]),
                                           abs(self.command_ranges["lin_vel_y"][0]), abs(self.command_ranges["lin_vel_y"][1]))
                    # 这里 `pop(i)` 的含义是：某个 curriculum 节点一旦触发，就不再重复检查。
                    self.cfg.commands.command_range_curriculum.pop(i)
                    self._update_env_command_ranges()
                    print(f"Command range updated at iter {current_iter}: {self.command_ranges}")
        # `remaining_dist` 是一个任务级启发式量：
        # 它估计“在当前 terrain 平台上，后续还允许分配多少 xy 行走任务”。
        # 这个量只和 xy 命令有关，因此如果后续新增垂向/跳跃命令，需要单独思考它们是否应该参与类似预算。
        remaining_dist = torch.clip(0.625 * self.cfg.terrain.terrain_length - torch.norm(self.commands_xy_accumulation[env_ids], dim=1) * self.cfg.commands.resampling_time, 0.0)
        # 每次重采样后，都把“下次还要过多少 step 才再采样”重置回 resampling_time 对应步数。
        self.commands_resampling_step[env_ids] = self.cfg.commands.resampling_time / self.dt
        if self.cfg.commands.dynamic_resample_commands:
            # arrive at boundary 0.625 times the width of the remaining distance
            if ((self.max_episode_length - self.episode_length_buf[env_ids]) == 0).any():
                raise ValueError("Some envs have zero remaining episode length during command resampling")
            # `vel_low_bound` 是一个很关键的工程技巧：
            # 剩余 episode 越短、剩余目标距离越大，则采到的命令速度下界越高。
            vel_low_bound = torch.clip(remaining_dist / ((self.max_episode_length - self.episode_length_buf[env_ids] + 1e-9) * self.dt), 0.0)
            self.commands[env_ids, 0] = sample_disjoint_intervals(
                env_ids,
                vel_low_bound,
                self.env_command_ranges["lin_vel_x"][env_ids, 0],
                self.env_command_ranges["lin_vel_x"][env_ids, 1],
                self.device
            )
            self.commands[env_ids, 1] = sample_disjoint_intervals(
                env_ids,
                vel_low_bound,
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                self.device
            )
            if self.cfg.commands.heading_command:
                r = torch.rand(len(env_ids), device=self.device)
                lower = self.env_command_ranges["heading"][env_ids, 0]
                upper = self.env_command_ranges["heading"][env_ids, 1]
                # heading 模式下第 4 维是目标朝向，而不是直接的 yaw 速度。
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
                self.device
            )
            self.commands[env_ids, 1] = sample_single_interval(
                env_ids,
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                self.device
            )
            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = sample_single_interval(
                    env_ids,
                    self.env_command_ranges["heading"][env_ids, 0],
                    self.env_command_ranges["heading"][env_ids, 1],
                    self.device
                )
            else:
                self.commands[env_ids, 2] = sample_single_interval(
                    env_ids,
                    self.env_command_ranges["ang_vel_yaw"][env_ids, 0],
                    self.env_command_ranges["ang_vel_yaw"][env_ids, 1],
                    self.device
                )

            # set small commands to zero
            # 非 dynamic 模式下，会把过小的 xy 命令直接清零，
            # 避免策略浪费容量在“几乎不动”的模糊命令上。
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        rand_prob = torch.rand(len(env_ids), device=self.device)
        min_prob, max_prob = 0.0, 0.0
        # set limitation lin vel
        if self.limit_vel_prob > 0.0:
            max_prob += self.limit_vel_prob
            lim_mask = (rand_prob >= min_prob) * (rand_prob < max_prob)
            lim_env_ids = env_ids[lim_mask]
            if len(lim_env_ids) > 0:
                # limit-vel 机制会把连续采样的命令，替换成离散边界命令，
                # 例如纯前进、纯后退、纯原地转向等，帮助策略覆盖边界工况。
                change_lim_env_ids = lim_env_ids
                if self.cfg.commands.limit_vel_invert_when_continuous:
                    was_limited = self.last_is_limit_vel[lim_env_ids]
                    invert_env_ids = lim_env_ids[was_limited]
                    # 如果上一次已经是 limit command，这次直接整体翻符号，
                    # 可以让机器人更频繁经历“由前进切到后退”“由左转切到右转”等反向场景。
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
                    self.env_command_ranges["lin_vel_y"][change_lim_env_ids, 1]
                )
                lin_vel_y_lim[self.limit_vel_comb[vel_idx, 1] == 0] = 0.0
                ang_vel_z_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 2] == -1,
                    self.env_command_ranges["ang_vel_yaw"][change_lim_env_ids, 0],
                    self.env_command_ranges["ang_vel_yaw"][change_lim_env_ids, 1]
                )
                ang_vel_z_lim[self.limit_vel_comb[vel_idx, 2] == 0] = 0.0
                self.commands[change_lim_env_ids, 0] = lin_vel_x_lim
                self.commands[change_lim_env_ids, 1] = lin_vel_y_lim
                self.commands[change_lim_env_ids, 2] = ang_vel_z_lim
                if self.cfg.commands.heading_command and self.cfg.commands.stop_heading_at_limit:
                    # heading mode 下，如果已经强行设成离散极值命令，
                    # 就不再继续更新 heading，避免 heading controller 把它改回去。
                    self.stop_heading[lim_env_ids] = True # stop heading to current heading
                self.last_is_limit_vel[env_ids] = False
                self.last_is_limit_vel[lim_env_ids] = True
            else:
                self.last_is_limit_vel[env_ids] = False
            min_prob += self.limit_vel_prob

        # set all commands to zero with some probability
        if self.cfg.commands.zero_command_curriculum is not None:
            self.zero_command_proba = self.get_current_scale(self.cfg.commands.zero_command_curriculum)
        if self.zero_command_proba > 0.0:
            max_prob += self.zero_command_proba
            # 为了不让机器人在接近 episode 末尾时突然被置零后“走不满任务距离”，
            # 这里会根据剩余距离与最高速度估计下一次 resample 时刻。
            next_resampling_step = torch.clip(
                self.max_episode_length - self.episode_length_buf[env_ids] - (remaining_dist / (0.8 * self.max_lin_vel * self.dt + 1e-9)),
                min=0.0,
                max=self.cfg.commands.resampling_time / self.dt,
            )
            zero_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) * (next_resampling_step > 0.0)
            zero_env_ids = env_ids[zero_mask]
            if len(zero_env_ids) > 0:
                # 零命令只清线速度，不一定清 yaw，
                # 因为后面还可能按概率附加“原地左/右转”的角速度任务。
                self.commands[zero_env_ids, :2] = 0.0
                self.commands_resampling_step[zero_env_ids] = next_resampling_step[zero_mask]
                if self.cfg.commands.limit_ang_vel_at_zero_command_prob > 0.0:
                    ang_vel_rand = torch.rand(len(zero_env_ids), device=self.device) # independent distribution
                    add_ang_mask = ang_vel_rand < self.cfg.commands.limit_ang_vel_at_zero_command_prob
                    add_ang_env_ids = zero_env_ids[add_ang_mask]
                    if len(add_ang_env_ids) > 0:
                        direction_rand = torch.rand(len(add_ang_env_ids), device=self.device)
                        self.commands[add_ang_env_ids, 2] = torch.where(
                            direction_rand < 0.5,
                            self.env_command_ranges["ang_vel_yaw"][add_ang_env_ids, 0],
                            self.env_command_ranges["ang_vel_yaw"][add_ang_env_ids, 1]
                        )
                        if self.cfg.commands.heading_command:
                            self.stop_heading[add_ang_env_ids] = True
            min_prob += self.zero_command_proba

        # turn over zero command time
        if self.cfg.init_state.turn_over and (self.turn_over_timer[env_ids] > 0).any():
            zero_mask = self.turn_over_timer[env_ids] > 0
            zero_env_ids = env_ids[zero_mask]
            # 翻身恢复保护期内强制清空命令，避免机器人刚起身又立刻被高速命令拖倒。
            self.commands[zero_env_ids, :3] = 0.0
            self.stop_heading[zero_env_ids] = True

        # 这里累计的是“每次采样得到的 xy command 向量和”，
        # terrain curriculum 会用它近似估计：这个 episode 理论上要求机器人走多远。
        self.commands_xy_accumulation[env_ids] += self.commands[env_ids, :2]

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # 这是“策略输出”真正接入物理世界的地方。
        # 可以把上游流程粗略理解成：
        # command -> observation -> policy action -> torque -> simulator。
        # 其中本函数只负责最后两步之间的映射。
        # 所以像“新增高度命令/跳跃命令”这类高层任务改动，
        # 往往优先影响的是 command / observation / reward，
        # 而不是这里的底层执行器映射。
        # 统一先做 `action_scale`，让 policy 输出落在一个相对稳定的范围。
        # 之后再根据 control_type 决定它代表角度偏移、速度目标还是直接力矩。
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        p_gains = self.p_gains * self.p_gains_multiplier # [new] multiplier 是随机量，在 action_scale 之后再乘能保持随机化和 action_scale 的解耦。
        d_gains = self.d_gains * self.d_gains_multiplier
        if control_type=="P":
            # Position mode: 目标角 = default pose + action offset + motor zero calibration。
            # 这是本仓库主线最常见的控制模式：
            # policy 不直接输出目标关节角，而是输出“相对默认站姿的偏移量”。
            torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_zero_offsets) - d_gains * self.dof_vel
        elif control_type=="V":
            # Velocity mode: action 被解释成目标关节速度。
            torques = p_gains*(actions_scaled - self.dof_vel) - d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            # Torque mode: action 几乎直接就是力矩命令，只保留比例缩放与最终裁剪。
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        # 最后统一按 URDF/资产给出的极限做裁剪，避免输出超出模拟器允许范围。
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # 关节 reset 不是精确回到唯一姿态，而是在默认角附近随机扰动。
        # 这样做能减少策略对“固定开局姿势”的依赖，让恢复能力更强。
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        # 新 episode 的关节速度清零，避免把上一局残余摆动带进来。
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # 这一句才是把新的 DOF 状态真正写回 simulator。
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # `root_states` 代表机身根节点状态，包含位置、姿态、线速度、角速度。
        # 对四足机器人来说，这几乎就是“机器人整个身体在世界里的起点”。
        if self.cfg.init_state.turn_over:
            self.turn_over_timer[env_ids] = 0.0
        # base position
        # reset 时先随机一个 yaw，避免策略过拟合固定朝向。
        if getattr(self.cfg.init_state, 'randomize_yaw', True):
            random_yaw = torch_rand_float(-np.pi, np.pi, (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            random_yaw = torch.zeros(len(env_ids), device=self.device)
        def get_quat(target_yaws, roll: float):
            roll_tensor = torch.full((len(target_yaws),), roll, device=self.device)
            pitch_tensor = torch.zeros((len(target_yaws),), device=self.device)
            quat = quat_from_euler_xyz(roll_tensor, pitch_tensor, target_yaws)
            return quat

        base_init_state = self.base_init_state.reshape(1, -1).repeat(len(env_ids), 1)
        if self.cfg.init_state.turn_over:
            # 如果开启 turn-over 训练，则按给定比例随机生成后翻/侧翻/正常朝向三种初始姿态。
            # 这相当于主动制造“机器人已经倒地”的训练样本。
            rand_prob = torch.rand(len(env_ids), device=self.device)
            proportions = self.cfg.init_state.turn_over_proportions
            init_heights = self.cfg.init_state.turn_over_init_heights

            min_prob, max_prob = 0.0, proportions[0]
            back_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) # backflip
            if back_mask.any():
                # 后翻时需要更低的初始高度，让机器人一开始就处于“翻倒在地”的物理状态附近。
                heights = torch_rand_float(init_heights['backflip'][0], init_heights['backflip'][1], (torch.sum(back_mask), 1), device=self.device).squeeze(1)
                base_init_state[back_mask, 2] = heights # z
                base_init_state[back_mask, 3:7] = get_quat(random_yaw[back_mask], np.pi)
                if self.cfg.init_state.turn_over:
                    self.turn_over_timer[env_ids[back_mask]] = self.cfg.commands.turn_over_zero_time['backflip']

            min_prob = max_prob
            max_prob += proportions[1]
            side_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) # sideflip
            if side_mask.any():
                side_ids = torch.nonzero(side_mask, as_tuple=False).flatten()
                heights = torch_rand_float(init_heights['sideflip'][0], init_heights['sideflip'][1], (len(side_ids), 1), device=self.device).squeeze(1)
                base_init_state[side_mask, 2] = heights # z
                side_rand_prob = torch.rand(len(side_ids), device=self.device)
                pos_side_mask = side_rand_prob < 0.5
                neg_side_mask = ~pos_side_mask
                if pos_side_mask.any():
                    # 侧翻既可能向左倒，也可能向右倒，两边都要覆盖到。
                    pos_ids = side_ids[pos_side_mask]
                    base_init_state[pos_ids, 3:7] = get_quat(random_yaw[pos_ids], np.pi/2)
                if neg_side_mask.any():
                    neg_ids = side_ids[neg_side_mask]
                    base_init_state[neg_ids, 3:7] = get_quat(random_yaw[neg_ids], -np.pi/2)
                if self.cfg.init_state.turn_over:
                    self.turn_over_timer[env_ids[side_mask]] = self.cfg.commands.turn_over_zero_time['sideflip']
            
            min_prob = max_prob
            max_prob += proportions[2]
            noflip_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) # noflip
            if noflip_mask.any():
                noflip_indices = torch.nonzero(noflip_mask, as_tuple=False).flatten()
                base_init_state[noflip_mask, 3:7] = get_quat(random_yaw[noflip_indices], 0.0)
        else:
            base_init_state[:, 3:7] = get_quat(random_yaw, 0.0)
                
        if self.custom_origins:
            self.root_states[env_ids] = base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # rough terrain 上再给一点 xy 抖动，让机器人不要总在平台中心同一点出生。
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # 上面的 `env_origins` 决定“每个并行环境在总地图中的出生位置”。
        # rough terrain 和 plane 模式下，它的来源不同，但用法一致。
        # base velocities
        # 初速度也做轻微随机化，避免 reset 后永远从完全静止的理想状态开始。
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # 这一步把 root state 同步回 simulator，和前面的 torch 缓存保持一致。
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        # 这是训练鲁棒性的常见技巧：定期给机器人一个“突然被撞了一下”的扰动。
        # 实现上不是施加物理外力，而是直接改写机身速度状态。
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_ang = self.cfg.domain_rand.max_push_ang_vel
        # push 的实现不是直接施加外力，而是瞬间改写 root velocity，
        # 从策略视角看，它等价于遭受了一次未知冲击。
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.root_states[:, 10:13] = torch_rand_float(-max_push_ang, max_push_ang, (self.num_envs, 3), device=self.device) # ang vel x/y/z
        
        env_ids_int32 = push_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

   
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        threshold = getattr(self.cfg.commands, "command_curriculum_threshold", 0.8)
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > threshold * self.reward_scales["tracking_lin_vel"]:
            step = getattr(self.cfg.commands, "command_curriculum_step", 0.5)
            dims = getattr(self.cfg.commands, "command_curriculum_dims", ["lin_vel_x"])
            max_ranges = getattr(self.cfg.commands, "max_command_curriculum_ranges", {})
            updated = False
            for dim in dims:
                if dim not in self.command_ranges:
                    continue
                max_range = max_ranges.get(dim, [-self.cfg.commands.max_curriculum, self.cfg.commands.max_curriculum])
                old_min, old_max = self.command_ranges[dim]
                self.command_ranges[dim][0] = np.clip(old_min - step, max_range[0], 0.0)
                self.command_ranges[dim][1] = np.clip(old_max + step, 0.0, max_range[1])
                updated = updated or self.command_ranges[dim][0] != old_min or self.command_ranges[dim][1] != old_max
            if updated:
                self.max_lin_vel = max(
                    abs(self.command_ranges["lin_vel_x"][0]),
                    abs(self.command_ranges["lin_vel_x"][1]),
                    abs(self.command_ranges["lin_vel_y"][0]),
                    abs(self.command_ranges["lin_vel_y"][1]),
                )
                self._update_env_command_ranges()


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # 基类 obs 前 3 维是 base linear velocity。
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        # 接下来 3 维是 base angular velocity。
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # projected_gravity 本质上是姿态估计量。
        noise_vec[6:9] = noise_scales.gravity * noise_level
        # command 由任务采样器产生，不视作传感器输入，因此不加噪。
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # 最后 12 维动作历史不加噪，防止破坏平滑控制相关信息。
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        # 下面这一组 tensor 是整个环境最核心的运行时状态缓存。
        # 几乎所有 reward / observation / reset 逻辑都会直接读写它们。
        # 对新手来说，最重要的是先区分两类来源：
        # 1. “镜像 simulator 的原始状态”：root/dof/contact/rigid-body
        # 2. “由原始状态派生/缓存出来的中间量”：base velocity、projected_gravity、commands、历史动作等
        # 第一类更像事实真值缓存，第二类更像为 reward/obs/课程学习服务的工作内存。
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # `root_states`: 机身根节点状态，shape = [num_envs, 13]。
        # 常用槽位语义：
        # [0:3]  world position
        # [3:7]  quaternion
        # [7:10] world linear velocity
        # [10:13] world angular velocity
        # 注意后续 `base_lin_vel` / `base_ang_vel` 会再把速度旋回机身坐标系。
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # dof_state 的最后一维长度为 2，分别是 position / velocity。
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        if self.cfg.terrain.measure_heights:
            # 预先生成高度采样点，并额外标记一块 base 下方的小窗口，
            # 供 `_get_base_height()` 估计机器人离地高度时使用。
            self.height_points = self._init_height_points()
            x_points = self.height_points[0, :, 0]
            y_points = self.height_points[0, :, 1]
            x_mask = (x_points >= -0.2) & (x_points <= 0.2)  # 0.4m length
            y_mask = (y_points >= -0.15) & (y_points <= 0.15)  # 0.3m width
            self.base_height_scan_mask = (x_mask & y_mask).float()
            self.num_base_height_scan_points = self.base_height_scan_mask.sum()
            assert self.num_base_height_scan_points > 0, "No height scan points within the specified area."
        self.measured_heights = 0

        # initialize some data used later on
        self.common_step_counter = 0
        # `common_step_counter`: 全局累计环境步数，不会在单个 env reset 时清零。
        # 它通常用于“按训练进度变化”的逻辑，例如 reward curriculum / command curriculum。
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        # `gravity_vec` / `forward_vec` 会反复参与坐标变换，因此预先为每个 env 准备好。
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        # `actions` / `last_actions` / `last_dof_vel` 这类张量通常用于 reward 中的平滑项。
        # --- 动作与执行器相关 buffer ---
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # `torques`: 最终送入 simulator 的电机力矩命令。
        # 它是“动作解释器” `_compute_torques()` 的输出，也是若干能耗/力矩惩罚 reward 的直接输入。
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # `actions` / `last_actions`: 当前策略输出与上一时刻输出。
        # 常用于动作平滑、二阶差分平滑等 reward，也可帮助观测隐式表达控制惯性。
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        # `last_dof_vel` / `last_root_vel`: 一帧历史缓存，常用于近似加速度或平滑项。
        # `self.commands` 是训练时“高层任务目标”的主缓存。
        # 当前默认语义是：[x 速度, y 速度, yaw 角速度, heading]。
        # 这里的第二维长度来自 `cfg.commands.num_commands`，因此扩容命令张量时第一步通常改这里。
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        # `commands`: 任务采样器给策略设置的高层目标，不是机器人当前真实状态。
        # 一个常见误解是把它当“观测的一部分”；实际上它先是内部目标，
        # 之后只有被显式拼进 `obs_buf` 的那几维，才真正对 actor 可见。
        # command 的 scale 只用于把 command 数值映射到 observation 的合理量级。
        # 这里目前被硬编码成 3 维，对应显式送入 observation 的 `x/y/yaw`。
        # 这也是扩展更多显式命令时最容易漏改的结构性限制之一。
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        # `commands_resampling_step` 像一个倒计时器：每走一步减 1，减到 0 就重新采 command。
        self.commands_resampling_step = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # 这个量不是机器人真实位移，而是“累计发出去的命令向量和”，
        # 常用于 curriculum 判断任务要求的理论行走距离。
        self.commands_xy_accumulation = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        # `commands_xy_accumulation` 不是传感器量，而是训练过程中的 bookkeeping buffer。
        # 它回答的问题更像：“这局理论上被要求走了多少 xy 距离？”
        self.zero_command_proba = 0.0
        # `feet_air_time` 按脚分别计时，shape = [num_envs, num_feet]。
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # `max_move_distance` 是本局走到过的最远距离，不是当前距离。
        # --- terrain / curriculum / reset 相关 buffer ---
        self.max_move_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.stop_heading = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_is_limit_vel = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.limit_vel_prob = self.cfg.commands.limit_vel_prob
        # `limit_vel_comb` 枚举所有离散极值组合，例如 [x=min, y=0, yaw=max]。
        self.limit_vel_comb = torch.tensor(list(product(
            self.cfg.commands.limit_vel["lin_vel_x"],
            self.cfg.commands.limit_vel["lin_vel_y"],
            self.cfg.commands.limit_vel["ang_vel_yaw"]
        )), device=self.device, requires_grad=False)
        self.last_robot_props_update_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        # `last_robot_props_update_step`: 预留给“按训练进度定期改 robot properties”的计时参考。
        # 它不是每个任务都在用，但属于 domain randomization 相关状态缓存的一部分。
        # `turn_over_timer` > 0 表示该 env 还在翻身恢复保护期内。
        self.turn_over_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # `env_command_ranges` 是 env 级别的最终 command 上下界，
        # 它已经综合了全局 command range 和 terrain-specific 限制。
        self.env_command_ranges = {
            'lin_vel_x': torch.tensor(self.command_ranges['lin_vel_x'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),
            'lin_vel_y': torch.tensor(self.command_ranges['lin_vel_y'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),
            'ang_vel_yaw': torch.tensor(self.command_ranges['ang_vel_yaw'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),
            'heading': torch.tensor(self.command_ranges['heading'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),
        }
        self._update_env_command_ranges()

        # joint positions offsets and PD gains
        # 这一组是“动作解释器”的静态参考系：
        # - `default_dof_pos` 决定 action=0 时的默认目标姿态
        # - `p_gains` / `d_gains` 决定 action 到 torque 的控制刚度
        # 新机器人接入时，这一段几乎总是最先需要核对的部分之一。
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            # `default_dof_pos` 后面会被 unsqueeze 成 [1, num_dof]，
            # 便于和 [num_envs, num_dof] 的 dof 张量自动广播相减。
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    # 这里允许用关键字匹配，例如配置里只写 `joint`，
                    # 就能同时匹配多个具体 dof 名字。
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
    
    def _update_env_command_ranges(self):
        """ Update environment-wise command ranges based on current command ranges and terrain type """
        # 这个函数的职责不是“采样命令”，而是先算出“每个 env 允许采样到什么范围”。
        # 真正的随机采样发生在 `_resample_commands()`。
        if not hasattr(self, 'terrain_ids'):
            self.env_command_ranges = {
                'lin_vel_x': torch.tensor(self.command_ranges['lin_vel_x'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),
                'lin_vel_y': torch.tensor(self.command_ranges['lin_vel_y'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),
                'ang_vel_yaw': torch.tensor(self.command_ranges['ang_vel_yaw'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),
                'heading': torch.tensor(self.command_ranges['heading'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),
            }
            # 没有 terrain_id 时，说明当前不是 rough terrain 多类型模式，
            # 所有 env 直接共享同一套 command range。
            return
        # rough terrain 下，同一个 batch 中不同 env 可能处在不同 terrain type，
        # 因此 command range 也需要按 env 分别裁剪。
        for terrain_id, terrain_command_ranges in enumerate(self.cfg.commands.terrain_max_command_ranges):
            env_ids = (self.terrain_ids == terrain_id).nonzero(as_tuple=False).flatten()
            if len(env_ids) == 0:
                continue
            # 下界取 max、上界取 min，本质上是在做两个区间的交集：
            # `全局 curriculum 范围 ∩ 当前 terrain 允许范围`。
            self.env_command_ranges['lin_vel_x'][env_ids, 0] = max(
                terrain_command_ranges['lin_vel_x'][0],
                self.command_ranges['lin_vel_x'][0],
            )
            self.env_command_ranges['lin_vel_x'][env_ids, 1] = min(
                terrain_command_ranges['lin_vel_x'][1],
                self.command_ranges['lin_vel_x'][1]
            )
            self.env_command_ranges['lin_vel_y'][env_ids, 0] = max(
                terrain_command_ranges['lin_vel_y'][0],
                self.command_ranges['lin_vel_y'][0]
            )
            self.env_command_ranges['lin_vel_y'][env_ids, 1] = min(
                terrain_command_ranges['lin_vel_y'][1],
                self.command_ranges['lin_vel_y'][1]
            )
            self.env_command_ranges['ang_vel_yaw'][env_ids, 0] = max(
                terrain_command_ranges['ang_vel_yaw'][0],
                self.command_ranges['ang_vel_yaw'][0]
            )
            self.env_command_ranges['ang_vel_yaw'][env_ids, 1] = min(
                terrain_command_ranges['ang_vel_yaw'][1],
                self.command_ranges['ang_vel_yaw'][1]
            )
            self.env_command_ranges['heading'][env_ids, 0] = max(
                terrain_command_ranges['heading'][0],
                self.command_ranges['heading'][0]
            )
            self.env_command_ranges['heading'][env_ids, 1] = min(
                terrain_command_ranges['heading'][1],
                self.command_ranges['heading'][1]
            )

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # 这是 reward 的“配置驱动绑定”机制：
        # 只要 `cfg.rewards.scales.xxx != 0`，环境就会自动去找 `_reward_xxx()`。
        # 因此 reward 的增删通常无需改主循环，只需维护配置和函数命名约定。
        # remove zero scales + multiply non-zero ones by dt
        def update_scales(scales):
            for key in list(scales.keys()):
                scale = scales[key]
                if scale==0:
                    scales.pop(key) 
                else:
                    scales[key] *= self.dt
        update_scales(self.reward_scales)
        if self.cfg.init_state.turn_over:
            update_scales(self.reward_turn_over_scales)
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        names = set()
        names.update(list(self.reward_scales.keys()))
        if self.cfg.init_state.turn_over:
            names.update(list(self.reward_turn_over_scales.keys()))
        for name in names:
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            # 如果这里报属性不存在，通常意味着：
            # 配置里写了某个 reward 名，但环境类里没有对应实现。
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in names}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
                # 这里完成从 URDF 到 batched env 的映射：
                # 机器人资产只加载一次，然后被复制到多个 environment 实例中。
                # 你可以把它想成“先造一个机器人模板，再批量复制成很多训练副本”。
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        # 先通过名字把“脚、hip、需要惩罚碰撞的 body、触发终止的 body”筛出来，
        # 后面再统一转成索引，供张量切片使用。
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        hip_names = [s for s in self.dof_names if 'hip' in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # domain rand
        self.motor_zero_offsets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains_multiplier = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains_multiplier = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.rewards.dynamic_sigma:
            self.dynamic_sigma_cfg = self.cfg.rewards.dynamic_sigma
            # 每类 terrain 对应一个 sigma 上限，用于难地形/高命令时放宽 tracking reward。
            self.terrain_max_sigmas = torch.tensor(self.dynamic_sigma_cfg["max_sigma"], device=self.device, requires_grad=False)

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # 每个循环体对应一个并行环境实例。
            # 它们共享同一个机器人模型定义，但状态、随机化结果、地形位置都彼此独立。
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            # 即使 env origin 已经固定，仍再加一点随机 xy 偏移，避免所有机器人都从完全同一个局部地形点出生。
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            if i == 0:
                self.default_body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # 把名字索引转成 tensor 索引，之后 reward/termination/observation 都通过这些索引切片。
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hip_names)):
            self.hip_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], hip_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        # `env_origins` 是并行环境的“出生点表”。
        # 后续 reset、距离统计、terrain curriculum 都要依赖它。
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1

            # random choice terrain levels and types for each env
            # self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            # self.terrain_types = torch.randint(0, self.cfg.terrain.num_cols, (self.num_envs,), device=self.device)

            # 当前实现采用 round robin，把 env 平均铺到不同 terrain level/type 上，
            # 这样一个 batch 内就能稳定覆盖多种地形，而不是纯随机碰运气。
            self.terrain_levels = torch.fmod(torch.arange(self.num_envs, device=self.device), max_init_level + 1)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs / self.cfg.terrain.num_cols), rounding_mode="floor").to(torch.long)
            self.terrain_cols2id = torch.tensor(self.terrain.cols2id, device=self.device)
            if len(self.terrain_cols2id):
                # `terrain_types` 是 terrain 列号；`terrain_ids` 是语义类别 id，
                # 后者会被 command range / dynamic sigma 等逻辑复用。
                self.terrain_ids = self.terrain_cols2id[self.terrain_types]

            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            # rough terrain 下，origin 来自 terrain 生成器给出的各个平台中心。
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            # 平地/无 rough terrain 时，不再依赖 terrain 平台，直接把 env 摆成规则网格。
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        """把配置树整理成 rollout 期间高频使用的运行时字段。"""

        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.reward_turn_over_scales = class_to_dict(self.cfg.rewards.turn_over_scales)
        # `command_ranges` 在这里被转成普通 dict，后续会在 curriculum / terrain 裁剪中反复更新。
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.max_lin_vel = max(abs(self.command_ranges["lin_vel_x"][0]), abs(self.command_ranges["lin_vel_x"][1]),
                               abs(self.command_ranges["lin_vel_y"][0]), abs(self.command_ranges["lin_vel_y"][1]))
        self.cfg.commands.command_range_curriculum = sorted(self.cfg.commands.command_range_curriculum, key=lambda x: x['iter'], reverse=True)

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done or self.cfg.terrain.mesh_type == 'plane':
            # don't change on initial reset
            return
        # 这里使用的是“本 episode 走到过的最远距离”，
        # 而不是 reset 前最后一帧的位置距离。
        # 这样机器人就算走远后又退回来，也仍会被认为“有能力通过当前地形”。
        distance = self.max_move_distance[env_ids]
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        if self.cfg.terrain.move_down_by_accumulated_xy_command:
            # 如果打开 accumulated_xy_command 模式，
            # “是否该降难度”不再看绝对位移，而看是否完成了累计命令要求的路程。
            # 直观上，它更像“按布置的作业量评分”，而不是只看最终走了多远。
            move_down = (distance < torch.norm(self.commands_xy_accumulation[env_ids], dim=1) * (self.cfg.commands.resampling_time * (1 - self.zero_command_proba)) * 0.5) * ~move_up
        else:
            # robots that walked less than half of their required distance go to simpler terrains
            move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        # 这能避免所有高水平 env 永远挤在最后一层，同时维持训练分布多样性。
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        # 每次 reset 后重新开始统计“这一局能走多远”。
        self.max_move_distance[env_ids] = 0.0
        

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        # 这些点定义了“机器人周围地形”如何被离散化采样。
        # 对 rough terrain 任务来说，高度观测的网格范围和密度会显著影响策略学到的步态。
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        # 先把 base-frame 采样点变到 world，再映射到 height map 索引。
        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale


    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        # When low body-height command is active, allow a larger posture deviation.
        if hasattr(self.cfg, 'commands') and hasattr(self.cfg.commands, 'body_height_command_idx') and hasattr(self.cfg.commands, 'body_height_command_threshold'):
            cmd_idx = self.cfg.commands.body_height_command_idx
            if self.commands.shape[1] > cmd_idx:
                low_height_mask = self.commands[:, cmd_idx] > self.cfg.commands.body_height_command_threshold
                rew = rew * (~low_height_mask)
        return rew

    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     base_height = self.root_states[:, 2]
    #     return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_base_height(self):
        # Penalize base height away from target
        # 这里不是直接用 root z，而是估计“机身相对于当前接触脚所在地面”的高度，
        # 对台阶/斜坡场景更稳健。
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        if not hasattr(self, 'last_contacts2'):
            self.last_contacts2 = torch.zeros_like(contact)
        # PhysX 在 mesh 上的接触有时会抖一下就消失，
        # 用当前帧和上一帧做 OR，可得到更稳定的“接触中”判断。
        contact_filt = torch.logical_or(contact, self.last_contacts2)  # (N, 4)
        self.last_contacts2 = contact
        feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        # 至少按 1 个接触脚来除，避免全悬空时出现除零。
        num_feet_contact = torch.sum(contact_filt, dim=1, keepdim=True).clamp(min=1.0)  # (N, 1)
        # 取当前接触脚的平均位置，把它当作“局部地面参考点”。
        feet_contact_pos = (feet_pos * contact_filt.unsqueeze(-1)).sum(dim=1) / num_feet_contact  # (N, 3)
        base_pos = self.root_states[:, 0:3]
        delta_pos = feet_contact_pos - base_pos
        # `projected_gravity` 指向机身坐标系下的“向下”方向，
        # 在它上面投影，相当于求 base 到局部地面的有效高度。
        base_height = (delta_pos * self.projected_gravity).sum(1)  # (N,)
        # 如果当前完全没有接触脚，就不给这项奖励，避免空中阶段被错误惩罚。
        rew = torch.square(base_height - self.cfg.rewards.base_height_target) * (contact_filt.sum(1) > 0)
        return rew

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        # 下限超界产生正惩罚，上限超界也产生正惩罚，
        # 中间安全区域内为 0。
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _get_dynamic_sigma(self, target_vel_abs, v_min, v_max):
        # 动态 sigma 的核心思想：命令越激进、地形越难，tracking reward 就适当放宽容错。
        # compute dynamic sigma based on terrain level
        default_sigma = self.cfg.rewards.tracking_sigma
        if not self.cfg.terrain.curriculum or self.cfg.rewards.dynamic_sigma is None or not hasattr(self, 'terrain_ids'):
            return torch.full_like(target_vel_abs, default_sigma)
        # `terrain_ids` 是“当前 env 属于哪一类 terrain”的编号，
        # 例如 wave / slope / stairs / flat 等。每类 terrain 可以有不同的最大 sigma。
        target_sigmas = self.terrain_max_sigmas[self.terrain_ids]
        sigma = torch.full_like(target_vel_abs, default_sigma)
        # based on velocity ranges, compute sigma
        # v_min <= v < v_max (linear interpolation)
        mask = (target_vel_abs >= v_min) & (target_vel_abs < v_max)
        if mask.any():
            # 在 [v_min, v_max) 区间内做线性插值：
            # 速度越接近上限，sigma 越接近当前 terrain 的目标 sigma。
            ratio = (target_vel_abs[mask] - v_min) / (v_max - v_min)
            sigma[mask] = default_sigma + ratio * (target_sigmas[mask] - default_sigma)
        # v >= v_max
        mask = target_vel_abs >= v_max
        if mask.any():
            sigma[mask] = target_sigmas[mask]
        # based on terrain level, compute sigma
        # terrain level 越高，说明当前 env 难度越大，于是把 sigma 从 default 慢慢推向 target。
        level_scale = torch.clamp(torch.exp((self.terrain_levels.float() + 1.0) / 10.0) - 1.0, max=1.0)
        sigma = default_sigma + level_scale * (sigma - default_sigma)
        return sigma

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # 当前 tracking reward 只关心前 2 维线速度命令。
        # 如果未来新增高度/跳跃类命令，它们不会自动出现在 reward 中，通常需要单独新增 `_reward_*()`。
        if self.cfg.rewards.dynamic_sigma is None:
            sigma_x = sigma_y = self.cfg.rewards.tracking_sigma
        else:
            vmin = self.dynamic_sigma_cfg["min_lin_vel"]
            vmax = self.dynamic_sigma_cfg["max_lin_vel"]
            sigma_x = self._get_dynamic_sigma(torch.abs(self.commands[:, 0]), vmin, vmax)
            sigma_y = self._get_dynamic_sigma(torch.abs(self.commands[:, 1]), vmin, vmax)
        # 这里不是直接对速度误差求和，而是分 x/y 两个方向分别除以各自 sigma，
        # 允许不同方向根据命令强度拥有不同容忍度。
        lin_vel_error_sq = torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2])
        scaled_error = lin_vel_error_sq[:, 0] / sigma_x + lin_vel_error_sq[:, 1] / sigma_y
        # reward 形式是 exp(-error)，误差越小越接近 1，越大越接近 0。
        return torch.exp(-scaled_error)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        # 当前只跟踪 `commands[:, 2]`，也就是 yaw 角速度目标。
        # 若改成“heading 显式进 obs + heading 单独奖励”或新增跳跃相位命令，这里都需要重新审视。
        if self.cfg.rewards.dynamic_sigma is None:
            sigma = self.cfg.rewards.tracking_sigma
        else:
            vmin = self.dynamic_sigma_cfg["min_ang_vel"]
            vmax = self.dynamic_sigma_cfg["max_ang_vel"]
            sigma = self._get_dynamic_sigma(torch.abs(self.commands[:, 2]), vmin, vmax)
        ang_vel_error_sq = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        # yaw 只是一维，因此这里的 tracking 形式比 xy 线速度更简单。
        return torch.exp(-ang_vel_error_sq/sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        # `first_contact` 只在“这只脚刚落地”的那一帧为真，
        # 这样 air-time 奖励只在落地瞬间结算一次，而不是接触期间每帧都重复加分。
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        # 脚在空中时间越久，落地时奖励越高，鼓励更明确的摆腿动作。
        air_time = torch.clamp(self.feet_air_time, max=0.9)
        rew_airTime = torch.sum((air_time - 0.3) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        # 一旦重新接触地面，该脚的 air-time 重新清零，开始下一轮计时。
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        # 如果脚的水平向接触力远大于竖直支撑力，通常意味着撞上了台阶立面/障碍物侧面。
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        # 只有在 command 接近 0 时才启用，防止机器人明明该站住却还在小幅抖腿。
        # Include yaw command so in-place turning does not get misclassified as "zero command".
        stand_still_mask = torch.norm(self.commands[:, :3], dim=1) < 0.1
        if hasattr(self.cfg, 'commands') and hasattr(self.cfg.commands, 'body_height_command_idx') and hasattr(self.cfg.commands, 'body_height_command_threshold'):
            cmd_idx = self.cfg.commands.body_height_command_idx
            if self.commands.shape[1] > cmd_idx:
                low_height_mask = self.commands[:, cmd_idx] > self.cfg.commands.body_height_command_threshold
                stand_still_mask = stand_still_mask & (~low_height_mask)
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * stand_still_mask

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        # 超过阈值的部分才开始惩罚，因此适度触地不会被无意义地压制。
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_action_smoothness(self):
        # a_t - 2a_{t-1} + a_{t-2}
        if not hasattr(self, 'last_last_actions'):
            self.last_last_actions = torch.zeros_like(self.last_actions)
        # 二阶差分越大，说明动作变化越“抖”，这项惩罚会越重。
        rew = torch.sum((self.actions - 2 * self.last_actions + self.last_last_actions).pow(2), dim=1)
        self.last_last_actions[:] = self.last_actions[:]
        return rew
    
    def _reward_dof_power(self):
        # Penalize power consumption
        # 功率近似为 torque * joint velocity，取绝对值后累加，
        # 用于鼓励更节能的步态。
        power = self.torques * self.dof_vel
        rew = torch.sum(torch.abs(power), dim=1)
        return rew

    def _get_base_height(self):
        if not self.cfg.terrain.measure_heights:
            return self.root_states[:, 2]
        # 根据高度扫描点计算base link到地面估计高度
        # `base_height_scan_mask` 只保留机身正下方附近的一小块采样点，
        # 避免把前方台阶/后方坑洞也混进“脚下地面高度”的估计里。
        masked_heights = self.measured_heights * self.base_height_scan_mask.unsqueeze(0)
        sum_heights = masked_heights.sum(dim=1)
        estimated_ground_z = sum_heights / self.num_base_height_scan_points

        base_z = self.root_states[:, 2] 
        base_height = base_z - estimated_ground_z  # (N,)
        return base_height

    def _reward_correct_base_height(self):
        base_height = self._get_base_height()
        # 与 `_reward_base_height()` 相比，这个版本直接用高度扫描估计地面高度，
        # 不依赖脚接触状态，因此在某些悬空/跨越场景里更稳定。
        rew = torch.square(base_height - self.cfg.rewards.base_height_target)
        return rew

    def _reward_feet_regulation(self):
        # CTS抬腿正则奖励, 在脚末端速度增大同时, 要求高度尽可能高
        base_height = self._get_base_height()  # 更新刚体空间位置 (开悟比赛无法修改环境, 只能在奖励中完成计算了)
        feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        feet_xy_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:9]
        base_pos = self.root_states[:, 0:3].unsqueeze(1)
        delta_feet = feet_pos - base_pos
        # 把脚端位置投影到重力方向上，得到“脚相对机身的竖直高度”。
        feet2base_height = (delta_feet * self.projected_gravity.unsqueeze(1)).sum(-1)  # 脚相对于身体的高度 (N, 4)
        # 再结合估计地面高度，得到脚相对地面的净高度。
        feet_height = torch.clamp(base_height.unsqueeze(1) - feet2base_height, min=0.0)  # 脚相对于地面的高度 (N, 4)
        # 速度大但脚抬得不够高时，指数项会更大，从而惩罚更明显；
        # 这会鼓励机器人在快速摆腿时顺带把脚抬高，减少绊地。
        rew = (feet_xy_vel.pow(2).sum(-1) * torch.exp(-feet_height / (0.025 * self.cfg.rewards.base_height_target))).sum(-1)
        return rew

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        # 这是比 `hip_to_default` 更泛化的版本：它约束所有关节，而不只约束 hip。
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_upright(self):
        # `projected_gravity[:, 2]` 越接近 -1，说明机身 z 轴越接近竖直向上。
        return (-1 - self.projected_gravity[:, 2]) / 2
    
    def _reward_legs_distance(self):
        # Penalize legs being too close to each other
        # feet_names: [FL_foot, FR_foot, RL_foot, RR_foot]
        feet_pos_world = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]  # (N, 4, 3)
        base_pos = self.root_states[:, 0:3]  # (N, 3)
        base_quat = self.base_quat  # (N, 4)
        feet_pos_relative_world = feet_pos_world - base_pos.unsqueeze(1)  # (N, 4, 3)
        # 先把世界坐标系下的脚端位置变回 base 坐标系，
        # 这样比较左右腿距离时就不会受机器人整体朝向影响。
        local_pos = quat_rotate_inverse(
            base_quat.repeat_interleave(4, dim=0),
            feet_pos_relative_world.reshape(-1, 3)
        ).reshape(self.num_envs, 4, 3)  # (N, 4, 3)

        # 只看 y 方向间距：前腿看 FL-FR，后腿看 RL-RR。
        dy_front =  local_pos[:, 0, 1] - local_pos[:, 1, 1]  # (N,)
        dy_rear =  local_pos[:, 2, 1] - local_pos[:, 3, 1]   # (N,)
        min_dist = self.cfg.rewards.min_legs_distance

        # 若实际间距小于阈值，就按平方误差惩罚；否则不罚。
        rew_front = torch.square(torch.clamp(min_dist - dy_front, min=0.0))
        rew_rear = torch.square(torch.clamp(min_dist - dy_rear, min=0.0))
        return rew_front + rew_rear
