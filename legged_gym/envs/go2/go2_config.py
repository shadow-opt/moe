import math
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, LeggedRobotCfgCTS, LeggedRobotCfgMoENGCTS, LeggedRobotCfgMoENGCTS, LeggedRobotCfgMCPCTS, LeggedRobotCfgACMoECTS, LeggedRobotCfgDualMoECTS, LeggedRobotCfgMoECTS

"""Go2 主线配置。

阅读建议：
1. 先把它当作 `LeggedRobotCfg` 的具体实例化；
2. 再对照 `go2_env.py` 看观测、奖励、噪声如何与这里一一对应；
3. 最后再看 PPO/CTS/MoE 训练配置，理解同一套环境参数可复用到多种训练器。
"""

class GO2Cfg(LeggedRobotCfg):
    """Go2 的环境配置。

    这里主要回答三个问题：
    - 机器人 reset 时长什么样；
    - rollout 过程中允许收到什么 command；
    - reward / randomization / terrain 如何共同塑造训练目标。
    """

    class init_state(LeggedRobotCfg.init_state):
        """Go2 初始姿态。

        `default_joint_angles` 同时承担两层含义：
        1. reset 后的默认站姿；
        2. position control 下 action=0 时的目标关节角。
        因此它既影响起步稳定性，也影响策略的“零点”。
        """

        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
        turn_over = False # initialize the robot in a flipped over position
        # turn_over 系列配置用于“翻倒后自行恢复/起身”类训练。
        # 当前默认关闭，因此主线 locomotion 训练不会触发这套 reset 逻辑。
        # turn_over_proportions = [0.1, 0.3, 0.6] # proportions for backflip, sideflip, noflip
        turn_over_proportions = [0.0, 0.2, 0.8] # proportions for backflip, sideflip, noflip
        turn_over_init_heights = { # initial heights range for each flip type
            'backflip': [0.10, 0.15],
            'sideflip': [0.16, 0.21],
        }
        # turn_over_proportions = [0.0, 1.0, 0.0] # proportions for backflip, sideflip, noflip

    class env(LeggedRobotCfg.env):
        """Go2 rollout 维度配置。"""

        num_envs = 8192
        # 对应 `Go2Robot.compute_observations()` 中 actor obs 的拼接结果。
        # 目前 45 维的组成是：3(base_ang_vel) + 3(projected_gravity) + 3(commands) + 12(dof_pos) + 12(dof_vel) + 12(actions)。
        # 如果以后把“高度命令”“跳跃命令”等显式拼进 actor obs，这里必须同步增加。
        num_observations = 45
        # privileged obs = actor obs(45)
        #                + base_lin_vel(3)
        #                + foot contact forces(4)
        #                + torques(12)
        #                + dof accelerations(12)
        #                + terrain heights(187)
        num_privileged_obs = 45 + 3 + 4 + 12 + 12 + 187  # 263
        # 新手调维度时，最稳妥的做法永远是“先改环境拼接，再回头改这里”，
        # 而不是只在配置里拍脑袋改一个数字。
        # num_privileged_obs = 45 + 3 + 187  # 235
        # num_privileged_obs = 48  # without height measurements
        episode_length_s = 25

    class domain_rand(LeggedRobotCfg.domain_rand):
        """Go2 的 domain randomization。

        这一组参数的主要目标是：
        - 让 policy 不过拟合某个固定摩擦/质量/电机模型；
        - 让 sim 中学到的步态对真机误差更鲁棒。
        """

        ### Robot properties ###
        randomize_friction = True
        friction_range = [0.0, 2.0]

        randomize_base_mass = True
        added_mass_range = [-1., 1.]

        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]

        randomize_base_com = True
        added_base_com_range = [-0.03, 0.03]

        randomize_restitution = True # restitution to robot links (Robot init)
        restitution_range = [0.0, 0.5]

        ### Environment reset ###
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]  
        damping_multiplier_range = [0.9, 1.1]    

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]

        randomize_motor_strength = True # (Env reset)
        motor_strength_range = [0.8, 1.2]

        ### Environment step ###
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6

        randomize_action_delay = True # use last_action with 0~20 ms delay, 4 decimation

    class control(LeggedRobotCfg.control):
        """Go2 默认采用 position PD 控制。"""

        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.0}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class terrain(LeggedRobotCfg.terrain):
        """主线 Go2 地形配置。

        这里最大的阅读坑点是：`terrain_proportions` 在 Python 里以后赋值为准，
        所以真正生效的是最后一条没有被注释掉的赋值，而不是上面的历史实验记录。
        """

        max_init_terrain_level = 5
        # [wave, slope, rough_slope, stairs up, stairs down, obstacles, stepping_stones, gap, flat]
        # terrain_proportions = [0.2, 0.05, 0.05, 0.30, 0.05, 0.25, 0.0, 0.0, 0.1]  # 更偏向wave
        # terrain_proportions = [0.05, 0.20, 0.05, 0.25, 0.10, 0.20, 0.0, 0.0, 0.15]  # 这个更偏向平地斜坡
        # terrain_proportions = [0.20, 0.05, 0.05, 0.30, 0.15, 0.20, 0.0, 0.0, 0.05]  # 更偏向wave和stairs
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        # terrain_proportions = [0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        terrain_proportions = [0.10, 0.05, 0.05, 0.20, 0.2, 0.20, 0.1, 0.05, 0.05]
        move_down_by_accumulated_xy_command = True # move down the terrain curriculum based on accumulated xy command distance instead of absolute distance
        
    class commands(LeggedRobotCfg.commands):
        """Go2 主线 command 采样策略。

        这一版不是最 vanilla 的 locomotion 设置，而是带有：
        - zero command curriculum
        - limit velocity sampling
        - dynamic command resampling
        - command range curriculum

        也就是会同时控制“命令有多难”“命令是否需要静止”“何时扩大速度范围”。
        """

        curriculum = False
        max_curriculum = 1.
        # 命令张量 `self.commands` 的内部容量。
        # 当前语义约定为：[lin_vel_x, lin_vel_y, ang_vel_yaw, heading]。
        # 但要特别注意：当前 actor observation 实际只显式使用前 3 维。
        # 第 4 维 `heading` 只有在 `heading_command=True` 时才作为内部目标使用。
        # 如果未来要新增“高度”“跳跃强度”等命令，常见做法是：
        # 1. 先在这里增加 `num_commands`；
        # 2. 再同步修改 command 采样、command scale、obs 拼接、奖励、导出部署链路。
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw (in heading mode ang_vel_yaw is recomputed from heading error)
        # `num_commands=4` 的真正含义是：环境内部始终为 heading 预留了第 4 个槽位；
        # 但当前主线 actor obs 只吃前 3 维，所以别把“内部有 4 维”误读成“策略已经看到了 4 维”。
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        # 训练早期先减少“停住不动”的任务，后期逐步增加 zero-command 比例，
        # 让策略既会走，也会稳。
        zero_command_curriculum = {'start_iter': 0, 'end_iter': 1500, 'start_value': 0.0, 'end_value': 0.1}
        limit_ang_vel_at_zero_command_prob = 0.2 # probability of add limiting angular velocity commands when zero command is sampled
        limit_vel_prob = 0.2 # probability of limiting linear velocity command
        limit_vel_invert_when_continuous = True # invert the limit logic when using continuous sample limit velocity commands
        # `limit_vel` 表示离散限制模板：从 min / 0 / max 中挑选，
        # 用于构造更极端、更规则的控制目标，帮助策略学会刹停、纯前进、纯转向等边界情况。
        limit_vel = {"lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]} # sample vel commands from min [-1] or zero [0] or max [1] range only
        stop_heading_at_limit = True # stop heading updates when vel is limited
        # dynamic_resample_commands 会根据剩余 episode 时间与剩余可走距离，
        # 给命令采样设置一个 lower bound，避免采到“理论上走不完”的过慢命令。
        dynamic_resample_commands = True # sample commands with low bounds
        command_range_curriculum = [{ # list for command range curriculums at specific training iterations
            'iter': 20000, # training iteration at which the command ranges are updated
            'lin_vel_x': [-1.0, 1.0], # min max [m/s]
            'lin_vel_y': [-1.0, 1.0], # min max [m/s]
            'ang_vel_yaw': [-1.5, 1.5], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        }, { # list for command range curriculums at specific training iterations
            'iter': 50000, # training iteration at which the command ranges are updated
            'lin_vel_x': [-2.0, 2.0], # min max [m/s]
            'lin_vel_y': [-1.0, 1.0], # min max [m/s]
            'ang_vel_yaw': [-2.0, 2.0], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        }]
        turn_over_zero_time = { # if turn_over is true, time robot must be stable before sampling new commands after a turn over
            "backflip": 5.0,
            "sideflip": 3.0,
        }
        # terrain-wise 上限，相当于“全局命令范围”和“地形可承受范围”的交集裁剪。
        # [wave, slope, rough slope, stairs up, stairs down, obstacles, stepping stones, gap, flat]
        terrain_max_command_ranges = [
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # wave
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # slope
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # rough slope
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs up
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs down
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # obstacles
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stepping stones
            {'lin_vel_x': [-0.5, 0.5], 'lin_vel_y': [-0.5, 0.5], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [-1.57, 1.57]},  # gap
            {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0], 'heading': [-1.57, 1.57]},  # flat
        ]

        class ranges:
            """初始命令范围。

            注意这只是训练起点，后续会被 `command_range_curriculum` 扩到更大范围。
            新增命令时，也通常要在这里补上对应的取值范围，否则 `_parse_cfg()` / `_update_env_command_ranges()` 无法统一处理。
            """

            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.5, 0.5] # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]   # min max [rad/s]
            heading = [-1.57, 1.57] # min max [rad]
        
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards(LeggedRobotCfg.rewards):
        """Go2 主线奖励设计。

        如果想理解“策略为什么学成当前步态”，这里是最重要的入口。
        建议配合 `LeggedRobot._prepare_reward_function()` 一起看：
        `scales` 里的每一个名字都必须在环境类中存在同名 `_reward_*` 实现。
        """

        soft_dof_pos_limit = 0.9
        base_height_target = 0.38
        only_positive_rewards = False
        max_contact_force = 147. # forces above this value are penalized, go2 weight 15kg
        # curriculum_rewards 用来让某些 reward scale 随训练迭代线性变化。
        # 常见用法是：训练前期多扶一把，后期再逐渐撤掉辅助项。
        curriculum_rewards = [
            {'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
            {'reward_name': 'correct_base_height', 'start_iter': 0, 'end_iter': 5000, 'start_value': 1.0, 'end_value': 10.0},
            # {'reward_name': 'dof_power', 'start_iter': 0, 'end_iter': 3000, 'start_value': 1.0, 'end_value': 0.1},
            # {'reward_name': 'upright', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
        ]
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        # dynamic_sigma 会随着命令速度和 terrain 难度改变 tracking reward 的容忍度。
        # 直观理解：速度越大 / 地形越难时，允许 tracking error 稍微大一些，
        # 否则高速复杂地形下 reward 会过于苛刻。
        dynamic_sigma = { # linear interpolation of sigma based on command velocity, **Must start terrain curriculum first**
            "min_lin_vel": 0.5, # min abs linear velocity to have default sigma
            "max_lin_vel": 1.5, # max abs linear velocity to have max sigma
            "min_ang_vel": 1.0, # min abs angular velocity to have default sigma
            "max_ang_vel": 2.0, # max abs angular velocity to have max sigma
            # wave, slope, rough_slope, stairs up, stairs down, obstacles, stepping_stones, gap, flat]
            # "max_sigma": [1/3, 1/4, 1/4, 1/2.7, 1/2.7, 1/2, 1, 1, 1/4]
            "max_sigma": [5/12, 1/4, 1/4, 1/2, 1/2, 3/4, 1, 1, 1/4]
        }
        min_legs_distance = 0.1  # min distance between legs to not be considered stumbling
        class scales:
            """奖励权重表。

            正值通常表示鼓励，负值表示惩罚。
            这些数值不会原样参与 reward，相乘前还会在 `LeggedRobot._prepare_reward_function()` 中乘上 `dt`。
            """

            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            dof_power = -2e-5
            torques = -1e-4
            correct_base_height = -1.0
            action_rate = -0.01
            action_smoothness = -0.01
            collision = -1.0
            dof_pos_limits = -2.0
            feet_regulation = -0.05
            # CTS reward trains to have very close feet distance, real robot performance is poor, but sim2sim can climb 20cm stairs, try to add hip_to_default reward or similar_to_default reward
            # training to y=1.5, font feet will collide noticeably, max y=1.0
            hip_to_default = -0.05
            # legs_distance = -1.5  # not good performance, avoid leg collision
            # similar_to_default = -0.01
            # feet_contact_forces = -1.0  # try to add but no effect, remove

        turn_over_roll_threshold = math.pi / 4 # threshold on roll to use turn over rewards
        class turn_over_scales:
            """翻身训练专用奖励权重。"""

            upright = 1.0
            # dof_acc = -2.5e-7
            # dof_power = -2e-5
            # action_rate = -0.001
            # action_smoothness = -0.001

    class noise(LeggedRobotCfg.noise):
        add_noise = True

class GO2CfgPPO(LeggedRobotCfgPPO):
    """Go2 + PPO 训练配置。"""

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_ppo'
        max_iterations = 150000
        save_interval = 500

class GO2CfgCTS(LeggedRobotCfgCTS):
    """Go2 + CTS 训练配置。"""

    class runner(LeggedRobotCfgCTS.runner):
        num_steps_per_env = 24
        run_name = ''
        experiment_name = 'go2_cts'
        max_iterations = 150000
        save_interval = 500
    
    class policy(LeggedRobotCfgCTS.policy):
        latent_dim = 32
        norm_type = 'l2norm'

class GO2CfgMoENGCTS(LeggedRobotCfgMoENGCTS):
    """Go2 + MoE No-Goal CTS 配置。"""

    class policy(LeggedRobotCfgMoENGCTS.policy):
        # 这里默认假设 command 只占 actor obs 中的 3 维槽位。
        # 如果未来扩展显式命令维度，相关 mask 也必须同步修改，否则 MoE/goal-mask 会错位。
        obs_no_goal_mask = [True] * 6 + [False] * 3 + [True] * 36  # mask for obs without command info
        student_expert_num = 8 # number of experts in the student model
    
    class algorithm(LeggedRobotCfgMoENGCTS.algorithm):
        load_balance_coef = 0.01

    class runner(LeggedRobotCfgMoENGCTS.runner):
        run_name = ''
        experiment_name = 'go2_moe_no_goal_cts'
        max_iterations = 150000
        save_interval = 500

class GO2CfgMCPCTS(LeggedRobotCfgMCPCTS):
    """Go2 + MCPCTS 配置。"""

    class policy(LeggedRobotCfgMCPCTS.policy):
        obs_no_goal_mask = [True] * 6 + [False] * 3 + [True] * 36  # mask for obs without command info
        student_expert_num = 8 # number of experts in the student model

    class runner(LeggedRobotCfgMCPCTS.runner):
        run_name = ''
        experiment_name = 'go2_mcp_cts'
        max_iterations = 150000
        save_interval = 500

class GO2CfgACMoECTS(LeggedRobotCfgACMoECTS):
    """Go2 + ACMoECTS 配置。"""

    class policy(LeggedRobotCfgACMoECTS.policy):
        expert_num = 8  # number of experts in the student model
    
    class runner(LeggedRobotCfgACMoECTS.runner):
        run_name = ''
        experiment_name = 'go2_ac_moe_cts'
        max_iterations = 150000
        save_interval = 500

class GO2CfgDualMoECTS(LeggedRobotCfgDualMoECTS):
    """Go2 + DualMoECTS 配置。"""

    class policy(LeggedRobotCfgDualMoECTS.policy):
        expert_num = 8  # number of experts in the student model
    
    class runner(LeggedRobotCfgDualMoECTS.runner):
        run_name = ''
        experiment_name = 'go2_dual_moe_cts'
        max_iterations = 150000
        save_interval = 500

class GO2CfgMoECTS(LeggedRobotCfgMoECTS):
    """Go2 + MoECTS 配置。"""

    class policy(LeggedRobotCfgMoECTS.policy):
        expert_num = 8  # number of experts in the student model
    
    class runner(LeggedRobotCfgMoECTS.runner):
        run_name = ''
        experiment_name = 'go2_moe_cts'
        max_iterations = 150000
        save_interval = 500
