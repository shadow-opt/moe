import math
from .base_config import BaseConfig

"""LeggedRobot 通用配置基类。

本文件的阅读方式建议是：
1. 先看 `LeggedRobotCfg`，理解环境在 reset / step / reward / observation 阶段会读取哪些字段；
2. 再看 `LeggedRobotCfgPPO` / `LeggedRobotCfgCTS` 等训练配置，理解“环境参数”和“算法参数”是如何分离的；
3. 最后回到具体机器人配置（例如 Go2），只关注它覆写了哪些默认值。

注意：这些类虽然写成嵌套 class，但在运行时会被 `BaseConfig` 递归实例化，
因此使用方式更像 `cfg.env.num_envs` 这样的配置对象，而不是纯静态类常量。
"""

class LeggedRobotCfg(BaseConfig):
    """通用四足机器人环境配置 schema。

    这个类定义的是“环境会用到哪些配置字段”，不是某一台具体机器人的最终参数。
    具体任务通常会继承它，并只覆写和自身相关的子配置。
    """

    class env:
        """环境级参数。

        这里的字段通常会在 `BaseTask`/`LeggedRobot` 初始化时直接决定 buffer 形状、
        episode 长度，以及 rollout 过程中返回给算法的 observation 维度。
        """

        num_envs = 4096
        num_observations = 48
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        # `num_observations` / `num_privileged_obs` 并不只是“文档数字”，
        # 它们会直接决定 `BaseTask` 分配多大的 `obs_buf` / `privileged_obs_buf`。
        # 因此一旦 `compute_observations()` 改了拼接布局，这里必须同步更新。
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        test = False

    class terrain:
        """地形配置。

        这部分会被 `Terrain` 和 `LeggedRobot.create_sim()` 共同消费：
        - `mesh_type` 决定是否创建 plane / heightfield / trimesh；
        - `terrain_proportions` 决定 rough terrain 中各列地形的比例；
        - `measure_heights` 与采样点列表会影响观测中的地形高度输入。
        """

        mesh_type = 'trimesh' # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        terrain_spacing = 0.5 # spacing between different terrain types [m]

        # [wave, slope, rough slope, stairs up, stairs down, obstacles, stepping stones, gap, flat]
        terrain_proportions = [0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        move_down_by_accumulated_xy_command = False # move down the terrain curriculum based on accumulated xy command distance instead of absolute distance

    class commands:
        """高层速度/朝向指令配置。

        `LeggedRobot._resample_commands()` 会大量读取这里的字段。
        从学习角度，可以把它理解成“策略被要求完成什么任务、任务难度如何随训练变化”。
        """

        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        # `num_commands` 决定的是内部 `self.commands` buffer 的容量，
        # 不等于 actor/critic 一定会显式看到同样多的命令维度。
        # 在本仓库里，“命令内部维度”和“送进 observation 的命令维度”经常不是一回事。
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        zero_command_curriculum = None
        # start training with zero commands and then gradually increase zero command probability
        # eg. {'start_iter': 0, 'end_iter': 1500, 'start_value': 0.0, 'end_value': 0.1}
        limit_ang_vel_at_zero_command_prob = 0.0 # probability of add limiting angular velocity commands when zero command is sampled
        limit_vel_prob = 0.0 # probability of limiting linear velocity command
        limit_vel_invert_when_continuous = True # invert the limit logic when using continuous sample limit velocity commands
        limit_vel = {"lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]} # sample vel commands from min [-1] or zero [0] or max [1] range only
        stop_heading_at_limit = True # stop heading updates when vel is limited
        dynamic_resample_commands = False # sample commands with low bounds
        command_range_curriculum = [] # list for command range curriculums at specific training iterations
        # eg: [{
        #     'iter': 20000, # training iteration at which the command ranges are updated
        #     'lin_vel_x': [-1.0, 1.0], # min max [m/s]
        #     'lin_vel_y': [-1.0, 1.0], # min max [m/s]
        #     'ang_vel_yaw': [-2.0, 2.0], # min max [rad/s]
        #     'heading': [-1.57, 1.57], # min max [rad]
        # }]
        turn_over_zero_time = { # if turn_over is true, time robot must be stable before sampling new commands after a turn over
            "backflip": 5.0,
            "sideflip": 3.0,
        }
        # 这里的顺序必须与 terrain 生成逻辑保持一致。
        # 每种 terrain 都可以再施加一层 command range 上限，
        # 用于实现“同一训练批次里，不同地形收到不同难度命令”。
        # [wave, slope, rough slope, stairs up, stairs down, obstacles, stepping stones, gap, flat]
        terrain_max_command_ranges = [
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.5, 1.5], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # wave
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.5, 1.5], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # slope
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.5, 1.5], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # rough slope
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs up
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs down
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # obstacles
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stepping stones
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # gap
            {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.5, 1.5], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # flat
        ]

        class ranges:
            """当前全局 command range。

            它是 command 采样的第一层范围，后续还可能被：
            1. `command_range_curriculum` 按训练迭代更新；
            2. `terrain_max_command_ranges` 按地形再次裁剪。
            """

            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        """reset 时机器人的初始状态。

        其中最重要的是 `default_joint_angles`：
        - 在 position control 下，policy 输出为 0 时，目标角度就是这些默认角；
        - 它同时定义了“默认站姿”，很多奖励也会围绕它来设计。
        """

        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}
        turn_over = False # if true, initialize the robot in a flipped over position
        turn_over_proportions = [0.0, 0.2, 0.8] # proportions for backflip, sideflip, no flip
        turn_over_init_heights = { # initial heights range for each flip type
            'backflip': [0.10, 0.15],
            'sideflip': [0.16, 0.21],
        }

    class control:
        """底层控制器配置。

        `LeggedRobot._compute_torques()` 会把 policy action 解释为：
        - P: 位置目标偏移
        - V: 速度目标
        - T: 直接输出力矩

        因此这里的 `action_scale`、`stiffness`、`damping` 和 `decimation`
        会直接决定动作的物理含义与控制频率。
        """

        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        """URDF/MJCF 资产与碰撞相关配置。"""

        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        """Domain Randomization 配置。

        可以按生效时机分三类理解：
        1. Robot properties：在 actor 创建时或属性更新时改物理参数；
        2. Environment reset：每次 reset 时重采样电机、PD、零偏；
        3. Environment step：在 rollout 中持续施加扰动，例如 push / action delay。
        """

        ### Robot properties ###
        robot_properties_update = None
        # eg: {'start_iter': 5000, 'interval': 5000}

        randomize_friction = True
        friction_range = [0.2, 1.25]

        randomize_base_mass = True
        added_mass_range = [-1., 1.]

        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]

        randomize_base_com = True
        added_base_com_range = [-0.03, 0.03]

        randomize_restitution = False # restitution to robot links (Robot init)
        restitution_range = [0.0, 0.2]

        ### Environment reset ###
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]  
        damping_multiplier_range = [0.9, 1.1]    

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]

        randomize_motor_strength = False # (Env reset)
        motor_strength_range = [0.8, 1.2]

        ### Environment step ###
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6

        randomize_action_delay = False # use last_action with 0~20 ms delay, 4 decimation

    class rewards:
        """奖励配置。

        关键约定：`scales` 中的键名会自动映射到环境类里的 `_reward_<name>()`。
        例如 `tracking_lin_vel` -> `_reward_tracking_lin_vel()`。
        因此新增奖励通常要同时改“配置”和“环境实现”两处。
        """

        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.

        class turn_over_scales:
            upright = 1.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized
        curriculum_rewards = None  # reward names to apply curriculum scaling to, List[dict]
        # eg: [{'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0}]
        dynamic_sigma = None # linear interpolation of sigma based on command velocity, **Must start terrain curriculum first**
        # eg: {
        #     "min_vel": 0.5, # min abs velocity to have default sigma
        #     "max_vel": 1.0, # max abs velocity to have max sigma
        #     # wave, slope, rough_slope, stairs up, stairs down, obstacles, stepping_stones, gap, flat]
        #     "max_sigma": [1/3, 1/4, 1/4, 1/2.7, 1/2.7, 1/2, 1, 1, 1/4]
        # }
        turn_over_roll_threshold = math.pi / 4 # threshold on roll to use turn over rewards
        min_legs_distance = 0.1  # min distance between legs to not be considered stumbling

    class normalization:
        """观测/动作归一化配置。

        这些 scale 不改变物理量本身，只改变喂给 policy 的数值尺度，
        目的是让不同模态的输入落到更适合神经网络学习的范围。
        """

        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 2.5
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        """观测噪声配置。"""

        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        """底层物理仿真配置。"""

        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    """标准 PPO 训练配置。

    这部分不描述机器人本身，而描述 rollout 长度、网络结构、优化超参数等。
    对二次开发来说，应当把它和 `LeggedRobotCfg` 分开理解：
    前者决定“环境长什么样”，后者决定“算法怎么学”。
    """

    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        """Actor / Critic 网络结构。"""

        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        """PPO 优化超参数。"""

        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        """训练循环与 checkpoint 管理配置。"""

        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

    class robogauge:
        enabled = False
        port = 9973

class LeggedRobotCfgCTS(BaseConfig):
    """CTS 系列训练配置基类。

    相比纯 PPO，它显式区分 teacher / student 表征学习，
    因此会多出 latent、encoder、teacher_env_ratio 等参数。
    """

    seed = 0
    runner_class_name = "OnPolicyRunnerCTS"
    history_length = 5
    class policy:
        """CTS policy 结构，包含 teacher/student encoder。"""

        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        teacher_encoder_hidden_dims = [512, 256]
        student_encoder_hidden_dims = [512, 256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        latent_dim = 32
        norm_type = 'l2norm' # normalization type for encoders: l2norm, simnorm
    
    class algorithm:
        """CTS 优化超参数。"""

        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        student_encoder_learning_rate = 1e-3
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        teacher_env_ratio = 0.75  # percentage of envs assigned to teacher
        # teacher_env_ratio = 1.00  # percentage of envs assigned to teacher

    class runner:
        """CTS 训练主循环配置。"""

        policy_class_name = 'ActorCriticCTS'
        algorithm_class_name = 'CTS'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

    class robogauge:
        enabled = False
        port = 9973

class LeggedRobotCfgMoENGCTS(LeggedRobotCfgCTS):
    """MoE + no-goal mask 版本的 CTS 配置。"""

    class policy(LeggedRobotCfgCTS.policy):
        obs_no_goal_mask = None # mask for observation without goal inputs
        student_expert_num = 8 # number of experts in the student model
    
    class algorithm(LeggedRobotCfgCTS.algorithm):
        load_balance_coef = 0.01  # coefficient for load balance loss

    class runner(LeggedRobotCfgCTS.runner):
        policy_class_name = 'ActorCriticMoENGCTS'
        algorithm_class_name = 'MoENGCTS'

class LeggedRobotCfgMCPCTS(LeggedRobotCfgCTS):
    """MCPCTS 配置。"""

    class policy(LeggedRobotCfgCTS.policy):
        obs_no_goal_mask = None # mask for observation without goal inputs
        student_expert_num = 8 # number of experts in the student model

    class runner(LeggedRobotCfgCTS.runner):
        policy_class_name = 'ActorCriticMCPCTS'
        algorithm_class_name = 'MCPCTS'

class LeggedRobotCfgACMoECTS(LeggedRobotCfgCTS):
    """Actor-Critic 双路都用 MoE 的 CTS 配置。"""

    class policy(LeggedRobotCfgCTS.policy):
        expert_num = 8 # number of experts in the student model

    class runner(LeggedRobotCfgCTS.runner):
        policy_class_name = 'ActorCriticACMoECTS'
        algorithm_class_name = 'ACMoECTS'

class LeggedRobotCfgDualMoECTS(LeggedRobotCfgCTS):
    """Dual MoE CTS 配置。"""

    class policy(LeggedRobotCfgCTS.policy):
        expert_num = 8 # number of experts in the student model
        student_encoder_hidden_dims = [512, 256, 256]

    class runner(LeggedRobotCfgCTS.runner):
        policy_class_name = 'ActorCriticDualMoECTS'
        algorithm_class_name = 'DualMoECTS'

class LeggedRobotCfgMoECTS(LeggedRobotCfgCTS):
    """通用 MoE CTS 配置。"""

    class policy(LeggedRobotCfgCTS.policy):
        expert_num = 8 # number of experts in the student model
        student_encoder_hidden_dims = [512, 256, 256]

    class algorithm(LeggedRobotCfgCTS.algorithm):
        load_balance_coef = 0.01  # coefficient for load balance loss

    class runner(LeggedRobotCfgCTS.runner):
        policy_class_name = 'ActorCriticMoECTS'
        algorithm_class_name = 'MoECTS'
