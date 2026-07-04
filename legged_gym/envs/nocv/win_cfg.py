from legged_gym.envs.go2.go2_config import GO2Cfg
from legged_gym.envs.go2.go2_config import GO2CfgMoECTS
from legged_gym.envs.go2.go2_config import GO2CfgCTS

class WINCfg(GO2Cfg):
    class init_state(GO2Cfg.init_state):
        turn_over = False
        turn_over_proportions = [0.0, 0.0, 0.0] # proportions for backflip, sideflip, noflip
        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #     'FL_hip_joint': 0.1,   # [rad]
        #     'RL_hip_joint': 0.1,   # [rad]
        #     'FR_hip_joint': -0.1,  # [rad]
        #     'RR_hip_joint': -0.1,   # [rad]

        #     'FL_thigh_joint': 0.8,     # [rad]
        #     'RL_thigh_joint': 1.0,   # [rad]
        #     'FR_thigh_joint': 0.8,     # [rad]
        #     'RR_thigh_joint': 1.0,   # [rad]

        #     'FL_calf_joint': -1.4,   # [rad]
        #     'RL_calf_joint': -1.4,    # [rad]
        #     'FR_calf_joint': -1.4,  # [rad]
        #     'RR_calf_joint': -1.4,    # [rad]
        # }

    class domain_rand(GO2Cfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-2, 2]
        randomize_link_mass = True
        multiplied_link_mass_range = [0.85, 1.15]
        


    class asset(GO2Cfg.asset):
        collapse_fixed_joints = False
        name = 'z2'
        """
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/j60/urdf/z2_heavy.urdf'
        foot_name = 'FOOT'
        flip_visual_attachments = True
        """
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/aaaaa_fixed/z2/urdf/z2.urdf'
        foot_name = 'foot'
        flip_visual_attachments = False
        
        terminate_after_contacts_on = ["base","trunk"]
        penalize_contacts_on = ["thigh", "calf", "hip"]
        self_collisions = 1 # 1关闭自碰撞，0开启自碰撞
        
        
    class control(GO2Cfg.control):
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
        stiffness = {'joint': 30.0}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class env(GO2Cfg.env):
        # actor obs = 3(base_ang_vel) + 3(projected_gravity) + 6(command obs) + 12(dof_pos) + 12(dof_vel) + 12(actions)
        num_observations = 48
        # privileged obs = actor obs(48) + base_lin_vel(3) + foot contact forces(4) + torques(12)
        #                + dof accelerations(12) + terrain heights(187)
        num_privileged_obs = 48 + 3 + 4 + 12 + 12 + 187
        num_one_step_obs = 48

    class terrain(GO2Cfg.terrain):
        # NoCV 训练更偏向楼梯/障碍类地形，但仍保留部分平地与斜坡，
        # 低高度档位只在 slope / rough_slope / flat 上启用。
        # mesh_type = 'plane'
        # terrain_proportions = [0.15, 0.05, 0.1, 0.3, 0.1, 0.0, 0.1, 0.0, 0.2]
        terrain_proportions = [0.15, 0.05, 0.1, 0.2, 0.1, 0.2, 0.0, 0.0, 0.2]
        # terrain_proportions = [0, 0, 0, 0, 0, 0, 0.3, 0.1, 0.3]
        # [wave, slope, rough_slope, stairs up, stairs down, obstacles, stones, gap, flat]
        
    class commands(GO2Cfg.commands):

        num_commands = 7 # 这是buffer，比command多1维 实际输入-1
        zero_command_curriculum = {'start_iter': 0, 'end_iter': 1500, 'start_value': 0.0, 'end_value': 0.07}
        full_stop_command_curriculum = {'start_iter': 0, 'end_iter': 1500, 'start_value': 0.0, 'end_value': 0.03}

        body_height_command_idx = 4 # buffer的第5维，索引第4维
        body_height_command_obs_scale = 1.0 # 缩放
        
        body_height_command_threshold = 0.5
        # 只有 command 值大于该阈值时，才视为“低高度档位”。
        # 当前 low command = 1.0，normal command = 0.0，因此 0.5 是自然分界。

        # 正常高度档位的 command 值。
        normal_body_height_command = 0.0
        # 低高度档位的 command 值。
        low_body_height_command = 1.0 # 对允许的地形，重采样 command 时有多大概率切到低高度档位。
        low_height_command_prob = 0.3
        low_height_force_normal_after_low = True
        low_height_require_full_release = True
        # 仅在这些 terrain id 上允许采样低高度档位：
        # 1 = slope, 2 = rough_slope, 8 = flat
        low_height_terrain_ids = [8]  # slope, rough_slope, flat
        # 当采到低高度档位时，对速度命令做额外缩放：
        # [lin_vel_x, lin_vel_y, ang_vel_yaw]
        # 这样可以让机器人在压低机身时适当放慢速度，提高可学性与稳定性。
        low_height_command_velocity_scale = [0.4, 0.2, 0.5]

        special_terrain_options_5 = [3]
        special_terrain_options_6 = [6]
        # 对于 stairs up  3
        # 对于 stones 6
        special_terrain_probs = 0.0
        special_terrain_command = 0
        normal_terrain_command = 0
        special_terrain_command_obs_scale = 1.0

                # 给命令采样设置一个 lower bound，避免采到“理论上走不完”的过慢命令。
        dynamic_resample_commands = True # sample commands with low bounds
        command_range_curriculum = [{ # list for command range curriculums at specific training iterations
            'iter': 20000, # training iteration at which the command ranges are updated
            'lin_vel_x': [-1.0, 1.0], # min max [m/s]
            'lin_vel_y': [-1.0, 1.0], # min max [m/s]
            'ang_vel_yaw': [-1.5, 1.5], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        },{ # list for command range curriculums at specific training iterations
            'iter': 40000, # training iteration at which the command ranges are updated
            'lin_vel_x': [-2.0, 2.0], # min max [m/s]
            'lin_vel_y': [-1.0, 1.0], # min max [m/s]
            'ang_vel_yaw': [-1.7, 1.7], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        }
        ]
        
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
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-0.0, 0.0], 'ang_vel_yaw': [-0, 0], 'heading': [-0.0, 0.0]},  # stairs up
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-0.0, 0.0], 'ang_vel_yaw': [-0, 0], 'heading': [-0.0, 0.0]},  # stairs down
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # obstacles
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [-1.57, 1.57]},  # stepping stones
            {'lin_vel_x': [-0.5, 0.5], 'lin_vel_y': [-0.5, 0.5], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [-1.57, 1.57]},  # gap
            {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0], 'heading': [-1.57, 1.57]},  # flat
        ]


    class rewards(GO2Cfg.rewards):
        # 正常档位继续沿用父类中的 `base_height_target`。
        # 低高度档位下，改为追踪这个更低的目标高度。
        low_base_height_target = 0.18
        base_height_target = 0.37
        soft_dof_pos_limit = 0.8
        soft_dof_vel_limit = 0.85
        soft_torque_limit = 0.75
        foot_slip_deadzone = 0.02
        foot_slip_excluded_terrain_ids = [3, 4]
        low_speed_feet_air_time_min = 0.05
        low_speed_feet_air_time_max = 0.35
        stand_still_default_pose_settle_sigma = 0.04
        curriculum_rewards = [
            {'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
            {'reward_name': 'correct_base_height', 'start_iter': 0, 'end_iter': 5000, 'start_value': 1.0, 'end_value': 10.0},
            {'reward_name': 'ang_vel_xy', 'start_iter': 10000, 'end_iter': 30000, 'start_value': 1.0, 'end_value': 1.8},
            # {'reward_name': 'foot_slip', 'start_iter': 30000, 'end_iter': 60000, 'start_value': 1.0, 'end_value': 2.0},
            # {'reward_name': 'x_command_hip_regular', 'start_iter': 30000, 'end_iter': 60000, 'start_value': 1.0, 'end_value': 10.0},
            {'reward_name': 'stand_still', 'start_iter': 10000, 'end_iter': 40000, 'start_value': 1.0, 'end_value': 3.0},
            # {'reward_name': 'hip_to_default', 'start_iter': 20000, 'end_iter': 70000, 'start_value': 1.0, 'end_value': 0.4},
            {'reward_name': 'lateral_yaw_tracking_error', 'start_iter': 0, 'end_iter': 70000, 'start_value': 1.0, 'end_value': 5.0},
            {'reward_name': 'hip_to_zero', 'start_iter': 0, 'end_iter': 70000, 'start_value': 1.0, 'end_value': 20.0},
            # {'reward_name': 'dof_power', 'start_iter': 0, 'end_iter': 3000, 'start_value': 1.0, 'end_value': 0.1},
            # {'reward_name': 'upright', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
        ]
        dynamic_sigma = { # linear interpolation of sigma based on command velocity, **Must start terrain curriculum first**
            "min_lin_vel": 0.5, # min abs linear velocity to have default sigma
            "max_lin_vel": 1.5, # max abs linear velocity to have max sigma
            "min_ang_vel": 1.0, # min abs angular velocity to have default sigma
            "max_ang_vel": 2.0, # max abs angular velocity to have max sigma
            # wave, slope, rough_slope, stairs up, stairs down, obstacles, stepping_stones, gap, flat]
            # "max_sigma": [1/3, 1/4, 1/4, 1/2.7, 1/2.7, 1/2, 1, 1, 1/4]
            "max_sigma": [5/12, 1/4, 1/4, 1/2, 1/2, 1, 3/4, 1, 1/4]
        }
        class scales(GO2Cfg.rewards.scales):
            # straight_path = 10.0 # [NOTE] 新增
            # straight_path_deviation = -3 # [NOTE] 新增
            ang_vel_xy = -0.05
            lateral_yaw_tracking_error = -0.3
            stand_still = -1.0
            stand_still_default_pose = 0.0

            hip_to_default = 0.0
            hip_to_zero = -0.5
            torques = -1e-4
            dof_vel_limits = -2.5
            dof_pos_limits = -4.0
            action_rate = -0.01
            feet_air_time = 1.0
            action_smoothness = -0.01
            foot_slip = -0.01
            low_speed_feet_air_time = 0.5
            feet_air_time_variance = -0.4
            feet_contact_without_cmd = 0.03
            x_command_hip_regular = -0.5
            
            
            
class WINCfgMoECTS(GO2CfgMoECTS):
    """WIN 对应的 MoE CTS 训练配置。

    这里只保留训练器侧配置继承；环境侧实际会配合 `WINCfg` 一起注册使用。
    """

    class runner(GO2CfgMoECTS.runner):
        run_name = 'new_inertial'
        experiment_name = 'win_moe_cts'
        max_iterations = 120000
        save_interval = 5000


class WINGo2Cfg(WINCfg):
    """WIN environment with GO2 reward scales plus selected low-height guards."""
    class commands(WINCfg.commands):
        flat_low_speed_command_prob = 0.2
        flat_low_speed_terrain_ids = [8]
        flat_low_speed_command_ranges = {
            "lin_vel_x": [-0.5, 0.5],
            "lin_vel_y": [-0.4, 0.4],
            "ang_vel_yaw": [-0.8, 0.8],
        }
        monotonic_command_prob = 0.3
        monotonic_command_type_probs = [0.45, 0.25, 0.30]
    class terrain(WINCfg.terrain):
        terrain_proportions = [0.15, 0.05, 0.1, 0.25, 0.1, 0.20, 0.0, 0.0, 0.15]
        
    class rewards(WINCfg.rewards):
        """ curriculum_rewards = [
            item for item in WINCfg.rewards.curriculum_rewards
            if item['reward_name'] != 'hip_to_default'
        ] """
        curriculum_rewards = [
            {'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
            {'reward_name': 'correct_base_height', 'start_iter': 0, 'end_iter': 5000, 'start_value': 1.0, 'end_value': 5.0},
            {'reward_name': 'ang_vel_xy', 'start_iter': 10000, 'end_iter': 30000, 'start_value': 1.0, 'end_value': 1.5},
            # {'reward_name': 'foot_slip', 'start_iter': 30000, 'end_iter': 60000, 'start_value': 1.0, 'end_value': 2.0},
            # {'reward_name': 'x_command_hip_regular', 'start_iter': 30000, 'end_iter': 60000, 'start_value': 1.0, 'end_value': 10.0},
            {'reward_name': 'stand_still', 'start_iter': 10000, 'end_iter': 40000, 'start_value': 1.0, 'end_value': 3.0},
            # {'reward_name': 'hip_to_default', 'start_iter': 20000, 'end_iter': 70000, 'start_value': 1.0, 'end_value': 0.4},
            {'reward_name': 'lateral_yaw_tracking_error', 'start_iter': 0, 'end_iter': 70000, 'start_value': 1.0, 'end_value': 5.0},
            {'reward_name': 'hip_to_zero', 'start_iter': 0, 'end_iter': 70000, 'start_value': 1.0, 'end_value': 20.0},
            # {'reward_name': 'dof_power', 'start_iter': 0, 'end_iter': 3000, 'start_value': 1.0, 'end_value': 0.1},
            # {'reward_name': 'upright', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
        ]
        
        class scales(GO2Cfg.rewards.scales):
            hip_to_default = 0.0
            stand_still = -1
            stand_still_default_pose = 0.0
            lateral_yaw_tracking_error = -0.3
            hip_to_zero = -0.5
            feet_air_time = 1.0
            feet_air_time_variance = -0.4
            feet_contact_without_cmd = 0.03


class WINGo2CfgMoECTS(WINCfgMoECTS):
    """MoE CTS runner config for the WIN + GO2 reward-scale variant."""

    class runner(WINCfgMoECTS.runner):
        run_name = 'go2_reward_scales'
        experiment_name = 'win_go2_moe_cts'
        max_iterations = 120000
        save_interval = 5000


class WINGo2StairCfg(WINGo2Cfg):
    """WIN GO2-reward variant using GO2 terrain-specific command ranges."""

    class commands(WINGo2Cfg.commands):
        terrain_max_command_ranges = GO2Cfg.commands.terrain_max_command_ranges


class WINGo2StairCfgMoECTS(WINGo2CfgMoECTS):
    """MoE CTS runner config for the stair-command WIN GO2 variant."""

    class runner(WINGo2CfgMoECTS.runner):
        run_name = 'go2_reward_scales_stair_commands'
        experiment_name = 'win_go2_stair_moe_cts'
        max_iterations = 120000
        save_interval = 5000


class WINLowSpeedCfg(WINCfg):
    """Low-speed WIN variant with tighter joint/action regularization."""

    class terrain(WINCfg.terrain):
        # [wave, slope, rough_slope, stairs up, stairs down, obstacles, stones, gap, flat]
        terrain_proportions = [0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4]

    class commands(WINCfg.commands):
        low_height_command_prob = 0.3
        low_height_terrain_ids = [8]
        low_height_command_velocity_scale = [0.4, 0.0, 0.0]

        dynamic_resample_commands = False
        command_range_curriculum = []

        terrain_max_command_ranges = [
            {'lin_vel_x': [-0.8, 0.8], 'lin_vel_y': [-0.4, 0.4], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # wave
            {'lin_vel_x': [-0.8, 0.8], 'lin_vel_y': [-0.4, 0.4], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # slope
            {'lin_vel_x': [-0.8, 0.8], 'lin_vel_y': [-0.4, 0.4], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # rough_slope
            {'lin_vel_x': [-0.8, 0.8], 'lin_vel_y': [-0.4, 0.4], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # stairs up
            {'lin_vel_x': [-0.8, 0.8], 'lin_vel_y': [-0.4, 0.4], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # stairs down
            {'lin_vel_x': [-0.8, 0.8], 'lin_vel_y': [-0.4, 0.4], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # obstacles
            {'lin_vel_x': [-0.8, 0.8], 'lin_vel_y': [-0.4, 0.4], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # stepping stones
            {'lin_vel_x': [-0.8, 0.8], 'lin_vel_y': [-0.4, 0.4], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # gap
            {'lin_vel_x': [-0.8, 0.8], 'lin_vel_y': [-0.4, 0.4], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # flat
        ]

        class ranges(WINCfg.commands.ranges):
            lin_vel_x = [-0.8, 0.8]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [0.0, 0.0]

    class rewards(WINCfg.rewards):
        soft_dof_pos_limit = 0.8
        curriculum_rewards = [
            {'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 500, 'start_value': 1.0, 'end_value': 0.0},
            {'reward_name': 'correct_base_height', 'start_iter': 0, 'end_iter': 2000, 'start_value': 1.0, 'end_value': 8.0},
            {'reward_name': 'ang_vel_xy', 'start_iter': 3000, 'end_iter': 10000, 'start_value': 1.0, 'end_value': 2.0},
            {'reward_name': 'stand_still', 'start_iter': 3000, 'end_iter': 12000, 'start_value': 1.0, 'end_value': 4.0},
            # {'reward_name': 'hip_to_default', 'start_iter': 4000, 'end_iter': 16000, 'start_value': 1.0, 'end_value': 0.4},
            {'reward_name': 'lateral_yaw_tracking_error', 'start_iter': 0, 'end_iter': 16000, 'start_value': 1.0, 'end_value': 5.0},
            {'reward_name': 'hip_to_zero', 'start_iter': 0, 'end_iter': 16000, 'start_value': 1.0, 'end_value': 20.0},
        ]

        class scales(WINCfg.rewards.scales):
            torques = -2.5e-4
            dof_pos_limits = -4.0
            action_rate = -0.03
            action_smoothness = -0.05
            similar_to_default = -0.02
            stand_still = -0.6


class WINLowSpeedCfgMoECTS(WINCfgMoECTS):
    """MoE CTS runner config for the low-speed WIN variant."""

    class runner(WINCfgMoECTS.runner):
        run_name = 'flat_low_speed_guarded'
        experiment_name = 'win_low_speed_moe_cts'
        max_iterations = 20000
        save_interval = 5000


class WINFlatSlowCfg(WINCfg):
    """WIN variant that keeps dynamic commands while adding extra flat low-speed and monotonic samples."""

    class commands(WINCfg.commands):
        flat_low_speed_command_prob = 0.2
        flat_low_speed_terrain_ids = [8]
        flat_low_speed_command_ranges = {
            "lin_vel_x": [-0.5, 0.5],
            "lin_vel_y": [-0.4, 0.4],
            "ang_vel_yaw": [-0.8, 0.8],
        }
        monotonic_command_prob = 0.3
        monotonic_command_type_probs = [0.45, 0.25, 0.30]


class WINFlatSlowCfgMoECTS(WINCfgMoECTS):
    """MoE CTS runner config for WIN with extra flat low-speed command coverage."""

    class runner(WINCfgMoECTS.runner):
        run_name = 'flat_slow'
        experiment_name = 'win_flat_slow_moe_cts'


class WINGuardedCfg(WINCfg):
    """WIN variant that keeps the original task distribution with stronger action and joint-limit guards."""

    class commands(WINCfg.commands):
        curriculum = True
        dynamic_resample_commands = False
        command_range_curriculum = []
        command_curriculum_step = 0.2
        command_curriculum_dims = ["lin_vel_x", "lin_vel_y", "ang_vel_yaw"]
        command_curriculum_threshold = 0.8
        monotonic_command_prob = 0.3
        monotonic_command_type_probs = [0.45, 0.25, 0.30]
        min_abs_lin_vel_x_by_terrain = {3: 0.55}
        max_command_curriculum_ranges = {
            "lin_vel_x": [-1.7, 1.7],
            "lin_vel_y": [-0.8, 0.8],
            "ang_vel_yaw": [-1.5, 1.5],
        }

        class ranges(WINCfg.commands.ranges):
            lin_vel_x = [-0.2, 0.2]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [-1.57, 1.57]

    class rewards(WINCfg.rewards):
        soft_dof_pos_limit = 0.8
        curriculum_rewards = [
            {'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
            {'reward_name': 'correct_base_height', 'start_iter': 0, 'end_iter': 5000, 'start_value': 1.0, 'end_value': 10.0},
            {'reward_name': 'ang_vel_xy', 'start_iter': 5000, 'end_iter': 15000, 'start_value': 1.0, 'end_value': 2.0},
            {'reward_name': 'stand_still', 'start_iter': 5000, 'end_iter': 20000, 'start_value': 1.0, 'end_value': 4.0},
            # {'reward_name': 'hip_to_default', 'start_iter': 10000, 'end_iter': 35000, 'start_value': 1.0, 'end_value': 0.4},
            {'reward_name': 'lateral_yaw_tracking_error', 'start_iter': 0, 'end_iter': 35000, 'start_value': 1.0, 'end_value': 5.0},
            {'reward_name': 'hip_to_zero', 'start_iter': 0, 'end_iter': 35000, 'start_value': 1.0, 'end_value': 20.0},
        ]

        class scales(WINCfg.rewards.scales):
            torques = -2.5e-4
            dof_pos_limits = -4.0
            action_rate = -0.03
            action_smoothness = -0.05
            foot_slip = -0.03
            low_speed_feet_air_time = 0.5
            feet_air_time_variance = -0.4


class WINGuardedCfgMoECTS(WINCfgMoECTS):
    """MoE CTS runner config for the guarded WIN variant."""

    class runner(WINCfgMoECTS.runner):
        run_name = 'compact_joint_action_guarded'
        experiment_name = 'win_guarded_moe_cts'
        max_iterations = 40000
        save_interval = 5000


class WINGuardedLongCfg(WINGuardedCfg):
    """Guarded WIN variant with the original WIN reward curriculum schedule."""

    class rewards(WINGuardedCfg.rewards):
        curriculum_rewards = WINCfg.rewards.curriculum_rewards


class WINGuardedLongCfgMoECTS(WINGuardedCfgMoECTS):
    """MoE CTS runner config for guarded WIN with original WIN training length."""

    class runner(WINGuardedCfgMoECTS.runner):
        run_name = 'guard_long'
        experiment_name = 'win_guard_long_moe_cts'
        max_iterations = WINCfgMoECTS.runner.max_iterations
