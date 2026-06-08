from .win_cts_cfg import WINVanillaCTS
from .win_cfg import WINCfg
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgHIM


class HimlocoCfg(WINCfg):
    class env(WINCfg.env):
        pass
    class init_state(WINCfg.init_state):
        pass
    class control(WINCfg.control):
        pass
    class asset(WINCfg.asset):
        pass 
    class domain_rand(WINCfg.domain_rand):
        pass
    class terrain(WINCfg.terrain):
        pass
    class commands(WINCfg.commands):
        command_range_curriculum = [{ # list for command range curriculums at specific training iterations
            'iter': 200, # training iteration at which the command ranges are updated
            'lin_vel_x': [-1.0, 1.0], # min max [m/s]
            'lin_vel_y': [-1.0, 1.0], # min max [m/s]
            'ang_vel_yaw': [-1.5, 1.5], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        }, { # list for command range curriculums at specific training iterations
            'iter': 600, # training iteration at which the command ranges are updated
            'lin_vel_x': [-2.0, 2.0], # min max [m/s]
            'lin_vel_y': [-1.0, 1.0], # min max [m/s]
            'ang_vel_yaw': [-2.0, 2.0], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        }]
        pass
    class rewards(WINCfg.rewards):
        curriculum_rewards = []
        pass
    

class HIMCfg(WINVanillaCTS):
    class commands(WINVanillaCTS.commands):
        num_commands = 7 # 这是buffer，比command多1维 实际输入-1

        body_height_command_idx = 4 # buffer的第5维，索引第4维
        body_height_command_obs_scale = 1.0 # 缩放
        
        body_height_command_threshold = 0.5
        # 只有 command 值大于该阈值时，才视为“低高度档位”。
        # 当前 low command = 1.0，normal command = 0.0，因此 0.5 是自然分界。

        # 正常高度档位的 command 值。
        normal_body_height_command = 0.0
        # 低高度档位的 command 值。
        low_body_height_command = 1.0 # 对允许的地形，重采样 command 时有多大概率切到低高度档位。
        low_height_command_prob = 0.4
        # 仅在这些 terrain id 上允许采样低高度档位：
        # 1 = slope, 2 = rough_slope, 8 = flat
        low_height_terrain_ids = [1, 2, 8]  # slope, rough_slope, flat
        # 当采到低高度档位时，对速度命令做额外缩放：
        # [lin_vel_x, lin_vel_y, ang_vel_yaw]
        # 这样可以让机器人在压低机身时适当放慢速度，提高可学性与稳定性。
        low_height_command_velocity_scale = [0.5, 0.5, 0.7]

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
            'iter': 300, # training iteration at which the command ranges are updated
            'lin_vel_x': [-1.0, 1.0], # min max [m/s]
            'lin_vel_y': [-1.0, 1.0], # min max [m/s]
            'ang_vel_yaw': [-1.5, 1.5], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        },{ # list for command range curriculums at specific training iterations
            'iter': 700, # training iteration at which the command ranges are updated
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
        # terrain_max_command_ranges = [
        #     {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # wave
        #     {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # slope
        #     {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # rough slope
        #     {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs up
        #     {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs down
        #     {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # obstacles
        #     {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [-1.57, 1.57]},  # stepping stones
        #     {'lin_vel_x': [-0.5, 0.5], 'lin_vel_y': [-0.5, 0.5], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [-1.57, 1.57]},  # gap
        #     {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0], 'heading': [-1.57, 1.57]},  # flat
        # ]
        terrain_max_command_ranges = [
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # wave
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # slope
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # rough slope
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-0.0, 0.0], 'ang_vel_yaw': [-0, 0], 'heading': [-0, 0]},  # stairs up
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-0.0, 0.0], 'ang_vel_yaw': [-0, 0], 'heading': [-0, 0]},  # stairs down
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # obstacles
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [-1.57, 1.57]},  # stepping stones
            {'lin_vel_x': [-0.5, 0.5], 'lin_vel_y': [-0.5, 0.5], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [-1.57, 1.57]},  # gap
            {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0], 'heading': [-1.57, 1.57]},  # flat
        ]
    
        class ranges(WINVanillaCTS.commands.ranges):
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0] # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]   # min max [rad/s]
            heading = [-1.57, 1.57] # min max [rad]

    class rewards(WINVanillaCTS.rewards):
        # 正常档位继续沿用父类中的 `base_height_target`。
        # 低高度档位下，改为追踪这个更低的目标高度。
        low_base_height_target = 0.2
        base_height_target = 0.34
        only_positive_rewards = True

        curriculum_rewards = [
        # 早期强约束身体不要乱跳，后期放开，让策略自己找步态。
        {'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 600, 'start_value': 1.0, 'end_value': 0.2},

        # him 有低高度 command，base height 目标很重要；训练中后期逐渐加重。
        {'reward_name': 'correct_base_height', 'start_iter': 0, 'end_iter': 1200, 'start_value': 1.0, 'end_value': 4.0},

        # roll/pitch 稳定性不要一开始太狠，否则会压制低身高动作；后期再收紧。
        {'reward_name': 'ang_vel_xy', 'start_iter': 300, 'end_iter': 1600, 'start_value': 1.0, 'end_value': 3.0},

        # 脚滑和动作平滑更像“收尾塑形”，太早加重容易学慢。
        {'reward_name': 'foot_slip', 'start_iter': 800, 'end_iter': 1800, 'start_value': 1.0, 'end_value': 3.0},
        {'reward_name': 'action_smoothness', 'start_iter': 800, 'end_iter': 1800, 'start_value': 1.0, 'end_value': 2.0},

        # 直走约束如果你希望 him 更会正向过地形，可以中后期加。
        {'reward_name': 'straight_path_deviation', 'start_iter': 1000, 'end_iter': 2000, 'start_value': 1.0, 'end_value': 2.0},
    ]
        class scales(WINVanillaCTS.rewards.scales):
            # straight_path = 10.0 # [NOTE] 新增
            # straight_path_deviation = -5 # [NOTE] 新增
            stand_still = -0.5
            collision = -1
            # orientation = -0.5
            # stumble = -2.
            # x_command_hip_regular = -0.5

class HIMCfgPPO(LeggedRobotCfgHIM):
    history_length = 6
    class algorithm( LeggedRobotCfgHIM.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgHIM.runner ):
        run_name = ''
        experiment_name = 'himloco'
        max_iterations = 2000 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations