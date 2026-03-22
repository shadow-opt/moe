from legged_gym.envs.go2.go2_config import GO2Cfg
from legged_gym.envs.go2.go2_config import GO2CfgMoECTS
from legged_gym.envs.go2.go2_config import GO2CfgCTS

class WINCfg(GO2Cfg):
    class init_state(GO2Cfg.init_state):
        turn_over = False
        turn_over_proportions = [0.0, 0.2, 0.8] # proportions for backflip, sideflip, noflip

    class asset(GO2Cfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/j60/urdf/z2.urdf'
        name = 'z2'
        foot_name = 'FOOT'
        terminate_after_contacts_on = ["base"]
        penalize_contacts_on = ["thigh", "calf", "hip"]
        self_collisions = 1 # 1关闭自碰撞，0开启自碰撞
        flip_visual_attachments = True

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

    class terrain(GO2Cfg.terrain):
        # NoCV 训练更偏向楼梯/障碍类地形，但仍保留部分平地与斜坡，
        # 低高度档位只在 slope / rough_slope / flat 上启用。
        terrain_proportions = [0.05, 0.05, 0.1, 0.2, 0.1, 0.2, 0.1, 0.0, 0.2]
        # [wave, slope, rough_slope, stairs up, stairs down, obstacles, stones, gap, flat]
        
    class commands(GO2Cfg.commands):

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
        special_terrain_probs = 0.5
        special_terrain_command = 1
        normal_terrain_command = 0
        special_terrain_command_obs_scale = 1.0

                # 给命令采样设置一个 lower bound，避免采到“理论上走不完”的过慢命令。
        dynamic_resample_commands = True # sample commands with low bounds
        command_range_curriculum = [{ # list for command range curriculums at specific training iterations
            'iter': 10000, # training iteration at which the command ranges are updated
            'lin_vel_x': [-1.0, 1.0], # min max [m/s]
            'lin_vel_y': [-1.0, 1.0], # min max [m/s]
            'ang_vel_yaw': [-1.5, 1.5], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        }, { # list for command range curriculums at specific training iterations
            'iter': 30000, # training iteration at which the command ranges are updated
            'lin_vel_x': [-1.5, 1.5], # min max [m/s]
            'lin_vel_y': [-1.0, 1.0], # min max [m/s]
            'ang_vel_yaw': [-1.7, 1.7], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        },{ # list for command range curriculums at specific training iterations
            'iter': 50000, # training iteration at which the command ranges are updated
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
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs up
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs down
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
        class scales(GO2Cfg.rewards.scales):
            straight_path = 1.0 # [NOTE] 新增
            straight_path_deviation = -2 # [NOTE] 新增
            stand_still = -2
            orientation = -2
            stumble = -2.
            # x_command_hip_regular = -0.5
            
            
            
class WINCfgMoECTS(GO2CfgMoECTS):
    """WIN 对应的 MoE CTS 训练配置。

    这里只保留训练器侧配置继承；环境侧实际会配合 `WINCfg` 一起注册使用。
    """

    class runner(GO2CfgMoECTS.runner):
        run_name = ''
        experiment_name = 'win_moe_cts'
        max_iterations = 80000
        save_interval = 5000