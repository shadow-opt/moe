from legged_gym.envs.nocv.nocv_cfg import NOCVCfg
from legged_gym.envs.nocv.jump_cfg import JUMPCfg
from legged_gym.envs.go2.go2_config import GO2CfgCTS

class NCTSCfg(NOCVCfg):
    class commands(NOCVCfg.commands):
        # 给命令采样设置一个 lower bound，避免采到“理论上走不完”的过慢命令。
        dynamic_resample_commands = True # sample commands with low bounds
        command_range_curriculum = [{ # list for command range curriculums at specific training iterations
            'iter': 6000, # training iteration at which the command ranges are updated
            'lin_vel_x': [-1.0, 1.0], # min max [m/s]
            'lin_vel_y': [-1.0, 1.0], # min max [m/s]
            'ang_vel_yaw': [-1.5, 1.5], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        }, { # list for command range curriculums at specific training iterations
            'iter': 15000, # training iteration at which the command ranges are updated
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
            {'lin_vel_x': [-0.5, 0.5], 'lin_vel_y': [-0.5, 0.5], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [-1.57, 1.57]},  # stepping stones
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

class JCTSCfg(JUMPCfg):
    class commands(JUMPCfg.commands):
        dynamic_resample_commands = True # sample commands with low bounds
        command_range_curriculum = [{ # list for command range curriculums at specific training iterations
            'iter': 6000, # training iteration at which the command ranges are updated
            'lin_vel_x': [-1.0, 1.0], # min max [m/s]
            'lin_vel_y': [-1.0, 1.0], # min max [m/s]
            'ang_vel_yaw': [-1.5, 1.5], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        }, { # list for command range curriculums at specific training iterations
            'iter': 15000, # training iteration at which the command ranges are updated
            'lin_vel_x': [-2.0, 2.0], # min max [m/s]
            'lin_vel_y': [-1.0, 1.0], # min max [m/s]
            'ang_vel_yaw': [-2.0, 2.0], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        }]


class NOCVCfgCTS(GO2CfgCTS):
    """Go2 + CTS 训练配置。"""

    class runner(GO2CfgCTS.runner):
        num_steps_per_env = 24
        run_name = 'nocv'
        experiment_name = 'nocv_cts'
        max_iterations = 40000
        save_interval = 1000
    
    class policy(GO2CfgCTS.policy):
        latent_dim = 32
        norm_type = 'l2norm'

class JUMPCfgCTS(NOCVCfgCTS):
    """Go2 + CTS 训练配置。"""

    class runner(NOCVCfgCTS.runner):
        run_name = 'jump'
        experiment_name = 'jump_cts'
        max_iterations = 40000
        save_interval = 1000
