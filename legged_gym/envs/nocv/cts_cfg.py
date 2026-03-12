from legged_gym.envs.nocv.nocv_cfg import NOCVCfg
from legged_gym.envs.nocv.jump_cfg import JUMPCfg
from legged_gym.envs.go2.go2_config import GO2CfgCTS

class NOCVCfgCTS(NOCVCfg):
    class commands(NOCVCfg.commands):

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





class NOCVCfgCTS(GO2CfgCTS):
    """Go2 + CTS 训练配置。"""

    class runner(GO2CfgCTS.runner):
        num_steps_per_env = 24
        run_name = 'nocv'
        experiment_name = 'nocv_cts'
        max_iterations = 24000
        save_interval = 500
    
    class policy(GO2CfgCTS.policy):
        latent_dim = 32
        norm_type = 'l2norm'

class JUMPCfgCTS(NOCVCfgCTS):
    """Go2 + CTS 训练配置。"""

    class runner(NOCVCfgCTS.runner):
        run_name = 'jump'
        experiment_name = 'nocv_cts'
        max_iterations = 30000
        save_interval = 500
