from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgMoECTS


class NOCVCfg(GO2Cfg):
    """NoCV 实验环境配置。

    本配置在主线 Go2 locomotion 基础上，增加了一个第 5 维 command：
    - 前 3 维仍是 `lin_vel_x / lin_vel_y / ang_vel_yaw`
    - 第 4 维继续保留给 `heading`
    - 第 5 维新增为 `body height mode`

    当前设计目标：
    1. 只在 `slope / rough_slope / flat` 上允许采样低高度档位；
    2. 低高度档位进入 actor obs 与 privileged obs；
    3. 当采到低高度档位时，同步把速度命令做温和缩放，避免“又低又快”过难；
    4. 奖励侧按 command 档位切换 base height target。
    """

    class env(GO2Cfg.env):
        # 主线 Go2 actor obs = 45，这里新增 1 维 body-height command，因此变为 46。
        num_observations = 46
        # 主线 Go2 privileged obs = 263，这里同样新增 1 维 command，因此变为 264。
        num_privileged_obs = 264

    class terrain(GO2Cfg.terrain):
        # NoCV 训练更偏向楼梯/障碍类地形，但仍保留部分平地与斜坡，
        # 因为低高度档位只在 slope / rough_slope / flat 上启用。
        terrain_proportions = [0.05, 0.05, 0.1, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05]
        # [wave, slope, rough_slope, stairs up, stairs down, obstacles, stepping_stones, gap, flat]

    class commands(GO2Cfg.commands):
        # 内部 command buffer 改为 5 维：
        # [lin_vel_x, lin_vel_y, ang_vel_yaw, heading, body_height_mode]
        num_commands = 5
        # 第 5 维（索引 4）作为新的高度档位 command。
        body_height_command_idx = 4
        # 该维进入 observation 时的缩放。
        # 当前是离散 0/1 档位，因此先保持 1.0，便于网络直接区分两种模式。
        body_height_command_obs_scale = 1.0
        # 只有 command 值大于该阈值时，才视为“低高度档位”。
        # 当前 low command = 1.0，normal command = 0.0，因此 0.5 是自然分界。
        body_height_command_threshold = 0.5
        # 正常高度档位的 command 值。
        normal_body_height_command = 0.0
        # 低高度档位的 command 值。
        low_body_height_command = 1.0
        # 对允许的地形，重采样 command 时有多大概率切到低高度档位。
        low_height_command_prob = 0.35
        # 仅在这些 terrain id 上允许采样低高度档位：
        # 1 = slope, 2 = rough_slope, 8 = flat
        low_height_terrain_ids = [1, 2, 8]  # slope, rough_slope, flat
        # 当采到低高度档位时，对速度命令做额外缩放：
        # [lin_vel_x, lin_vel_y, ang_vel_yaw]
        # 这样可以让机器人在压低机身时适当放慢速度，提高可学性与稳定性。
        low_height_command_velocity_scale = [0.7, 0.7, 0.8]
    
    class rewards(GO2Cfg.rewards):
        # 正常档位继续沿用父类中的 `base_height_target`。
        # 低高度档位下，改为追踪这个更低的目标高度。
        low_base_height_target = 0.20


class NOCVCfgMoECTS(GO2CfgMoECTS):
    """NoCV 对应的 MoE CTS 训练配置。

    这里只保留训练器侧配置继承；环境侧实际会配合 `NOCVCfg` 一起注册使用。
    """

    pass