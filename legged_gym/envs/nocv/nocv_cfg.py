from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgMoECTS, GO2CfgCTS


class NOCVCfg(GO2Cfg):
    """NoCV 实验环境配置。

    本配置在主线 Go2 locomotion 基础上，扩展出与 jump 任务一致的 command 结构：
    - 前 3 维仍是 `lin_vel_x / lin_vel_y / ang_vel_yaw`
    - 第 4 维继续保留给 `heading`
    - 第 5 维新增为 `body height mode`
    - 第 6~9 维预留给 jump：`jump_dx / jump_dy / jump_dz / jump_trigger`

    当前设计目标：
    1. 只在 `slope / rough_slope / flat` 上允许采样低高度档位；
    2. 低高度档位进入 actor obs 与 privileged obs；
    3. 当采到低高度档位时，同步把速度命令做温和缩放，避免“又低又快”过难；
    4. 奖励侧按 command 档位切换 base height target；
    5. 与 jump 任务保持相同 command/obs 布局，其中 jump 相关输入恒为 0。
    """


    class asset(GO2Cfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/qianxihouzhouzu/urdf/qianxihouzhouzu.urdf'
        name = 'qxhzz'
        foot_name = 'foot'
        terminate_after_contacts_on = ["base", "trunk"]
        penalize_contacts_on = ["thigh", "calf", "hip"]
        self_collisions = 1
        flip_visual_attachments = False

    class env(GO2Cfg.env):
        # actor obs = 3(base_ang_vel) + 3(projected_gravity) + 8(command obs) + 12(dof_pos) + 12(dof_vel) + 12(actions)
        num_observations = 50
        # privileged obs = actor obs(50) + base_lin_vel(3) + foot contact forces(4) + torques(12)
        #                + dof accelerations(12) + terrain heights(187) + jump state placeholders(3)
        num_privileged_obs = 271

    class terrain(GO2Cfg.terrain):
        # NoCV 训练更偏向楼梯/障碍类地形，但仍保留部分平地与斜坡，
        # 因为低高度档位只在 slope / rough_slope / flat 上启用。
        terrain_proportions = [0.05, 0.05, 0.1, 0.2, 0.2, 0.2, 0.0, 0.0, 0.2]
        # [wave, slope, rough_slope, stairs up, stairs down, obstacles, stepping_stones, gap, flat]

    class commands(GO2Cfg.commands):
        # 内部 command buffer 改为 9 维：
        # [lin_vel_x, lin_vel_y, ang_vel_yaw, heading, body_height_mode,
        #  jump_dx, jump_dy, jump_dz, jump_trigger]
        num_commands = 9
        # 第 5 维（索引 4）作为新的高度档位 command。
        body_height_command_idx = 4
        jump_dx_command_idx = 5
        jump_dy_command_idx = 6
        jump_dz_command_idx = 7
        jump_trigger_command_idx = 8
        # 该维进入 observation 时的缩放。
        # 当前是离散 0/1 档位，因此先保持 1.0，便于网络直接区分两种模式。
        body_height_command_obs_scale = 1.0
        jump_command_obs_scale = [1.0, 1.0, 1.0, 1.0]
        # 只有 command 值大于该阈值时，才视为“低高度档位”。
        # 当前 low command = 1.0，normal command = 0.0，因此 0.5 是自然分界。
        body_height_command_threshold = 0.5
        jump_command_threshold = 0.5
        # 正常高度档位的 command 值。
        normal_body_height_command = 0.0
        # 低高度档位的 command 值。
        low_body_height_command = 1.0
        normal_jump_command = 0.0
        active_jump_command = 1.0
        # 对允许的地形，重采样 command 时有多大概率切到低高度档位。
        low_height_command_prob = 0.4
        # 仅在这些 terrain id 上允许采样低高度档位：
        # 1 = slope, 2 = rough_slope, 8 = flat
        low_height_terrain_ids = [1, 2, 8]  # slope, rough_slope, flat
        # 当采到低高度档位时，对速度命令做额外缩放：
        # [lin_vel_x, lin_vel_y, ang_vel_yaw]
        # 这样可以让机器人在压低机身时适当放慢速度，提高可学性与稳定性。
        low_height_command_velocity_scale = [0.5, 0.5, 0.7]

        class ranges(GO2Cfg.commands.ranges):
            jump_dx = [0.0, 0.0]
            jump_dy = [0.0, 0.0]
            jump_dz = [0.0, 0.0]
            jump_trigger = [0.0, 0.0]
    
    class rewards(GO2Cfg.rewards):
        # 正常档位继续沿用父类中的 `base_height_target`。
        # 低高度档位下，改为追踪这个更低的目标高度。
        low_base_height_target = 0.18
        class scales(GO2Cfg.rewards.scales):
            straight_path = 2.0 # [NOTE] 新增奖励函数
            straight_path_deviation = -0.5 # [NOTE] 新增
            
            
        


class NOCVCfgMoECTS(GO2CfgMoECTS):
    """NoCV 对应的 MoE CTS 训练配置。

    这里只保留训练器侧配置继承；环境侧实际会配合 `NOCVCfg` 一起注册使用。
    """

    class runner(GO2CfgMoECTS.runner):
        run_name = ''
        experiment_name = 'nocv_moe_cts'
        max_iterations = 130000
        save_interval = 500

