from legged_gym.envs.go2.go2_config import GO2Cfg
from legged_gym.envs.go2.go2_config import GO2CfgMoECTS
from legged_gym.envs.go2.go2_config import GO2CfgCTS

class WINCfg(GO2Cfg):
    
    class asset(GO2Cfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/qianxihouzhouzu/urdf/qianxihouzhouzu.urdf'
        name = 'qxhzz'
        foot_name = 'foot'
        terminate_after_contacts_on = ["base", "trunk"]
        penalize_contacts_on = ["thigh", "calf", "hip"]
        self_collisions = 1 # 1关闭自碰撞，0开启自碰撞
        flip_visual_attachments = False

    class env(GO2Cfg.env):
        # actor obs = 3(base_ang_vel) + 3(projected_gravity) + 6(command obs) + 12(dof_pos) + 12(dof_vel) + 12(actions)
        num_observations = 48
        # privileged obs = actor obs(48) + base_lin_vel(3) + foot contact forces(4) + torques(12)
        #                + dof accelerations(12) + terrain heights(187)
        num_privileged_obs = 48 + 3 + 4 + 12 + 12 + 187

    class terrain(GO2Cfg.terrain):
        # NoCV 训练更偏向楼梯/障碍类地形，但仍保留部分平地与斜坡，
        # 低高度档位只在 slope / rough_slope / flat 上启用。
        terrain_proportions = [0.05, 0.05, 0.1, 0.2, 0.15, 0.15, 0.1, 0.0, 0.2]
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

        special_terrain_options_5 = [3, 4, 5]
        special_terrain_options_6 = [6]
        # 对于 stairs up/down 和 obstacles 5
        # 对于 stones 6
        special_terrain_probs = 0.6
        special_terrain_command = 1
        normal_terrain_command = 0
        special_terrain_command_obs_scale = 1.0


    class rewards(GO2Cfg.rewards):
        # 正常档位继续沿用父类中的 `base_height_target`。
        # 低高度档位下，改为追踪这个更低的目标高度。
        low_base_height_target = 0.18
        class scales(GO2Cfg.rewards.scales):
            straight_path = 1.0 # [NOTE] 新增
            straight_path_deviation = -0.7 # [NOTE] 新增
            
            
            
class WINCfgMoECTS(GO2CfgMoECTS):
    """WIN 对应的 MoE CTS 训练配置。

    这里只保留训练器侧配置继承；环境侧实际会配合 `WINCfg` 一起注册使用。
    """

    class runner(GO2CfgMoECTS.runner):
        run_name = ''
        experiment_name = 'win_moe_cts'
        max_iterations = 100000
        save_interval = 5000