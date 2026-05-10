from legged_gym.envs.nocv.win_cfg import WINCfg
from legged_gym.envs.nocv.win_cfg import WINCfgMoECTS

class WINGapCfg(WINCfg):
    class init_state(WINCfg.init_state):
        randomize_yaw = False


    class terrain(WINCfg.terrain):
        terrain_proportions = [0.2, 0.1, 0.1, 0, 0, 0, 0.0, 0.4, 0.2]
        # [wave, slope, rough_slope, stairs up, stairs down, obstacles, stones, gap, flat]
        
    class commands(WINCfg.commands):
        low_height_command_prob = 0.0
        special_terrain_command = -2
        special_terrain_command_obs_scale = 1
        normal_terrain_command = 0
        special_terrain_options_6 = [7]
        special_terrain_options_5 = []
        special_terrain_probs = 1.0
        limit_vel_prob = 0.0
        zero_command_curriculum = None
        limit_ang_vel_at_zero_command_prob = 0.0
        terrain_max_command_ranges = [
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # wave
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # slope
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # rough slope
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs up
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs down
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # obstacles
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [-1.57, 1.57]},  # stepping stones
            {'lin_vel_x': [0, 0.3], 'lin_vel_y': [0.0, 0.0], 'ang_vel_yaw': [0.0, 0.0], 'heading': [0.0, 0.0]},  # gap
            {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0], 'heading': [-1.57, 1.57]},  # flat
        ]
        # 不需要scale，直接在terrain配置special_terrain_command_velocity_scale = [0.5, 0.0, 0.0]

    class rewards(WINCfg.rewards):
        low_foot_margin = 0.02
        low_foot_depth_scale = 0.05

        class scales(WINCfg.rewards.scales):
            low_foot = -0.3
            x_command_hip_regular = 0

        
