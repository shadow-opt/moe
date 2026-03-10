from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgMoECTS

class NOCVCfg(GO2Cfg):
    class env(GO2Cfg.env):
        num_observations = 46
        num_privileged_obs = 264

    class terrain(GO2Cfg.terrain):
        terrain_proportions = [0.10, 0.05, 0.05, 0.20, 0.2, 0.20, 0.1, 0.05, 0.05]
        # [wave, slope, rough_slope, stairs up, stairs down, obstacles, stepping_stones, gap, flat]

    class commands(GO2Cfg.commands):
        num_commands = 5
        body_height_command_idx = 4
        body_height_command_obs_scale = 1.0
        body_height_command_threshold = 0.5
        normal_body_height_command = 0.0
        low_body_height_command = 1.0
        low_height_command_prob = 0.35
        low_height_terrain_ids = [1, 2, 8]  # slope, rough_slope, flat

    class rewards(GO2Cfg.rewards):
        low_base_height_target = 0.32

class NOCVCfgMoECTS(GO2CfgMoECTS):
    pass