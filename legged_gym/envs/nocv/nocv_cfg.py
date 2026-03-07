from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgMoECTS

class NOCVCfg(GO2Cfg):
    class env(GO2Cfg.env):
        num_observations = super().env.num_observations + 1
        num_privileged_obs = super().env.num_privileged_obs + 1

    class terrain(GO2Cfg.terrain):
        terrain_proportions = [0.10, 0.05, 0.05, 0.20, 0.2, 0.20, 0.1, 0.05, 0.05]
        # [wave, slope, rough_slope, stairs up, stairs down, obstacles, stepping_stones, gap, flat]
        
        
    pass

class NOCVCfgMoECTS(GO2CfgMoECTS):
    pass