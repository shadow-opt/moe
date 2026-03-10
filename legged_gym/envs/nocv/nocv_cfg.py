from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgMoECTS

class NOCVCfg(GO2Cfg):
    class env(GO2Cfg.env):
        num_observations = super().env.num_observations + 5
        num_privileged_obs = super().env.num_privileged_obs + 5
        # [实验分支提示]
        # 这里只在配置层把 observation 维度各加了 5，
        # 但若对应环境类没有同步改 `compute_observations()` / `privileged_obs_buf` 拼接，
        # 就会出现“配置维度”和“真实张量维度”不一致的问题。
        # 因此二次开发时不要只改 cfg 数字，必须同时回看环境实现。

    class terrain(GO2Cfg.terrain):
        terrain_proportions = [0.10, 0.05, 0.05, 0.20, 0.2, 0.20, 0.1, 0.05, 0.05]
        # [wave, slope, rough_slope, stairs up, stairs down, obstacles, stepping_stones, gap, flat]
        
        
    pass

class NOCVCfgMoECTS(GO2CfgMoECTS):
    pass