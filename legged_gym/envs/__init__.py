from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_env import Go2Robot
from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO, GO2CfgCTS, GO2CfgMoECTS, GO2CfgMoENGCTS, GO2CfgMCPCTS, GO2CfgACMoECTS, GO2CfgDualMoECTS
from legged_gym.envs.nocv.nocv_env import NoCVRobot
from legged_gym.envs.nocv.nocv_cfg import NOCVCfg, NOCVCfgMoECTS
from legged_gym.envs.nocv.jump_env import JumpRobot
from legged_gym.envs.nocv.jump_cfg import JUMPCfg, JUMPCfgMoECTS
from legged_gym.envs.nocv.cts_cfg import NOCVCfgCTS, JUMPCfgCTS

from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

"""环境任务注册入口。

这里是“task 字符串真正落地”的地方。
例如命令行里写 `--task go2`，最终就会在这里解析成：
- 环境类：`Go2Robot`
- 环境配置：`GO2Cfg`
- 训练配置：`GO2CfgPPO`

如果后续想新建一个“支持更多命令的新任务”，常见做法不是直接改训练脚本，
而是新增一套配置/环境类后，在这里再注册一个新的任务名。
"""

task_registry.register("go2", Go2Robot, GO2Cfg(), GO2CfgPPO())
task_registry.register("go2_cts", Go2Robot, GO2Cfg(), GO2CfgCTS())
task_registry.register("go2_moe_cts", Go2Robot, GO2Cfg(), GO2CfgMoECTS())
task_registry.register("go2_moe_ng_cts", Go2Robot, GO2Cfg(), GO2CfgMoENGCTS())
task_registry.register("go2_mcp_cts", Go2Robot, GO2Cfg(), GO2CfgMCPCTS())
task_registry.register("go2_ac_moe_cts", Go2Robot, GO2Cfg(), GO2CfgACMoECTS())
task_registry.register("go2_dual_moe_cts", Go2Robot, GO2Cfg(), GO2CfgDualMoECTS())
task_registry.register("nocv", NoCVRobot, NOCVCfg(), NOCVCfgMoECTS())
task_registry.register("jump", JumpRobot, JUMPCfg(), JUMPCfgMoECTS())
task_registry.register("n_cts", NoCVRobot, NOCVCfg(), NOCVCfgCTS())
task_registry.register("j_cts", JumpRobot, JUMPCfg(), JUMPCfgCTS())

