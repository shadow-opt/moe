import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

"""训练入口脚本。

建议按下面顺序阅读训练链路：
1. `train()`：只负责把任务名解析成环境和训练器；
2. `task_registry.make_env()`：决定具体实例化哪个环境类、加载哪份配置；
3. `LeggedRobot` / `Go2Robot`：真正维护命令、观测、奖励、reset 逻辑；
4. `runner.learn()`：进入 PPO/CTS 等算法的 rollout + update 循环。

最容易误解的一点：训练脚本本身并不直接把“手柄命令/速度命令”喂给机器人。
训练时的 `command` 主要由环境内部的 `self.commands` 张量维护，并在环境 step 中按配置重采样。
"""

def train(args):
    # 这里的 `args.task` 只是一个任务名字符串，例如 `go2` 或 `go2_moe_cts`。
    # 真正的环境类和配置，会在 `task_registry` 中查表得到。
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # 训练器（runner）内部会持有算法、buffer、日志目录等训练态对象。
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    # resume 时需要把环境的公共步数同步到 runner 当前迭代，否则 curriculum 会从头算。
    env.common_step_counter = runner.current_learning_iteration * env.num_steps_per_env  # resume env step counter
    # reward curriculum 依赖公共步数，因此恢复训练时先强制刷新一次。
    env.update_reward_curriculum(force_update=True)  # force update reward curriculum at start
    # 进入主训练循环后，真正的“命令采样 -> 动作 -> 仿真 -> reward -> update”闭环开始运行。
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
