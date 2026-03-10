import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np
import sys

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner, OnPolicyRunnerCTS

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

"""任务注册表。

可以把这个文件理解成“任务名 -> 环境类/配置/训练器配置”的总索引。
训练脚本和播放脚本通常只知道一个 `task` 字符串，真正要跑哪套环境、哪份参数，都在这里决定。
"""

class TaskRegistry():
    def __init__(self):
        # `task_classes` 保存任务名到环境类的映射，例如 `go2 -> Go2Robot`。
        self.task_classes = {}
        # `env_cfgs` / `train_cfgs` 分别保存环境配置和算法配置。
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        # 注册阶段通常发生在 `legged_gym/envs/__init__.py` 被 import 时。
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # 训练配置里的 seed 会同步给环境配置，保证环境随机性与算法实验设置一致。
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # `args` 里主要是 CLI 覆盖项，例如 `--task`、`--headless`、`--num_envs`。
        if args is None:
            args = get_args()
        # 第一步：根据任务名找到环境类。
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # 第二步：取出该任务的默认环境配置。
            env_cfg, _ = self.get_cfgs(name)
        # 第三步：允许命令行参数覆盖默认配置。
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # 第四步：把嵌套配置转成 Isaac Gym 需要的 sim 参数。
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        # 第五步：实例化环境对象。
        # 注意命令采样逻辑并不在这里，而是在环境类内部的 `_init_buffers()` / `_resample_commands()`。
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # `make_alg_runner()` 和 `make_env()` 分工明确：前者只创建算法/日志/恢复逻辑，不碰环境命令实现。
        if args is None:
            args = get_args()
        # 优先使用显式传入的 train_cfg，否则按任务名查默认训练配置。
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # CLI 也可以覆盖训练器配置，例如 resume、checkpoint 等。
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        # 训练配置最终会被转成普通 dict，交给 runner/algorithm 使用。
        # 这里还有一个阅读上的关键点：
        # - 环境侧 buffer 的尺寸，已经在 `make_env()` 阶段由 env_cfg 固定；
        # - 算法侧 rollout/history buffer 的尺寸，则在 runner 初始化阶段由 train_cfg 决定。
        # 两边会在这里汇合，因此这是“环境维度”和“算法维度”对齐的总入口之一。
        #
        # [风险提示]
        # 当前代码直接使用了 `train_cfg_dict`，但本函数里并没有看到它的显式定义。
        # 从语义上看，这里原本应当把 `train_cfg` 递归转成 dict 后再传给 runner；
        # 因此若后续运行时报未定义错误，应优先检查这里。
        runner = eval(train_cfg.runner_class_name)(env, train_cfg_dict, log_dir, device=args.rl_device)
        # save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume:
            # 恢复模型只影响策略参数，不会自动重写环境内部的命令逻辑；
            # 因此一旦命令维度或 observation layout 发生变化，旧模型通常不能直接兼容。
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()