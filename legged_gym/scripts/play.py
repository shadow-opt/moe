import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry, Logger
from legged_gym.utils.exporter import export_policy_as_jit, export_policy_as_onnx, export_policy_as_pkl

import numpy as np
import torch

"""推理/回放入口脚本。

它和 `train.py` 的最大区别是：
1. 通常会关闭大部分随机化与噪声，方便稳定观察策略；
2. 可以直接覆写 `env.commands`，手工指定想测试的命令。

因此当后续扩展“更多命令”时，这个文件往往是最容易漏改的测试入口之一。
"""

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # 这里把训练态环境改造成“更可视化、可复现”的测试态环境。
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 200)
    # env_cfg.terrain.mesh_type = 'plane'
    # env_cfg.terrain.num_rows = 8
    env_cfg.terrain.num_cols = 10
    # env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_pd_gains = False
    env_cfg.domain_rand.randomize_motor_zero_offset = False
    env_cfg.init_state.randomize_yaw = False


    env_cfg.env.test = True

    # 创建环境后，`obs = env.get_observations()` 会拿到 reset 后的首帧观测。
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # 推理时通常直接恢复最新训练好的模型。
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        if hasattr(runner.alg, 'actor_critic'):
            model = runner.alg.actor_critic
        else:
            model = runner.alg.model
        export_policy_as_jit(model, path)
        export_policy_as_onnx(model, path)
        export_policy_as_pkl(model, path)
        print('Exported policy as jit script / onnx to: ', path)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())

        # if FIX_COMMAND:
        if True:
            # `jump` 任务需要同时固定 jump 相关 command，
            # 否则这里只改前三维速度命令，无法验证“可控跳高/跳远”。
            if args.task in ("jump", "j_cts") and env.commands.shape[1] >= 9:
                jump_dx_mid = 0.5 * (env_cfg.commands.ranges.jump_dx[0] + env_cfg.commands.ranges.jump_dx[1])
                jump_dy_mid = 0.0
                jump_dz_mid = 0.5 * (env_cfg.commands.ranges.jump_dz[0] + env_cfg.commands.ranges.jump_dz[1])
                env.commands[:, 0] = 0.0
                env.commands[:, 1] = 0.0
                env.commands[:, 2] = 0.0
                env.commands[:, env.body_height_command_idx] = env_cfg.commands.low_body_height_command
                env.commands[:, env.jump_dx_command_idx] = jump_dx_mid
                env.commands[:, env.jump_dy_command_idx] = jump_dy_mid
                env.commands[:, env.jump_dz_command_idx] = jump_dz_mid
                env.commands[:, env.jump_trigger_command_idx] = env_cfg.commands.active_jump_command
            else:
                # 这里直接把每个 env 的命令固定成“向前走”。
                env.commands[:, 0] = 2.0
                env.commands[:, 1] = 0.0
                env.commands[:, 2] = 0.0
                # env.commands[:, 4] = 2.5
                # env.commands[:, 3] = 1

        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    FIX_COMMAND = True
    args = get_args()
    play(args)
