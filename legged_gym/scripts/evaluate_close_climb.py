import json

import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


OUTCOME_KEYS = (
    "climb_success",
    "climb_fall",
    "climb_stuck",
    "climb_timeout",
    "climb_out_of_track",
)


def evaluate(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    if not hasattr(env_cfg, "close_climb_phase"):
        raise ValueError("evaluate_close_climb.py requires a win_close_climb task")
    env_cfg.env.test = True
    env_cfg.noise.add_noise = False
    env_cfg.init_state.randomize_yaw = False
    if args.num_envs is None:
        env_cfg.env.num_envs = 256

    if args.eval_mode == "wall":
        env_cfg.terrain.terrain_proportions = [0.0] * 9 + [1.0]
        env_cfg.terrain.thin_wall_fixed_height = args.eval_wall_height
    else:
        env_cfg.terrain.terrain_proportions = [0.0] * 8 + [1.0, 0.0]
        env_cfg.terrain.thin_wall_fixed_height = None

    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    if not args.warmstart_path:
        train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(env, args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)

    totals = {key: 0.0 for key in OUTCOME_KEYS}
    wall_episodes = 0.0
    success_time_sum = 0.0
    hip_peak_sum = 0.0
    hip_excess_sum = 0.0
    flat_samples = 0
    flat_velocity_error = 0.0
    flat_height_error = 0.0

    for _ in range(args.eval_steps):
        with torch.no_grad():
            actions = policy(obs.detach())
            obs, _, _, dones, infos = env.step(actions.detach())

        if args.eval_mode == "flat":
            velocity_error = torch.square(env.commands[:, :3] - torch.cat((env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]), dim=1))
            target_height = env._get_commanded_base_height_target()
            base_height = env.root_states[:, 2] - env.env_origins[:, 2]
            flat_velocity_error += velocity_error.sum().item()
            flat_height_error += torch.square(base_height - target_height).sum().item()
            flat_samples += env.num_envs

        episode = infos.get("episode", {}) if dones.any() else {}
        count = float(episode.get("climb_wall_episodes", 0.0))
        if count > 0:
            wall_episodes += count
            for key in OUTCOME_KEYS:
                totals[key] += float(episode.get(key, 0.0)) * count
            success_count = float(episode.get("climb_success", 0.0)) * count
            success_time_sum += float(episode.get("climb_success_time", 0.0)) * success_count
            hip_peak_sum += float(episode.get("climb_hip_peak_force", 0.0)) * count
            hip_excess_sum += float(episode.get("climb_hip_excess", 0.0)) * count

    if args.eval_mode == "wall":
        denom = max(wall_episodes, 1.0)
        success_count = totals["climb_success"]
        report = {
            "mode": "wall",
            "wall_height_m": args.eval_wall_height,
            "episodes": int(wall_episodes),
            **{key[len("climb_"):]: value / denom for key, value in totals.items()},
            "mean_success_time_s": success_time_sum / max(success_count, 1.0),
            "mean_hip_peak_force_n": hip_peak_sum / denom,
            "mean_hip_force_excess_ns": hip_excess_sum / denom,
        }
    else:
        report = {
            "mode": "flat",
            "samples": flat_samples,
            "velocity_tracking_rmse": (flat_velocity_error / max(3 * flat_samples, 1)) ** 0.5,
            "height_tracking_rmse_m": (flat_height_error / max(flat_samples, 1)) ** 0.5,
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    evaluate(get_args())
