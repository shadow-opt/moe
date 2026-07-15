import glob
import hashlib
import json
import os
import re
import subprocess
import copy
from collections import defaultdict

import isaacgym
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.exporter import export_policy_as_jit


DEFAULT_BASELINE = os.path.join(
    LEGGED_GYM_ROOT_DIR,
    "logs/win_flat_slow_moe_cts/Jul05_17-22-50_flat_slow/model_85000.pt",
)


def _command(vx=0.0, vy=0.0, yaw=0.0, body=0.0):
    # Internal env buffer includes heading at index 3.
    return [vx, vy, yaw, 0.0, body, 0.0, 0.0]


SCENARIOS = (
    {"name": "stand", "kind": "stand", "command": _command()},
    {"name": "jump_fwd_03", "kind": "jump", "command": _command(0.3, body=-1.0)},
    {"name": "jump_fwd_05", "kind": "jump", "command": _command(0.5, body=-1.0)},
    {"name": "jump_fwd_08", "kind": "jump", "command": _command(0.8, body=-1.0)},
    {"name": "jump_back_03", "kind": "jump", "command": _command(-0.3, body=-1.0)},
    {"name": "jump_back_05", "kind": "jump", "command": _command(-0.5, body=-1.0)},
    {"name": "jump_back_08", "kind": "jump", "command": _command(-0.8, body=-1.0)},
    {"name": "jump_fwd_10", "kind": "jump_extended", "command": _command(1.0, body=-1.0)},
    {"name": "jump_back_10", "kind": "jump_extended", "command": _command(-1.0, body=-1.0)},
    {"name": "walk_fwd", "kind": "walk", "command": _command(0.5)},
    {"name": "walk_back", "kind": "walk", "command": _command(-0.5)},
    {"name": "walk_left", "kind": "walk", "command": _command(vy=0.5)},
    {"name": "walk_right", "kind": "walk", "command": _command(vy=-0.5)},
    {"name": "turn_left", "kind": "walk", "command": _command(yaw=1.0)},
    {"name": "turn_right", "kind": "walk", "command": _command(yaw=-1.0)},
    {"name": "walk_mixed", "kind": "walk", "command": _command(0.5, 0.3, 0.5)},
    {"name": "low_height", "kind": "walk", "command": _command(0.3, body=1.0)},
    {"name": "walk_jump_walk", "kind": "transition", "command": _command(0.5)},
    {"name": "stand_jump_stand", "kind": "transition", "command": _command()},
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_iteration(path):
    match = re.search(r"model_(\d+)\.pt$", path)
    return int(match.group(1)) if match else -1


def _load_model(model, path, device, env=None):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if env is not None and hasattr(env, "load_training_state"):
        env.load_training_state(checkpoint.get("env_training_state"))
    return checkpoint


def _configure_eval_env(env_cfg, num_envs):
    env_cfg.env.num_envs = num_envs
    env_cfg.env.test = False
    env_cfg.env.collect_substep_contact_metrics = True
    env_cfg.noise.add_noise = False
    env_cfg.init_state.randomize_yaw = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_pd_gains = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_motor_zero_offset = False
    env_cfg.domain_rand.randomize_action_delay = False
    env_cfg.domain_rand.push_robots = False


def _scenario_assignment(num_envs, device):
    scenario_ids = torch.arange(num_envs, device=device) % len(SCENARIOS)
    commands = torch.tensor(
        [SCENARIOS[int(index)]["command"] for index in scenario_ids.cpu()],
        dtype=torch.float,
        device=device,
    )
    return scenario_ids, commands


def _foot_sides(env):
    names = [env.body_names[int(index)] for index in env.feet_indices]
    left = [index for index, name in enumerate(names) if name.startswith(("FL", "RL"))]
    right = [index for index, name in enumerate(names) if name.startswith(("FR", "RR"))]
    if len(left) != 2 or len(right) != 2:
        raise RuntimeError(f"Could not derive left/right feet from {names}")
    return left, right


def _empty_metrics(num_envs, device):
    metrics = {}
    for scenario in SCENARIOS:
        metrics[scenario["name"]] = defaultdict(float)
        metrics[scenario["name"]]["force_samples"] = []
        metrics[scenario["name"]]["impulse_samples"] = []
        metrics[scenario["name"]]["touchdown_spreads"] = []
        metrics[scenario["name"]]["left_right_touchdown"] = []
        metrics[scenario["name"]]["expert_usage"] = None
    metrics["_state"] = {
        "in_flight": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "touchdown_steps": torch.full((num_envs, 4), -1, dtype=torch.long, device=device),
        "initial_xy": None,
    }
    return metrics


def _apply_transition_schedule(env, scenario_ids, step):
    updates = []
    if step == 250:
        updates = [
            ("walk_jump_walk", _command(0.5, body=-1.0)),
            ("stand_jump_stand", _command(-0.5, body=-1.0)),
        ]
    elif step == 650:
        updates = [
            ("walk_jump_walk", _command(0.5)),
            ("stand_jump_stand", _command()),
        ]
    for name, command in updates:
        scenario_index = next(i for i, value in enumerate(SCENARIOS) if value["name"] == name)
        env_ids = (scenario_ids == scenario_index).nonzero(as_tuple=False).flatten()
        values = torch.tensor(command, device=env.device).repeat(len(env_ids), 1)
        env.set_requested_commands(env_ids, values)
        env.commands_resampling_step[env_ids] = env.max_episode_length + 1


def _accumulate_step(metrics, env, scenario_ids, weights, dones, step, left_feet, right_feet):
    contact = env.contact_forces[:, env.feet_indices, 2] > env.cfg.commands.jump_contact_threshold
    all_air = ~torch.any(contact, dim=1)
    all_contact = torch.all(contact, dim=1)
    partial = ~(all_air | all_contact)
    state = metrics["_state"]
    just_airborne = all_air & ~state["in_flight"]
    state["in_flight"] |= all_air
    state["touchdown_steps"][just_airborne] = -1
    for foot in range(4):
        first = state["in_flight"] & contact[:, foot] & (state["touchdown_steps"][:, foot] < 0)
        state["touchdown_steps"][first, foot] = step
    completed = state["in_flight"] & torch.all(state["touchdown_steps"] >= 0, dim=1)

    for index, scenario in enumerate(SCENARIOS):
        mask = scenario_ids == index
        count = int(mask.sum().item())
        if count == 0:
            continue
        result = metrics[scenario["name"]]
        result["samples"] += count
        result["fall_count"] += int(dones[mask].sum().item())
        target = env.commands[mask, :3]
        actual = torch.cat((env.base_lin_vel[mask, :2], env.base_ang_vel[mask, 2:3]), dim=1)
        result["velocity_error_sq"] += torch.square(target - actual).sum().item()
        result["mean_actual_vx_sum"] += env.base_lin_vel[mask, 0].sum().item()
        result["mean_command_vx_sum"] += env.commands[mask, 0].sum().item()
        result["torque_clip_count"] += env.torque_clip_count_step[mask].sum().item()
        result["torque_samples"] += count * env.num_actions * env.cfg.control.decimation
        lower = env.dof_pos_limits[:, 0]
        upper = env.dof_pos_limits[:, 1]
        violation = (env.dof_pos[mask] < lower) | (env.dof_pos[mask] > upper)
        result["joint_limit_violations"] += violation.sum().item()

        active_jump = mask & (env.commands[:, env.body_height_command_idx] < -0.5)
        jump_count = int(active_jump.sum().item())
        if jump_count:
            result["jump_samples"] += jump_count
            result["all_air_count"] += all_air[active_jump].sum().item()
            result["all_contact_count"] += all_contact[active_jump].sum().item()
            result["partial_contact_count"] += partial[active_jump].sum().item()
            result["takeoff_count"] += just_airborne[active_jump].sum().item()
            result["force_samples"].append(env.substep_foot_force_max[active_jump].detach().cpu())
            result["impulse_samples"].append(env.substep_foot_normal_impulse[active_jump].detach().cpu())

        completed_mask = completed & mask
        if completed_mask.any():
            touchdown = state["touchdown_steps"][completed_mask]
            spreads = touchdown.max(dim=1).values - touchdown.min(dim=1).values
            left_time = touchdown[:, left_feet].float().mean(dim=1)
            right_time = touchdown[:, right_feet].float().mean(dim=1)
            result["touchdown_spreads"].extend(spreads.cpu().tolist())
            result["left_right_touchdown"].extend(torch.abs(left_time - right_time).cpu().tolist())
            result["landing_count"] += int(completed_mask.sum().item())

        if scenario["kind"] == "stand":
            drift = torch.norm(env.root_states[mask, :2] - state["initial_xy"][mask], dim=1)
            result["stand_drift_sum"] += drift.sum().item()
            result["stand_tilt_sum"] += torch.norm(env.rpy[mask, :2], dim=1).sum().item()

        usage = weights[mask].sum(dim=0).detach().cpu()
        result["expert_usage"] = usage if result["expert_usage"] is None else result["expert_usage"] + usage
        result["expert_samples"] += count

    state["in_flight"][completed] = False
    state["touchdown_steps"][completed] = -1
    state["in_flight"][dones] = False
    state["touchdown_steps"][dones] = -1


def _finalize_metrics(metrics):
    reports = {}
    for scenario in SCENARIOS:
        raw = metrics[scenario["name"]]
        samples = max(raw["samples"], 1.0)
        jump_samples = max(raw["jump_samples"], 1.0)
        forces = torch.cat(raw["force_samples"]).flatten() if raw["force_samples"] else torch.zeros(1)
        impulses = torch.cat(raw["impulse_samples"]).flatten() if raw["impulse_samples"] else torch.zeros(1)
        usage = raw["expert_usage"] / max(raw["expert_samples"], 1.0)
        report = {
            "kind": scenario["kind"],
            "fall_count": int(raw["fall_count"]),
            "velocity_rmse": (raw["velocity_error_sq"] / (3.0 * samples)) ** 0.5,
            "mean_actual_vx": raw["mean_actual_vx_sum"] / samples,
            "mean_command_vx": raw["mean_command_vx_sum"] / samples,
            "takeoffs": int(raw["takeoff_count"]),
            "landings": int(raw["landing_count"]),
            "all_air_fraction": raw["all_air_count"] / jump_samples,
            "all_contact_fraction": raw["all_contact_count"] / jump_samples,
            "partial_contact_fraction": raw["partial_contact_count"] / jump_samples,
            "touchdown_spread_mean_steps": sum(raw["touchdown_spreads"]) / max(len(raw["touchdown_spreads"]), 1),
            "left_right_touchdown_mean_steps": sum(raw["left_right_touchdown"]) / max(len(raw["left_right_touchdown"]), 1),
            "contact_force_max_n": forces.max().item(),
            "contact_force_p95_n": torch.quantile(forces, 0.95).item(),
            "contact_force_p99_n": torch.quantile(forces, 0.99).item(),
            "normal_impulse_max_ns": impulses.max().item(),
            "torque_clip_fraction": raw["torque_clip_count"] / max(raw["torque_samples"], 1.0),
            "joint_limit_violations": int(raw["joint_limit_violations"]),
            "stand_drift_mean_m": raw["stand_drift_sum"] / samples,
            "stand_tilt_mean_rad": raw["stand_tilt_sum"] / samples,
            "expert_usage": usage.tolist(),
        }
        target_vx = scenario["command"][0]
        report["direction_correct"] = target_vx == 0.0 or report["mean_actual_vx"] * target_vx > 0.0
        reports[scenario["name"]] = report
    return reports


def evaluate_model(env, model, seeds, steps, include_extended_jump=False):
    device = env.device
    scenario_ids, initial_commands = _scenario_assignment(env.num_envs, device)
    metrics = _empty_metrics(env.num_envs, device)
    left_feet, right_feet = _foot_sides(env)
    all_ids = torch.arange(env.num_envs, device=device)

    for seed in seeds:
        torch.manual_seed(seed)
        obs, _ = env.reset()
        env.set_requested_commands(all_ids, initial_commands)
        env.commands_resampling_step[:] = env.max_episode_length + 1
        env.compute_observations()
        obs = env.get_observations()
        history = torch.zeros(
            env.num_envs, model.history_length, env.num_obs, device=device
        )
        metrics["_state"]["initial_xy"] = env.root_states[:, :2].clone()
        metrics["_state"]["in_flight"].zero_()
        metrics["_state"]["touchdown_steps"].fill_(-1)

        for step in range(steps):
            _apply_transition_schedule(env, scenario_ids, step)
            with torch.no_grad():
                history = torch.cat([history[:, 1:], obs.unsqueeze(1)], dim=1)
                latent, weights = model.student_moe_encoder(history.flatten(1))
                actions = model.actor(torch.cat([latent, obs], dim=1))
                obs, _, _, dones, _ = env.step(actions)
            _accumulate_step(
                metrics, env, scenario_ids, weights, dones.bool(), step, left_feet, right_feet
            )
            history[dones > 0] = 0.0
    reports = _finalize_metrics(metrics)
    for value in reports.values():
        if value["kind"] == "jump_extended":
            value["kind"] = "jump" if include_extended_jump else "optional"
    return reports


def _aggregate_walk_rmse(reports):
    values = [value["velocity_rmse"] for value in reports.values() if value["kind"] == "walk"]
    return sum(values) / max(len(values), 1)


def _evaluate_gates(reports, baseline_reports):
    jump_reports = [value for value in reports.values() if value["kind"] == "jump"]
    transition_reports = [value for value in reports.values() if value["kind"] == "transition"]
    all_reports = [value for value in reports.values() if value["kind"] != "optional"]
    baseline_walk_rmse = _aggregate_walk_rmse(baseline_reports)
    candidate_walk_rmse = _aggregate_walk_rmse(reports)
    degradation = (candidate_walk_rmse - baseline_walk_rmse) / max(baseline_walk_rmse, 1e-8)
    gates = {
        "no_falls": all(value["fall_count"] == 0 for value in all_reports),
        "jump_direction": all(value["direction_correct"] for value in jump_reports),
        "jump_partial_contacts": all(value["partial_contact_fraction"] < 0.05 for value in jump_reports),
        "walking_degradation": degradation < 0.20,
        "torque_clipping": all(value["torque_clip_fraction"] < 0.01 for value in all_reports),
        "joint_limits": all(value["joint_limit_violations"] == 0 for value in all_reports),
        "transition_falls": all(value["fall_count"] == 0 for value in transition_reports),
    }
    jump_mae = sum(
        abs(value["mean_actual_vx"] - value["mean_command_vx"])
        for value in jump_reports
    ) / max(len(jump_reports), 1)
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "walking_rmse": candidate_walk_rmse,
        "baseline_walking_rmse": baseline_walk_rmse,
        "walking_degradation": degradation,
        "jump_mean_absolute_velocity_error": jump_mae,
    }


def _jit_parity(model, policy_path, device):
    scripted = torch.jit.load(policy_path, map_location="cpu").eval()
    scripted.reset()
    native = copy.deepcopy(model).to("cpu").eval()
    generator = torch.Generator(device="cpu").manual_seed(1234)
    sequence = torch.randn(32, 1, 48, generator=generator)
    history = torch.zeros(1, native.history_length, 48)
    maxima = {"action": 0.0, "latent": 0.0, "gate": 0.0}
    with torch.no_grad():
        for obs_cpu in sequence:
            obs = obs_cpu
            history = torch.cat([history[:, 1:], obs.unsqueeze(1)], dim=1)
            latent, gate = native.student_moe_encoder(history.flatten(1))
            action = native.actor(torch.cat([latent, obs], dim=1))
            jit_action, (jit_gate, jit_latent) = scripted(obs_cpu)
            maxima["action"] = max(maxima["action"], torch.max(torch.abs(action.cpu() - jit_action)).item())
            maxima["latent"] = max(maxima["latent"], torch.max(torch.abs(latent.cpu() - jit_latent)).item())
            maxima["gate"] = max(maxima["gate"], torch.max(torch.abs(gate.cpu() - jit_gate)).item())
    scripted.reset()
    reset_action, _ = scripted(sequence[0])
    with torch.no_grad():
        zero_history = torch.zeros_like(history)
        zero_history = torch.cat([zero_history[:, 1:], sequence[0].unsqueeze(1)], dim=1)
        latent, _ = native.student_moe_encoder(zero_history.flatten(1))
        native_reset = native.actor(torch.cat([latent, sequence[0]], dim=1))
    maxima["reset_action"] = torch.max(torch.abs(native_reset.cpu() - reset_action)).item()
    return maxima


def export_best(model, env, checkpoint_path, output_dir, evaluation):
    os.makedirs(output_dir, exist_ok=True)
    policy_path = os.path.join(output_dir, "policy.pt")
    export_policy_as_jit(model, output_dir, filename="policy.pt")
    parity = _jit_parity(model, policy_path, env.device)
    if max(parity.values()) > 1e-5:
        raise RuntimeError(f"Native/JIT parity failed: {parity}")
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=LEGGED_GYM_ROOT_DIR, text=True
        ).strip()
    except Exception:
        git_commit = "unknown"
    manifest = {
        "schema_version": 1,
        "task": "win_jump_moe_cts",
        "git_commit": git_commit,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "policy_path": os.path.abspath(policy_path),
        "policy_sha256": _sha256(policy_path),
        "num_observations": 48,
        "num_privileged_observations": 266,
        "num_actions": 12,
        "history_length": 5,
        "observation_layout": [
            "base_ang_vel[3]",
            "projected_gravity[3]",
            "command[vx,vy,yaw,body_mode,jump_sin,jump_cos]",
            "dof_pos[12]",
            "dof_vel[12]",
            "last_action[12]",
        ],
        "command_modes": {"jump": -1.0, "normal": 0.0, "low_height": 1.0},
        "observation_scales": {
            "lin_vel": env.obs_scales.lin_vel,
            "ang_vel": env.obs_scales.ang_vel,
            "dof_pos": env.obs_scales.dof_pos,
            "dof_vel": env.obs_scales.dof_vel,
            "command": env.commands_scale.tolist(),
        },
        "model_joint_names": list(env.dof_names),
        "default_joint_angles": env.default_dof_pos[0].tolist(),
        "joint_position_limits": env.dof_pos_limits.tolist(),
        "joint_velocity_limits": env.dof_vel_limits.tolist(),
        "joint_torque_limits": env.torque_limits.tolist(),
        "control": {
            "kp": 30.0,
            "kd": 1.0,
            "action_scale": env.cfg.control.action_scale,
            "policy_dt": env.dt,
            "simulation_dt": env.sim_params.dt,
            "decimation": env.cfg.control.decimation,
        },
        "jit_parity_max_abs_error": parity,
        "evaluation": evaluation,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest_path


def evaluate(args):
    if args.task != "win_jump_moe_cts":
        raise ValueError("evaluate_z2_jump.py requires --task win_jump_moe_cts")
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    num_envs = args.num_envs or 256
    _configure_eval_env(env_cfg, num_envs)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    args.resume = False
    args.warmstart_path = None
    runner, _ = task_registry.make_alg_runner(
        env, args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    model = runner.alg.model
    seeds = [int(value) for value in args.eval_seeds.split(",") if value.strip()]

    baseline_path = os.path.abspath(args.baseline_checkpoint or DEFAULT_BASELINE)
    _load_model(model, baseline_path, env.device, env)
    baseline_reports = evaluate_model(env, model, seeds, args.jump_eval_steps)

    if args.checkpoint_path:
        checkpoint_paths = [os.path.abspath(args.checkpoint_path)]
    elif args.checkpoint_dir:
        checkpoint_paths = sorted(
            glob.glob(os.path.join(os.path.abspath(args.checkpoint_dir), "model_*.pt")),
            key=_checkpoint_iteration,
        )
        checkpoint_paths = [
            path for path in checkpoint_paths
            if 0 <= _checkpoint_iteration(path) <= 20000
            and _checkpoint_iteration(path) % 4000 == 0
        ]
    else:
        raise ValueError("Provide --checkpoint_path or --checkpoint_dir")
    if not checkpoint_paths:
        raise FileNotFoundError("No 0/4k/.../20k checkpoints found")

    output_dir = os.path.abspath(
        args.eval_output_dir or os.path.join(os.path.dirname(checkpoint_paths[0]), "jump_evaluation")
    )
    os.makedirs(output_dir, exist_ok=True)
    summaries = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = _load_model(model, checkpoint_path, env.device, env)
        unlocked = bool((checkpoint.get("env_training_state") or {}).get("jump_speed_unlocked", False))
        reports = evaluate_model(
            env,
            model,
            seeds,
            args.jump_eval_steps,
            include_extended_jump=unlocked,
        )
        summary = _evaluate_gates(reports, baseline_reports)
        result = {
            "checkpoint": checkpoint_path,
            "iteration": _checkpoint_iteration(checkpoint_path),
            "seeds": seeds,
            "steps_per_seed": args.jump_eval_steps,
            "jump_speed_unlocked": unlocked,
            "summary": summary,
            "scenarios": reports,
        }
        result_path = os.path.join(output_dir, f"model_{result['iteration']}.json")
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        summaries.append(result)
        print(json.dumps({"checkpoint": checkpoint_path, **summary}, indent=2, sort_keys=True))

    passing = [result for result in summaries if result["summary"]["passed"]]
    best = min(
        passing,
        key=lambda value: value["summary"]["jump_mean_absolute_velocity_error"],
        default=None,
    )
    index = {
        "baseline_checkpoint": baseline_path,
        "evaluated": [result["checkpoint"] for result in summaries],
        "best_checkpoint": best["checkpoint"] if best else None,
    }
    if args.export_best and best is not None:
        _load_model(model, best["checkpoint"], env.device, env)
        index["manifest"] = export_best(
            model,
            env,
            best["checkpoint"],
            os.path.join(output_dir, "best"),
            best["summary"],
        )
    with open(os.path.join(output_dir, "index.json"), "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
    return index


if __name__ == "__main__":
    evaluate(get_args())
