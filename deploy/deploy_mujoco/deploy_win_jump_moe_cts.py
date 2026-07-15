import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from win_jump_command import JumpCommandFSM, MODE_JUMP


ROOT = Path(__file__).resolve().parents[2]


def _single_profile(vx=0.0, vy=0.0, yaw=0.0, body_mode=0.0):
    return [
        {
            "time": 0.0,
            "vx": vx,
            "vy": vy,
            "yaw": yaw,
            "body_mode": body_mode,
        }
    ]


BUILTIN_PROFILES = {
    "stand": _single_profile(),
    "jump_fwd_03": _single_profile(0.3, body_mode=-1.0),
    "jump_fwd_05": _single_profile(0.5, body_mode=-1.0),
    "jump_fwd_08": _single_profile(0.8, body_mode=-1.0),
    "jump_fwd_10": _single_profile(1.0, body_mode=-1.0),
    "jump_back_03": _single_profile(-0.3, body_mode=-1.0),
    "jump_back_05": _single_profile(-0.5, body_mode=-1.0),
    "jump_back_08": _single_profile(-0.8, body_mode=-1.0),
    "jump_back_10": _single_profile(-1.0, body_mode=-1.0),
    "walk_fwd": _single_profile(0.5),
    "walk_back": _single_profile(-0.5),
    "walk_left": _single_profile(vy=0.5),
    "walk_right": _single_profile(vy=-0.5),
    "turn_left": _single_profile(yaw=1.0),
    "turn_right": _single_profile(yaw=-1.0),
    "walk_mixed": _single_profile(0.5, 0.3, 0.5),
    "low_height": _single_profile(0.3, body_mode=1.0),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expand_path(value):
    return os.path.abspath(value.replace("{LEGGED_GYM_ROOT_DIR}", str(ROOT)))


def load_inputs(config_path, manifest_override=None):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["profiles"].update(BUILTIN_PROFILES)
    manifest_path = expand_path(manifest_override or config["manifest_path"])
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    config["xml_path"] = expand_path(config["xml_path"])
    return config, manifest, manifest_path


def quaternion_conjugate(q):
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quaternion_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def rotate_inverse(q, vector):
    pure = np.asarray([0.0, *vector], dtype=np.float64)
    return quaternion_multiply(
        quaternion_multiply(quaternion_conjugate(q), pure), q
    )[1:]


def validate_interface(model, config, manifest):
    required = {
        "policy_sha256",
        "policy_path",
        "model_joint_names",
        "default_joint_angles",
        "num_observations",
        "num_actions",
        "history_length",
        "control",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Manifest missing fields: {missing}")
    policy_path = expand_path(manifest["policy_path"])
    if sha256(policy_path) != manifest["policy_sha256"]:
        raise ValueError("Policy SHA256 does not match manifest")
    if manifest["num_observations"] != 48 or manifest["num_actions"] != 12:
        raise ValueError("Runner requires the 48-observation, 12-action interface")
    if manifest["history_length"] != 5:
        raise ValueError("Runner requires CTS history length 5")

    control = manifest["control"]
    expected_dt = config["simulation_dt"] * config["control_decimation"]
    checks = {
        "kp": 30.0,
        "kd": 1.0,
        "action_scale": 0.25,
        "policy_dt": expected_dt,
    }
    for key, expected in checks.items():
        if not math.isclose(float(control[key]), expected, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"Manifest control {key}={control[key]} != {expected}")

    qpos_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]
    actuator_names = [
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            model.actuator_trnid[actuator_id, 0],
        )
        for actuator_id in range(model.nu)
    ]
    model_names = list(manifest["model_joint_names"])
    if set(model_names) != set(qpos_names) or set(model_names) != set(actuator_names):
        raise ValueError(
            f"Joint set mismatch: policy={model_names}, qpos={qpos_names}, actuators={actuator_names}"
        )
    return policy_path, model_names, qpos_names, actuator_names


def joint_addresses(model, joint_names):
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joint_names
    ]
    return (
        np.asarray([model.jnt_qposadr[index] for index in joint_ids]),
        np.asarray([model.jnt_dofadr[index] for index in joint_ids]),
    )


def build_joint_maps(model_names, qpos_names, actuator_names):
    if set(model_names) != set(qpos_names) or set(model_names) != set(actuator_names):
        raise ValueError("Policy, qpos, and actuator joint sets must match")
    model_from_qpos = np.asarray([qpos_names.index(name) for name in model_names])
    actuator_from_model = np.asarray([model_names.index(name) for name in actuator_names])
    return model_from_qpos, actuator_from_model


def foot_contact_forces(model, data, foot_geom_ids):
    forces = np.zeros(4, dtype=np.float64)
    wrench = np.zeros(6, dtype=np.float64)
    geom_to_foot = {geom_id: index for index, geom_id in enumerate(foot_geom_ids)}
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        foot = geom_to_foot.get(contact.geom1, geom_to_foot.get(contact.geom2))
        if foot is None:
            continue
        mujoco.mj_contactForce(model, data, contact_index, wrench)
        forces[foot] += abs(wrench[0])
    return forces


def apply_randomization(model, data, cfg, rng, enabled):
    num_actions = 12
    result = {
        "kp_scale": np.ones(num_actions),
        "kd_scale": np.ones(num_actions),
        "motor_strength": np.ones(num_actions),
        "zero_offset": np.zeros(num_actions),
        "delay_substeps": 0,
    }
    if not enabled:
        return result
    rand = cfg["randomization"]
    friction = rng.uniform(*rand["friction"])
    model.geom_friction[:, 0] = friction
    restitution = rng.uniform(*rand["restitution"])
    # MuJoCo's positive solref second parameter is a damping ratio; lower gives more bounce.
    model.geom_solref[:, 1] = max(0.1, 1.0 - restitution)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    link_scale = rng.uniform(*rand["link_mass_scale"])
    model.body_mass[1:] *= link_scale
    model.body_mass[base_id] += rng.uniform(*rand["added_base_mass"])
    model.body_ipos[base_id] += rng.uniform(*rand["base_com"], size=3)
    result["kp_scale"][:] = rng.uniform(*rand["pd_scale"], size=num_actions)
    result["kd_scale"][:] = rng.uniform(*rand["pd_scale"], size=num_actions)
    result["motor_strength"][:] = rng.uniform(*rand["motor_strength"], size=num_actions)
    result["zero_offset"][:] = rng.uniform(*rand["motor_zero_offset"], size=num_actions)
    delay_s = rng.uniform(*rand["action_delay_s"])
    result["delay_substeps"] = int(round(delay_s / cfg["simulation_dt"]))
    mujoco.mj_setConst(model, data)
    return result


def build_observation(
    data,
    command,
    q_model,
    dq_model,
    default_angles,
    last_action,
):
    quat = np.asarray(data.qpos[3:7])
    body_ang_vel = rotate_inverse(quat, data.qvel[3:6])
    gravity = rotate_inverse(quat, [0.0, 0.0, -1.0])
    command_scaled = command * np.asarray([2.0, 2.0, 0.25, 1.0, 1.0, 1.0])
    return np.concatenate(
        (
            body_ang_vel * 0.25,
            gravity,
            command_scaled,
            q_model - default_angles,
            dq_model * 0.05,
            last_action,
        )
    ).astype(np.float32)


def run_trial(config, manifest, profile_name, seed=0, randomized=False):
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(config["xml_path"])
    data = mujoco.MjData(model)
    model.opt.timestep = config["simulation_dt"]
    policy_path, model_names, qpos_names, actuator_names = validate_interface(
        model, config, manifest
    )
    qpos_addr, qvel_addr = joint_addresses(model, qpos_names)
    model_from_qpos, actuator_from_model = build_joint_maps(
        model_names, qpos_names, actuator_names
    )
    default_angles = np.asarray(manifest["default_joint_angles"], dtype=np.float64)
    torque_limits = np.asarray(manifest["joint_torque_limits"], dtype=np.float64)
    position_limits = np.asarray(manifest["joint_position_limits"], dtype=np.float64)
    randomization = apply_randomization(model, data, config, rng, randomized)

    data.qpos[:3] = [0.0, 0.0, 0.42]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qpos_values = default_angles[np.asarray([model_names.index(name) for name in qpos_names])]
    data.qpos[qpos_addr] = qpos_values
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    policy = torch.jit.load(policy_path, map_location="cpu").eval()
    policy.reset()
    fsm = JumpCommandFSM(policy_dt=config["simulation_dt"] * config["control_decimation"])
    schedule = config["profiles"][profile_name]
    schedule_index = 0
    last_action = np.zeros(12, dtype=np.float64)
    current_action = np.zeros(12, dtype=np.float64)
    previous_action = np.zeros(12, dtype=np.float64)
    foot_geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("FL_foot", "RL_foot", "FR_foot", "RR_foot")
    ]
    contact_force_max = []
    contact_impulse = np.zeros(4)
    impulse_samples = []
    velocity_error = []
    actual_vx = []
    command_vx = []
    partial_contacts = 0
    jump_contact_samples = 0
    torque_clip_count = 0
    torque_sample_count = 0
    joint_limit_violations = 0
    falls = 0
    timed_out = False
    max_steps = int(round(config["duration_s"] / config["simulation_dt"]))
    decimation = int(config["control_decimation"])
    contacts = np.ones(4, dtype=bool)
    push_interval = int(round(config["randomization"]["push_interval_s"] / config["simulation_dt"]))

    for sim_step in range(max_steps):
        policy_tick = sim_step % decimation == 0
        if policy_tick:
            time_s = sim_step * config["simulation_dt"]
            while schedule_index < len(schedule) and time_s >= schedule[schedule_index]["time"]:
                item = schedule[schedule_index]
                fsm.request(item["vx"], item["vy"], item["yaw"], item["body_mode"])
                schedule_index += 1
            command = fsm.step(contacts)
            if fsm.timed_out:
                timed_out = True
                break
            q_qpos = data.qpos[qpos_addr]
            dq_qpos = data.qvel[qvel_addr]
            q_model = q_qpos[model_from_qpos]
            dq_model = dq_qpos[model_from_qpos]
            obs = build_observation(
                data, command, q_model, dq_model, default_angles, last_action
            )
            previous_action = last_action.copy()
            with torch.no_grad():
                output = policy(torch.from_numpy(obs).unsqueeze(0))
                current_action = output[0].squeeze(0).numpy().astype(np.float64)
            local_velocity = rotate_inverse(data.qpos[3:7], data.qvel[:3])
            velocity_error.append((command[:3] - [local_velocity[0], local_velocity[1], rotate_inverse(data.qpos[3:7], data.qvel[3:6])[2]]) ** 2)
            actual_vx.append(local_velocity[0])
            command_vx.append(command[0])
            if fsm.active_mode == MODE_JUMP:
                jump_contact_samples += 1
                partial_contacts += int(contacts.any() and not contacts.all())
            last_action = current_action.copy()
            if sim_step > 0:
                impulse_samples.append(contact_impulse.copy())
            contact_impulse[:] = 0.0

        q_qpos = data.qpos[qpos_addr]
        dq_qpos = data.qvel[qvel_addr]
        q_model = q_qpos[model_from_qpos]
        dq_model = dq_qpos[model_from_qpos]
        substep_in_control = sim_step % decimation
        used_action = previous_action if substep_in_control < randomization["delay_substeps"] else current_action
        target = default_angles + randomization["zero_offset"] + 0.25 * used_action
        torque_model = (
            30.0 * randomization["kp_scale"] * (target - q_model)
            - 1.0 * randomization["kd_scale"] * dq_model
        ) * randomization["motor_strength"]
        clipped = np.abs(torque_model) > torque_limits
        torque_clip_count += int(clipped.sum())
        torque_sample_count += 12
        torque_model = np.clip(torque_model, -torque_limits, torque_limits)
        data.ctrl[:] = torque_model[actuator_from_model]

        if randomized and sim_step > 0 and sim_step % push_interval == 0:
            data.qvel[:2] += rng.uniform(
                -config["randomization"]["push_linear"],
                config["randomization"]["push_linear"],
                size=2,
            )
            data.qvel[3:6] += rng.uniform(
                -config["randomization"]["push_angular"],
                config["randomization"]["push_angular"],
                size=3,
            )

        mujoco.mj_step(model, data)
        foot_forces = foot_contact_forces(model, data, foot_geom_ids)
        contacts = foot_forces > config["contact_threshold_n"]
        contact_force_max.append(foot_forces.copy())
        contact_impulse += foot_forces * config["simulation_dt"]
        q_model = data.qpos[qpos_addr][model_from_qpos]
        joint_limit_violations += int(
            ((q_model < position_limits[:, 0]) | (q_model > position_limits[:, 1])).sum()
        )
        gravity = rotate_inverse(data.qpos[3:7], [0.0, 0.0, -1.0])
        tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
        if data.qpos[2] < config["fall_height_m"] or tilt > config["fall_tilt_rad"]:
            falls += 1
            break

    forces = np.asarray(contact_force_max) if contact_force_max else np.zeros((1, 4))
    impulse_samples.append(contact_impulse.copy())
    impulses = np.asarray(impulse_samples) if impulse_samples else np.zeros((1, 4))
    errors = np.asarray(velocity_error) if velocity_error else np.zeros((1, 3))
    return {
        "seed": seed,
        "randomized": randomized,
        "profile": profile_name,
        "falls": falls,
        "transition_timeout": timed_out,
        "velocity_rmse": float(np.sqrt(errors.mean())),
        "mean_actual_vx": float(np.mean(actual_vx)) if actual_vx else 0.0,
        "mean_command_vx": float(np.mean(command_vx)) if command_vx else 0.0,
        "direction_correct": (
            not command_vx
            or abs(float(np.mean(command_vx))) < 1e-6
            or float(np.mean(command_vx)) * float(np.mean(actual_vx)) > 0.0
        ),
        "partial_contact_fraction": partial_contacts / max(jump_contact_samples, 1),
        "contact_force_max_n": float(np.max(forces)),
        "contact_force_p95_n": float(np.quantile(forces, 0.95)),
        "contact_force_p99_n": float(np.quantile(forces, 0.99)),
        "normal_impulse_max_ns": float(np.max(impulses)),
        "torque_clip_fraction": torque_clip_count / max(torque_sample_count, 1),
        "joint_limit_violations": joint_limit_violations,
        "sanitized_commands": fsm.sanitized_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Fail-closed Z2 48-d MoE-CTS MuJoCo runner")
    parser.add_argument(
        "--config",
        default=str(ROOT / "deploy/deploy_mujoco/configs/win_jump_moe_cts.yaml"),
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--profile", default="walk_jump_walk")
    parser.add_argument("--suite", action="store_true", help="Run the full command matrix.")
    parser.add_argument("--monte-carlo", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config, manifest, _ = load_inputs(args.config, args.manifest)
    if args.profile not in config["profiles"]:
        raise ValueError(f"Unknown profile {args.profile}; available={sorted(config['profiles'])}")
    profiles = sorted(config["profiles"]) if args.suite else [args.profile]
    results = []
    for profile in profiles:
        results.append(run_trial(config, manifest, profile, seed=args.seed, randomized=False))
        for index in range(args.monte_carlo):
            results.append(
                run_trial(config, manifest, profile, seed=args.seed + index + 1, randomized=True)
            )
    report = {
        "manifest": manifest["policy_path"],
        "profiles": profiles,
        "trials": results,
        "all_nominal_gates_passed": all(
            result["falls"] == 0
            and not result["transition_timeout"]
            and result["torque_clip_fraction"] < 0.01
            and result["joint_limit_violations"] == 0
            and (not result["profile"].startswith("jump_") or result["direction_correct"])
            and (not result["profile"].startswith("jump_") or result["partial_contact_fraction"] < 0.05)
            for result in results
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
