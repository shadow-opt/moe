import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import os
import imageio
from pathlib import Path
from argparse import ArgumentParser
import pygame


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def validate_config(config):
    cmd_scale = np.asarray(config["cmd_scale"], dtype=np.float32)
    cmd_init = np.asarray(config["cmd_init"], dtype=np.float32)
    num_actions = int(config["num_actions"])
    num_obs = int(config["num_obs"])
    expected_num_obs = 6 + len(cmd_scale) + 3 * num_actions

    if len(cmd_scale) != len(cmd_init):
        raise ValueError(
            f"cmd_scale length ({len(cmd_scale)}) must match cmd_init length ({len(cmd_init)})"
        )
    if len(cmd_scale) != 8:
        raise ValueError(
            "Jump deployment expects 8 command observation values: "
            "[vx, vy, wz, body_height_mode, jump_dx, jump_dy, jump_dz, jump_trigger]"
        )
    if num_obs != expected_num_obs:
        raise ValueError(
            f"num_obs={num_obs} does not match expected observation size {expected_num_obs}"
        )

    jump_limits = {
        "dx": np.asarray(config.get("jump_dx_limits", [-0.35, 0.35]), dtype=np.float32),
        "dy": np.asarray(config.get("jump_dy_limits", [-0.20, 0.20]), dtype=np.float32),
        "dz": np.asarray(config.get("jump_dz_limits", [0.12, 0.42]), dtype=np.float32),
        "takeoff_velocity_threshold": float(config.get("takeoff_velocity_threshold", 0.15)),
        "takeoff_height_threshold": float(config.get("takeoff_height_threshold", 0.03)),
        "landing_height_margin": float(config.get("landing_height_margin", 0.015)),
        "min_jump_hold_s": float(config.get("min_jump_hold_s", 0.20)),
    }
    for name in ("dx", "dy", "dz"):
        if jump_limits[name].shape != (2,):
            raise ValueError(f"jump_{name}_limits must contain [min, max]")
        if jump_limits[name][0] > jump_limits[name][1]:
            raise ValueError(f"jump_{name}_limits must satisfy min <= max")

    return cmd_scale, cmd_init, jump_limits


def clip_with_warning(value, limits, name):
    clipped = float(np.clip(value, limits[0], limits[1]))
    if not np.isclose(clipped, value):
        print(
            f"[WARN] {name}={value:.3f} 超出训练范围 [{limits[0]:.3f}, {limits[1]:.3f}]，"
            f"已裁剪为 {clipped:.3f}"
        )
    return clipped


def set_walk_command(cmd, vx, vy, wz):
    cmd[0] = vx
    cmd[1] = vy
    cmd[2] = wz
    cmd[3] = 0.0
    cmd[4] = 0.0
    cmd[5] = 0.0
    cmd[6] = 0.0
    cmd[7] = 0.0


def draw_moe_weights(screen, weights, width, height):
    screen.fill((255, 255, 255))

    num_experts = len(weights)
    if num_experts == 0:
        return

    margin = 5
    bar_width = (width - 2 * margin) / num_experts
    max_bar_height = height - 2 * margin
    for i, w in enumerate(weights):
        w_clamped = max(0.0, min(1.0, w))
        bar_height = int(w_clamped * max_bar_height)

        x = margin + i * bar_width
        y = height - margin - bar_height

        pygame.draw.rect(screen, (50, 100, 255), (x, y, bar_width - 2, bar_height))

    pygame.display.flip()


def set_jump_command(cmd, jump_state):
    cmd[0] = 0.0
    cmd[1] = 0.0
    cmd[2] = 0.0
    cmd[3] = 1.0
    cmd[4] = jump_state["dx"]
    cmd[5] = jump_state["dy"]
    cmd[6] = jump_state["dz"]
    cmd[7] = 1.0


def update_command_with_jump(cmd, joystick, max_cmd, button_state, jump_state, jump_args, robot_state):
    """更新手柄输入：摇杆控制速度，RB 键触发跳跃。

    cmd layout: [vx, vy, wz, body_height_mode, jump_dx, jump_dy, jump_dz, jump_trigger]

    跳跃状态机：
    - IDLE: 正常行走，RB 按下时进入 JUMPING
    - JUMPING: 跳跃命令生效，检测到起跳并回落后回到 IDLE
    """
    pygame.event.pump()
    dead_zone = 0.1
    now = time.time()
    walk_vx = 0.0
    walk_vy = 0.0
    walk_wz = 0.0
    rb_rising = False

    if joystick is not None:
        lx = joystick.get_axis(0)
        ly = joystick.get_axis(1)
        rx = joystick.get_axis(3)
        if abs(lx) < dead_zone: lx = 0
        if abs(ly) < dead_zone: ly = 0
        if abs(rx) < dead_zone: rx = 0
        walk_vx = -ly * max_cmd[0]
        walk_vy = -lx * max_cmd[1]
        walk_wz = -rx * max_cmd[2]

        rb_pressed = joystick.get_button(5)
        rb_rising = rb_pressed and not button_state.get("rb", False)
        button_state["rb"] = rb_pressed
    else:
        button_state["rb"] = False

    if jump_state["mode"] == "IDLE":
        set_walk_command(cmd, walk_vx, walk_vy, walk_wz)
        if rb_rising:
            jump_dx = clip_with_warning(
                walk_vx / max(max_cmd[0], 1e-6) * jump_args["dx_limit"],
                jump_args["dx_limits"],
                "jump_dx",
            )
            jump_dy = clip_with_warning(
                walk_vy / max(max_cmd[1], 1e-6) * jump_args["dy_limit"],
                jump_args["dy_limits"],
                "jump_dy",
            )
            jump_dz = clip_with_warning(jump_args["dz"], jump_args["dz_limits"], "jump_dz")

            jump_state["mode"] = "JUMPING"
            jump_state["trigger_time"] = now
            jump_state["dx"] = jump_dx
            jump_state["dy"] = jump_dy
            jump_state["dz"] = jump_dz
            jump_state["has_taken_off"] = False
            jump_state["launch_height"] = robot_state["base_height"] if robot_state is not None else 0.0
            set_jump_command(cmd, jump_state)
    elif jump_state["mode"] == "JUMPING":
        set_jump_command(cmd, jump_state)

        elapsed = now - jump_state["trigger_time"]
        if robot_state is not None:
            if (not jump_state["has_taken_off"]) and (
                robot_state["base_lin_vel_z"] >= jump_args["takeoff_velocity_threshold"]
                or robot_state["base_height"] >= jump_state["launch_height"] + jump_args["takeoff_height_threshold"]
            ):
                jump_state["has_taken_off"] = True

            landed = (
                jump_state["has_taken_off"]
                and elapsed >= jump_args["min_jump_hold_s"]
                and robot_state["base_height"] <= jump_state["launch_height"] + jump_args["landing_height_margin"]
                and robot_state["base_lin_vel_z"] <= 0.0
            )
        else:
            landed = False

        timed_out = elapsed >= jump_args["timeout"]
        if landed or timed_out:
            jump_state["mode"] = "IDLE"
            jump_state["has_taken_off"] = False
            set_walk_command(cmd, walk_vx, walk_vy, walk_wz)

    return cmd


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-video", action="store_true", help="Whether to save video of the simulation.")
    parser.add_argument("--visualize-moe-weights", action="store_true", help="Whether to visualize mixture of experts weights.")
    parser.add_argument("--config", default="jump.yaml", help="Deployment config file under deploy/deploy_mujoco/configs/.")
    parser.add_argument("--policy-path", default=None, help="Optional override for policy path.")
    parser.add_argument("--jump-dz", type=float, default=0.20, help="Target jump height (dz).")
    parser.add_argument("--jump-dx-range", type=float, default=0.35, help="Max jump dx from stick deflection.")
    parser.add_argument("--jump-dy-range", type=float, default=0.20, help="Max jump dy from stick deflection.")
    parser.add_argument("--jump-cooldown", type=float, default=3.0, help="Seconds after trigger before jump commands reset.")
    args = parser.parse_args()
    save_video = args.save_video
    visualize_moe_weights = args.visualize_moe_weights
    config_file = args.config

    # Pygame 初始化
    pygame.init()

    use_joystick = False
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        use_joystick = True
        print(f"Detected Joystick: {joystick.get_name()}")
    else:
        print("No Joystick detected. Using default commands from config.")

    screen = None
    win_width, win_height = 400, 200
    if visualize_moe_weights:
        screen = pygame.display.set_mode((win_width, win_height))
        pygame.display.set_caption("MoE Weights Visualization")

    config_path = Path(LEGGED_GYM_ROOT_DIR) / "deploy" / "deploy_mujoco" / "configs" / config_file
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        if args.policy_path is not None:
            policy_path = args.policy_path.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale, cmd, jump_limits = validate_config(config)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        idx_model2mj = idx_mj2model = list(range(num_actions))
        if 'mujoco_joint_names' in config and 'model_joint_names' in config:
            mujoco_joint_names = config["mujoco_joint_names"]
            model_joint_names = config["model_joint_names"]
            idx_model2mj = [model_joint_names.index(joint) for joint in mujoco_joint_names]
            idx_mj2model = [mujoco_joint_names.index(joint) for joint in model_joint_names]

    jump_args = {
        "dz": clip_with_warning(args.jump_dz, jump_limits["dz"], "jump_dz"),
        "dx_limit": float(args.jump_dx_range),
        "dy_limit": float(args.jump_dy_range),
        "dx_limits": jump_limits["dx"],
        "dy_limits": jump_limits["dy"],
        "dz_limits": jump_limits["dz"],
        "timeout": float(args.jump_cooldown),
        "takeoff_velocity_threshold": jump_limits["takeoff_velocity_threshold"],
        "takeoff_height_threshold": jump_limits["takeoff_height_threshold"],
        "landing_height_margin": jump_limits["landing_height_margin"],
        "min_jump_hold_s": jump_limits["min_jump_hold_s"],
    }

    if not os.path.exists(policy_path):
        raise FileNotFoundError(
            f"Policy file not found: {policy_path}. "
            f"Please export a jump policy or pass --policy-path to override it."
        )
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Mujoco model file not found: {xml_path}")

    video_save_dir = str(Path(__file__).parent / "videos")
    os.makedirs(video_save_dir, exist_ok=True)

    model_name = os.path.basename(policy_path).split('.')[0]
    cmd_str = f"dz_{jump_args['dz']}"
    video_filename = f"{model_name}_{cmd_str}.mp4"
    video_path = os.path.join(video_save_dir, video_filename)
    print(f"Video recording will be saved to: {video_path}")
    print(f"Jump params: dz={jump_args['dz']}, dx_limit={jump_args['dx_limit']}, "
          f"dy_limit={jump_args['dy_limit']}, timeout={jump_args['timeout']}s")

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    last_action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    button_state = {}
    jump_state = {
        "mode": "IDLE",
        "trigger_time": 0.0,
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
        "has_taken_off": False,
        "launch_height": 0.0,
    }

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    renderer = mujoco.Renderer(m, height=360, width=640)

    # load policy
    policy = torch.jit.load(policy_path)

    if save_video:
        video_fps = 50
        sim_fps = 1.0 / m.opt.timestep
        frame_skip = int(sim_fps / video_fps)
        if frame_skip < 1:
            frame_skip = 1
        writer = imageio.get_writer(video_path, fps=video_fps)
        print(f"Sim FPS: {sim_fps:.2f}, Video FPS: {video_fps}, Frame Skip: {frame_skip}, Save at: {video_path}")

    with mujoco.viewer.launch_passive(m, d) as viewer:

        # set viewer.camera to follow robot
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -30.0
        viewer.cam.azimuth = 0.0

        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            if use_joystick and counter % control_decimation == 0:
                robot_state = {
                    "base_height": float(d.qpos[2]),
                    "base_lin_vel_z": float(d.qvel[2]),
                }
                cmd = update_command_with_jump(
                    cmd,
                    joystick,
                    config["max_cmd"],
                    button_state,
                    jump_state,
                    jump_args,
                    robot_state,
                )
                mode_label = jump_state["mode"]
                if mode_label == "JUMPING":
                    print(
                        f"[{mode_label:7s}] dx={jump_state['dx']:.2f}, dy={jump_state['dy']:.2f}, "
                        f"dz={jump_state['dz']:.2f}, taken_off={int(jump_state['has_taken_off'])}",
                        end='\r'
                    )
                else:
                    print(
                        f"[{mode_label:7s}] Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}  "
                        f"(RB to jump)",
                        end='\r'
                    )
            elif visualize_moe_weights and counter % control_decimation == 0:
                pygame.event.pump()

            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau

            mujoco.mj_step(m, d)

            if save_video and counter % frame_skip == 0:
                try:
                    renderer.update_scene(d, camera=viewer.cam)
                    frame = renderer.render()
                    writer.append_data(frame)
                except Exception as e:
                    print(f"Error rendering frame: {e}")

            counter += 1
            if counter % control_decimation == 0:
                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                lin_vel = d.qvel[:3]
                ang_vel = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                lin_vel = lin_vel * lin_vel_scale
                ang_vel = ang_vel * ang_vel_scale

                command_obs_dim = len(cmd_scale)
                command_obs_start = 6
                dof_pos_start = command_obs_start + command_obs_dim
                dof_vel_start = dof_pos_start + num_actions
                action_start = dof_vel_start + num_actions

                obs[:3] = ang_vel
                obs[3:6] = gravity_orientation
                obs[command_obs_start:dof_pos_start] = cmd * cmd_scale
                obs[dof_pos_start:dof_vel_start] = qj[idx_mj2model]
                obs[dof_vel_start:action_start] = dqj[idx_mj2model]
                obs[action_start:action_start + num_actions] = action[idx_mj2model]
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                last_action = action
                result = policy(obs_tensor)

                if isinstance(result, tuple):
                    action, (weights, latent) = result
                    action = action.detach().numpy().squeeze()[idx_model2mj]
                    weights = weights.detach().numpy().squeeze()

                    if visualize_moe_weights and screen is not None:
                        draw_moe_weights(screen, weights, win_width, win_height)

                else:
                    action = result.detach().numpy().squeeze()[idx_model2mj]

                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            viewer.sync()

    if save_video:
        writer.close()

    pygame.quit()
    print(f"\nVideo saved successfully to {video_path}")
