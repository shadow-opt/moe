import sys
from pathlib import Path
PATH_PARENT = Path(__file__).parent
sys.path.append(str(PATH_PARENT))
from utils import MujocoRenderUtils
from joint_limit_monitor import JointLimitMonitor, prepare_velocity_limits

import os
import time
import threading
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import os
import imageio
from argparse import ArgumentParser
from matplotlib import pyplot as plt

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

def quat_rotate_inverse(q, v):
    q = np.array(q, np.float32)
    v = np.array(v, np.float32)
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def reorder_by_joint_names(values, source_joint_names, target_joint_names):
    source_indices = {joint_name: index for index, joint_name in enumerate(source_joint_names)}
    return np.asarray(values, dtype=np.float32)[[source_indices[joint_name] for joint_name in target_joint_names]]

def get_xbox_command(joystick, max_cmd):
    pygame.event.pump()
    dead_zone = 0.1
    lx = joystick.get_axis(0)
    ly = joystick.get_axis(1)
    rx = joystick.get_axis(3)
    if abs(lx) < dead_zone: lx = 0
    if abs(ly) < dead_zone: ly = 0
    if abs(rx) < dead_zone: rx = 0
    cmd_x = -ly * max_cmd[0]
    cmd_y = -lx * max_cmd[1]
    cmd_yaw = -rx * max_cmd[2]
    return np.array([cmd_x, cmd_y, cmd_yaw], dtype=np.float32)


def update_velocity_command_from_xbox(
    command_obs,
    joystick,
    max_cmd,
    button_state,
    jump_control=False,
):
    """Update [vx, vy, yaw, height, climb, wall_height] from an Xbox controller."""
    pygame.event.pump()
    dead_zone = 0.1
    if joystick is not None:
        lx = joystick.get_axis(0)
        ly = joystick.get_axis(1)
        rx = joystick.get_axis(3)
        if abs(lx) < dead_zone: lx = 0
        if abs(ly) < dead_zone: ly = 0
        if abs(rx) < dead_zone: rx = 0

        command_obs[0] = -ly * max_cmd[0]
        command_obs[1] = -lx * max_cmd[1]
        command_obs[2] = -rx * max_cmd[2]

        if jump_control:
            y_pressed = joystick.get_button(3)
            if y_pressed and not button_state.get("y", False):
                toggle_jump_command(command_obs)
                enabled = command_obs[3] < -0.5
                print(f"\nJump mode requested: {'ON' if enabled else 'OFF'}", flush=True)
            button_state["y"] = y_pressed
            if command_obs[3] < -0.5:
                command_obs[1:3] = 0.0
        else:
            # A 键 (button 0) 切换低高度模式
            a_pressed = joystick.get_button(0)
            if a_pressed and not button_state.get("a", False):
                command_obs[3] = 0.0 if command_obs[3] > 0.5 else 1.5
            button_state["a"] = a_pressed

            b_pressed = joystick.get_button(1)
            if b_pressed and not button_state.get("b", False):
                command_obs[4] = 0.0 if command_obs[4] > 0.5 else 1.0
            button_state["b"] = b_pressed
    else:
        command_obs[:3] = 0.0
    if not jump_control:
        apply_climb_command(command_obs)
    return command_obs


def apply_climb_command(command_obs):
    if len(command_obs) < 6:
        return command_obs
    if command_obs[4] > 0.5:
        command_obs[:3] = [0.30, 0.0, 0.0]
        command_obs[5] = 0.30
    else:
        command_obs[5] = 0.0
    return command_obs


def toggle_jump_command(command_obs):
    """Toggle the 6-D WIN command between walk and jump requests."""
    if len(command_obs) != 6:
        raise ValueError("Jump control requires a 6-D command")
    if command_obs[3] < -0.5:
        command_obs[3:] = 0.0
    else:
        command_obs[1:3] = 0.0
        command_obs[3] = -1.0
        command_obs[4:6] = 0.0
    return command_obs


class JumpCommandManager:
    def __init__(self, policy_dt, config):
        self.policy_dt = float(policy_dt)
        self.cycle_time = float(config.get("cycle_time", 1.5))
        self.pause_time = float(config.get("pause_time", 0.75))
        self.cycle_steps = max(int(round(self.cycle_time / self.policy_dt)), 1)
        self.pause_steps = max(int(round(self.pause_time / self.policy_dt)), 0)
        self.phase_steps = -1
        self.pause_remaining = 0
        self.jump_active = False
        self.motion_enabled = False

    def update(self, requested_command):
        request = np.asarray(requested_command[:4], dtype=np.float32).copy()
        jump_requested = request[3] < -0.5
        if not jump_requested:
            self.phase_steps = -1
            self.pause_remaining = 0
            self.jump_active = False
            self.motion_enabled = False
            command = np.zeros(6, dtype=np.float32)
            command[:4] = request
            return command

        if not self.jump_active:
            self.phase_steps = -1
            self.pause_remaining = 0
            self.jump_active = True
        request[1:3] = 0.0
        request[3] = -1.0
        speed = abs(float(request[0]))
        if speed >= 0.3:
            self.motion_enabled = True
        elif speed < 0.2:
            self.motion_enabled = False
        if self.motion_enabled:
            if self.pause_remaining > 0:
                self.pause_remaining -= 1
                request[0] = 0.0
                phase_sin = 0.0
                phase_cos = 0.0
            else:
                self.phase_steps += 1
                if self.phase_steps >= self.cycle_steps:
                    self.phase_steps = -1
                    self.pause_remaining = self.pause_steps
                    request[0] = 0.0
                    phase_sin = 0.0
                    phase_cos = 0.0
                else:
                    phase = self.phase_steps * self.policy_dt / self.cycle_time
                    phase_sin = np.sin(2.0 * np.pi * phase)
                    phase_cos = np.cos(2.0 * np.pi * phase)
        else:
            self.phase_steps = -1
            self.pause_remaining = 0
            phase_sin = 0.0
            phase_cos = 0.0
        return np.asarray([*request, phase_sin, phase_cos], dtype=np.float32)


class KeyboardCommandController:
    def __init__(self, max_cmd, jump_control=False):
        try:
            from pynput import keyboard as pynput_keyboard
        except ImportError as exc:
            raise ImportError(
                "Keyboard control requires pynput. Install it in the moe env with: pip install pynput"
            ) from exc

        self.max_cmd = np.asarray(max_cmd, dtype=np.float32)
        self.jump_control = bool(jump_control)
        self.keys = set()
        self.edge_keys = []
        self.lock = threading.Lock()
        self.keyboard = pynput_keyboard
        self.listener = pynput_keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def normalize_key(self, key):
        special_keys = {
            self.keyboard.Key.up: "up",
            self.keyboard.Key.down: "down",
            self.keyboard.Key.left: "left",
            self.keyboard.Key.right: "right",
            self.keyboard.Key.space: "space",
        }
        if key in special_keys:
            return special_keys[key]
        char = getattr(key, "char", None)
        if char is not None:
            return char.lower()
        return None

    def on_press(self, key):
        key_name = self.normalize_key(key)
        if key_name is None:
            return
        with self.lock:
            if key_name not in self.keys and key_name in ("h", "c", "r", "y"):
                self.edge_keys.append(key_name)
            self.keys.add(key_name)

    def on_release(self, key):
        key_name = self.normalize_key(key)
        if key_name is None:
            return
        with self.lock:
            self.keys.discard(key_name)

    def update(self, command_obs):
        with self.lock:
            keys = set(self.keys)
            edge_keys = self.edge_keys
            self.edge_keys = []

        if "r" in edge_keys:
            command_obs[:] = 0.0
        elif "y" in edge_keys and self.jump_control:
            toggle_jump_command(command_obs)
            enabled = command_obs[3] < -0.5
            print(f"\nJump mode requested: {'ON' if enabled else 'OFF'}", flush=True)
        elif "h" in edge_keys and not self.jump_control and len(command_obs) > 3:
            command_obs[3] = 0.0 if command_obs[3] > 0.5 else 1.5
        elif "c" in edge_keys and not self.jump_control and len(command_obs) > 4:
            command_obs[4] = 0.0 if command_obs[4] > 0.5 else 1.0

        if "space" in keys:
            command_obs[:] = 0.0
        else:
            vx_axis = float(("w" in keys or "up" in keys) - ("s" in keys or "down" in keys))
            vy_axis = float(("a" in keys or "left" in keys) - ("d" in keys or "right" in keys))
            wz_axis = float(("q" in keys) - ("e" in keys))
            if self.jump_control and command_obs[3] < -0.5:
                if vx_axis:
                    command_obs[0] = vx_axis * self.max_cmd[0]
                command_obs[1:3] = 0.0
            else:
                command_obs[0] = vx_axis * self.max_cmd[0]
                command_obs[1] = vy_axis * self.max_cmd[1]
                command_obs[2] = wz_axis * self.max_cmd[2]

        if not self.jump_control:
            apply_climb_command(command_obs)
        return command_obs

    def stop(self):
        self.listener.stop()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="win.yaml", help="Config file name under deploy/deploy_mujoco/configs")
    parser.add_argument(
        "--control",
        choices=["xbox", "keyboard", "config"],
        default="xbox",
        help="Command input source: xbox joystick, keyboard hotkeys in the MuJoCo viewer, or static config cmd_init.",
    )
    parser.add_argument("--save-video", action="store_true", help="Whether to save video of the simulation.")
    parser.add_argument("--visualize-moe-weights", action="store_true", help="Whether to visualize mixture of experts weights.")
    parser.add_argument("--save-moe-latent", action="store_true", help="Whether to save mixture of experts latent vectors.")
    args = parser.parse_args()
    save_video = args.save_video
    visualize_moe_weights = args.visualize_moe_weights
    save_moe_latent = args.save_moe_latent
    config_file = args.config
    control_mode = args.control
    # config_file = "win_go2.yaml"
    use_joystick = False
    joystick = None
    button_state = {}
    keyboard_controller = None
    if control_mode == "xbox":
        try:
            import pygame
        except ImportError as exc:
            raise ImportError(
                "Xbox control requires pygame. Use --control keyboard or install pygame."
            ) from exc
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            use_joystick = True
            print(f"Detected Joystick: {joystick.get_name()}")
        else:
            print("No Joystick detected. Using default commands from config.")
    elif control_mode == "keyboard":
        print("Keyboard control enabled: W/S=vx, A/D=vy, Q/E=wz, Space=stop, H=height, C=climb, R=reset.")
    else:
        print("Using static commands from config cmd_init.")

    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}"
    print(f"Loading config: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        configured_velocity_limits = config.get("joint_velocity_limits")

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        max_cmd = np.array(config["max_cmd"], dtype=np.float32)
        clip_observations = float(config.get("clip_observations", 100.0))
        clip_actions = float(config.get("clip_actions", 100.0))

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        viewer_camera_mode = config.get("viewer_camera_mode", "fixed")
        viewer_camera_name = config.get("viewer_camera_name", "chase_cam")

        cmd = np.array(config["cmd_init"], dtype=np.float32)
        jump_control_config = config.get("jump_control", {})
        jump_control_enabled = bool(jump_control_config.get("enabled", False))
        init_base_pos = np.array(config.get("init_base_pos", [0.0, 0.0, 0.42]), dtype=np.float32)
        init_base_quat = np.array(config.get("init_base_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)

        config_mujoco_joint_names = None
        model_joint_names = None
        if 'mujoco_joint_names' in config and 'model_joint_names' in config:
            config_mujoco_joint_names = config["mujoco_joint_names"]
            model_joint_names = config["model_joint_names"]

    if len(default_angles) != num_actions:
        raise ValueError(f"default_angles len {len(default_angles)} != num_actions {num_actions}")
    if len(kps) != num_actions or len(kds) != num_actions:
        raise ValueError(
            f"kps/kds len mismatch: len(kps)={len(kps)}, len(kds)={len(kds)}, num_actions={num_actions}"
        )
    if len(cmd) != len(cmd_scale):
        raise ValueError(f"cmd_init len {len(cmd)} != cmd_scale len {len(cmd_scale)}")
    if len(max_cmd) != 3:
        raise ValueError(f"max_cmd len {len(max_cmd)} != 3")
    if len(init_base_pos) != 3:
        raise ValueError(f"init_base_pos len {len(init_base_pos)} != 3")
    if len(init_base_quat) != 4:
        raise ValueError(f"init_base_quat len {len(init_base_quat)} != 4")
    expected_num_obs = 3 + 3 + len(cmd) + 3 * num_actions
    if num_obs != expected_num_obs:
        raise ValueError(f"num_obs mismatch: config={num_obs}, expected={expected_num_obs}")
    if jump_control_enabled and len(cmd) != 6:
        raise ValueError("jump_control requires 6-D cmd_init and cmd_scale")
    if control_mode == "keyboard" and jump_control_enabled:
        print("Jump control enabled: Y toggles jump mode; Space exits jump and stops.")

    video_save_dir = str(PATH_PARENT / "videos")
    os.makedirs(video_save_dir, exist_ok=True)

    model_name = os.path.basename(policy_path).split('.')[0]
    cmd_str = f"cmd_{cmd[0]}_{cmd[1]}_{cmd[2]}"

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    last_action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    qpos_joint_names = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(m.njnt)
        if m.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]
    actuator_joint_names = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[actuator_id, 0])
        for actuator_id in range(m.nu)
    ]
    if model_joint_names is None:
        model_joint_names = qpos_joint_names
    if config_mujoco_joint_names is None:
        config_mujoco_joint_names = qpos_joint_names
    if len(qpos_joint_names) != num_actions or len(actuator_joint_names) != num_actions:
        raise ValueError(
            f"XML joint/actuator count mismatch: len(qpos_joint_names)={len(qpos_joint_names)}, "
            f"len(actuator_joint_names)={len(actuator_joint_names)}, num_actions={num_actions}"
        )
    if set(model_joint_names) != set(qpos_joint_names):
        raise ValueError(
            f"model_joint_names mismatch with XML qpos joints: model_joint_names={model_joint_names}, "
            f"qpos_joint_names={qpos_joint_names}"
        )
    if set(config_mujoco_joint_names) != set(qpos_joint_names):
        raise ValueError(
            f"config mujoco_joint_names mismatch with XML qpos joints: "
            f"config_mujoco_joint_names={config_mujoco_joint_names}, qpos_joint_names={qpos_joint_names}"
        )
    if set(actuator_joint_names) != set(qpos_joint_names):
        raise ValueError(
            f"XML actuator joints mismatch with XML qpos joints: actuator_joint_names={actuator_joint_names}, "
            f"qpos_joint_names={qpos_joint_names}"
        )

    idx_model2qpos = [model_joint_names.index(joint) for joint in qpos_joint_names]
    idx_qpos2model = [qpos_joint_names.index(joint) for joint in model_joint_names]
    idx_ctrl_from_qpos = [qpos_joint_names.index(joint) for joint in actuator_joint_names]

    joint_velocity_limits, velocity_limit_warning = prepare_velocity_limits(
        configured_velocity_limits, config_mujoco_joint_names, qpos_joint_names
    )

    default_angles = reorder_by_joint_names(default_angles, config_mujoco_joint_names, qpos_joint_names)
    kps = reorder_by_joint_names(kps, config_mujoco_joint_names, qpos_joint_names)
    kds = reorder_by_joint_names(kds, config_mujoco_joint_names, qpos_joint_names)
    target_dof_pos = default_angles.copy()

    qpos_joint_ids = [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        for joint_name in qpos_joint_names
    ]
    joint_qpos_indices = np.array([m.jnt_qposadr[joint_id] for joint_id in qpos_joint_ids])
    joint_dof_indices = np.array([m.jnt_dofadr[joint_id] for joint_id in qpos_joint_ids])
    position_ranges = np.full((num_actions, 2), np.nan, dtype=np.float64)
    torque_ranges = np.full((num_actions, 2), np.nan, dtype=np.float64)
    for index, joint_id in enumerate(qpos_joint_ids):
        if m.jnt_limited[joint_id]:
            position_ranges[index] = m.jnt_range[joint_id]
        if m.jnt_actfrclimited[joint_id]:
            torque_ranges[index] = m.jnt_actfrcrange[joint_id]
    limit_monitor = JointLimitMonitor(
        qpos_joint_names,
        position_ranges=position_ranges,
        velocity_limits=joint_velocity_limits,
        torque_ranges=torque_ranges,
    )
    if velocity_limit_warning is not None:
        print(velocity_limit_warning)
    for warning in limit_monitor.configuration_warnings():
        if velocity_limit_warning is None or "metric=velocity" not in warning:
            print(warning)
    print(limit_monitor.startup_summary())

    jump_manager = None
    operator_cmd = cmd.copy()
    if jump_control_enabled:
        jump_manager = JumpCommandManager(
            simulation_dt * control_decimation,
            jump_control_config,
        )

    init_base_quat_norm = np.linalg.norm(init_base_quat)
    if init_base_quat_norm <= 0.0:
        raise ValueError(f"init_base_quat must be non-zero, got {init_base_quat}")
    d.qpos[:3] = init_base_pos
    d.qpos[3:7] = init_base_quat / init_base_quat_norm
    d.qpos[7:] = default_angles
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    print(f"XML qpos joint order: {qpos_joint_names}")
    print(f"XML actuator joint order: {actuator_joint_names}")
    print(f"Model joint order: {model_joint_names}")
    print(f"Model->qpos map: {idx_model2qpos}")
    print(f"Qpos->model map: {idx_qpos2model}")
    print(f"Qpos->ctrl map: {idx_ctrl_from_qpos}")
    print(f"default_angles in qpos order: {default_angles.tolist()}")

    renderer = mujoco.Renderer(m, height=360, width=640)
    
    # load policy
    policy = torch.jit.load(policy_path)
    if hasattr(policy, "reset"):
        policy.reset()

    video_fps = 50
    if save_video:
        video_filename = f"{model_name}_{cmd_str}.mp4"
        video_path = os.path.join(video_save_dir, video_filename)
        print(f"Video recording will be saved to: {video_path}")
        sim_fps = 1.0 / m.opt.timestep
        frame_skip = int(sim_fps / video_fps)
        if frame_skip < 1:
            frame_skip = 1
        writer = imageio.get_writer(video_path, fps=video_fps)
        print(f"Sim FPS: {sim_fps:.2f}, Video FPS: {video_fps}, Frame Skip: {frame_skip}, Save at: {video_path}")
    mujoco_render_utils = MujocoRenderUtils(video_fps, m.opt.timestep)

    if visualize_moe_weights:
        plt.ion()
        fig, ax = plt.subplots(figsize=(5,3))
        ax.set_title(f"Command: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}")
        bars = None
    
    if save_moe_latent:
        latent_save_dir = str(PATH_PARENT / "data_latents")
        os.makedirs(latent_save_dir, exist_ok=True)
        latent_filename = f"{model_name}_{cmd_str}_latents.npy"
        latent_path = os.path.join(latent_save_dir, latent_filename)
        all_latents = []

    if control_mode == "keyboard":
        keyboard_controller = KeyboardCommandController(
            max_cmd,
            jump_control=jump_control_enabled,
        )

    with mujoco.viewer.launch_passive(m, d) as viewer:
        if viewer_camera_mode.lower() == "fixed":
            cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, viewer_camera_name)
            if cam_id < 0:
                raise ValueError(
                    f"camera '{viewer_camera_name}' not found in XML model: {xml_path}. "
                    "Please add it under robot base body (e.g. chase_cam)."
                )
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cam_id
            print(f"Viewer camera fixed to XML camera '{viewer_camera_name}' (id={cam_id}).")
        else:
            raise ValueError(
                f"Unsupported viewer_camera_mode: {viewer_camera_mode}. "
                "Supported values: fixed"
            )

        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            vel = d.qvel[:3]
            ang_vel = d.qvel[3:6]
            local_vel = quat_rotate_inverse(d.qpos[3:7], vel)
            local_ang_vel = quat_rotate_inverse(d.qpos[3:7], ang_vel)
            show_str = f"Speed: Vx={local_vel[0]:.2f}, Vy={local_vel[1]:.2f}, Wz={local_ang_vel[2]:.2f}, "
            step_start = time.time()

            if control_mode == "xbox" and use_joystick and counter % control_decimation == 0:
                # cmd = get_xbox_command(joystick, max_cmd)
                operator_cmd = update_velocity_command_from_xbox(
                    operator_cmd,
                    joystick,
                    max_cmd,
                    button_state,
                    jump_control=jump_control_enabled,
                )
                cmd = operator_cmd
                show_str += f"Cmd: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}"
                print(show_str, end='\r')
            elif control_mode == "keyboard" and counter % control_decimation == 0:
                operator_cmd = keyboard_controller.update(operator_cmd)
                cmd = operator_cmd
                show_str += f"Cmd: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}"
                print(show_str, end='\r')

            if jump_manager is not None and counter % control_decimation == 0:
                cmd = jump_manager.update(operator_cmd)

            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau[idx_ctrl_from_qpos]
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            mujoco_render_utils.update(cmd, d)
            limit_event = limit_monitor.update(
                d.time,
                d.qpos[joint_qpos_indices],
                d.qvel[joint_dof_indices],
                tau,
                d.qfrc_actuator[joint_dof_indices],
            )
            if limit_event is not None:
                print(f"\n{limit_event}", flush=True)

            if save_video and counter % frame_skip == 0:
                try:
                    renderer.update_scene(d, camera=viewer.cam)
                    mujoco_render_utils.update_external_rendering(renderer, ctype='renderer')
                    frame = renderer.render()
                    writer.append_data(frame)
                except Exception as e:
                    print(f"Error rendering frame: {e}")

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

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

                obs[:3] = ang_vel
                obs[3:6] = gravity_orientation
                obs[6:12] = cmd * cmd_scale
                obs[12 : 12 + num_actions] = qj[idx_qpos2model]
                obs[12 + num_actions : 12 + 2 * num_actions] = dqj[idx_qpos2model]
                obs[12 + 2 * num_actions : 12 + 3 * num_actions] = action
                np.clip(obs, -clip_observations, clip_observations, out=obs)
                
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                last_action = action
                result = policy(obs_tensor)
                if isinstance(result, tuple):
                    action, (weights, latent) = result  # moe
                    action = action.detach().cpu().numpy().squeeze()
                    weights = weights.detach().cpu().numpy().squeeze()
                    latent = latent.detach().cpu().numpy().squeeze()
                    if visualize_moe_weights:
                        if bars is None:
                            x = np.arange(len(weights))
                            bars = ax.bar(x, weights)
                            ax.set_ylim(0, 1)
                        else:
                            for bar, w in zip(bars, weights):
                                bar.set_height(w)
                        
                        plt.draw()
                        plt.pause(0.001) # 这会造成大约 1ms 的延迟
                    if save_moe_latent:
                        all_latents.append(latent)
                else:
                    action = result.detach().cpu().numpy().squeeze()
                action = np.clip(action, -clip_actions, clip_actions)
                # transform action to target_dof_pos
                target_dof_pos = action[idx_model2qpos] * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            mujoco_render_utils.update_external_rendering(viewer, ctype='viewer')
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            # time_until_next_step = m.opt.timestep - (time.time() - step_start) - 0.1
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)
            
            # 计算这一步剩余的时间
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # writer.close()
    if save_video:
        print(f"Video saved successfully to {video_path}")
        writer.close()
    if save_moe_latent and len(all_latents) > 0:
        all_latents = np.array(all_latents)
        np.save(latent_path, all_latents)
        print(f"Latent vectors saved successfully to {latent_path}")
    if keyboard_controller is not None:
        keyboard_controller.stop()
    if control_mode == "xbox":
        pygame.quit()
