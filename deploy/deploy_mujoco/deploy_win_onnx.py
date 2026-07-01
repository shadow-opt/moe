import sys
from pathlib import Path

PATH_PARENT = Path(__file__).parent
sys.path.append(str(PATH_PARENT))
from utils import MujocoRenderUtils

import os
import time
import threading
import mujoco
import mujoco.viewer
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import yaml
import imageio
from argparse import ArgumentParser
import pygame
import onnxruntime as ort
from matplotlib import pyplot as plt


OBS_FRAME_DIM = 48
OBS_HISTORY_LEN = 5
OBS_INPUT_DIM = OBS_FRAME_DIM * OBS_HISTORY_LEN
PACK_SLICES = (
    (0, 3),
    (3, 6),
    (6, 12),
    (12, 24),
    (24, 36),
    (36, 48),
)


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def quat_rotate(q, v):
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a + b + c


def quat_rotate_inverse(q, v):
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def quat_from_rpy(roll, pitch, yaw):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def reorder_by_joint_names(values, source_joint_names, target_joint_names):
    source_indices = {
        joint_name: index for index, joint_name in enumerate(source_joint_names)
    }
    return np.asarray(values, dtype=np.float32)[
        [source_indices[joint_name] for joint_name in target_joint_names]
    ]


def update_velocity_command_from_xbox(command_obs, joystick, max_cmd, button_state):
    pygame.event.pump()
    dead_zone = 0.1
    if joystick is not None:
        lx = joystick.get_axis(0)
        ly = joystick.get_axis(1)
        rx = joystick.get_axis(3)
        if abs(lx) < dead_zone:
            lx = 0
        if abs(ly) < dead_zone:
            ly = 0
        if abs(rx) < dead_zone:
            rx = 0

        command_obs[0] = -ly * max_cmd[0]
        command_obs[1] = -lx * max_cmd[1]
        command_obs[2] = -rx * max_cmd[2]

        a_pressed = joystick.get_button(0)
        if a_pressed and not button_state.get("a", False):
            command_obs[3] = 0.0 if command_obs[3] > 0.5 else 1.0
        button_state["a"] = a_pressed
    else:
        command_obs[:3] = 0.0
    command_obs[4:] = 0.0
    return command_obs


class KeyboardCommandController:
    def __init__(self, max_cmd):
        try:
            from pynput import keyboard as pynput_keyboard
        except ImportError as exc:
            raise ImportError(
                "Keyboard control requires pynput. Install it in the moe env with: "
                "pip install pynput"
            ) from exc

        self.max_cmd = np.asarray(max_cmd, dtype=np.float32)
        self.keys = set()
        self.edge_keys = []
        self.lock = threading.Lock()
        self.keyboard = pynput_keyboard
        self.listener = pynput_keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
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
            if key_name not in self.keys and key_name in ("h", "r"):
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
        elif "h" in edge_keys and len(command_obs) > 3:
            command_obs[3] = 0.0 if command_obs[3] > 0.5 else 1.0

        if "space" in keys:
            command_obs[:3] = 0.0
        else:
            vx_axis = float(("w" in keys or "up" in keys) - ("s" in keys or "down" in keys))
            vy_axis = float(("a" in keys or "left" in keys) - ("d" in keys or "right" in keys))
            wz_axis = float(("q" in keys) - ("e" in keys))
            command_obs[0] = vx_axis * self.max_cmd[0]
            command_obs[1] = vy_axis * self.max_cmd[1]
            command_obs[2] = wz_axis * self.max_cmd[2]

        if len(command_obs) > 4:
            command_obs[4:] = 0.0
        return command_obs

    def stop(self):
        self.listener.stop()


class OnnxPolicy:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self._validate_model()

    def _shape_matches(self, shape, expected):
        return len(shape) == len(expected) and all(
            actual in (wanted, "None", None) for actual, wanted in zip(shape, expected)
        )

    def _validate_model(self):
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(f"ONNX model must have 1 input, got {len(inputs)}")

        input_info = inputs[0]
        if input_info.type != "tensor(float)":
            raise ValueError(f"ONNX input must be tensor(float), got {input_info.type}")
        if not self._shape_matches(input_info.shape, [1, OBS_INPUT_DIM]):
            raise ValueError(
                f"ONNX input must be [1, {OBS_INPUT_DIM}], got {input_info.shape}"
            )

        if len(outputs) < 1:
            raise ValueError("ONNX model must expose at least actions output")
        if not self._shape_matches(outputs[0].shape, [1, 12]):
            raise ValueError(f"ONNX actions output must be [1, 12], got {outputs[0].shape}")
        if len(outputs) > 1 and not self._shape_matches(outputs[1].shape, [1, 8]):
            raise ValueError(f"ONNX moe_weights output must be [1, 8], got {outputs[1].shape}")
        if len(outputs) > 2 and not self._shape_matches(outputs[2].shape, [1, 32]):
            raise ValueError(f"ONNX latent output must be [1, 32], got {outputs[2].shape}")

        print(f"ONNX input: {input_info.name} {input_info.shape} {input_info.type}")
        print(
            "ONNX outputs: "
            + ", ".join(f"{out.name} {out.shape} {out.type}" for out in outputs)
        )
        return input_info.name

    def infer(self, packed_obs):
        outputs = self.session.run(None, {self.input_name: packed_obs})
        action = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        weights = None
        latent = None
        if len(outputs) > 1:
            weights = np.asarray(outputs[1], dtype=np.float32).reshape(-1)
        if len(outputs) > 2:
            latent = np.asarray(outputs[2], dtype=np.float32).reshape(-1)
        return action, weights, latent


def push_history(history, frame):
    history[:-1] = history[1:]
    history[-1] = frame


def pack_term_major(history):
    return np.concatenate(
        [history[:, begin:end].reshape(-1) for begin, end in PACK_SLICES]
    ).astype(np.float32).reshape(1, OBS_INPUT_DIM)


def load_config(config_file, policy_path_override):
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}"
    print(f"Loading config: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    policy_path = policy_path_override or config["policy_path"]
    config["policy_path"] = policy_path.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    config["xml_path"] = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    return config


def validate_config(config):
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    if num_actions != 12:
        raise ValueError(f"num_actions must be 12 for RK WIN ONNX model, got {num_actions}")
    if num_obs != OBS_FRAME_DIM:
        raise ValueError(f"num_obs must be {OBS_FRAME_DIM}, got {num_obs}")
    if len(config["cmd_init"]) != len(config["cmd_scale"]):
        raise ValueError(
            f"cmd_init len {len(config['cmd_init'])} != cmd_scale len {len(config['cmd_scale'])}"
        )
    if len(config["cmd_init"]) != 6:
        raise ValueError(f"cmd_init must be 6-dim, got {len(config['cmd_init'])}")
    if len(config["max_cmd"]) != 3:
        raise ValueError(f"max_cmd must be 3-dim, got {len(config['max_cmd'])}")
    for key in ("kps", "kds", "default_angles"):
        if len(config[key]) != num_actions:
            raise ValueError(f"{key} len {len(config[key])} != num_actions {num_actions}")


def create_observation(
    d,
    default_angles,
    cmd,
    cmd_scale,
    action_model_order,
    idx_qpos2model,
    num_actions,
    ang_vel_scale,
    dof_pos_scale,
    dof_vel_scale,
    clip_observations,
    imu_quat_convention,
):
    obs = np.zeros(OBS_FRAME_DIM, dtype=np.float32)
    qj = d.qpos[7:]
    dqj = d.qvel[6:]
    quat = d.qpos[3:7]
    ang_vel = d.qvel[3:6]

    qj = (qj - default_angles) * dof_pos_scale
    dqj = dqj * dof_vel_scale
    if imu_quat_convention == "body-to-world":
        gravity_orientation = get_gravity_orientation(quat)
    elif imu_quat_convention == "lpms-world-to-body-correct":
        lpms_quat = quat.copy()
        lpms_quat[1:] *= -1.0
        gravity_orientation = quat_rotate(lpms_quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    elif imu_quat_convention == "lpms-world-to-body-misused":
        lpms_quat = quat.copy()
        lpms_quat[1:] *= -1.0
        gravity_orientation = get_gravity_orientation(lpms_quat)
    else:
        raise ValueError(f"Unsupported imu_quat_convention: {imu_quat_convention}")
    ang_vel = ang_vel * ang_vel_scale

    obs[:3] = ang_vel
    obs[3:6] = gravity_orientation
    obs[6:12] = cmd * cmd_scale
    obs[12 : 12 + num_actions] = qj[idx_qpos2model]
    obs[12 + num_actions : 12 + 2 * num_actions] = dqj[idx_qpos2model]
    obs[12 + 2 * num_actions : 12 + 3 * num_actions] = action_model_order
    np.clip(obs, -clip_observations, clip_observations, out=obs)
    return obs


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="win_rk_onnx.yaml",
        help="Config file name under deploy/deploy_mujoco/configs",
    )
    parser.add_argument("--policy-path", type=str, default=None, help="Override ONNX model path")
    parser.add_argument(
        "--control",
        choices=["xbox", "keyboard", "config"],
        default="keyboard",
        help="Command input source.",
    )
    parser.add_argument("--duration", type=float, default=None, help="Override simulation duration")
    parser.add_argument("--headless", action="store_true", help="Run without launching Mujoco viewer")
    parser.add_argument("--save-video", action="store_true", help="Whether to save video of the simulation.")
    parser.add_argument("--visualize-moe-weights", action="store_true", help="Visualize mixture of experts weights.")
    parser.add_argument("--save-moe-latent", action="store_true", help="Save mixture of experts latent vectors.")
    parser.add_argument(
        "--init-base-rpy-deg",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        default=None,
        help="Override initial base orientation in degrees, useful for IMU convention A/B tests.",
    )
    parser.add_argument(
        "--imu-quat-convention",
        choices=["body-to-world", "lpms-world-to-body-correct", "lpms-world-to-body-misused"],
        default="body-to-world",
        help=(
            "Observation gravity convention. body-to-world matches Mujoco/Isaac training; "
            "lpms-world-to-body-correct inverts the Mujoco quat then consumes it as LPMS G->body; "
            "lpms-world-to-body-misused simulates feeding an LPMS G->body quaternion into the old RK inverse formula."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config, args.policy_path)
    validate_config(config)

    policy_path = config["policy_path"]
    xml_path = config["xml_path"]
    simulation_duration = args.duration if args.duration is not None else config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    max_cmd = np.array(config["max_cmd"], dtype=np.float32)
    clip_observations = float(config.get("clip_observations", 100.0))
    clip_actions = float(config.get("clip_actions", 100.0))
    num_actions = config["num_actions"]
    cmd = np.array(config["cmd_init"], dtype=np.float32)
    init_base_pos = np.array(config.get("init_base_pos", [0.0, 0.0, 0.42]), dtype=np.float32)
    init_base_quat = np.array(config.get("init_base_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
    if args.init_base_rpy_deg is not None:
        init_base_quat = quat_from_rpy(*np.deg2rad(np.array(args.init_base_rpy_deg, dtype=np.float32)))
    viewer_camera_mode = config.get("viewer_camera_mode", "fixed")
    viewer_camera_name = config.get("viewer_camera_name", "chase_cam")
    config_mujoco_joint_names = config.get("mujoco_joint_names")
    model_joint_names = config.get("model_joint_names")

    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"ONNX policy not found: {policy_path}")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Mujoco XML not found: {xml_path}")

    pygame.init()
    pygame.joystick.init()
    joystick = None
    button_state = {}
    keyboard_controller = None
    if args.control == "xbox":
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            print(f"Detected Joystick: {joystick.get_name()}")
        else:
            print("No Joystick detected. Using default commands from config.")
    elif args.control == "keyboard" and not args.headless:
        print("Keyboard: W/S or Up/Down=vx, A/D or Left/Right=vy, Q/E=wz, Space=stop, H=height, R=reset.")
        keyboard_controller = KeyboardCommandController(max_cmd)
    elif args.control == "keyboard" and args.headless:
        print("Headless mode ignores keyboard events; using config cmd_init.")
    else:
        print("Using static commands from config cmd_init.")

    model_name = os.path.basename(policy_path).split(".")[0]
    cmd_str = f"cmd_{cmd[0]}_{cmd[1]}_{cmd[2]}"
    video_save_dir = str(PATH_PARENT / "videos")
    os.makedirs(video_save_dir, exist_ok=True)

    action_model_order = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    history = np.zeros((OBS_HISTORY_LEN, OBS_FRAME_DIM), dtype=np.float32)

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
            f"XML joint/actuator count mismatch: qpos={len(qpos_joint_names)}, "
            f"actuators={len(actuator_joint_names)}, num_actions={num_actions}"
        )
    if set(model_joint_names) != set(qpos_joint_names):
        raise ValueError(
            f"model_joint_names mismatch with XML qpos joints: {model_joint_names} vs {qpos_joint_names}"
        )
    if set(config_mujoco_joint_names) != set(qpos_joint_names):
        raise ValueError(
            f"mujoco_joint_names mismatch with XML qpos joints: {config_mujoco_joint_names} vs {qpos_joint_names}"
        )
    if set(actuator_joint_names) != set(qpos_joint_names):
        raise ValueError(
            f"XML actuator joints mismatch with qpos joints: {actuator_joint_names} vs {qpos_joint_names}"
        )

    idx_model2qpos = [model_joint_names.index(joint) for joint in qpos_joint_names]
    idx_qpos2model = [qpos_joint_names.index(joint) for joint in model_joint_names]
    idx_ctrl_from_qpos = [qpos_joint_names.index(joint) for joint in actuator_joint_names]

    default_angles = reorder_by_joint_names(default_angles, config_mujoco_joint_names, qpos_joint_names)
    kps = reorder_by_joint_names(kps, config_mujoco_joint_names, qpos_joint_names)
    kds = reorder_by_joint_names(kds, config_mujoco_joint_names, qpos_joint_names)
    target_dof_pos = default_angles.copy()

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
    print(f"IMU quat convention for obs gravity: {args.imu_quat_convention}")

    policy = OnnxPolicy(policy_path)
    renderer = None
    writer = None
    frame_skip = 1
    video_fps = 50
    if args.save_video:
        renderer = mujoco.Renderer(m, height=360, width=640)
        video_filename = f"{model_name}_{cmd_str}.mp4"
        video_path = os.path.join(video_save_dir, video_filename)
        sim_fps = 1.0 / m.opt.timestep
        frame_skip = max(1, int(sim_fps / video_fps))
        writer = imageio.get_writer(video_path, fps=video_fps)
        print(f"Video recording will be saved to: {video_path}")

    mujoco_render_utils = MujocoRenderUtils(video_fps, m.opt.timestep)

    bars = None
    if args.visualize_moe_weights:
        plt.ion()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_title(f"Command: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}")

    all_latents = []
    latent_path = None
    if args.save_moe_latent:
        latent_save_dir = str(PATH_PARENT / "data_latents")
        os.makedirs(latent_save_dir, exist_ok=True)
        latent_path = os.path.join(latent_save_dir, f"{model_name}_{cmd_str}_latents.npy")

    counter = 0
    start = time.time()

    def step_once(viewer=None):
        nonlocal counter, cmd, target_dof_pos, action_model_order, bars

        vel = d.qvel[:3]
        ang_vel = d.qvel[3:6]
        local_vel = quat_rotate_inverse(d.qpos[3:7], vel)
        local_ang_vel = quat_rotate_inverse(d.qpos[3:7], ang_vel)
        show_str = (
            f"Speed: Vx={local_vel[0]:.2f}, Vy={local_vel[1]:.2f}, "
            f"Wz={local_ang_vel[2]:.2f}, "
        )
        step_start = time.time()

        if counter % control_decimation == 0:
            if args.control == "xbox" and joystick is not None:
                cmd = update_velocity_command_from_xbox(cmd, joystick, max_cmd, button_state)
            elif args.control == "keyboard" and keyboard_controller is not None:
                cmd = keyboard_controller.update(cmd)
            show_str += f"Cmd: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}"
            print(show_str, end="\r")

        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau[idx_ctrl_from_qpos]
        mujoco.mj_step(m, d)
        mujoco_render_utils.update(cmd, d)

        if args.save_video and counter % frame_skip == 0:
            try:
                if viewer is not None:
                    renderer.update_scene(d, camera=viewer.cam)
                else:
                    renderer.update_scene(d)
                mujoco_render_utils.update_external_rendering(renderer, ctype="renderer")
                writer.append_data(renderer.render())
            except Exception as exc:
                print(f"Error rendering frame: {exc}")

        counter += 1
        if counter % control_decimation == 0:
            obs = create_observation(
                d,
                default_angles,
                cmd,
                cmd_scale,
                action_model_order,
                idx_qpos2model,
                num_actions,
                ang_vel_scale,
                dof_pos_scale,
                dof_vel_scale,
                clip_observations,
                args.imu_quat_convention,
            )
            push_history(history, obs)
            packed_obs = pack_term_major(history)
            action, weights, latent = policy.infer(packed_obs)
            action_model_order = np.clip(action, -clip_actions, clip_actions)
            target_dof_pos = action_model_order[idx_model2qpos] * action_scale + default_angles

            if args.visualize_moe_weights and weights is not None:
                if bars is None:
                    x = np.arange(len(weights))
                    bars = ax.bar(x, weights)
                    ax.set_ylim(0, 1)
                else:
                    for bar, weight in zip(bars, weights):
                        bar.set_height(weight)
                plt.draw()
                plt.pause(0.001)
            if args.save_moe_latent and latent is not None:
                all_latents.append(latent)

        if viewer is not None:
            mujoco_render_utils.update_external_rendering(viewer, ctype="viewer")
            viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    try:
        if args.headless:
            while time.time() - start < simulation_duration:
                step_once()
        else:
            with mujoco.viewer.launch_passive(m, d) as viewer:
                if viewer_camera_mode.lower() == "fixed":
                    cam_id = mujoco.mj_name2id(
                        m, mujoco.mjtObj.mjOBJ_CAMERA, viewer_camera_name
                    )
                    if cam_id < 0:
                        raise ValueError(
                            f"camera '{viewer_camera_name}' not found in XML model: {xml_path}"
                        )
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = cam_id
                    print(f"Viewer camera fixed to XML camera '{viewer_camera_name}' (id={cam_id}).")
                else:
                    raise ValueError(
                        f"Unsupported viewer_camera_mode: {viewer_camera_mode}. Supported values: fixed"
                    )

                while viewer.is_running() and time.time() - start < simulation_duration:
                    step_once(viewer)
    finally:
        if keyboard_controller is not None:
            keyboard_controller.stop()
        if writer is not None:
            writer.close()
            print(f"\nVideo saved successfully to {video_path}")
        if args.save_moe_latent and len(all_latents) > 0:
            np.save(latent_path, np.asarray(all_latents, dtype=np.float32))
            print(f"\nLatent vectors saved successfully to {latent_path}")


if __name__ == "__main__":
    main()
