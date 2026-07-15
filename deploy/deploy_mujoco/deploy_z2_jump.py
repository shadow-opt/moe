import argparse
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "z2_jump.yaml"
PROFILE_NAMES = ("jump", "spring_jump")


def resolve_path(value):
    return Path(str(value).replace("{LEGGED_GYM_ROOT_DIR}", str(ROOT_DIR))).expanduser()


def load_config(path=DEFAULT_CONFIG):
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / "configs" / config_path
    with config_path.open("r") as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def validate_config(config):
    required_sections = {"xml_path", "simulation", "policy_interface", "model_joint_names", "controls", "profiles"}
    missing = sorted(required_sections - set(config))
    if missing:
        raise ValueError("Missing config fields: {}".format(missing))

    interface = config["policy_interface"]
    num_actions = int(interface["num_actions"])
    single_obs = int(interface["num_single_obs"])
    frame_stack = int(interface["frame_stack"])
    if (num_actions, single_obs, frame_stack) != (12, 47, 10):
        raise ValueError(
            "Expected jump policy interface actions=12, single_obs=47, frame_stack=10; got {}, {}, {}".format(
                num_actions, single_obs, frame_stack
            )
        )

    joint_names = list(config["model_joint_names"])
    if len(joint_names) != num_actions or len(set(joint_names)) != num_actions:
        raise ValueError("model_joint_names must contain 12 unique names")

    if set(config["profiles"]) != set(PROFILE_NAMES):
        raise ValueError("profiles must contain exactly {}".format(PROFILE_NAMES))
    for name in PROFILE_NAMES:
        profile = config["profiles"][name]
        if len(profile["default_angles"]) != num_actions:
            raise ValueError("{}.default_angles must contain {} values".format(name, num_actions))
        if len(profile["init_base_pos"]) != 3:
            raise ValueError("{}.init_base_pos must contain three values".format(name))
        if float(profile["torque_limit"]) <= 0.0:
            raise ValueError("{}.torque_limit must be positive".format(name))

    simulation = config["simulation"]
    if not math.isclose(
        float(simulation["dt"]) * int(simulation["control_decimation"]),
        0.02,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("simulation dt * control_decimation must equal the training policy period 0.02 s")

    spring = config["profiles"]["spring_jump"]
    if int(spring["preparation_steps"]) != 50 or int(spring["episode_steps"]) != 250:
        raise ValueError("spring_jump must use 50 preparation steps and 250 episode steps")


class JointOrderMap:
    def __init__(self, model_names, qpos_names, actuator_names):
        self.model_names = list(model_names)
        self.qpos_names = list(qpos_names)
        self.actuator_names = list(actuator_names)
        expected = set(self.model_names)
        if len(expected) != len(self.model_names):
            raise ValueError("Model joint names must be unique")
        for label, names in (("qpos", self.qpos_names), ("actuator", self.actuator_names)):
            if len(names) != len(set(names)) or set(names) != expected:
                raise ValueError("{} joint names do not match policy joints: {}".format(label, names))

        self._qpos_for_model = np.asarray(
            [self.qpos_names.index(name) for name in self.model_names], dtype=np.int64
        )
        self._model_for_qpos = np.asarray(
            [self.model_names.index(name) for name in self.qpos_names], dtype=np.int64
        )
        self._qpos_for_actuator = np.asarray(
            [self.qpos_names.index(name) for name in self.actuator_names], dtype=np.int64
        )

    def qpos_to_model(self, values):
        return np.asarray(values)[self._qpos_for_model]

    def model_to_qpos(self, values):
        return np.asarray(values)[self._model_for_qpos]

    def qpos_to_actuator(self, values):
        return np.asarray(values)[self._qpos_for_actuator]


def quat_rotate_inverse(quaternion_wxyz, vector):
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    q_w = q[0]
    q_vec = q[1:]
    return (
        v * (2.0 * q_w * q_w - 1.0)
        - np.cross(q_vec, v) * q_w * 2.0
        + q_vec * np.dot(q_vec, v) * 2.0
    )


def quaternion_to_euler(quaternion_wxyz):
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = math.asin(float(pitch_arg))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


class ObservationBuilder:
    def __init__(self, config, profile_name, default_angles):
        interface = config["policy_interface"]
        scales = interface["obs_scales"]
        self.profile_name = profile_name
        self.default_angles = np.asarray(default_angles, dtype=np.float32)
        self.frame_stack = int(interface["frame_stack"])
        self.num_single_obs = int(interface["num_single_obs"])
        self.clip_observations = float(interface["clip_observations"])
        self.ang_vel_scale = float(scales["ang_vel"])
        self.dof_pos_scale = float(scales["dof_pos"])
        self.dof_vel_scale = float(scales["dof_vel"])
        self.euler_scale = float(scales["euler"])
        self.velocity_command_scale = np.asarray(scales["velocity_command"], dtype=np.float32)
        self.control_dt = float(config["simulation"]["dt"]) * int(config["simulation"]["control_decimation"])
        self.phase_period = float(config["profiles"]["jump"].get("phase_period", 1.5))
        self.history = deque(maxlen=self.frame_stack)
        self.reset()

    def reset(self):
        self.history.clear()
        for _ in range(self.frame_stack):
            self.history.append(np.zeros(self.num_single_obs, dtype=np.float32))

    def build_single_frame(
        self,
        q_model,
        dq_model,
        base_quat_wxyz,
        world_ang_vel,
        previous_action_model,
        command,
        policy_step,
    ):
        if self.profile_name == "jump":
            phase = float(policy_step) * self.control_dt / self.phase_period
            command_prefix = np.concatenate(
                (
                    np.asarray([math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)], dtype=np.float32),
                    np.asarray(command, dtype=np.float32) * self.velocity_command_scale,
                )
            )
        else:
            command_prefix = np.concatenate(
                (np.zeros(2, dtype=np.float32), np.asarray(command, dtype=np.float32))
            )

        local_ang_vel = quat_rotate_inverse(base_quat_wxyz, world_ang_vel).astype(np.float32)
        euler = quaternion_to_euler(base_quat_wxyz)
        frame = np.concatenate(
            (
                command_prefix,
                local_ang_vel * self.ang_vel_scale,
                euler * self.euler_scale,
                (np.asarray(q_model, dtype=np.float32) - self.default_angles) * self.dof_pos_scale,
                np.asarray(dq_model, dtype=np.float32) * self.dof_vel_scale,
                np.asarray(previous_action_model, dtype=np.float32),
            )
        ).astype(np.float32)
        if frame.shape != (self.num_single_obs,):
            raise ValueError("Single observation shape {}, expected ({},)".format(frame.shape, self.num_single_obs))
        return np.clip(frame, -self.clip_observations, self.clip_observations)

    def append(self, *args, **kwargs):
        frame = self.build_single_frame(*args, **kwargs)
        self.history.append(frame)
        return np.concatenate(tuple(self.history)).astype(np.float32)


class SpringEpisodeController:
    def __init__(self, target_displacement, preparation_steps, episode_steps):
        self.target_displacement = float(target_displacement)
        self.preparation_steps = int(preparation_steps)
        self.episode_steps = int(episode_steps)
        self.running = False
        self.step = 0

    def reset(self):
        self.running = False
        self.step = 0

    def start(self):
        self.running = True
        self.step = 0

    def command(self):
        trigger = 1.0 if self.running and self.step >= self.preparation_steps else 0.0
        return np.asarray([self.target_displacement, 0.0, trigger], dtype=np.float32)

    def advance(self):
        if not self.running:
            return False
        self.step += 1
        if self.step >= self.episode_steps:
            self.running = False
            return True
        return False


@dataclass
class ControlUpdate:
    velocity_command: np.ndarray
    reset: bool = False
    start_spring: bool = False
    select_profile: Optional[str] = None


class ConfigController:
    def __init__(self, config):
        self.command = np.asarray(config["controls"]["config_command"], dtype=np.float32)

    def poll(self):
        return ControlUpdate(self.command.copy())

    def close(self):
        return None


class KeyboardController:
    def __init__(self, config):
        try:
            from pynput import keyboard
        except ImportError as exc:
            raise RuntimeError("Keyboard control requires pynput and an available X display") from exc

        self.keyboard = keyboard
        self.command = np.asarray(config["controls"]["keyboard_command"], dtype=np.float32)
        self.keys = set()
        self.edge_keys = []
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _normalize(self, key):
        special = {self.keyboard.Key.space: "space"}
        if key in special:
            return special[key]
        char = getattr(key, "char", None)
        return char.lower() if char is not None else None

    def _on_press(self, key):
        name = self._normalize(key)
        if name is None:
            return
        with self.lock:
            if name not in self.keys and name in ("space", "r", "1", "2"):
                self.edge_keys.append(name)
            self.keys.add(name)

    def _on_release(self, key):
        name = self._normalize(key)
        if name is not None:
            with self.lock:
                self.keys.discard(name)

    def poll(self):
        with self.lock:
            keys = set(self.keys)
            edges = self.edge_keys
            self.edge_keys = []
        velocity = np.asarray(
            [
                (float("w" in keys) - float("s" in keys)) * self.command[0],
                (float("a" in keys) - float("d" in keys)) * self.command[1],
                (float("q" in keys) - float("e" in keys)) * self.command[2],
            ],
            dtype=np.float32,
        )
        selected = "jump" if "1" in edges else "spring_jump" if "2" in edges else None
        return ControlUpdate(
            velocity,
            reset="r" in edges,
            start_spring="space" in edges,
            select_profile=selected,
        )

    def close(self):
        self.listener.stop()


class XboxController:
    def __init__(self, config):
        import pygame

        self.pygame = pygame
        pygame.display.init()
        pygame.joystick.init()
        self.joystick = None
        self.previous_buttons = {}
        self.controls = config["controls"]
        self.max_command = np.asarray(self.controls["xbox_max_command"], dtype=np.float32)
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

    @property
    def available(self):
        return self.joystick is not None

    @property
    def name(self):
        return self.joystick.get_name() if self.joystick is not None else ""

    def _rising(self, label, index):
        pressed = bool(self.joystick.get_button(int(index)))
        rising = pressed and not self.previous_buttons.get(label, False)
        self.previous_buttons[label] = pressed
        return rising

    def poll(self):
        if self.joystick is None:
            return ControlUpdate(np.zeros(3, dtype=np.float32))
        self.pygame.event.pump()
        axes = np.asarray(
            [-self.joystick.get_axis(1), -self.joystick.get_axis(0), -self.joystick.get_axis(3)],
            dtype=np.float32,
        )
        deadzone = float(self.controls["joystick_deadzone"])
        axes[np.abs(axes) < deadzone] = 0.0
        velocity = axes * self.max_command
        if np.linalg.norm(velocity[:2]) <= float(self.controls["planar_command_deadband"]):
            velocity[:2] = 0.0

        buttons = self.controls["xbox_buttons"]
        reset = self._rising("x", buttons["x_reset"])
        start_spring = self._rising("y", buttons["y_spring"])
        select_profile = None
        if self._rising("lb", buttons["lb_jump"]):
            select_profile = "jump"
        if self._rising("rb", buttons["rb_spring"]):
            select_profile = "spring_jump"
        return ControlUpdate(velocity, reset, start_spring, select_profile)

    def close(self):
        self.pygame.quit()


class Z2JumpSimulation:
    def __init__(self, config, profile_name="jump"):
        import mujoco
        import torch

        self.mujoco = mujoco
        self.torch = torch
        self.config = config
        self.interface = config["policy_interface"]
        self.model = mujoco.MjModel.from_xml_path(str(resolve_path(config["xml_path"])))
        self.model.opt.timestep = float(config["simulation"]["dt"])
        self.data = mujoco.MjData(self.model)
        self.decimation = int(config["simulation"]["control_decimation"])
        self.num_actions = int(self.interface["num_actions"])
        self.action_scale = float(self.interface["action_scale"])
        self.clip_actions = float(self.interface["clip_actions"])

        self.qpos_joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(self.model.njnt)
            if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self.actuator_joint_names = [
            mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(self.model.actuator_trnid[actuator_id, 0]),
            )
            for actuator_id in range(self.model.nu)
        ]
        self.joint_map = JointOrderMap(
            config["model_joint_names"], self.qpos_joint_names, self.actuator_joint_names
        )
        joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.qpos_joint_names
        ]
        self.qpos_indices = np.asarray([self.model.jnt_qposadr[joint_id] for joint_id in joint_ids])
        self.dof_indices = np.asarray([self.model.jnt_dofadr[joint_id] for joint_id in joint_ids])
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.foot_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        }
        self.foot_geom_ids.discard(-1)

        self.policies = {}
        expected_input = int(self.interface["num_single_obs"]) * int(self.interface["frame_stack"])
        for name in PROFILE_NAMES:
            policy_path = resolve_path(config["profiles"][name]["policy_path"])
            if not policy_path.is_file():
                raise FileNotFoundError("Policy file not found: {}".format(policy_path))
            policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
            with torch.inference_mode():
                output = policy(torch.zeros(1, expected_input, dtype=torch.float32))
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, self.num_actions):
                raise ValueError("{} policy must map (1,{}) to (1,{})".format(name, expected_input, self.num_actions))
            self.policies[name] = policy

        spring = config["profiles"]["spring_jump"]
        self.spring_episode = SpringEpisodeController(
            spring["target_displacement"], spring["preparation_steps"], spring["episode_steps"]
        )
        self.profile_name = None
        self.profile = None
        self.builder = None
        self.velocity_command = np.zeros(3, dtype=np.float32)
        self.action_model = np.zeros(self.num_actions, dtype=np.float32)
        self.target_qpos = np.zeros(self.num_actions, dtype=np.float32)
        self.physics_step = 0
        self.policy_step = 0
        self.pending_episode_complete = False
        self.fall_reported = False
        self.metrics = {}
        self.reset(profile_name)

    @property
    def at_policy_boundary(self):
        return self.physics_step % self.decimation == 0

    def reset(self, profile_name=None, start_spring=False):
        if profile_name is not None:
            if profile_name not in PROFILE_NAMES:
                raise ValueError("Unknown profile: {}".format(profile_name))
            self.profile_name = profile_name
        self.profile = self.config["profiles"][self.profile_name]
        default_model = np.asarray(self.profile["default_angles"], dtype=np.float32)
        default_qpos = self.joint_map.model_to_qpos(default_model)

        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = np.asarray(self.profile["init_base_pos"], dtype=np.float64)
        self.data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[self.qpos_indices] = default_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

        self.builder = ObservationBuilder(self.config, self.profile_name, default_model)
        self.action_model = np.zeros(self.num_actions, dtype=np.float32)
        self.target_qpos = default_qpos.copy()
        self.physics_step = 0
        self.policy_step = 0
        self.pending_episode_complete = False
        self.fall_reported = False
        self.spring_episode.reset()
        if start_spring and self.profile_name == "spring_jump":
            self.spring_episode.start()
        policy = self.policies[self.profile_name]
        if hasattr(policy, "reset"):
            policy.reset()
        self.metrics = {
            "initial_height": float(self.data.qpos[2]),
            "max_height": float(self.data.qpos[2]),
            "airborne_time": 0.0,
            "has_ground_contact": False,
            "has_been_airborne": False,
            "landed": False,
        }

    def start_spring(self):
        self.reset("spring_jump", start_spring=True)

    def _spring_command(self):
        return self.spring_episode.command()

    def _infer_policy(self):
        q_qpos = self.data.qpos[self.qpos_indices]
        dq_qpos = self.data.qvel[self.dof_indices]
        q_model = self.joint_map.qpos_to_model(q_qpos)
        dq_model = self.joint_map.qpos_to_model(dq_qpos)
        command = self.velocity_command if self.profile_name == "jump" else self._spring_command()
        observation = self.builder.append(
            q_model,
            dq_model,
            self.data.qpos[3:7],
            self.data.qvel[3:6],
            self.action_model,
            command,
            self.policy_step,
        )
        tensor = self.torch.from_numpy(observation).unsqueeze(0)
        with self.torch.inference_mode():
            action = self.policies[self.profile_name](tensor)
        self.action_model = np.clip(
            action.detach().cpu().numpy().reshape(-1),
            -self.clip_actions,
            self.clip_actions,
        ).astype(np.float32)
        target_model = np.asarray(self.profile["default_angles"], dtype=np.float32) + self.action_model * self.action_scale
        self.target_qpos = self.joint_map.model_to_qpos(target_model)
        self.policy_step += 1
        if self.profile_name == "spring_jump" and self.spring_episode.advance():
            self.pending_episode_complete = True

    def _base_contact(self):
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if body1 == self.base_body_id or body2 == self.base_body_id:
                return True
        return False

    def _feet_in_contact(self):
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.geom1 in self.foot_geom_ids or contact.geom2 in self.foot_geom_ids:
                return True
        return False

    def _update_metrics(self):
        height = float(self.data.qpos[2])
        self.metrics["max_height"] = max(self.metrics["max_height"], height)
        feet_contact = self._feet_in_contact()
        if feet_contact:
            if self.metrics["has_been_airborne"]:
                self.metrics["landed"] = True
            self.metrics["has_ground_contact"] = True
        elif self.metrics["has_ground_contact"]:
            self.metrics["airborne_time"] += float(self.model.opt.timestep)
            self.metrics["has_been_airborne"] = True

    def step(self):
        if self.at_policy_boundary:
            self._infer_policy()

        q = self.data.qpos[self.qpos_indices]
        dq = self.data.qvel[self.dof_indices]
        kp = float(self.profile["kp"])
        kd = float(self.profile["kd"])
        raw_tau_qpos = (self.target_qpos - q) * kp - dq * kd
        limit = float(self.profile["torque_limit"])
        tau_qpos = np.clip(raw_tau_qpos, -limit, limit)
        self.data.ctrl[:] = self.joint_map.qpos_to_actuator(tau_qpos)
        self.mujoco.mj_step(self.model, self.data)
        self.physics_step += 1
        self._update_metrics()

        reset_height = float(self.profile["reset_height"])
        if not self.fall_reported and (
            float(self.data.qpos[2]) <= reset_height or self._base_contact()
        ):
            self.fall_reported = True
            return "fall"
        if self.pending_episode_complete and self.at_policy_boundary:
            self.pending_episode_complete = False
            return "episode_complete"
        return None

    def summary(self):
        return {
            "profile": self.profile_name,
            "sim_time": float(self.data.time),
            "max_height": float(self.metrics["max_height"]),
            "height_gain": float(self.metrics["max_height"] - self.metrics["initial_height"]),
            "airborne_time": float(self.metrics["airborne_time"]),
            "landed": bool(self.metrics["landed"]),
        }


def make_controller(control_mode, config):
    if control_mode == "keyboard":
        return KeyboardController(config)
    if control_mode == "xbox":
        controller = XboxController(config)
        if controller.available:
            print("Detected Xbox controller: {}".format(controller.name))
            return controller
        controller.close()
        print("No Xbox controller detected; falling back to config command")
    return ConfigController(config)


def process_control(simulation, update):
    simulation.velocity_command = np.asarray(update.velocity_command, dtype=np.float32)
    if update.select_profile is not None and update.select_profile != simulation.profile_name:
        simulation.reset(update.select_profile)
        print("Selected policy: {}".format(simulation.profile_name))
    if update.reset:
        simulation.reset()
        print("Reset policy: {}".format(simulation.profile_name))
    if update.start_spring:
        if simulation.profile_name == "spring_jump":
            simulation.start_spring()
            print("Spring episode armed: dx=0.8 m, trigger_delay=1.0 s")
        else:
            print("Spring trigger ignored while jump policy is active")


def run_headless(simulation, duration, auto_start_spring=False):
    if simulation.profile_name == "jump":
        simulation.velocity_command = np.asarray(
            simulation.config["controls"]["config_command"], dtype=np.float32
        )
    if auto_start_spring and simulation.profile_name == "spring_jump":
        simulation.start_spring()
    elapsed = 0.0
    events = []
    simulation_dt = float(simulation.model.opt.timestep)
    while elapsed < float(duration):
        event = simulation.step()
        elapsed += simulation_dt
        if event is not None:
            events.append(event)
    summary = simulation.summary()
    summary["events"] = events
    return summary


def run_viewer(simulation, controller, duration):
    import mujoco
    import mujoco.viewer

    camera_name = simulation.config["simulation"]["viewer_camera"]
    camera_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError("Camera '{}' not found in MuJoCo model".format(camera_name))

    print("Keyboard: W/S/A/D/Q/E move, Space spring, R reset, 1/2 select policy")
    print("Xbox: sticks move, Y spring, X reset, LB/RB select policy")
    start_wall_time = time.time()
    with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id
        while viewer.is_running() and time.time() - start_wall_time < float(duration):
            step_start = time.time()
            if simulation.at_policy_boundary:
                process_control(simulation, controller.poll())
            event = simulation.step()
            if event is not None:
                result = simulation.summary()
                print("\n{}: {}".format(event, result))
            viewer.sync()
            sleep_time = float(simulation.model.opt.timestep) - (time.time() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Go2 jump policies on the Z2 MuJoCo model")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path or config filename")
    parser.add_argument("--policy", choices=PROFILE_NAMES, default="jump")
    parser.add_argument("--control", choices=("xbox", "keyboard", "config"), default="xbox")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=None, help="Simulation duration in seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    duration = float(args.duration) if args.duration is not None else float(config["simulation"]["duration"])
    simulation = Z2JumpSimulation(config, args.policy)
    print("Config: {}".format(config["_config_path"]))
    print("Scene: {}".format(resolve_path(config["xml_path"])))
    print("Policy: {}".format(args.policy))
    if args.headless:
        summary = run_headless(simulation, duration, auto_start_spring=args.policy == "spring_jump")
        print("HEADLESS_RESULT {}".format(summary))
        return

    controller = make_controller(args.control, config)
    try:
        if isinstance(controller, ConfigController) and args.policy == "spring_jump":
            simulation.start_spring()
        run_viewer(simulation, controller, duration)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
