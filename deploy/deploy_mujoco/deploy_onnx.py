import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import yaml
import os
import imageio
from pathlib import Path
from argparse import ArgumentParser
import pygame
import onnxruntime as ort
import warnings

try:
	import glfw
	glfw.ERROR_REPORTING = {65537: 'ignore', None: 'warn'}
	warnings.filterwarnings("ignore", message=".*GLFW library is not initialized.*", category=glfw.GLFWError)
except Exception:
	glfw = None


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


def pd_control(target_q, q, kp, target_dq, dq, kd):
	return (target_q - q) * kp + (target_dq - dq) * kd


def get_xbox_command(joystick, max_cmd):
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
		cmd_x = -ly * max_cmd[0]
		cmd_y = -lx * max_cmd[1]
		cmd_yaw = -rx * max_cmd[2]
		return np.array([cmd_x, cmd_y, cmd_yaw], dtype=np.float32)
	return np.zeros(3, dtype=np.float32)


def build_command_vector(cmd_current, joystick_cmd):
	cmd_vec = cmd_current.copy().astype(np.float32)
	if cmd_vec.size == 0:
		return cmd_vec
	overwrite_dim = min(3, cmd_vec.shape[0], joystick_cmd.shape[0])
	cmd_vec[:overwrite_dim] = joystick_cmd[:overwrite_dim]
	return cmd_vec







def pack_stacked_obs(obs_hist, term_dims, stack_layout):
	if stack_layout == "time_major":
		return obs_hist.reshape(-1).astype(np.float32)
	if stack_layout == "field_major":
		chunks = []
		offset = 0
		for dim in term_dims:
			chunks.append(obs_hist[:, offset:offset + dim].reshape(-1))
			offset += dim
		return np.concatenate(chunks, axis=0).astype(np.float32)
	raise ValueError(f"Unsupported stack layout: {stack_layout}")


def draw_moe_weights(screen, weights, width, height):
	screen.fill((255, 255, 255))
	num_experts = len(weights)
	if num_experts == 0:
		return

	margin = 5
	bar_width = (width - 2 * margin) / num_experts
	max_bar_height = height - 2 * margin

	for i, w in enumerate(weights):
		w_clamped = max(0.0, min(1.0, float(w)))
		bar_height = int(w_clamped * max_bar_height)
		x = margin + i * bar_width
		y = height - margin - bar_height
		pygame.draw.rect(screen, (50, 100, 255), (x, y, bar_width - 2, bar_height))

	pygame.display.flip()


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--save-video", action="store_true", help="Whether to save video of the simulation.")
	parser.add_argument("--visualize-moe-weights", action="store_true", help="Whether to visualize mixture of experts weights.")
	parser.add_argument("--config", type=str, default="nocv_moe_cts.yaml", help="Config filename under deploy/deploy_mujoco/configs")
	args = parser.parse_args()

	save_video = args.save_video
	visualize_moe_weights = args.visualize_moe_weights
	config_file = args.config

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

	with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
		policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
		xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

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
		cmd_scale = np.array(config["cmd_scale"][:], dtype=np.float32)
		cmd_dim = int(cmd_scale.shape[0])

		num_actions = config["num_actions"]
		cmd_cfg = np.array(config["cmd_init"], dtype=np.float32)
		cmd = np.zeros(cmd_dim, dtype=np.float32)
		copy_dim = min(cmd_cfg.shape[0], cmd_dim)
		cmd[:copy_dim] = cmd_cfg[:copy_dim]
		idx_model2mj = idx_mj2model = list(range(num_actions))
		if "mujoco_joint_names" in config and "model_joint_names" in config:
			mujoco_joint_names = config["mujoco_joint_names"]
			model_joint_names = config["model_joint_names"]
			idx_model2mj = [model_joint_names.index(joint) for joint in mujoco_joint_names]
			idx_mj2model = [mujoco_joint_names.index(joint) for joint in model_joint_names]

	base_obs_dim = 3 + 3 + cmd_dim + num_actions + num_actions + num_actions
	obs_term_dims = [3, 3, cmd_dim, num_actions, num_actions, num_actions]
	default_history_len = int(config.get("history_len", 5))
	obs_dim = default_history_len * base_obs_dim
	stack_layout = str(config.get("onnx_stack_layout", "field_major")).lower()
	if stack_layout not in ("field_major", "time_major"):
		raise ValueError(f"Invalid onnx_stack_layout={stack_layout}. Expected 'field_major' or 'time_major'.")

	video_save_dir = str(Path(__file__).parent / "videos")
	os.makedirs(video_save_dir, exist_ok=True)
	model_name = os.path.basename(policy_path).split('.')[0]
	cmd_x = cmd[0] if cmd_dim > 0 else 0.0
	cmd_y = cmd[1] if cmd_dim > 1 else 0.0
	cmd_yaw = cmd[2] if cmd_dim > 2 else 0.0
	cmd_str = f"cmd_{cmd_x}_{cmd_y}_{cmd_yaw}"
	video_filename = f"{model_name}_{cmd_str}.mp4"
	video_path = os.path.join(video_save_dir, video_filename)
	if save_video:
		print(f"Video recording will be saved to: {video_path}")

	action = np.zeros(num_actions, dtype=np.float32)
	target_dof_pos = default_angles.copy()

	obs_hist = np.zeros((default_history_len, base_obs_dim), dtype=np.float32)
	obs = np.zeros(obs_dim, dtype=np.float32)

	counter = 0

	m = mujoco.MjModel.from_xml_path(xml_path)
	d = mujoco.MjData(m)
	m.opt.timestep = simulation_dt
	renderer = mujoco.Renderer(m, height=360, width=640)

	available_providers = ort.get_available_providers()
	if "CUDAExecutionProvider" in available_providers:
		providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
		print(f"ONNXRuntime available providers: {available_providers}")
	else:
		providers = ["CPUExecutionProvider"]
		print(f"ONNXRuntime available providers: {available_providers}")
		print("[Info] CUDAExecutionProvider unavailable, using CPUExecutionProvider.")

	policy = ort.InferenceSession(policy_path, providers=providers)
	input_meta = policy.get_inputs()[0]
	input_name = input_meta.name
	input_shape = input_meta.shape
	print(f"ONNX input name: {input_name}")
	print(f"ONNX input shape: {input_shape}")

	model_obs_dim = None
	if len(input_shape) >= 2 and isinstance(input_shape[1], int):
		model_obs_dim = input_shape[1]

	if model_obs_dim is None:
		model_obs_dim = obs_dim
		print(f"[Warning] ONNX input shape has dynamic second dim. Fallback to deploy obs dim {obs_dim}.")

	if model_obs_dim == base_obs_dim:
		history_len = 1
		use_stacked_history = False
	elif model_obs_dim % base_obs_dim == 0:
		history_len = model_obs_dim // base_obs_dim
		use_stacked_history = True
	else:
		raise ValueError(
			f"Unsupported ONNX obs dim: {model_obs_dim}. Expected single-frame {base_obs_dim} or N*{base_obs_dim} for stacked history."
		)

	if "num_obs" in config:
		cfg_num_obs = int(config["num_obs"])
		if cfg_num_obs == model_obs_dim:
			pass
		elif cfg_num_obs % base_obs_dim == 0 and model_obs_dim == base_obs_dim:
			print(f"[Info] Config num_obs={cfg_num_obs} is stacked env obs, while ONNX uses single-frame dim={model_obs_dim}.")
		elif model_obs_dim % base_obs_dim == 0 and cfg_num_obs == base_obs_dim:
			print(f"[Info] Config num_obs={cfg_num_obs} is single-frame, while ONNX uses stacked dim={model_obs_dim}.")
		else:
			print(f"[Warning] Config num_obs={cfg_num_obs} is incompatible with ONNX input dim={model_obs_dim} (base_obs_dim={base_obs_dim}).")

	if use_stacked_history and history_len != default_history_len:
		obs_hist = np.zeros((history_len, base_obs_dim), dtype=np.float32)
		obs_dim = history_len * base_obs_dim
		obs = np.zeros(obs_dim, dtype=np.float32)
	else:
		history_len = default_history_len
	if use_stacked_history:
		print(f"[Info] Using history-stacked observation input ({model_obs_dim}) with history_len={history_len}, layout={stack_layout}.")
	else:
		print(f"[Info] Using single-frame observation input ({model_obs_dim}).")

	if save_video:
		video_fps = 50
		sim_fps = 1.0 / m.opt.timestep
		frame_skip = max(1, int(sim_fps / video_fps))
		writer = imageio.get_writer(video_path, fps=video_fps)
		print(f"Sim FPS: {sim_fps:.2f}, Video FPS: {video_fps}, Frame Skip: {frame_skip}, Save at: {video_path}")
	else:
		writer = None

	try:
		with mujoco.viewer.launch_passive(m, d) as viewer:
			viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
			viewer.cam.trackbodyid = 1
			viewer.cam.distance = 3.0
			viewer.cam.elevation = -30.0
			viewer.cam.azimuth = 0.0

			start = time.time()
			while viewer.is_running() and time.time() - start < simulation_duration:
				if use_joystick and counter % control_decimation == 0:
					joystick_cmd = get_xbox_command(joystick, config["max_cmd"])
					cmd = build_command_vector(cmd, joystick_cmd)
					print(f"Cmd: Vx={cmd[0] if cmd_dim > 0 else 0.0:.2f}, Vy={cmd[1] if cmd_dim > 1 else 0.0:.2f}, Wz={cmd[2] if cmd_dim > 2 else 0.0:.2f}", end='\r')
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
					qj = d.qpos[7:]
					dqj = d.qvel[6:]
					quat = d.qpos[3:7]
					ang_vel = d.qvel[3:6]

					qj = (qj - default_angles) * dof_pos_scale
					dqj = dqj * dof_vel_scale
					gravity_orientation = get_gravity_orientation(quat)
					ang_vel = ang_vel * ang_vel_scale

					frame_obs = np.zeros(base_obs_dim, dtype=np.float32)
					offset = 0
					frame_obs[offset:offset + 3] = ang_vel.astype(np.float32)
					offset += 3
					frame_obs[offset:offset + 3] = gravity_orientation.astype(np.float32)
					offset += 3
					frame_obs[offset:offset + cmd_dim] = (cmd * cmd_scale).astype(np.float32)
					offset += cmd_dim
					frame_obs[offset:offset + num_actions] = qj[idx_mj2model].astype(np.float32)
					offset += num_actions
					frame_obs[offset:offset + num_actions] = dqj[idx_mj2model].astype(np.float32)
					offset += num_actions
					frame_obs[offset:offset + num_actions] = action[idx_mj2model].astype(np.float32)

					if use_stacked_history:
						obs_hist = np.roll(obs_hist, -1, axis=0)
						obs_hist[-1] = frame_obs
						obs[:] = pack_stacked_obs(obs_hist, obs_term_dims, stack_layout)
						obs_input = np.expand_dims(obs, axis=0).astype(np.float32)
					else:
						obs_input = np.expand_dims(frame_obs, axis=0).astype(np.float32)

					if obs_input.shape[1] != model_obs_dim:
						raise ValueError(f"Observation shape mismatch: built {obs_input.shape[1]}, model expects {model_obs_dim}.")

					result = policy.run(None, {input_name: obs_input})

					action_out = np.asarray(result[0]).squeeze()
					action = action_out[idx_model2mj].astype(np.float32)

					if len(result) > 1 and visualize_moe_weights and screen is not None:
						weights = np.asarray(result[1]).squeeze()
						draw_moe_weights(screen, weights, win_width, win_height)

					target_dof_pos = action * action_scale + default_angles

				viewer.sync()
	finally:
		if writer is not None:
			writer.close()
		if hasattr(renderer, "close"):
			try:
				renderer.close()
			except Exception:
				pass
		pygame.quit()

	if save_video:
		print(f"Video saved successfully to {video_path}")
