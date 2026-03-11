from legged_gym.envs.nocv.nocv_cfg import NOCVCfg, NOCVCfgMoECTS


class JUMPCfg(NOCVCfg):
	"""NoCV 可控跳跃任务配置。

	在 `NOCV` 的 5 维 command 基础上，继续追加 4 维跳跃命令：
	- `jump_dx`
	- `jump_dy`
	- `jump_dz`
	- `jump_trigger`

	因此内部 command 语义变为：
	`[lin_vel_x, lin_vel_y, ang_vel_yaw, heading, body_height_mode, jump_dx, jump_dy, jump_dz, jump_trigger]`
	"""

	class env(NOCVCfg.env):
		num_observations = 50
		num_privileged_obs = 271
		episode_length_s = 6.0
		reset_height = 0.18

		# ---- 落地姿态终止阈值 ----
		# 仅在 `has_jumped=True`（即已检测到落地）后生效，
		# 用于惩罚翻滚、侧翻、高角速度失稳等不良落地姿态。
		jump_land_pitch_threshold = 0.4   # rad (~23°)
		jump_land_roll_threshold = 0.4    # rad (~23°)
		jump_land_ang_vel_threshold = 4.0 # rad/s, 落地后机身旋转失稳

	class terrain(NOCVCfg.terrain):
		# jump 任务先只在 flat 上训练，避免把跳跃信号和复杂地形适应同时耦合。
		terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

	class commands(NOCVCfg.commands):
		num_commands = 9
		resampling_time = 10.0
		dynamic_resample_commands = False
		curriculum = False
		command_range_curriculum = []
		# jump 任务下，普通 NoCV 的低高度随机采样关闭；
		# 低姿态仅由 jump 逻辑在需要时显式打开。
		low_height_command_prob = 0.0
		low_height_terrain_ids = [8]

		jump_dx_command_idx = 5
		jump_dy_command_idx = 6
		jump_dz_command_idx = 7
		jump_trigger_command_idx = 8
		jump_terrain_ids = [8]

		jump_command_obs_scale = [1.0, 1.0, 1.0, 1.0]
		jump_command_threshold = 0.5
		jump_command_prob = 1.0
		normal_jump_command = 0.0
		active_jump_command = 1.0
		jump_locomotion_command_scale = [0.0, 0.0, 0.0]

		class ranges(NOCVCfg.commands.ranges):
			jump_dx = [-0.35, 0.35]
			jump_dy = [-0.20, 0.20]
			jump_dz = [0.12, 0.28]
			jump_trigger = [0.0, 1.0]

	class rewards(NOCVCfg.rewards):
		jump_apex_sigma = 0.04
		jump_land_sigma = 0.20

		class scales(NOCVCfg.rewards.scales):
			jump_takeoff_vel = 1.5
			jump_apex_height = 2.0
			jump_land_target = 3.0
			jump_flight = 0.5


class JUMPCfgMoECTS(NOCVCfgMoECTS):
	class runner(NOCVCfgMoECTS.runner):
		run_name = ''
		experiment_name = 'jump_moe_cts'
		max_iterations = 130000
		save_interval = 500