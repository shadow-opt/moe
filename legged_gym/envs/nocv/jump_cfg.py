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

	当前设计约束：
	1. jump 任务只在 flat tile（terrain id = 8）上训练；
	2. 每个 episode 只执行一次 jump command；
	3. `jump_dx / jump_dy / jump_dz` 分别控制跳远、侧向落点和跳高目标。
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
		# 只保留 flat tile。仍沿用 trimesh + terrain id 语义，
		# 这样 `jump_terrain_ids` / `low_height_terrain_ids` 的门控逻辑无需改动。
		terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

	class commands(NOCVCfg.commands):
		num_commands = 9
		resampling_time = 10.0
		dynamic_resample_commands = False
		curriculum = False
		command_range_curriculum = []

		# jump 任务不再复用 NoCV 的随机低姿态子任务；
		# 蹲伏准备完全由 jump trigger 显式驱动。
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
			jump_dz = [0.12, 0.42]
			jump_trigger = [0.0, 1.0]

	class rewards(NOCVCfg.rewards):
		jump_apex_sigma = 0.04
		jump_land_sigma = 0.20
		# 落地瞬间希望前后足不要拉得太开。
		# 这里约束的是 base 坐标系下“前足中心 - 后足中心”的 x 向距离上限。
		jump_land_stance_length_max = 0.24
		jump_land_stance_length_sigma = 0.02

		class scales(NOCVCfg.rewards.scales):
			# jump 模式下，垂向速度 / 常规高度 / feet regulation 会与起跳和腾空目标直接对冲，
			# 因此在 jump 专用任务中关闭，改由 jump 专属奖励与落地稳定性终止约束承担。
			lin_vel_z = 0.0
			correct_base_height = 0.0
			feet_regulation = 0.0
			jump_takeoff_vel = 1.5
			jump_apex_height = 2.0
			jump_land_target = 3.0
			jump_land_compact = 0.8
			jump_flight = 0.5


class JUMPCfgMoECTS(NOCVCfgMoECTS):
	class runner(NOCVCfgMoECTS.runner):
		run_name = 'jump'
		experiment_name = 'nocv_moe_cts'
		max_iterations = 150000
		save_interval = 500