import copy

from legged_gym.envs.go2.go2_config import GO2Cfg
from legged_gym.envs.nocv.win_cfg import WINFlatSlowCfg, WINFlatSlowCfgMoECTS


class WINJumpCfg(WINFlatSlowCfg):
    """Flat Z2 locomotion plus command-gated continuous jumping."""

    class env(WINFlatSlowCfg.env):
        num_envs = 8192
        episode_length_s = 25.0

    class terrain(WINFlatSlowCfg.terrain):
        mesh_type = "trimesh"
        curriculum = True
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    class commands(WINFlatSlowCfg.commands):
        # Internal buffer: [vx, vy, yaw, heading, body_mode, jump_sin, jump_cos].
        num_commands = 7
        zero_command_curriculum = None
        full_stop_command_curriculum = None
        command_range_curriculum = []
        resampling_time = 5.0

        jump_mode_command = -1.0
        jump_sample_probability = 0.60
        walk_sample_probability = 0.30
        stand_sample_probability = 0.10
        idle_jump_sample_probability = 0.0
        jump_min_abs_vx = 0.30
        jump_initial_max_abs_vx = 0.80
        jump_unlocked_max_abs_vx = 1.00
        jump_enable_threshold = 0.30
        jump_disable_threshold = 0.20
        jump_cycle_time = 1.50
        jump_pause_time = 0.0
        turn_command_sample_probability = 0.0
        cyclic_jump_phase = False
        jump_prep_steps = 10
        jump_landing_contact_steps = 3
        jump_exit_timeout_s = 2.0
        jump_contact_threshold = 5.0
        jump_curriculum_threshold = 0.80
        jump_curriculum_windows = 5
        jump_curriculum_min_samples = 4096

        class ranges(WINFlatSlowCfg.commands.ranges):
            lin_vel_x = [-1.1, 1.1]
            lin_vel_y = [-0.8, 0.8]
            ang_vel_yaw = [-1.5, 1.5]
            heading = [-1.57, 1.57]

        terrain_max_command_ranges = [
            {
                "lin_vel_x": [-1.1, 1.1],
                "lin_vel_y": [-0.8, 0.8],
                "ang_vel_yaw": [-1.5, 1.5],
                "heading": [-1.57, 1.57],
            }
            for _ in range(9)
        ]

    class domain_rand(WINFlatSlowCfg.domain_rand):
        # Match the selected Jul05 flat-slow checkpoint distribution.
        randomize_friction = True
        friction_range = [0.0, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 0.5]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        randomize_base_com = True
        added_base_com_range = [-0.03, 0.03]
        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]
        randomize_action_delay = True
        push_robots = True
        push_interval_s = 4.0
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6
        hard_torque_limit_after_randomization = True

    class rewards(WINFlatSlowCfg.rewards):
        curriculum_rewards = None
        dynamic_sigma = None
        tracking_sigma = 0.25
        max_contact_force = 100.0
        soft_dof_pos_limit = 0.8
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.75
        jump_base_height_target = 0.30
        jump_target_feet_height = 0.05

        class scales:
            # Frozen model_85000 flat-slow rewards.
            action_rate = -0.01
            action_smoothness = -0.01
            ang_vel_xy = -0.09
            collision = -1.0
            correct_base_height = -10.0
            dof_acc = -2.5e-7
            dof_pos_limits = -4.0
            dof_power = -2.0e-5
            dof_vel_limits = -0.01
            feet_air_time = 1.0
            feet_air_time_variance = -0.3
            feet_contact_without_cmd = 0.03
            feet_regulation = -0.05
            foot_slip = -0.01
            hip_to_default = -0.05
            hip_to_zero = -10.0
            lateral_yaw_tracking_error = -1.5
            lin_vel_z = 0.0
            low_height_correct_base_height = -20.0
            low_speed_feet_air_time = 0.5
            stand_still = -4.0
            stand_still_default_pose = 0.0
            torques = -1.0e-4
            tracking_ang_vel = 0.55
            tracking_lin_vel = 1.1
            x_command_hip_regular = -0.5

            # Go2 jump recipe, routed only to active jump samples.
            jump_contact_pattern = 2.0
            jump_tracking_lin_vel = 2.0
            jump_tracking_ang_vel = 2.0
            jump_lin_vel_z = 0.05
            jump_ang_vel_xy = 0.2
            jump_orientation = 0.6
            jump_base_height = 1.0
            jump_feet_air_time = 1.0
            jump_default_pos = -0.1
            jump_default_hip_pos = 0.3
            jump_feet_contact_forces = -0.01
            jump_feet_clearance = 0.5
            jump_torques = -2.0e-4
            jump_dof_acc = -5.5e-4
            jump_action_rate = -0.01


class WINJumpCfgMoECTS(WINFlatSlowCfgMoECTS):
    class algorithm(WINFlatSlowCfgMoECTS.algorithm):
        learning_rate = 1.0e-4
        min_learning_rate = 1.0e-5
        max_learning_rate = 1.0e-4
        student_encoder_learning_rate = 1.0e-3
        walk_behavior_coef = 0.1
        jump_symmetry = True
        jump_symmetry_start_coef = 0.1
        jump_symmetry_end_coef = 1.0
        jump_symmetry_ramp_iterations = 5000
        jump_body_mode_obs_index = 9

    class runner(WINFlatSlowCfgMoECTS.runner):
        run_name = "jump_posttrain"
        experiment_name = "win_jump_moe_cts"
        num_steps_per_env = 24
        max_iterations = 20000
        save_interval = 4000
        save_initial_checkpoint = True
        exact_save_intervals = True


class WINJumpCyclicCfg(WINJumpCfg):
    class commands(WINJumpCfg.commands):
        cyclic_jump_phase = True


class WINJumpCyclicCfgMoECTS(WINJumpCfgMoECTS):
    class runner(WINJumpCfgMoECTS.runner):
        run_name = "jump_posttrain_cyclic_phase"
        experiment_name = "win_jump_cyclic_moe_cts"


class WINJumpCyclicScratchCfg(WINJumpCyclicCfg):
    class commands(WINJumpCyclicCfg.commands):
        low_height_command_prob = 0.0
        low_height_terrain_ids = []

    class rewards(WINJumpCyclicCfg.rewards):
        curriculum_rewards = copy.deepcopy(GO2Cfg.rewards.curriculum_rewards)

        class scales(WINJumpCyclicCfg.rewards.scales):
            lin_vel_z = GO2Cfg.rewards.scales.lin_vel_z
            correct_base_height = GO2Cfg.rewards.scales.correct_base_height


class WINJumpCyclicScratchCfgMoECTS(WINJumpCyclicCfgMoECTS):
    class algorithm(WINJumpCyclicCfgMoECTS.algorithm):
        learning_rate = 1.0e-3
        min_learning_rate = 1.0e-5
        max_learning_rate = 1.0e-3
        student_encoder_learning_rate = 1.0e-3
        walk_behavior_coef = 0.0

    class runner(WINJumpCyclicCfgMoECTS.runner):
        run_name = "jump_scratch_cyclic_phase"
        experiment_name = "win_jump_cyclic_scratch_moe_cts"


class WINJumpRecoveryCfg(WINJumpCyclicCfg):
    """Short post-training task that teaches jump mode to idle at zero vx."""

    class commands(WINJumpCyclicCfg.commands):
        # 25% of jump-mode samples keep body_mode=-1 while commanding zero
        # velocity and zero phase, matching a released deployment joystick.
        idle_jump_sample_probability = 0.25
        jump_pause_time = 0.75
        turn_command_sample_probability = 0.20
        turn_command_min_abs_yaw = 0.40
        turn_command_max_abs_yaw = 1.00


class WINJumpRecoveryCfgMoECTS(WINJumpCyclicCfgMoECTS):
    class algorithm(WINJumpCyclicCfgMoECTS.algorithm):
        learning_rate = 1.0e-4
        max_learning_rate = 1.0e-4
        idle_jump_behavior_coef = 1.0

    class runner(WINJumpCyclicCfgMoECTS.runner):
        run_name = "jump_zero_vx_recovery"
        experiment_name = "win_jump_recovery_moe_cts"
        max_iterations = 2000
        save_interval = 250
