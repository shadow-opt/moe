from legged_gym.envs.nocv.win_cfg import WINFlatSlowCfg, WINFlatSlowCfgMoECTS


WALL_HEIGHTS = [0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35]


def _terrain_command_ranges():
    common = {
        "lin_vel_x": [-2.0, 2.0],
        "lin_vel_y": [-1.0, 1.0],
        "ang_vel_yaw": [-1.7, 1.7],
        "heading": [-1.57, 1.57],
    }
    return [dict(common) for _ in range(10)]


class WINCloseClimbCfg(WINFlatSlowCfg):
    """Shared 48-observation WIN locomotion and real thin-wall task."""

    close_climb_phase = "transition"

    class env(WINFlatSlowCfg.env):
        num_envs = 4096
        episode_length_s = 10.0

    class terrain(WINFlatSlowCfg.terrain):
        mesh_type = "trimesh"
        curriculum = True
        horizontal_scale = 0.025
        vertical_scale = 0.005
        border_size = 2.0
        terrain_length = 4.0
        terrain_width = 1.6
        terrain_spacing = 0.5
        num_rows = 7
        num_cols = 20
        max_init_terrain_level = 0
        # [wave, slope, rough slope, stairs up/down, obstacles, stones, gap, flat, thin_wall]
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.6]
        thin_wall_heights = WALL_HEIGHTS
        thin_wall_fixed_height = None
        thin_wall_thickness = 0.05
        thin_wall_front_offset = 0.425

    class commands(WINFlatSlowCfg.commands):
        # Internal: [vx, vy, yaw, heading, height_mode, climb_mode, wall_height_m].
        num_commands = 7
        climb_command_idx = 5
        wall_height_command_idx = 6
        wall_height_command_obs_scale = 3.0
        wall_terrain_id = 9
        wall_approach_velocity = 0.30
        wall_trigger_distance_range = [0.40, 0.45]
        acquisition_reset_distance_range = [0.40, 0.45]
        transition_reset_distance_range = [0.70, 1.00]
        command_range_curriculum = []
        dynamic_resample_commands = False
        terrain_max_command_ranges = _terrain_command_ranges()

        class ranges(WINFlatSlowCfg.commands.ranges):
            lin_vel_x = [-2.0, 2.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.7, 1.7]
            heading = [-1.57, 1.57]

    class init_state(WINFlatSlowCfg.init_state):
        randomize_yaw = False

    class domain_rand(WINFlatSlowCfg.domain_rand):
        friction_range = [0.4, 1.6]
        push_robots = False
        randomize_action_delay = True

    class rewards(WINFlatSlowCfg.rewards):
        # Freeze the mature multipliers from the 85k flat checkpoint.
        curriculum_rewards = None
        dynamic_sigma = None
        close_climb_wall_id = 9
        close_climb_lateral_limit = 0.30
        close_climb_stuck_duration_s = 2.0
        close_climb_no_progress_delta = 0.005
        close_climb_fall_base_height = 0.16
        close_climb_fall_roll = 0.90
        close_climb_fall_pitch = 0.90
        close_climb_stable_duration_s = 0.50
        close_climb_post_success_s = 1.0
        close_climb_post_disengage_s = 0.5
        close_climb_success_base_margin = 0.50
        close_climb_success_feet_margin = 0.10
        close_climb_stable_base_height = [0.28, 0.50]
        close_climb_stable_roll = 0.35
        close_climb_stable_pitch = 0.45
        close_climb_stable_yaw_rate = 1.0
        close_climb_front_hip_force_threshold = 30.0
        close_climb_rear_hip_force_threshold = 5.0
        close_climb_rear_hip_force_multiplier = 5.0
        foot_slip_excluded_terrain_ids = [3, 4, 9]

        class scales(WINFlatSlowCfg.rewards.scales):
            lin_vel_z = 0.0
            ang_vel_xy = -0.09
            stand_still = -4.0
            lateral_yaw_tracking_error = -1.5
            hip_to_zero = -10.0
            climb_progress = 4.0
            climb_rear_feet_progress = 2.0
            climb_clearance = 5.0
            climb_stable_success = 10.0
            climb_wall_contact = 0.25
            climb_orientation = -1.0
            climb_centerline = -6.0
            climb_yaw = -1.0
            climb_hip_force = -0.02
            climb_stuck = -2.0


class WINCloseClimbAcquireCfg(WINCloseClimbCfg):
    close_climb_phase = "acquire"

    class terrain(WINCloseClimbCfg.terrain):
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.75]

    class domain_rand(WINCloseClimbCfg.domain_rand):
        friction_range = [0.8, 1.2]
        randomize_base_mass = False
        randomize_link_mass = False
        randomize_base_com = False
        randomize_restitution = False
        randomize_pd_gains = False
        randomize_motor_zero_offset = False
        randomize_motor_strength = True
        motor_strength_range = [0.95, 1.05]
        randomize_action_delay = False
        push_robots = False


class WINCloseClimbEvalCfg(WINCloseClimbCfg):
    close_climb_phase = "eval"

    class env(WINCloseClimbCfg.env):
        test = True

    class terrain(WINCloseClimbCfg.terrain):
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        thin_wall_fixed_height = 0.30

    class domain_rand(WINCloseClimbCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        randomize_link_mass = False
        randomize_base_com = False
        randomize_restitution = False
        randomize_pd_gains = False
        randomize_motor_zero_offset = False
        randomize_motor_strength = False
        randomize_action_delay = False
        push_robots = False


class WINCloseClimbAcquireCfgMoECTS(WINFlatSlowCfgMoECTS):
    class algorithm(WINFlatSlowCfgMoECTS.algorithm):
        learning_rate = 3e-4
        student_encoder_learning_rate = 1e-3

    class runner(WINFlatSlowCfgMoECTS.runner):
        run_name = "close_climb_acquire"
        experiment_name = "win_close_climb_acquire"
        num_steps_per_env = 24
        max_iterations = 5000
        save_interval = 5000


class WINCloseClimbTransitionCfgMoECTS(WINFlatSlowCfgMoECTS):
    class algorithm(WINFlatSlowCfgMoECTS.algorithm):
        learning_rate = 1e-4
        student_encoder_learning_rate = 3e-4

    class runner(WINFlatSlowCfgMoECTS.runner):
        run_name = "close_climb_transition"
        experiment_name = "win_close_climb_transition"
        num_steps_per_env = 24
        max_iterations = 5000
        save_interval = 5000


class WINCloseClimbEvalCfgMoECTS(WINCloseClimbTransitionCfgMoECTS):
    class runner(WINCloseClimbTransitionCfgMoECTS.runner):
        run_name = "close_climb_eval"
        # Evaluation resumes checkpoints produced by the transition stage.
        experiment_name = "win_close_climb_transition"
        max_iterations = 0
