from legged_gym.envs.nocv.jump_cfg import JUMPCfg, JUMPCfgMoECTS


class CurriculumJUMPCfg(JUMPCfg):
    class env(JUMPCfg.env):
        num_observations = 50
        num_privileged_obs = 271
        episode_length_s = 6.0

    class commands(JUMPCfg.commands):
        command_range_curriculum = []
        jump_command_prob = 1.0
        jump_locomotion_command_scale = [1.0, 1.0, 1.0]
        jump_phase_cycle_time = 0.60

        performance_curriculum = True
        curriculum_success_window = 20
        curriculum_success_high = 0.80
        curriculum_success_low = 0.50
        curriculum_min_valid_episodes = 8
        curriculum_start_level = 0

        curriculum_levels = [
            {
                "jump_dx": [-0.10, 0.10],
                "jump_dy": [-0.05, 0.05],
                "jump_dz": [0.05, 0.10],
            },
            {
                "jump_dx": [-0.15, 0.15],
                "jump_dy": [-0.08, 0.08],
                "jump_dz": [0.07, 0.12],
            },
            {
                "jump_dx": [-0.20, 0.20],
                "jump_dy": [-0.10, 0.10],
                "jump_dz": [0.09, 0.15],
            },
            {
                "jump_dx": [-0.25, 0.25],
                "jump_dy": [-0.12, 0.12],
                "jump_dz": [0.10, 0.18],
            },
            {
                "jump_dx": [-0.30, 0.30],
                "jump_dy": [-0.15, 0.15],
                "jump_dz": [0.11, 0.21],
            },
            {
                "jump_dx": [-0.35, 0.35],
                "jump_dy": [-0.20, 0.20],
                "jump_dz": [0.12, 0.25],
            },
        ]

    class rewards(JUMPCfg.rewards):
        only_positive_rewards = False
        only_positive_rewards_ji22_style = True
        ji22_neg_reward_sigma = 0.2
        task_pos_sigma = 0.05
        task_ori_sigma = 0.05
        task_max_height_sigma = 0.05
        curriculum_success_landing_sigma = 0.05
        curriculum_success_min_apex_height = 0.12
        vel_tracking_sigma = 0.1
        dof_pos_sigma = 0.1
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 1.0
        max_contact_force = 200.0

        class scales(JUMPCfg.rewards.scales):
            termination = -20.0
            tracking_lin_vel = 30.0
            tracking_ang_vel = 5.0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 6.0
            torques = -1e-3
            dof_vel = 0.0
            dof_acc = -1e-6
            dof_power = 0.0
            base_height = 0.0
            correct_base_height = 20.0
            feet_air_time = 0.0
            collision = 0.0
            feet_stumble = 0.0
            action_rate = -0.2
            action_smoothness = 0.0
            stand_still = 0.0
            dof_pos_limits = -10.0
            feet_regulation = 0.0
            hip_to_default = 0.0
            similar_to_default = 12.0
            straight_path = 0.0
            straight_path_deviation = 0.0

            jump_flight = 100.0
            jump_pattern = 2.0
            jump_swing_clearance = 0.5
            jump_target_vel = 1.0
            foot_clearance = 0.0
            line_z = 0.8
            dof_hip_pos = -0.3
            land_pos = 3.0
            jump_takeoff_vel = 0.0
            jump_apex_height = 0.0
            jump_land_target = 0.0
            jump_land_compact = 0.0
            jump_phase_contact = 0.0

            task_pos = 1500.0
            task_ori = 1500.0
            task_max_height = 5000.0


class CurriculumJUMPCfgMoECTS(JUMPCfgMoECTS):
    class runner(JUMPCfgMoECTS.runner):
        run_name = 'curriculum_jump'
        experiment_name = 'curriculum_jump_moe_cts'
        max_iterations = 40000
        save_interval = 1000
