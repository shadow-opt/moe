import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "deploy" / "deploy_mujoco" / "deploy_z2_jump.py"
SPEC = importlib.util.spec_from_file_location("deploy_z2_jump", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Z2JumpDeployTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = MODULE.load_config()
        cls.simulation = MODULE.Z2JumpSimulation(cls.config)

    def test_config_references_real_scene_and_torchscript_policies(self):
        self.assertTrue(MODULE.resolve_path(self.config["xml_path"]).is_file())
        self.assertEqual(set(self.config["profiles"]), {"jump", "spring_jump"})
        for profile in self.config["profiles"].values():
            self.assertTrue(MODULE.resolve_path(profile["policy_path"]).is_file())
        self.assertEqual(self.config["policy_interface"]["num_single_obs"], 47)
        self.assertEqual(self.config["policy_interface"]["frame_stack"], 10)

    def test_joint_mapping_swaps_fr_and_rl_without_changing_leg_values(self):
        model_names = ["FL_a", "FL_b", "FR_a", "FR_b", "RL_a", "RL_b", "RR_a", "RR_b"]
        qpos_names = ["FL_a", "FL_b", "RL_a", "RL_b", "FR_a", "FR_b", "RR_a", "RR_b"]
        mapping = MODULE.JointOrderMap(model_names, qpos_names, qpos_names)
        qpos_values = np.asarray([10, 11, 30, 31, 20, 21, 40, 41])

        model_values = mapping.qpos_to_model(qpos_values)

        self.assertEqual(model_values.tolist(), [10, 11, 20, 21, 30, 31, 40, 41])
        self.assertEqual(mapping.model_to_qpos(model_values).tolist(), qpos_values.tolist())
        self.assertEqual(mapping.qpos_to_actuator(qpos_values).tolist(), qpos_values.tolist())

    def test_jump_observation_uses_training_field_order_and_zero_history(self):
        defaults = np.asarray(self.config["profiles"]["jump"]["default_angles"], dtype=np.float32)
        builder = MODULE.ObservationBuilder(self.config, "jump", defaults)
        previous_action = np.arange(12, dtype=np.float32)
        stacked = builder.append(
            defaults,
            np.zeros(12, dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            np.asarray([4.0, 8.0, 12.0]),
            previous_action,
            np.asarray([0.5, -0.25, 1.0]),
            policy_step=0,
        )

        self.assertEqual(stacked.shape, (470,))
        self.assertTrue(np.all(stacked[: 9 * 47] == 0.0))
        frame = stacked[-47:]
        np.testing.assert_allclose(frame[:5], [0.0, 1.0, 1.0, -0.5, 0.25], atol=1e-6)
        np.testing.assert_allclose(frame[5:8], [1.0, 2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(frame[8:35], 0.0, atol=1e-6)
        np.testing.assert_allclose(frame[35:47], previous_action, atol=1e-6)

    def test_spring_observation_uses_fixed_unscaled_displacement(self):
        defaults = np.asarray(
            self.config["profiles"]["spring_jump"]["default_angles"], dtype=np.float32
        )
        builder = MODULE.ObservationBuilder(self.config, "spring_jump", defaults)
        frame = builder.build_single_frame(
            defaults,
            np.zeros(12),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3),
            np.zeros(12),
            np.asarray([0.8, 0.0, 1.0]),
            policy_step=123,
        )

        np.testing.assert_allclose(frame[:5], [0.0, 0.0, 0.8, 0.0, 1.0], atol=1e-6)

    def test_spring_episode_trigger_and_timeout_are_exact(self):
        episode = MODULE.SpringEpisodeController(0.8, preparation_steps=50, episode_steps=250)
        episode.start()

        for _ in range(50):
            np.testing.assert_allclose(episode.command(), [0.8, 0.0, 0.0], atol=1e-7)
            self.assertFalse(episode.advance())
        self.assertEqual(episode.step, 50)
        np.testing.assert_allclose(episode.command(), [0.8, 0.0, 1.0], atol=1e-7)

        for _ in range(199):
            self.assertFalse(episode.advance())
        self.assertEqual(episode.step, 249)
        self.assertTrue(episode.advance())
        self.assertEqual(episode.step, 250)
        self.assertFalse(episode.running)
        np.testing.assert_allclose(episode.command(), [0.8, 0.0, 0.0], atol=1e-7)

    def test_real_model_reset_uses_profile_height_defaults_and_joint_mapping(self):
        for profile_name, expected_height in (("jump", 0.42), ("spring_jump", 0.39)):
            with self.subTest(profile=profile_name):
                self.simulation.reset(profile_name)
                np.testing.assert_allclose(
                    self.simulation.data.qpos[:3], [3.7, -9.0, expected_height], atol=1e-9
                )
                q_model = self.simulation.joint_map.qpos_to_model(
                    self.simulation.data.qpos[self.simulation.qpos_indices]
                )
                np.testing.assert_allclose(
                    q_model, self.config["profiles"][profile_name]["default_angles"], atol=1e-7
                )
                self.assertTrue(all(np.all(frame == 0.0) for frame in self.simulation.builder.history))

    def test_real_policy_short_rollout_is_finite_and_respects_torque_limit(self):
        cases = (("jump", 80, False), ("spring_jump", 240, True))
        for profile_name, steps, start_spring in cases:
            with self.subTest(profile=profile_name):
                self.simulation.reset(profile_name, start_spring=start_spring)
                self.simulation.velocity_command = np.asarray([0.5, 0.0, 0.0], dtype=np.float32)
                events = []
                for _ in range(steps):
                    event = self.simulation.step()
                    if event is not None:
                        events.append(event)
                        break

                self.assertNotIn("fall", events)
                self.assertTrue(np.isfinite(self.simulation.data.qpos).all())
                self.assertTrue(np.isfinite(self.simulation.data.qvel).all())
                self.assertTrue(np.isfinite(self.simulation.data.ctrl).all())
                self.assertTrue(np.isfinite(self.simulation.action_model).all())
                self.assertLessEqual(
                    np.max(np.abs(self.simulation.data.ctrl)),
                    self.config["profiles"][profile_name]["torque_limit"] + 1e-6,
                )

    def test_airborne_metric_ignores_initial_drop_before_first_foot_contact(self):
        self.simulation.reset("jump")
        for _ in range(10):
            self.simulation.step()

        self.assertFalse(self.simulation.metrics["has_ground_contact"])
        self.assertEqual(self.simulation.metrics["airborne_time"], 0.0)

    def test_headless_jump_uses_deterministic_config_command(self):
        self.simulation.reset("jump")

        result = MODULE.run_headless(self.simulation, duration=0.02)

        np.testing.assert_allclose(
            self.simulation.velocity_command,
            self.config["controls"]["config_command"],
            atol=1e-7,
        )
        self.assertEqual(result["events"], [])

    def test_headless_fall_is_reported_without_automatic_reset(self):
        self.simulation.reset("jump")
        self.simulation.data.qpos[2] = 0.10
        self.simulation.mujoco.mj_forward(self.simulation.model, self.simulation.data)

        result = MODULE.run_headless(self.simulation, duration=0.04)

        self.assertEqual(result["events"].count("fall"), 1)
        self.assertGreater(self.simulation.data.time, 0.0)
        self.assertNotAlmostEqual(self.simulation.data.qpos[2], 0.42)


if __name__ == "__main__":
    unittest.main()
