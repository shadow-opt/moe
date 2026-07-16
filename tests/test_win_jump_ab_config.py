import copy
import unittest
from types import SimpleNamespace

import isaacgym  # Import before torch-backed environment modules.
import torch

from legged_gym.envs import task_registry
from legged_gym.envs.go2.go2_config import GO2Cfg
from legged_gym.envs.nocv.win_jump_cfg import (
    WINJumpCfg,
    WINJumpCfgMoECTS,
    WINJumpCyclicCfg,
    WINJumpCyclicCfgMoECTS,
    WINJumpCyclicScratchCfg,
    WINJumpCyclicScratchCfgMoECTS,
)
from legged_gym.envs.nocv.win_jump_env import MODE_JUMP, MODE_WALK, WINJumpRobot
from legged_gym.scripts.evaluate_z2_jump import SUPPORTED_TASKS
from legged_gym.utils.helpers import class_to_dict


class WinJumpABConfigTests(unittest.TestCase):
    def test_environment_configs_only_differ_in_phase_mode(self):
        legacy = copy.deepcopy(class_to_dict(WINJumpCfg()))
        cyclic = copy.deepcopy(class_to_dict(WINJumpCyclicCfg()))

        self.assertFalse(legacy["commands"].pop("cyclic_jump_phase"))
        self.assertTrue(cyclic["commands"].pop("cyclic_jump_phase"))
        self.assertEqual(legacy, cyclic)

    def test_training_configs_only_differ_in_log_identity(self):
        legacy = copy.deepcopy(class_to_dict(WINJumpCfgMoECTS()))
        cyclic = copy.deepcopy(class_to_dict(WINJumpCyclicCfgMoECTS()))

        self.assertEqual(legacy["runner"].pop("run_name"), "jump_posttrain")
        self.assertEqual(
            cyclic["runner"].pop("run_name"),
            "jump_posttrain_cyclic_phase",
        )
        self.assertEqual(
            legacy["runner"].pop("experiment_name"),
            "win_jump_moe_cts",
        )
        self.assertEqual(
            cyclic["runner"].pop("experiment_name"),
            "win_jump_cyclic_moe_cts",
        )
        self.assertEqual(legacy, cyclic)

    def test_tasks_share_environment_class_and_evaluator_support(self):
        self.assertIs(
            task_registry.get_task_class("win_jump_moe_cts"),
            task_registry.get_task_class("win_jump_cyclic_moe_cts"),
        )
        self.assertIs(
            task_registry.get_task_class("win_jump_cyclic_moe_cts"),
            task_registry.get_task_class("win_jump_cyclic_scratch_moe_cts"),
        )
        self.assertEqual(
            SUPPORTED_TASKS,
            {"win_jump_moe_cts", "win_jump_cyclic_moe_cts"},
        )

    def test_scratch_environment_has_only_intended_differences(self):
        cyclic = copy.deepcopy(class_to_dict(WINJumpCyclicCfg()))
        scratch = copy.deepcopy(class_to_dict(WINJumpCyclicScratchCfg()))

        self.assertEqual(scratch["commands"]["low_height_command_prob"], 0.0)
        self.assertEqual(scratch["commands"]["low_height_terrain_ids"], [])
        self.assertEqual(
            scratch["rewards"]["curriculum_rewards"],
            GO2Cfg.rewards.curriculum_rewards,
        )
        self.assertEqual(scratch["rewards"]["scales"]["lin_vel_z"], -2.0)
        self.assertEqual(scratch["rewards"]["scales"]["correct_base_height"], -1.0)
        self.assertEqual(scratch["rewards"]["scales"]["jump_lin_vel_z"], 0.05)

        for key in ("low_height_command_prob", "low_height_terrain_ids"):
            scratch["commands"][key] = cyclic["commands"][key]
        scratch["rewards"]["curriculum_rewards"] = cyclic["rewards"]["curriculum_rewards"]
        for key in ("lin_vel_z", "correct_base_height"):
            scratch["rewards"]["scales"][key] = cyclic["rewards"]["scales"][key]
        self.assertEqual(cyclic, scratch)

    def test_scratch_training_has_only_intended_differences(self):
        cyclic = copy.deepcopy(class_to_dict(WINJumpCyclicCfgMoECTS()))
        scratch = copy.deepcopy(class_to_dict(WINJumpCyclicScratchCfgMoECTS()))

        expected_algorithm = {
            "learning_rate": 1.0e-3,
            "min_learning_rate": 1.0e-5,
            "max_learning_rate": 1.0e-3,
            "student_encoder_learning_rate": 1.0e-3,
            "walk_behavior_coef": 0.0,
        }
        for key, value in expected_algorithm.items():
            self.assertEqual(scratch["algorithm"][key], value)
            scratch["algorithm"][key] = cyclic["algorithm"][key]

        self.assertEqual(scratch["runner"]["max_iterations"], 20000)
        self.assertEqual(scratch["runner"]["save_interval"], 4000)
        self.assertTrue(scratch["runner"]["save_initial_checkpoint"])
        self.assertTrue(scratch["runner"]["exact_save_intervals"])
        for key in ("run_name", "experiment_name"):
            scratch["runner"][key] = cyclic["runner"][key]
        self.assertEqual(cyclic, scratch)

    def test_go2_reward_curriculum_boundaries(self):
        robot = object.__new__(WINJumpRobot)
        robot.num_steps_per_env = 24
        configs = {
            config["reward_name"]: config
            for config in WINJumpCyclicScratchCfg.rewards.curriculum_rewards
        }

        for iteration, expected in ((0, 1.0), (750, 0.5), (1500, 0.0)):
            robot.common_step_counter = iteration * robot.num_steps_per_env
            self.assertAlmostEqual(robot.get_current_scale(configs["lin_vel_z"]), expected)
        for iteration, expected in ((0, 1.0), (2500, 5.5), (5000, 10.0)):
            robot.common_step_counter = iteration * robot.num_steps_per_env
            self.assertAlmostEqual(
                robot.get_current_scale(configs["correct_base_height"]),
                expected,
            )

    def test_reward_curriculum_scales_walk_without_changing_jump(self):
        robot = object.__new__(WINJumpRobot)
        robot.rew_buf = torch.zeros(2)
        robot.active_mode = torch.tensor([MODE_WALK, MODE_JUMP])
        robot.reward_names = ["tracking_lin_vel", "jump_tracking_lin_vel"]
        robot.reward_functions = [lambda: torch.ones(2), lambda: torch.ones(2)]
        robot.reward_scales = {"tracking_lin_vel": 1.0, "jump_tracking_lin_vel": 1.0}
        robot.reward_curriculum_scales = {"tracking_lin_vel": 2.0}
        robot.episode_sums = {name: torch.zeros(2) for name in robot.reward_names}
        robot.cfg = SimpleNamespace(
            rewards=SimpleNamespace(only_positive_rewards=False),
        )

        robot.compute_reward()

        torch.testing.assert_close(robot.rew_buf, torch.tensor([2.0, 1.0]))

        robot.rew_buf.zero_()
        robot.reward_curriculum_scales = {}
        robot.compute_reward()

        torch.testing.assert_close(robot.rew_buf, torch.tensor([1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
