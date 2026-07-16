import unittest
from types import SimpleNamespace

import isaacgym  # Import before torch-backed environment modules.
import torch

from legged_gym.envs import task_registry
from legged_gym.envs.nocv.win_cfg import (
    WINStairVelGuardWorldCfg,
    WINStairVelGuardWorldCfgMoECTS,
)
from legged_gym.envs.nocv.win_env import WINStairVelGuardWorldRobot
from legged_gym.utils.helpers import class_to_dict


class WinStairVelGuardTests(unittest.TestCase):
    def test_task_configuration(self):
        env = class_to_dict(WINStairVelGuardWorldCfg())
        train = class_to_dict(WINStairVelGuardWorldCfgMoECTS())

        self.assertEqual(
            env["terrain"]["terrain_proportions"],
            [0.10, 0.05, 0.05, 0.35, 0.15, 0.10, 0.0, 0.0, 0.20],
        )
        self.assertAlmostEqual(sum(env["terrain"]["terrain_proportions"]), 1.0)
        self.assertEqual(env["commands"]["ranges"]["lin_vel_x"], [-2.0, 2.0])
        self.assertEqual(env["commands"]["ranges"]["lin_vel_y"], [-1.0, 1.0])
        self.assertEqual(env["commands"]["ranges"]["ang_vel_yaw"], [-1.7, 1.7])
        self.assertEqual(env["commands"]["command_range_curriculum"], [])
        self.assertEqual(env["commands"]["zero_command_curriculum"]["end_value"], 0.07)
        self.assertEqual(env["commands"]["full_stop_command_curriculum"]["end_value"], 0.03)
        self.assertEqual(env["rewards"]["scales"]["feet_regulation"], 0.0)
        self.assertEqual(env["rewards"]["scales"]["feet_regulation_world"], -0.05)
        self.assertEqual(env["rewards"]["scales"]["dof_vel_limits"], -5.0)
        self.assertEqual(
            env["rewards"]["curriculum_rewards"],
            [{
                "reward_name": "dof_vel_limits",
                "start_iter": 0,
                "end_iter": 2000,
                "start_value": 0.2,
                "end_value": 1.0,
            }],
        )

        self.assertEqual(train["algorithm"]["learning_rate"], 3e-4)
        self.assertEqual(train["algorithm"]["min_learning_rate"], 3e-5)
        self.assertEqual(train["algorithm"]["max_learning_rate"], 1e-3)
        self.assertEqual(train["algorithm"]["walk_behavior_coef"], 0.0)
        self.assertEqual(train["runner"]["max_iterations"], 5000)
        self.assertEqual(train["runner"]["save_interval"], 1000)
        self.assertTrue(train["runner"]["save_initial_checkpoint"])
        self.assertTrue(train["runner"]["exact_save_intervals"])
        self.assertIs(
            task_registry.get_task_class("win_stair_vel_guard_world_moe_cts"),
            WINStairVelGuardWorldRobot,
        )

    def test_stairs_up_sampler_is_half_pure_forward(self):
        torch.manual_seed(0)
        count = 10000
        robot = object.__new__(WINStairVelGuardWorldRobot)
        robot.device = "cpu"
        robot.cfg = SimpleNamespace(
            commands=SimpleNamespace(
                stairs_up_forward_command_prob=0.5,
                stairs_up_forward_lin_vel_x=[0.4, 0.9],
            )
        )
        robot.terrain_ids = torch.full((count,), 3, dtype=torch.long)
        robot.commands = torch.zeros(count, 7)
        robot.commands[:, :3] = torch.tensor([0.1, 0.2, 0.3])
        robot.commands[:1000, :3] = torch.tensor([0.0, 0.0, 0.3])
        robot.commands_xy_accumulation = torch.zeros(count, 2)

        robot._apply_stairs_up_forward_commands(torch.arange(count))

        pure_forward = (robot.commands[:, 1] == 0.0) & (robot.commands[:, 2] == 0.0)
        ratio = pure_forward.float().mean().item()
        self.assertGreater(ratio, 0.48)
        self.assertLess(ratio, 0.52)
        self.assertTrue(torch.all(robot.commands[pure_forward, 0] >= 0.4))
        self.assertTrue(torch.all(robot.commands[pure_forward, 0] <= 0.9))
        self.assertTrue(torch.all(robot.commands[:1000, :3] == torch.tensor([0.0, 0.0, 0.3])))
        untouched_translating = (~pure_forward) & (torch.arange(count) >= 1000)
        self.assertTrue(
            torch.all(
                robot.commands[untouched_translating, :3]
                == torch.tensor([0.1, 0.2, 0.3])
            )
        )

    def test_velocity_barrier_starts_at_eighty_percent_and_does_not_saturate(self):
        robot = object.__new__(WINStairVelGuardWorldRobot)
        robot.cfg = SimpleNamespace(
            rewards=SimpleNamespace(
                soft_dof_vel_limit=0.8,
                velocity_limit_barrier_curvature=5.0,
            )
        )
        robot.dof_vel_limits = torch.full((1,), 15.0)
        robot.dof_vel = torch.tensor([[11.9], [12.0], [12.1], [12.5], [15.0], [16.0]])

        penalty = robot._reward_dof_vel_limits()

        torch.testing.assert_close(penalty[:2], torch.zeros(2))
        self.assertGreater(penalty[2].item(), 0.0)
        self.assertTrue(torch.all(penalty[2:] > penalty[1:-1]))
        expected_at_limit = 0.2 + 5.0 * 0.2 ** 2
        self.assertAlmostEqual(penalty[4].item(), expected_at_limit, places=6)
        self.assertGreater(penalty[5].item(), penalty[4].item())

    def test_velocity_curriculum_boundaries(self):
        robot = object.__new__(WINStairVelGuardWorldRobot)
        robot.num_steps_per_env = 24
        config = WINStairVelGuardWorldCfg.rewards.curriculum_rewards[0]
        for iteration, expected in ((0, 0.2), (1000, 0.6), (2000, 1.0), (5000, 1.0)):
            robot.common_step_counter = iteration * robot.num_steps_per_env
            self.assertAlmostEqual(robot.get_current_scale(config), expected)


if __name__ == "__main__":
    unittest.main()
