import copy
import unittest

import isaacgym  # Import before torch-backed environment modules.

from legged_gym.envs import task_registry
from legged_gym.envs.nocv.win_jump_cfg import (
    WINJumpCfg,
    WINJumpCfgMoECTS,
    WINJumpCyclicCfg,
    WINJumpCyclicCfgMoECTS,
)
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
        self.assertEqual(
            SUPPORTED_TASKS,
            {"win_jump_moe_cts", "win_jump_cyclic_moe_cts"},
        )


if __name__ == "__main__":
    unittest.main()
