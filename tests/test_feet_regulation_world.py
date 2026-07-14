import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "legged_gym"
    / "envs"
    / "nocv"
    / "reward_utils.py"
)
SPEC = importlib.util.spec_from_file_location("nocv_reward_utils", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_feet_height_uses_world_vertical_clearance():
    base_height = torch.tensor([0.37])
    base_pos = torch.tensor([[0.0, 0.0, 0.37]])
    feet_pos = torch.tensor(
        [[[0.20, 0.10, 0.05], [-0.20, -0.10, 0.02]]]
    )

    height = MODULE.feet_height_from_world_gravity(
        base_height,
        feet_pos,
        base_pos,
        torch.tensor([[0.0, 0.0, -9.81]]),
    )

    torch.testing.assert_close(height, torch.tensor([[0.05, 0.02]]))


def test_feet_height_is_invariant_to_world_xy_rotation():
    base_height = torch.tensor([0.37, 0.37])
    base_pos = torch.tensor([[0.0, 0.0, 0.37], [0.0, 0.0, 0.37]])
    feet_pos = torch.tensor(
        [
            [[0.20, 0.10, 0.04]],
            [[-0.10, 0.20, 0.04]],
        ]
    )

    height = MODULE.feet_height_from_world_gravity(
        base_height,
        feet_pos,
        base_pos,
        torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
    )

    torch.testing.assert_close(height[0], height[1])
