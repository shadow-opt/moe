import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOGIC = _load(
    "win_jump_logic",
    ROOT / "legged_gym" / "envs" / "nocv" / "win_jump_logic.py",
)
TORQUE = _load(
    "torque_utils",
    ROOT / "legged_gym" / "utils" / "torque_utils.py",
)


def test_top_level_sampler_has_exact_interval_boundaries():
    draws = torch.tensor([0.0, 0.599999, 0.6, 0.899999, 0.9, 0.999999])
    modes = LOGIC.classify_mode_draws(draws, 0.6, 0.3)
    assert modes.tolist() == [
        LOGIC.MODE_JUMP,
        LOGIC.MODE_JUMP,
        LOGIC.MODE_WALK,
        LOGIC.MODE_WALK,
        LOGIC.MODE_STAND,
        LOGIC.MODE_STAND,
    ]


def test_top_level_sampler_converges_to_60_30_10():
    generator = torch.Generator().manual_seed(7)
    modes = LOGIC.classify_mode_draws(torch.rand(200000, generator=generator), 0.6, 0.3)
    ratios = torch.bincount(modes, minlength=3).float() / modes.numel()
    torch.testing.assert_close(ratios, torch.tensor([0.3, 0.6, 0.1]), atol=0.005, rtol=0.0)


def test_motor_strength_is_clipped_after_randomization():
    torques = torch.tensor([[30.0, -30.0, 10.0]])
    strengths = torch.tensor([[1.2, 1.2, 0.8]])
    limits = torch.tensor([30.0, 30.0, 30.0])
    randomized, applied, clipped = TORQUE.apply_motor_strength_and_limit(
        torques, strengths, limits, True
    )
    torch.testing.assert_close(randomized, torch.tensor([[36.0, -36.0, 8.0]]))
    torch.testing.assert_close(applied, torch.tensor([[30.0, -30.0, 8.0]]))
    assert clipped.tolist() == [[True, True, False]]
