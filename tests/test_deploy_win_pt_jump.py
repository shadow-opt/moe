import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "deploy" / "deploy_mujoco" / "deploy_win_pt.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("deploy_win_pt", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_y_toggle_command_enters_and_leaves_jump_mode():
    command = np.zeros(6, dtype=np.float32)
    MODULE.toggle_jump_command(command, default_vx=0.5)
    np.testing.assert_array_equal(command, [0.5, 0.0, 0.0, -1.0, 0.0, 0.0])

    MODULE.toggle_jump_command(command, default_vx=0.5)
    np.testing.assert_array_equal(command, [0.5, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_jump_manager_emits_training_command_after_ten_prep_ticks():
    manager = MODULE.JumpCommandManager(
        policy_dt=0.02,
        config={"default_vx": 0.5, "cycle_time": 1.5, "prep_steps": 10},
    )
    request = np.asarray([0.0, 0.4, 0.5, -1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(9):
        np.testing.assert_array_equal(manager.update(request, [True] * 4), np.zeros(6))

    command = manager.update(request, [True] * 4)
    np.testing.assert_allclose(command, [0.5, 0.0, 0.0, -1.0, 0.0, 1.0], atol=1e-7)


def test_jump_config_uses_direct_48d_policy_without_manifest():
    config_path = ROOT / "deploy" / "deploy_mujoco" / "configs" / "win_jump.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["num_obs"] == 48
    assert len(config["cmd_init"]) == 6
    assert config["jump_control"]["enabled"] is True
    assert "policy_path" in config
    assert "manifest" not in config
    assert "sha256" not in config
