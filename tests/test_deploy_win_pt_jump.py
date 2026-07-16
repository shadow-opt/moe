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
    MODULE.toggle_jump_command(command)
    np.testing.assert_array_equal(command, [0.0, 0.0, 0.0, -1.0, 0.0, 0.0])

    MODULE.toggle_jump_command(command)
    np.testing.assert_array_equal(command, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_jump_manager_emits_training_command_after_ten_prep_ticks():
    manager = MODULE.JumpCommandManager(
        policy_dt=0.02,
        config={"cycle_time": 1.5, "prep_steps": 10},
    )
    request = np.asarray([0.5, 0.4, 0.5, -1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(9):
        np.testing.assert_array_equal(manager.update(request, [True] * 4), np.zeros(6))

    command = manager.update(request, [True] * 4)
    np.testing.assert_allclose(command, [0.5, 0.0, 0.0, -1.0, 0.0, 1.0], atol=1e-7)


def test_xbox_y_button_toggles_jump_request():
    class Joystick:
        def __init__(self):
            self.buttons = {3: 1}

        def get_axis(self, _index):
            return 0.0

        def get_button(self, index):
            return self.buttons.get(index, 0)

    class Pygame:
        class event:
            @staticmethod
            def pump():
                pass

    MODULE.pygame = Pygame()
    command = np.zeros(6, dtype=np.float32)
    button_state = {}
    MODULE.update_velocity_command_from_xbox(
        command,
        Joystick(),
        np.asarray([0.8, 0.6, 1.3]),
        button_state,
        jump_control=True,
    )
    np.testing.assert_array_equal(command, [0.0, 0.0, 0.0, -1.0, 0.0, 0.0])


def test_jump_config_uses_direct_48d_policy_without_manifest():
    config_path = ROOT / "deploy" / "deploy_mujoco" / "configs" / "win_jump.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["num_obs"] == 48
    assert len(config["cmd_init"]) == 6
    assert config["jump_control"]["enabled"] is True
    assert "default_vx" not in config["jump_control"]
    assert "policy_path" in config
    assert "manifest" not in config
    assert "sha256" not in config
