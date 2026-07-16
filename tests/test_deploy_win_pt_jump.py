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


def test_jump_manager_switches_immediately_without_contact_gating():
    manager = MODULE.JumpCommandManager(
        policy_dt=0.02,
        config={"cycle_time": 1.5},
    )
    request = np.asarray([0.5, 0.4, 0.5, -1.0, 0.0, 0.0], dtype=np.float32)
    command = manager.update(request)
    np.testing.assert_allclose(command, [0.5, 0.0, 0.0, -1.0, 0.0, 1.0], atol=1e-7)

    command = manager.update([0.2, 0.1, 0.3, 0.0])
    np.testing.assert_allclose(command, [0.2, 0.1, 0.3, 0.0, 0.0, 0.0])


def test_jump_manager_does_not_advance_phase_at_zero_velocity():
    manager = MODULE.JumpCommandManager(0.02, {"cycle_time": 1.5})
    command = manager.update([0.0, 0.0, 0.0, -1.0])
    np.testing.assert_array_equal(command, [0.0, 0.0, 0.0, -1.0, 0.0, 0.0])


def test_jump_manager_inserts_zero_command_pause_between_hops():
    manager = MODULE.JumpCommandManager(
        0.02, {"cycle_time": 0.06, "pause_time": 0.04}
    )
    request = [0.5, 0.0, 0.0, -1.0]
    outputs = [manager.update(request) for _ in range(7)]
    assert [float(output[0]) for output in outputs] == [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5]
    for output in outputs[3:6]:
        np.testing.assert_array_equal(output, [0.0, 0.0, 0.0, -1.0, 0.0, 0.0])


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
    assert "prep_steps" not in config["jump_control"]
    assert "landing_steps" not in config["jump_control"]
    assert "policy_path" in config
    assert "manifest" not in config
    assert "sha256" not in config
