import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "deploy"
    / "deploy_mujoco"
    / "win_jump_command.py"
)
SPEC = importlib.util.spec_from_file_location("win_jump_command", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_jump_prep_is_ten_ticks_and_phase_starts_at_zero():
    fsm = MODULE.JumpCommandFSM()
    fsm.request(0.5, 0.2, 0.3, -1.0)

    for _ in range(9):
        command = fsm.step([True] * 4)
        np.testing.assert_array_equal(command, np.zeros(6, dtype=np.float32))

    command = fsm.step([True] * 4)
    assert fsm.active_mode == MODULE.MODE_JUMP
    np.testing.assert_allclose(command, [0.5, 0.0, 0.0, -1.0, 0.0, 1.0], atol=1e-7)
    assert fsm.sanitized_count == 1


def test_jump_exit_waits_for_airborne_then_three_contact_ticks():
    fsm = MODULE.JumpCommandFSM()
    fsm.request(0.5, body_mode=-1.0)
    for _ in range(10):
        fsm.step([True] * 4)

    fsm.request(0.4, body_mode=0.0)
    assert fsm.active_mode == MODULE.MODE_JUMP
    fsm.step([False] * 4)
    for _ in range(2):
        fsm.step([True] * 4)
        assert fsm.active_mode == MODULE.MODE_JUMP
    command = fsm.step([True] * 4)
    assert fsm.active_mode == MODULE.MODE_WALK
    np.testing.assert_allclose(command, [0.4, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_jump_velocity_hysteresis_and_timeout():
    fsm = MODULE.JumpCommandFSM(exit_timeout_s=0.04)
    fsm.request(0.5, body_mode=-1.0)
    for _ in range(10):
        fsm.step([True] * 4)
    assert fsm.motion_enabled

    fsm.latched_vx = 0.25
    fsm.step([True] * 4)
    assert fsm.motion_enabled
    fsm.latched_vx = 0.1
    fsm.step([True] * 4)
    assert not fsm.motion_enabled

    fsm.latched_vx = 0.5
    fsm.request(0.0, body_mode=0.0)
    fsm.step([True] * 4)
    fsm.step([True] * 4)
    assert fsm.timed_out
