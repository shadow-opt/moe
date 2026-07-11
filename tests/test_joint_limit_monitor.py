import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "deploy" / "deploy_mujoco" / "joint_limit_monitor.py"
SPEC = importlib.util.spec_from_file_location("joint_limit_monitor", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
JointLimitMonitor = MODULE.JointLimitMonitor
prepare_velocity_limits = MODULE.prepare_velocity_limits


def make_monitor(position_ranges=None, velocity_limits=None, torque_ranges=None):
    return JointLimitMonitor(
        ["joint_a", "joint_b"],
        position_ranges=position_ranges
        if position_ranges is not None
        else [[-0.5, 0.5], [-0.1, 3.2]],
        velocity_limits=velocity_limits if velocity_limits is not None else [15.0, 10.0],
        torque_ranges=torque_ranges
        if torque_ranges is not None
        else [[-30.0, 30.0], [-20.0, 40.0]],
    )


def update(monitor, positions=None, velocities=None, tau_raw=None, tau_applied=None, time=1.0):
    return monitor.update(
        time,
        positions if positions is not None else [0.0, 1.55],
        velocities if velocities is not None else [0.0, 0.0],
        tau_raw if tau_raw is not None else [0.0, 0.0],
        tau_applied if tau_applied is not None else [0.0, 0.0],
    )


def test_position_uses_midpoint_normalization_for_asymmetric_ranges():
    monitor = make_monitor()

    event = update(monitor, positions=[0.45, 3.035])

    assert "joint=joint_a metric=position state=NORMAL->WARN" in event
    assert "joint=joint_b metric=position state=NORMAL->WARN" in event
    assert event.count("utilization=0.900") == 2


def test_all_metrics_use_warning_and_critical_thresholds():
    monitor = make_monitor()

    warning = update(
        monitor,
        positions=[0.45, 1.55],
        velocities=[0.0, -9.0],
        tau_raw=[0.0, 36.0],
        tau_applied=[-27.0, 0.0],
    )
    critical = update(
        monitor,
        positions=[0.475, 1.55],
        velocities=[0.0, -9.5],
        tau_raw=[0.0, 38.0],
        tau_applied=[-28.5, 0.0],
        time=2.0,
    )

    assert warning.count("state=NORMAL->WARN") == 4
    assert critical.count("state=WARN->CRITICAL") == 4


def test_hysteresis_prevents_chatter_and_reports_downgrade_and_recovery():
    monitor = make_monitor()

    update(monitor, velocities=[14.25, 0.0])
    assert update(monitor, velocities=[14.10, 0.0]) is None
    downgrade = update(monitor, velocities=[13.90, 0.0])
    assert "state=CRITICAL->WARN" in downgrade
    assert update(monitor, velocities=[13.30, 0.0]) is None
    recovery = update(monitor, velocities=[13.19, 0.0])
    assert "state=WARN->NORMAL" in recovery


def test_torque_event_contains_raw_applied_values_and_saturation():
    monitor = make_monitor()

    event = update(monitor, tau_raw=[35.0, 0.0], tau_applied=[30.0, 0.0])

    assert event.count("tau_raw=35.0000") == 2
    assert event.count("tau_applied=30.0000") == 2
    assert event.count("saturated=true") == 2
    assert event.startswith("LIMIT_EVENT sim_time=1.000000 changes=[")


def test_missing_limits_disable_only_the_affected_joint_metric():
    monitor = make_monitor(
        position_ranges=[[np.nan, np.nan], [-0.1, 3.2]],
        velocity_limits=[np.nan, 10.0],
        torque_ranges=[[np.nan, np.nan], [-20.0, 40.0]],
    )

    warnings = monitor.configuration_warnings()
    event = update(
        monitor,
        positions=[100.0, 3.035],
        velocities=[100.0, 9.0],
        tau_raw=[100.0, 36.0],
        tau_applied=[100.0, 36.0],
    )

    assert len(warnings) == 3
    assert all("joint_a" in warning for warning in warnings)
    assert "joint=joint_a" not in event
    assert event.count("joint=joint_b") == 4


def test_multiple_changes_are_aggregated_into_one_line():
    monitor = make_monitor()

    event = update(monitor, velocities=[13.5, 9.0])

    assert "\n" not in event
    assert event.count("metric=velocity") == 2
    assert " ; " in event


def test_velocity_limits_are_reordered_by_joint_name():
    limits, warning = prepare_velocity_limits(
        [10.0, 20.0, 30.0],
        ["front", "rear", "middle"],
        ["rear", "middle", "front"],
    )

    assert warning is None
    assert limits.tolist() == [20.0, 30.0, 10.0]


def test_invalid_velocity_limit_shape_disables_velocity_monitoring():
    limits, warning = prepare_velocity_limits(
        [10.0],
        ["front", "rear"],
        ["front", "rear"],
    )

    assert limits is None
    assert "reason=shape_mismatch" in warning
