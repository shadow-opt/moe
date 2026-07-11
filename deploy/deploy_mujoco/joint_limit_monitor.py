from dataclasses import dataclass
from enum import IntEnum

import numpy as np


def prepare_velocity_limits(configured_limits, source_joint_names, target_joint_names):
    if configured_limits is None:
        return None, "LIMIT_MONITOR_WARNING metric=velocity reason=missing_joint_velocity_limits"
    try:
        values = np.asarray(configured_limits, dtype=np.float64)
    except (TypeError, ValueError):
        return None, "LIMIT_MONITOR_WARNING metric=velocity reason=invalid_values"
    expected_shape = (len(source_joint_names),)
    if values.shape != expected_shape:
        return None, (
            "LIMIT_MONITOR_WARNING metric=velocity reason=shape_mismatch "
            f"expected={expected_shape} actual={values.shape}"
        )
    source_indices = {name: index for index, name in enumerate(source_joint_names)}
    return values[[source_indices[name] for name in target_joint_names]], None


class LimitState(IntEnum):
    NORMAL = 0
    WARN = 1
    CRITICAL = 2


@dataclass(frozen=True)
class LimitEvent:
    joint_index: int
    metric: str
    previous_state: LimitState
    state: LimitState
    value: float
    limit: float
    utilization: float


class JointLimitMonitor:
    METRICS = ("position", "velocity", "tau_raw", "tau_applied")

    def __init__(
        self,
        joint_names,
        position_ranges,
        velocity_limits,
        torque_ranges,
        warning_threshold=0.90,
        critical_threshold=0.95,
        hysteresis=0.02,
    ):
        self.joint_names = tuple(joint_names)
        self.num_joints = len(self.joint_names)
        self.warning_threshold = float(warning_threshold)
        self.critical_threshold = float(critical_threshold)
        self.hysteresis = float(hysteresis)
        if not 0.0 < self.warning_threshold < self.critical_threshold:
            raise ValueError("Expected 0 < warning_threshold < critical_threshold")
        if not 0.0 <= self.hysteresis < self.warning_threshold:
            raise ValueError("Expected 0 <= hysteresis < warning_threshold")

        self.position_ranges = self._ranges_or_nan(position_ranges)
        self.velocity_limits = self._limits_or_nan(velocity_limits)
        self.torque_ranges = self._ranges_or_nan(torque_ranges)
        self.states = {
            metric: np.full(self.num_joints, LimitState.NORMAL, dtype=np.int8)
            for metric in self.METRICS
        }

    def _ranges_or_nan(self, ranges):
        if ranges is None:
            return np.full((self.num_joints, 2), np.nan, dtype=np.float64)
        values = np.asarray(ranges, dtype=np.float64)
        if values.shape != (self.num_joints, 2):
            raise ValueError(
                f"Expected ranges shape {(self.num_joints, 2)}, got {values.shape}"
            )
        return values.copy()

    def _limits_or_nan(self, limits):
        if limits is None:
            return np.full(self.num_joints, np.nan, dtype=np.float64)
        values = np.asarray(limits, dtype=np.float64)
        if values.shape != (self.num_joints,):
            raise ValueError(
                f"Expected limits shape {(self.num_joints,)}, got {values.shape}"
            )
        return values.copy()

    def configuration_warnings(self):
        warnings = []
        valid_position = (
            np.isfinite(self.position_ranges).all(axis=1)
            & (self.position_ranges[:, 1] > self.position_ranges[:, 0])
        )
        valid_velocity = np.isfinite(self.velocity_limits) & (self.velocity_limits > 0.0)
        valid_torque = (
            np.isfinite(self.torque_ranges).all(axis=1)
            & (self.torque_ranges[:, 0] < 0.0)
            & (self.torque_ranges[:, 1] > 0.0)
        )
        for metric, valid in (
            ("position", valid_position),
            ("velocity", valid_velocity),
            ("torque", valid_torque),
        ):
            missing = [name for name, enabled in zip(self.joint_names, valid) if not enabled]
            if missing:
                warnings.append(
                    f"LIMIT_MONITOR_WARNING metric={metric} disabled_joints={','.join(missing)}"
                )
        return warnings

    def startup_summary(self):
        enabled = {
            "position": int(
                np.sum(
                    np.isfinite(self.position_ranges).all(axis=1)
                    & (self.position_ranges[:, 1] > self.position_ranges[:, 0])
                )
            ),
            "velocity": int(
                np.sum(np.isfinite(self.velocity_limits) & (self.velocity_limits > 0.0))
            ),
            "torque": int(
                np.sum(
                    np.isfinite(self.torque_ranges).all(axis=1)
                    & (self.torque_ranges[:, 0] < 0.0)
                    & (self.torque_ranges[:, 1] > 0.0)
                )
            ),
        }
        return (
            "LIMIT_MONITOR_READY "
            f"warning={self.warning_threshold:.2f} critical={self.critical_threshold:.2f} "
            f"hysteresis={self.hysteresis:.2f} "
            f"enabled=position:{enabled['position']}/{self.num_joints},"
            f"velocity:{enabled['velocity']}/{self.num_joints},"
            f"torque:{enabled['torque']}/{self.num_joints}"
        )

    def update(self, sim_time, positions, velocities, tau_raw, tau_applied):
        values = {
            "position": self._vector(positions, "positions"),
            "velocity": self._vector(velocities, "velocities"),
            "tau_raw": self._vector(tau_raw, "tau_raw"),
            "tau_applied": self._vector(tau_applied, "tau_applied"),
        }
        utilizations, limits = self._compute_utilizations(values)
        events = []
        for metric in self.METRICS:
            for joint_index in range(self.num_joints):
                utilization = utilizations[metric][joint_index]
                if not np.isfinite(utilization):
                    continue
                previous_state = LimitState(self.states[metric][joint_index])
                state = self._next_state(previous_state, utilization)
                if state == previous_state:
                    continue
                self.states[metric][joint_index] = state
                events.append(
                    LimitEvent(
                        joint_index=joint_index,
                        metric=metric,
                        previous_state=previous_state,
                        state=state,
                        value=float(values[metric][joint_index]),
                        limit=float(limits[metric][joint_index]),
                        utilization=float(utilization),
                    )
                )
        if not events:
            return None
        return self._format_event(float(sim_time), events, values, utilizations)

    def _vector(self, values, label):
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (self.num_joints,):
            raise ValueError(
                f"Expected {label} shape {(self.num_joints,)}, got {vector.shape}"
            )
        return vector

    def _compute_utilizations(self, values):
        lower = self.position_ranges[:, 0]
        upper = self.position_ranges[:, 1]
        half_range = 0.5 * (upper - lower)
        midpoint = 0.5 * (upper + lower)
        position_utilization = np.divide(
            np.abs(values["position"] - midpoint),
            half_range,
            out=np.full(self.num_joints, np.nan),
            where=np.isfinite(half_range) & (half_range > 0.0),
        )
        position_limit = np.where(values["position"] >= midpoint, upper, lower)

        velocity_utilization = np.divide(
            np.abs(values["velocity"]),
            self.velocity_limits,
            out=np.full(self.num_joints, np.nan),
            where=np.isfinite(self.velocity_limits) & (self.velocity_limits > 0.0),
        )
        velocity_limit = np.copysign(self.velocity_limits, values["velocity"])

        raw_utilization, raw_limit = self._torque_utilization(values["tau_raw"])
        applied_utilization, applied_limit = self._torque_utilization(values["tau_applied"])
        return (
            {
                "position": position_utilization,
                "velocity": velocity_utilization,
                "tau_raw": raw_utilization,
                "tau_applied": applied_utilization,
            },
            {
                "position": position_limit,
                "velocity": velocity_limit,
                "tau_raw": raw_limit,
                "tau_applied": applied_limit,
            },
        )

    def _torque_utilization(self, torque):
        lower = self.torque_ranges[:, 0]
        upper = self.torque_ranges[:, 1]
        signed_limit = np.where(torque >= 0.0, upper, lower)
        magnitude_limit = np.abs(signed_limit)
        utilization = np.divide(
            np.abs(torque),
            magnitude_limit,
            out=np.full(self.num_joints, np.nan),
            where=np.isfinite(magnitude_limit) & (magnitude_limit > 0.0),
        )
        return utilization, signed_limit

    def _next_state(self, previous_state, utilization):
        if previous_state == LimitState.NORMAL:
            if utilization >= self.critical_threshold:
                return LimitState.CRITICAL
            if utilization >= self.warning_threshold:
                return LimitState.WARN
            return LimitState.NORMAL
        if previous_state == LimitState.WARN:
            if utilization >= self.critical_threshold:
                return LimitState.CRITICAL
            if utilization < self.warning_threshold - self.hysteresis:
                return LimitState.NORMAL
            return LimitState.WARN
        if utilization >= self.critical_threshold - self.hysteresis:
            return LimitState.CRITICAL
        if utilization >= self.warning_threshold - self.hysteresis:
            return LimitState.WARN
        return LimitState.NORMAL

    def _format_event(self, sim_time, events, values, utilizations):
        changes = []
        for event in events:
            fields = [
                f"joint={self.joint_names[event.joint_index]}",
                f"metric={event.metric}",
                f"state={event.previous_state.name}->{event.state.name}",
                f"value={event.value:.4f}",
                f"limit={event.limit:.4f}",
                f"utilization={event.utilization:.3f}",
            ]
            if event.metric in ("tau_raw", "tau_applied"):
                index = event.joint_index
                raw_utilization = utilizations["tau_raw"][index]
                applied_utilization = utilizations["tau_applied"][index]
                same_direction = (
                    values["tau_raw"][index] == 0.0
                    or values["tau_applied"][index] == 0.0
                    or np.sign(values["tau_raw"][index]) == np.sign(values["tau_applied"][index])
                )
                saturated = (
                    same_direction
                    and np.isfinite(raw_utilization)
                    and np.isfinite(applied_utilization)
                    and raw_utilization >= 1.0
                    and applied_utilization >= 0.99
                )
                fields.extend(
                    [
                        f"tau_raw={values['tau_raw'][index]:.4f}",
                        f"tau_raw_utilization={raw_utilization:.3f}",
                        f"tau_applied={values['tau_applied'][index]:.4f}",
                        f"tau_applied_utilization={applied_utilization:.3f}",
                        f"saturated={str(saturated).lower()}",
                    ]
                )
            changes.append("{" + " ".join(fields) + "}")
        return f"LIMIT_EVENT sim_time={sim_time:.6f} changes=[{' ; '.join(changes)}]"
