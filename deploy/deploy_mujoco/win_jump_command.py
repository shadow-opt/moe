import math

import numpy as np


MODE_WALK = 0
MODE_JUMP = 1
MODE_STAND = 2
MODE_PREP = 3


class JumpCommandFSM:
    """Single-robot deployment FSM matching WINJumpRobot command semantics."""

    def __init__(
        self,
        policy_dt=0.02,
        cycle_time=1.5,
        prep_steps=10,
        landing_steps=3,
        exit_timeout_s=2.0,
        disable_threshold=0.2,
        enable_threshold=0.3,
    ):
        self.policy_dt = float(policy_dt)
        self.cycle_time = float(cycle_time)
        self.prep_steps = int(prep_steps)
        self.landing_steps = int(landing_steps)
        self.exit_timeout_s = float(exit_timeout_s)
        self.disable_threshold = float(disable_threshold)
        self.enable_threshold = float(enable_threshold)
        self.reset()

    def reset(self):
        self.requested_mode = MODE_STAND
        self.active_mode = MODE_STAND
        self.target = np.zeros(4, dtype=np.float32)
        self.latched_vx = 0.0
        self.prep_remaining = 0
        self.phase_steps = 0
        self.motion_enabled = False
        self.exit_pending = False
        self.seen_airborne = False
        self.landing_streak = 0
        self.exit_elapsed = 0.0
        self.timed_out = False
        self.sanitized_count = 0

    def request(self, vx=0.0, vy=0.0, yaw=0.0, body_mode=0.0):
        values = np.asarray([vx, vy, yaw, body_mode], dtype=np.float32)
        if body_mode < -0.5:
            self.requested_mode = MODE_JUMP
            if vy != 0.0 or yaw != 0.0:
                self.sanitized_count += 1
            values[1:3] = 0.0
            values[3] = -1.0
        elif abs(vx) < 0.1 and abs(vy) < 0.1 and abs(yaw) < 0.1 and abs(body_mode) < 0.5:
            self.requested_mode = MODE_STAND
            values[:] = 0.0
        else:
            self.requested_mode = MODE_WALK
            values[3] = 1.0 if body_mode > 0.5 else 0.0
        self.target = values

        if self.requested_mode == MODE_JUMP:
            if self.active_mode == MODE_JUMP:
                self.latched_vx = float(values[0])
                self.exit_pending = False
            else:
                self.active_mode = MODE_PREP
                self.prep_remaining = self.prep_steps
                self.phase_steps = 0
                self.motion_enabled = False
        elif self.active_mode == MODE_JUMP:
            self.exit_pending = True
            self.exit_elapsed = 0.0
            self.landing_streak = 0
        else:
            self.active_mode = self.requested_mode

    def _enter_jump(self):
        self.active_mode = MODE_JUMP
        self.latched_vx = float(self.target[0])
        # step() advances phase later in the same tick; -1 yields initial [0, 1].
        self.phase_steps = -1
        self.motion_enabled = abs(self.latched_vx) >= self.enable_threshold
        self.exit_pending = False
        self.seen_airborne = False
        self.landing_streak = 0
        self.exit_elapsed = 0.0

    def _leave_jump(self):
        self.active_mode = self.requested_mode
        self.phase_steps = 0
        self.motion_enabled = False
        self.exit_pending = False
        self.seen_airborne = False
        self.landing_streak = 0
        self.exit_elapsed = 0.0

    def step(self, foot_contacts):
        contacts = np.asarray(foot_contacts, dtype=bool)
        if contacts.shape != (4,):
            raise ValueError(f"Expected four foot contacts, got {contacts.shape}")

        if self.active_mode == MODE_PREP:
            self.prep_remaining -= 1
            if self.prep_remaining <= 0:
                self._enter_jump()

        if self.active_mode == MODE_JUMP:
            speed = abs(self.latched_vx)
            if speed >= self.enable_threshold:
                self.motion_enabled = True
            elif speed < self.disable_threshold:
                self.motion_enabled = False
            if self.motion_enabled:
                self.phase_steps += 1
            else:
                self.phase_steps = 0
            self.seen_airborne |= self.motion_enabled and not contacts.any()

            if self.exit_pending:
                landed = self.seen_airborne and contacts.all()
                self.landing_streak = self.landing_streak + 1 if landed else 0
                self.exit_elapsed += self.policy_dt
                if self.landing_streak >= self.landing_steps:
                    self._leave_jump()
                elif self.exit_elapsed >= self.exit_timeout_s:
                    self.timed_out = True

        return self.observation_command()

    def observation_command(self):
        if self.active_mode == MODE_PREP:
            return np.zeros(6, dtype=np.float32)
        if self.active_mode != MODE_JUMP:
            command = np.zeros(6, dtype=np.float32)
            command[:4] = self.target
            return command
        phase = self.phase_steps * self.policy_dt / self.cycle_time
        command = np.asarray(
            [
                self.latched_vx,
                0.0,
                0.0,
                -1.0,
                math.sin(2.0 * math.pi * phase),
                math.cos(2.0 * math.pi * phase),
            ],
            dtype=np.float32,
        )
        if not self.motion_enabled:
            command[4:6] = 0.0
        return command
