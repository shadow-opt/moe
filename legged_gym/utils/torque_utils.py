import torch


def apply_motor_strength_and_limit(torques, motor_strengths, torque_limits, hard_limit):
    """Apply motor-strength randomization, then optionally enforce hardware limits."""
    randomized = torques * motor_strengths
    clipped = torch.abs(randomized) > torque_limits
    if hard_limit:
        applied = torch.clip(randomized, -torque_limits, torque_limits)
    else:
        applied = randomized
    return randomized, applied, clipped
