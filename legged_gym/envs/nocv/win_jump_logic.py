import torch


MODE_WALK = 0
MODE_JUMP = 1
MODE_STAND = 2


def classify_mode_draws(draws, jump_probability, walk_probability):
    """Map uniform draws to the jump/walk/stand top-level categories."""
    if jump_probability < 0.0 or walk_probability < 0.0:
        raise ValueError("Mode probabilities must be non-negative")
    if jump_probability + walk_probability > 1.0:
        raise ValueError("Jump and walk probabilities may not exceed one")
    modes = torch.full_like(draws, MODE_STAND, dtype=torch.long)
    modes[draws < jump_probability] = MODE_JUMP
    walk = (draws >= jump_probability) & (
        draws < jump_probability + walk_probability
    )
    modes[walk] = MODE_WALK
    return modes
