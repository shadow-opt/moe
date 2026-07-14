import torch


def feet_height_from_world_gravity(
    base_height,
    feet_pos_world,
    base_pos_world,
    gravity_world,
):
    """Compute foot clearance using positions and gravity in the world frame."""
    down_world = gravity_world / torch.linalg.vector_norm(
        gravity_world,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-6)
    delta_feet_world = feet_pos_world - base_pos_world.unsqueeze(1)
    feet_to_base_height = torch.sum(
        delta_feet_world * down_world.unsqueeze(1),
        dim=-1,
    )
    return torch.clamp(base_height.unsqueeze(1) - feet_to_base_height, min=0.0)
