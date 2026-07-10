import numpy as np


def thin_wall_terrain(terrain, wall_height, wall_thickness=0.05, wall_front_offset=0.425):
    """Add a flat-topped wall spanning the complete width of one terrain tile."""
    height_units = int(np.round(wall_height / terrain.vertical_scale))
    thickness_pixels = max(1, int(np.round(wall_thickness / terrain.horizontal_scale)))
    center_x = terrain.width // 2
    front_x = center_x + int(np.round(wall_front_offset / terrain.horizontal_scale))
    rear_x = min(front_x + thickness_pixels, terrain.width)
    if front_x < 0 or rear_x <= front_x:
        raise ValueError(
            f"thin wall does not fit terrain: front={front_x}, rear={rear_x}, length={terrain.width}"
        )
    terrain.height_field_raw[front_x:rear_x, :] = height_units
    return terrain
