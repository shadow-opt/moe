import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "legged_gym" / "utils" / "thin_wall.py"
SPEC = importlib.util.spec_from_file_location("thin_wall_geometry", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeTerrain:
    horizontal_scale = 0.025
    vertical_scale = 0.005
    width = 160
    length = 64

    def __init__(self):
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)


def test_all_wall_curriculum_heights_are_exact_and_isolated():
    for wall_height in (0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35):
        terrain = MODULE.thin_wall_terrain(FakeTerrain(), wall_height)
        nonzero = np.argwhere(terrain.height_field_raw > 0)
        x_pixels = np.unique(nonzero[:, 0])

        assert len(x_pixels) == 2
        assert np.isclose(len(x_pixels) * terrain.horizontal_scale, 0.05)
        assert np.isclose((x_pixels[0] - terrain.width // 2) * terrain.horizontal_scale, 0.425)
        assert nonzero[:, 1].min() == 0
        assert nonzero[:, 1].max() == terrain.length - 1
        assert np.all(terrain.height_field_raw[x_pixels, :] == round(wall_height / terrain.vertical_scale))
        assert np.all(terrain.height_field_raw[:x_pixels.min(), :] == 0)
        assert np.all(terrain.height_field_raw[x_pixels.max() + 1:, :] == 0)
