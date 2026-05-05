from collections import defaultdict
import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.spacing = cfg.terrain_spacing
        self.spacing_pixels = int(self.spacing / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels + max(0, cfg.num_cols-1) * self.spacing_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels + max(0, cfg.num_rows-1) * self.spacing_pixels) + 2 * self.border
        self.name2cols = defaultdict(set)  # terrain type to column index
        self.cols2id = []  # column index to terrain id

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)
            self.name2cols[terrain.terrain_name].add(j)
            self.cols2id.append(terrain.terrain_id)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        IS_HARD = True
        if IS_HARD:
            # hard
            slope = 0.1 + difficulty * 0.52  # max: 29.6 degrees
            step_height = 0.05 + 0.23 * difficulty  # max: 0.257 m
            discrete_obstacles_height = 0.05 + difficulty * 0.25  # max: 0.275 m
        else:
            # default (easy)
            slope = difficulty * 0.4  # max: 19.8 degrees
            step_height = 0.05 + 0.18 * difficulty  # max: 0.212 m
            discrete_obstacles_height = 0.05 + difficulty * 0.2  # max: 0.23 m

        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 0.1 + 0.07 * difficulty
        # pit_depth = 1. * difficulty  # 删除坑洞
        amplitude = 0.1 + 0.2 * difficulty
        
        if choice < self.proportions[0]:
            terrain.terrain_name = "wave"
            terrain.terrain_id = 0
            terrain_utils.wave_terrain(terrain, num_waves=5, amplitude=amplitude)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[1]:  # 平滑坡
            terrain.terrain_name = "slope"
            terrain.terrain_id = 1
            if choice < (self.proportions[0] + self.proportions[1])/ 2:  # 一半正坡, 一半负坡
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[2]:  # 粗糙坡
            terrain.terrain_name = "rough_slope"
            terrain.terrain_id = 2
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[4]:  # 下楼梯
            terrain.terrain_name = "stairs_down"
            terrain.terrain_id = 4
            if choice<self.proportions[3]:  # 上楼梯
                terrain.terrain_name = "stairs_up"
                terrain.terrain_id = 3
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[5]:  # 障碍物
            terrain.terrain_name = "obstacles"
            terrain.terrain_id = 5
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[6]:
            terrain.terrain_name = "stones"
            terrain.terrain_id = 6
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05 - 0.05 * difficulty, max_height=0.05 + 0.05 * difficulty, step=0.005, downsampled_scale=0.2)
            stone_terrain(terrain, step_width=0.31, step_height=step_height*0.3, platform_size=2.)
            terrain_utils.wave_terrain(terrain, num_waves=5, amplitude=amplitude)
            


            # terrain_utils.random_uniform_terrain(terrain, min_height=-0.05 - 0.05 * difficulty, max_height=0.05 + 0.05 * difficulty, step=0.005, downsampled_scale=0.2)
            # terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[7]:  # 间隙
            terrain.terrain_name = "gap"
            terrain.terrain_id = 7

            bridge_cfg = plank_bridge_curriculum(difficulty, terrain.horizontal_scale)
            plank_bridge_terrain(
                terrain,
                gap_size=bridge_cfg["gap_size"],
                plank_length=bridge_cfg["plank_length"],
                plank_width=bridge_cfg["plank_width"],
                height=0.0,
                pit_depth=bridge_cfg["pit_depth"],
                platform_len=bridge_cfg["platform_len"],
            )


            # gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
            # gap_plus_terrain(terrain, gap_size=gap_size)
        else:  # 平地
            terrain.terrain_name = "flat"
            terrain.terrain_id = 8
            pit_terrain(terrain, depth=0.0, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * (self.length_per_env_pixels + self.spacing_pixels)
        end_x = start_x + self.length_per_env_pixels
        start_y = self.border + j * (self.width_per_env_pixels + self.spacing_pixels)
        end_y = start_y + self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length + i * self.spacing
        env_origin_y = (j + 0.5) * self.env_width + j * self.spacing
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def gap_plus_terrain(terrain, gap_size = 0.1, platform_size=4.0):
    platform_size = 4.5
    # gap_size = 0.15
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = platform_size // 2
    x2 = x1 + gap_size
    y1 = platform_size // 2
    y2 = y1 + gap_size
   
    x3 = x2 + int(0.4/terrain.horizontal_scale) + 1
    x4 = x3 + gap_size

    y3 = y2 + int(0.4/terrain.horizontal_scale) +1
    y4 = y3 + gap_size

    terrain.height_field_raw[center_x-x4 : center_x + x4, center_y-y4 : center_y + y4] = -1000
    terrain.height_field_raw[center_x-x3 : center_x + x3, center_y-y3 : center_y + y3] = 0

    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def stone_terrain(terrain, step_width, step_height, platform_size=2.):
    # def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.):
    # """
    # Generate stairs

    # Parameters:
    #     terrain (terrain): the terrain
    #     step_width (float):  the width of the step [meters]
    #     step_height (float): the step_height [meters]
    #     platform_size (float): size of the flat platform at the center of the terrain [meters]
    # Returns:
    #     terrain (SubTerrain): update terrain
    # """
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length
    height = step_height
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height = -height
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
    return terrain


def plank_bridge_curriculum(difficulty, horizontal_scale):
    """Smooth curriculum for proprioceptive-only policy.

    Difficulty in [0, 1]:
    1) Warmup: keep gap=0 for stable early learning.
    2) Ramp: smoothstep increases gap and gently tightens geometry.
    """
    d = float(np.clip(difficulty, 0.0, 1.0))
    warmup = 0.0
    if d <= warmup:
        progress = 0.2
    else:
        t = (d - warmup) / (1.0 - warmup)
        progress = t * t * (3.0 - 2.0 * t)  # smoothstep

    # Quantized by terrain horizontal resolution (typically 0.1 m).
    raw_gap = 0.20 * progress
    gap_pixels = int(np.round(raw_gap / horizontal_scale))
    gap_size = gap_pixels * horizontal_scale

    return {
        "gap_size": gap_size,                             # 0.0 -> 0.1 -> 0.2
        "plank_length": 0.80 - 0.30 * progress,          # 0.80 -> 0.50
        "plank_width": 4.80, # - 1.20 * progress,            4.8
        "platform_len": 2.40 - 0.40 * progress,          # 2.40 -> 2.00
        "pit_depth": 0.60 + 1.40 * progress,             # 0.60 -> 2.00
    }


def plank_bridge_terrain(terrain, gap_size=0.15, plank_length=0.5, plank_width=1.0, height=0.5, pit_depth=2.0, platform_len=2.0):
    """
    生成木板桥地形，并在中心保留出生平台
    :param terrain: 地形对象
    :param gap_size: 木板间的空隙 [m]
    :param plank_length: 木板长度 [m]
    :param plank_width: 木板宽度 [m]
    :param height: 木板高度 [m]
    :param pit_depth: 缝隙深度 [m]
    :param platform_len: 中心出生平台的长度 [m] (机器人出生在中心，必须留平地)
    """
    # 1. 坐标转换
    gap_pixels = max(0, int(np.round(gap_size / terrain.horizontal_scale)))
    plank_len_pixels = max(1, int(plank_length / terrain.horizontal_scale))
    width_pixels = max(1, int(plank_width / terrain.horizontal_scale))
    height_raw = max(0, int(height / terrain.vertical_scale))
    platform_pixels = max(1, int(platform_len / terrain.horizontal_scale))
    
    # 2. 初始化为地面高度（不悬空）
    pit_depth_raw = int(pit_depth / terrain.vertical_scale)
    terrain.height_field_raw[:, :] = height_raw

    # 3. 计算桥的Y轴范围 (宽度)
    mid_y = terrain.width // 2
    y_start = max(0, mid_y - width_pixels // 2)
    y_end = min(terrain.width, mid_y + width_pixels // 2)

    # 4. 在桥宽范围内按周期挖缝隙（缝隙下沉，木板保持地面高度）
    if gap_pixels > 0:
        current_x = plank_len_pixels
        while current_x < terrain.length:
            gap_end = min(current_x + gap_pixels, terrain.length)
            terrain.height_field_raw[current_x:gap_end, y_start:y_end] = -pit_depth_raw
            current_x += plank_len_pixels + gap_pixels

    # 5. 强制填平中心区域作为出生点
    mid_x = terrain.length // 2
    plat_start = mid_x - platform_pixels // 2
    plat_end = mid_x + platform_pixels // 2
    
    # 边界保护
    plat_start = max(0, plat_start)
    plat_end = min(terrain.length, plat_end)
    
    # 将中心区域强制设为平地高度，覆盖掉可能存在的间隙
    terrain.height_field_raw[plat_start:plat_end, y_start:y_end] = height_raw

    return terrain