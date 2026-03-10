
import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        # 这一层 buffer 属于“所有任务共通的 RL 接口壳”，
        # 它们先于具体机器人环境被创建，后续由子类在 step/reset/reward/obs 阶段持续写入。
        # 可以按用途分成两类：
        # 1. 发给算法的接口 buffer：`obs_buf` / `privileged_obs_buf` / `rew_buf` / `reset_buf`
        # 2. episode 调度 buffer：`episode_length_buf` / `time_out_buf`
        # 真正的机器人物理状态（例如 dof_pos / root_states）并不在这里，
        # 而是在 `LeggedRobot._init_buffers()` 中从 simulator 取回后再包装。
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        # `obs_buf`: actor 每一步真正看到的观测，shape = [num_envs, num_obs]。
        # 它会在环境的 `compute_observations()` 中被整块重写。
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # `rew_buf`: 当前这一步的总 reward，shape = [num_envs]。
        # 算法只拿它做本步学习信号，不保存长期累计；长期统计通常在 `episode_sums` 中做。
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        # `reset_buf`: 哪些 env 需要在本 step 结束后 reset。
        # 约定上 1/True 表示“需要重置”，0/False 表示“继续 rollout”。
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # `episode_length_buf`: 每个 env 当前 episode 已经走了多少个 policy step。
        # 不是物理子步数，而是外层环境 step 数。
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # `time_out_buf`: timeout 专用终止标记。
        # 常见用法是把“时间到自然结束”和“碰撞/摔倒失败结束”区分开，
        # 以便算法对 timeout 不施加 termination penalty。
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
            # `privileged_obs_buf`: asymmetric training 场景下只给 critic/teacher 的额外观测。
            # 若为 None，通常意味着 actor 和 critic 共用同一份 observation。
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    def get_observations(self):
        return self.obs_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        # `reset()` 是给训练器/播放脚本用的整批 reset 快捷入口。
        # 它内部仍然走 `reset_idx()` + `step(zero_action)`，
        # 这样可以确保 reset 后所有派生 buffer（obs、reward、历史状态）都处于一致状态。
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)