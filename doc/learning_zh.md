# Go2 RL GYM 学习与二次开发手册
> 面向对象：会基本 Python、能读一点代码，但对强化学习、Isaac Gym、四足机器人训练框架还不熟的开发者。  
> 目标：帮你先建立整体认识，再带你顺着“训练 → Play → 导出 → Mujoco → 真机 → 二次开发”这条主线读懂本仓库。



[TOC]


---

## 0. 阅读这份文档前，你应该先知道什么

这个仓库不是一个“只有训练脚本”的小项目，而是一套完整链路：

- 用 **Isaac Gym** 做大规模并行训练；
- 用 **rsl_rl** 提供强化学习算法主体；
- 用 **Mujoco** 做 Sim2Sim 验证；
- 用 **Unitree SDK** 做 Go2 真机部署；
- 在基础 PPO 之外，还实现了 **CTS / MoE / MCP / Dual-MoE** 等扩展算法。

如果你是第一次接触这个仓库，最容易犯的错误不是“代码看不懂”，而是：

1. 没搞清楚哪个目录负责什么；
2. 不知道 `--task` 最终会走到哪套环境和训练配置；
3. 修改了训练端的观测或动作，却忘了同步修改部署端；
4. 直接跑 `play.py`，却没意识到它默认会恢复已有模型；
5. 想二次开发，却不知道应该改配置、改环境、改算法，还是改部署脚本。

这份文档就是为了解决这些问题。

---

## 1. 推荐学习路径

如果你是第一次接触本项目，建议按下面顺序学习：

1. **先看仓库地图**：知道每个目录是干什么的；
2. **再看任务注册**：理解 `--task=go2_moe_cts` 到底映射到哪里；
3. **再看训练入口**：理解一次训练是如何启动的；
4. **再看 Go2 环境配置**：知道 obs、reward、terrain、commands 在哪里；
5. **再看环境主循环**：弄懂 action 如何变成力矩，reward 如何汇总；
6. **再看 Play/导出/部署**：知道为什么训练端和部署端必须保持一致；
7. **最后再动手改代码**：先改 reward，再改 obs，再考虑新增 task 或改算法。

一句话总结：**先搞懂“配置和调用链”，再动“网络和算法”。**

---

## 2. 仓库整体架构

本仓库大体可以拆成四层：

### 2.1 用户入口层

- [README_zh.md](../README_zh.md)
- [README.md](../README.md)
- [doc/setup_zh.md](setup_zh.md)

这一层是给用户看的：怎么装、怎么训、怎么 play、怎么部署。

### 2.2 训练环境层

- [legged_gym/envs](../legged_gym/envs)
- [legged_gym/envs/base](../legged_gym/envs/base)
- [legged_gym/envs/go2](../legged_gym/envs/go2)

这一层定义“机器人在仿真中怎么走”：

- 观测是什么；
- 动作是什么；
- 奖励怎么计算；
- 地形怎么生成；
- 命令怎么采样；
- 什么时候 reset。

### 2.3 算法训练层

- [rsl_rl/rsl_rl/algorithms](../rsl_rl/rsl_rl/algorithms)
- [rsl_rl/rsl_rl/modules](../rsl_rl/rsl_rl/modules)
- [rsl_rl/rsl_rl/runners](../rsl_rl/rsl_rl/runners)

这一层定义“怎么学”：

- 是 PPO 还是 CTS / MoE；
- actor / critic 网络长什么样；
- rollout 怎么收集；
- loss 怎么算；
- checkpoint 怎么保存。

### 2.4 部署与验证层

- [deploy/deploy_mujoco](../deploy/deploy_mujoco)
- [deploy/deploy_real](../deploy/deploy_real)
- [deploy/pre_train/go2](../deploy/pre_train/go2)

这一层定义“训练好的策略怎么拿出来用”：

- 在 Mujoco 里复现实验；
- 在真机上加载 TorchScript 模型；
- 用 YAML 配置 obs / action / joints / 控制频率。

---

## 3. 仓库地图：先知道每个目录负责什么

下面是新手最需要认识的目录。

### 3.1 顶层目录

| 目录/文件 | 作用 |
| --- | --- |
| [README_zh.md](../README_zh.md) | 中文总入口 |
| [doc/setup_zh.md](setup_zh.md) | 安装与环境准备 |
| [setup.py](../setup.py) | 根项目安装依赖 |
| [UPDATE.md](../UPDATE.md) | 历史更新说明 |
| [logs](../logs) | 训练日志与模型输出 |

### 3.2 环境相关目录

| 目录/文件 | 作用 |
| --- | --- |
| [legged_gym/envs/__init__.py](../legged_gym/envs/__init__.py) | 任务注册中心 |
| [legged_gym/envs/base/legged_robot.py](../legged_gym/envs/base/legged_robot.py) | 四足环境主逻辑 |
| [legged_gym/envs/base/legged_robot_config.py](../legged_gym/envs/base/legged_robot_config.py) | 通用配置基类 |
| [legged_gym/envs/go2/go2_env.py](../legged_gym/envs/go2/go2_env.py) | Go2 自定义 obs/reward |
| [legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py) | Go2 环境与训练配置 |

### 3.3 训练脚本目录

| 目录/文件 | 作用 |
| --- | --- |
| [legged_gym/scripts/train.py](../legged_gym/scripts/train.py) | 训练入口 |
| [legged_gym/scripts/play.py](../legged_gym/scripts/play.py) | 可视化 / 导出入口 |
| [legged_gym/utils/task_registry.py](../legged_gym/utils/task_registry.py) | 创建 env 和 runner |
| [legged_gym/utils/helpers.py](../legged_gym/utils/helpers.py) | CLI 参数、resume、配置覆盖 |
| [legged_gym/utils/exporter.py](../legged_gym/utils/exporter.py) | 导出 TorchScript / ONNX / PKL |

### 3.4 算法目录

| 目录/文件 | 作用 |
| --- | --- |
| [rsl_rl/rsl_rl/runners/on_policy_runner.py](../rsl_rl/rsl_rl/runners/on_policy_runner.py) | PPO runner |
| [rsl_rl/rsl_rl/runners/on_policy_runner_cts.py](../rsl_rl/rsl_rl/runners/on_policy_runner_cts.py) | CTS/MoE 系 runner |
| [rsl_rl/rsl_rl/modules](../rsl_rl/rsl_rl/modules) | 策略网络结构 |
| [rsl_rl/rsl_rl/algorithms](../rsl_rl/rsl_rl/algorithms) | 各算法更新逻辑 |

### 3.5 部署目录

| 目录/文件 | 作用 |
| --- | --- |
| [deploy/deploy_mujoco/deploy_go2.py](../deploy/deploy_mujoco/deploy_go2.py) | Mujoco Sim2Sim |
| [deploy/deploy_mujoco/deploy_go2_moe.py](../deploy/deploy_mujoco/deploy_go2_moe.py) | MoE 权重可视化版 |
| [deploy/deploy_mujoco/configs/go2.yaml](../deploy/deploy_mujoco/configs/go2.yaml) | Mujoco 部署配置 |
| [deploy/deploy_real/deploy_real_go2.py](../deploy/deploy_real/deploy_real_go2.py) | 真机 Python 部署 |
| [deploy/deploy_real/config_go2.py](../deploy/deploy_real/config_go2.py) | 真机配置加载器 |
| [deploy/deploy_real/configs/go2.yaml](../deploy/deploy_real/configs/go2.yaml) | 真机部署配置 |

---

## 4. `--task` 到底是怎么找到对应环境的

很多新手第一次看到训练命令：

```bash
python legged_gym/scripts/train.py --task=go2_moe_cts
```

容易以为 `--task` 只是个名字。实际上它是整条训练链路的入口。

### 4.1 注册位置

任务注册在 [legged_gym/envs/__init__.py](../legged_gym/envs/__init__.py)。这里用 `task_registry.register()` 把：

- task 名称；
- 环境类；
- 环境配置；
- 训练配置；

绑定在一起。

当前注册关系可以理解为：

- `go2` → `Go2Robot + GO2Cfg + GO2CfgPPO`
- `go2_cts` → `Go2Robot + GO2Cfg + GO2CfgCTS`
- `go2_moe_cts` → `Go2Robot + GO2Cfg + GO2CfgMoECTS`
- `go2_moe_ng_cts` → `Go2Robot + GO2Cfg + GO2CfgMoENGCTS`
- `go2_mcp_cts` → `Go2Robot + GO2Cfg + GO2CfgMCPCTS`
- `go2_ac_moe_cts` → `Go2Robot + GO2Cfg + GO2CfgACMoECTS`
- `go2_dual_moe_cts` → `Go2Robot + GO2Cfg + GO2CfgDualMoECTS`

### 4.2 这意味着什么

这意味着：

1. **同一个环境类 `Go2Robot` 可以配多种训练算法**；
2. 环境参数和训练参数是分开的；
3. 新增任务时，通常不需要重写全部代码，很多时候只需要：
	- 派生一个新的配置类；
	- 再注册一个新的 task 名称。

### 4.3 新手常见误区

仓库里有一些额外配置文件，比如其他 Go2 配置变体，但**只有被注册到** [legged_gym/envs/__init__.py](../legged_gym/envs/__init__.py) **里的任务，才能直接通过 `--task=xxx` 使用**。

---

## 5. 从训练命令到学习循环：完整调用链

### 5.1 训练入口很薄

[legged_gym/scripts/train.py](../legged_gym/scripts/train.py) 很短，核心只有几步：

1. `task_registry.make_env()` 创建环境；
2. `task_registry.make_alg_runner()` 创建训练器；
3. 同步环境步数与奖励课程；
4. `runner.learn()` 开始训练。

也就是说：**训练入口只是组装器，真正的逻辑在 `task_registry`、env 和 runner 里。**

### 5.2 `TaskRegistry` 干了什么

[legged_gym/utils/task_registry.py](../legged_gym/utils/task_registry.py) 是训练系统最值得先读懂的文件之一。

它主要做三件事：

#### A. `register()`

把 task 名称和对应实现登记起来。

#### B. `make_env()`

负责：

- 根据 task 名字找到环境类与配置；
- 用命令行参数覆盖配置；
- 设置随机种子；
- 解析 Isaac Gym 仿真参数；
- 最终实例化环境对象。

#### C. `make_alg_runner()`

负责：

- 根据 task 找到训练配置；
- 用命令行参数覆盖训练配置；
- 确定日志目录；
- 创建 runner；
- 如果 `resume=True`，自动恢复已有模型。

所以你可以把 `TaskRegistry` 理解成一个**任务工厂**。

---

## 6. Go2 的配置文件是你最常改的地方

如果你后续要二次开发，最先熟悉的文件通常不是算法代码，而是 [legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py)。

这个文件分两部分：

1. `GO2Cfg`：环境配置；
2. `GO2CfgPPO` / `GO2CfgCTS` / `GO2CfgMoECTS` 等：训练配置。

---

## 7. `GO2Cfg`：环境配置逐块讲解

### 7.1 `init_state`

这里定义机器人初始状态：

- 初始 base 位置；
- 12 个关节的默认角度；
- 是否启用翻倒初始化；
- backflip / sideflip / noflip 比例。

你在部署端看到的 `default_angles`，本质上也要和这里保持一致。

### 7.2 `env`

这里定义环境维度：

- `num_envs = 8192`：并行环境数；
- `num_observations = 45`：actor 输入维度；
- `num_privileged_obs = 263`：critic 输入维度；
- `episode_length_s = 25`：每回合最长时长。

这几个数字很关键，尤其是 `45`。

因为后面的部署脚本里，Mujoco 和真机都是手工拼接 45 维 obs。如果你把这里改成 48 或 60，却忘了改部署端，那就一定会出错。

### 7.3 `domain_rand`

这里定义域随机化：

- 摩擦系数；
- 机身质量；
- 各 link 质量；
- 质心偏移；
- restitution；
- PD 增益；
- motor zero offset；
- motor strength；
- 外力推扰动；
- action delay。

这是 sim2real 成功率很重要的一部分。

你可以把它理解成：**训练时故意把物理条件扰乱，让策略别只会在“理想世界”里走。**

### 7.4 `control`

这里决定动作如何变成关节控制：

- `control_type = 'P'`：位置式 PD 控制；
- `stiffness` / `damping`：PD 参数；
- `action_scale = 0.25`；
- `decimation = 4`：每次策略动作持续多少个仿真 step。

最重要的关系式是：

$$
	ext{target\_angle} = \text{default\_angle} + \text{action\_scale} \cdot \text{action}
$$

部署端也是按这套逻辑写的。

### 7.5 `terrain`

这里定义地形课程：

- 初始难度；
- 地形类型比例；
- 是否按累计运动距离调整课程。

本项目不是只训练平地，它会混合 slope、stairs、obstacles、flat 等多种地形。

### 7.6 `commands`

这里定义速度命令如何采样：

- 命令维度；
- 重采样时间；
- 是否 heading mode；
- 零命令 curriculum；
- velocity 限制概率；
- 动态命令范围；
- 不同训练阶段的命令范围升级；
- 不同地形的最大命令范围。

这是“机器人为什么有时前进、有时横移、有时转向”的来源。

### 7.7 `asset`

这里定义机器人 URDF 和碰撞相关规则：

- URDF 路径；
- 足端名称；
- 哪些部位碰撞要惩罚；
- 哪些部位碰撞后要终止。

### 7.8 `rewards`

这里定义奖励整体风格：

- `tracking_sigma`；
- `dynamic_sigma`；
- `base_height_target`；
- `max_contact_force`；
- `curriculum_rewards`；
- 最关键的 `scales`。

`scales` 里你会看到大量奖励项，比如：

- `tracking_lin_vel`
- `tracking_ang_vel`
- `lin_vel_z`
- `ang_vel_xy`
- `dof_acc`
- `dof_power`
- `torques`
- `correct_base_height`
- `action_rate`
- `action_smoothness`
- `collision`
- `dof_pos_limits`
- `feet_regulation`
- `hip_to_default`

最重要的理解是：**这里定义的是“每个奖励项的权重”，不是奖励公式本身。**

---

## 8. 奖励函数到底在哪实现

奖励的实现主要分两层：

### 8.1 汇总器在基类里

[legged_gym/envs/base/legged_robot.py](../legged_gym/envs/base/legged_robot.py) 中：

- `_prepare_reward_function()` 会扫描所有非零奖励项；
- 根据名字自动寻找 `_reward_xxx()`；
- `compute_reward()` 逐个调用并加权求和。

也就是说，如果你在配置里写了：

```python
hip_to_default = -0.05
```

那么框架就会自动去找：

```python
_reward_hip_to_default()
```

### 8.2 Go2 的自定义奖励在子类里

[legged_gym/envs/go2/go2_env.py](../legged_gym/envs/go2/go2_env.py) 里定义了 Go2 专有奖励，例如：

- `Go2Robot._reward_hip_to_default()`
- `Go2Robot._reward_x_command_hip_regular()`

这就是一种很标准的扩展方式：

1. 在配置里加 reward scale；
2. 在环境类里补 `_reward_xxx()`；
3. 框架自动收集并参与总 reward。

**这是最适合新手入门的第一类二次开发。**

---

## 9. 观测是怎么构造的

### 9.1 基类默认 obs

基类 [legged_gym/envs/base/legged_robot.py](../legged_gym/envs/base/legged_robot.py) 默认会构造：

- base linear velocity
- base angular velocity
- projected gravity
- commands
- dof pos
- dof vel
- previous actions

### 9.2 Go2 覆盖了默认 obs

但 [legged_gym/envs/go2/go2_env.py](../legged_gym/envs/go2/go2_env.py) 重写了 `compute_observations()`。

Go2 的 actor obs 是 45 维，只包含：

1. base angular velocity（3）
2. projected gravity（3）
3. commands（3）
4. dof pos（12）
5. dof vel（12）
6. previous action（12）

所以总维度是：

$$
3 + 3 + 3 + 12 + 12 + 12 = 45
$$

### 9.3 privileged obs 是给 critic 的

Go2 同时还构造了更大的 `privileged_obs_buf`，用于 critic：

- 额外包含 base linear velocity；
- 足端接触力；
- 电机力矩；
- 电机加速度；
- 地形高度测量。

这就是为什么配置里会看到 `num_privileged_obs = 263`。

### 9.4 新手必须牢记的一条规则

**只要你改了 `compute_observations()`，你就要同时检查下面三处：**

1. [legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py) 的 `num_observations` / `num_privileged_obs`；
2. [deploy/deploy_mujoco/deploy_go2.py](../deploy/deploy_mujoco/deploy_go2.py) 的 obs 拼接逻辑；
3. [deploy/deploy_real/deploy_real_go2.py](../deploy/deploy_real/deploy_real_go2.py) 的 obs 拼接逻辑。

否则训练端和部署端会不一致。

---

## 10. 环境每一步到底做了什么

理解 [legged_gym/envs/base/legged_robot.py](../legged_gym/envs/base/legged_robot.py) 的 `step()` 和 `post_physics_step()`，你就理解了环境主循环。

### 10.1 `step(actions)`

核心流程是：

1. clip action；
2. 根据 `decimation` 循环推进多个仿真 step；
3. 如果开启 action delay，就可能先沿用上一帧 action；
4. `_compute_torques()` 把动作变成力矩；
5. 把 torque 发给 Isaac Gym；
6. 完成一个策略 step 后，调用 `post_physics_step()`。

### 10.2 `_compute_torques()`

这里是动作到控制的关键。

当前 Go2 用的是 `P` 控制：

$$
	au = K_p(q_{target} - q + offset) - K_d \dot{q}
$$

其中：

$$
q_{target} = q_{default} + action\_scale \cdot action
$$

### 10.3 `post_physics_step()`

这里会做：

- 刷新 root state / contact force / rigid body state；
- 维护 episode step 和全局 step；
- 更新奖励课程；
- 计算 base 姿态与速度；
- `_post_physics_step_callback()` 做命令重采样和高度测量；
- 检查终止；
- 计算 reward；
- reset 已完成的环境；
- 计算新 obs；
- 记录 last action / last velocity。

### 10.4 `reset_idx()`

这个函数会在环境 reset 时做很多事：

- 随机化电机强度；
- 随机化电机零位偏差；
- 随机化 PD 增益；
- 更新 terrain curriculum；
- 重置关节和 base 状态；
- 重新采样 commands；
- 统计上一回合的奖励信息写入 `infos['episode']`。

这也是 TensorBoard 里很多统计指标的来源。

---

## 11. 命令采样与 curriculum 是怎么工作的

速度命令不是固定写死的，而是由 [legged_gym/envs/base/legged_robot.py](../legged_gym/envs/base/legged_robot.py) 的 `_resample_commands()` 动态采样。

它做的事情比你想象中多：

- 根据训练迭代数逐步放开命令范围；
- 根据剩余路程和剩余 episode 长度动态调整下界；
- 按概率采样“极限速度命令”；
- 按概率采样“零命令”；
- 在翻倒初始化场景下强制一段时间保持零命令。

所以当你觉得“为什么机器人总是在某些速度范围里动”，先别急着看 reward，先看 `commands` 配置和 `_resample_commands()`。

---

## 12. 训练配置：PPO、CTS、MoE 的区别先怎么看

在 [legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py) 下半部分，你会看到多套训练配置类：

- `GO2CfgPPO`
- `GO2CfgCTS`
- `GO2CfgMoENGCTS`
- `GO2CfgMCPCTS`
- `GO2CfgACMoECTS`
- `GO2CfgDualMoECTS`
- `GO2CfgMoECTS`

它们不是重新定义环境，而是重新定义：

- runner 类型；
- algorithm 类型；
- policy 类型；
- history 长度；
- latent 维度；
- expert 数量；
- load-balance 等算法参数；
- 实验名称和保存间隔。

### 12.1 先用直观理解，不要一上来啃论文

你可以先这样理解：

- **PPO**：最标准的 actor-critic；
- **CTS**：增加 teacher-student/history 结构，让 student 学会只靠有限信息控制；
- **MoE**：不是一个单一专家，而是多个 expert 共同工作，通过 gating 或混合权重决定谁更擅长当前情况；
- **MCP / Dual-MoE / AC-MoE**：是在 student 编码器、actor 结构或 mixture 方式上的不同变体。

对二次开发来说，最重要的不是先完全搞懂论文，而是先知道：

1. 用的是哪个 runner；
2. 模型 forward 输出是不是只有 action；
3. 导出时是否还要带上 latent / weights。

---

## 13. 为什么 CTS / MoE 不能按普通 PPO 去理解

### 13.1 runner 不同

普通 PPO 走的是：

- [rsl_rl/rsl_rl/runners/on_policy_runner.py](../rsl_rl/rsl_rl/runners/on_policy_runner.py)

CTS / MoE 走的是：

- [rsl_rl/rsl_rl/runners/on_policy_runner_cts.py](../rsl_rl/rsl_rl/runners/on_policy_runner_cts.py)

### 13.2 history 是关键差异

在 `OnPolicyRunnerCTS` 里会维护：

- `history_length`
- `history` buffer

每一步 rollout 都会把最新 obs 拼进历史窗口，再交给模型。

这意味着很多 CTS / MoE 模型并不是简单的：

$$
obs \rightarrow action
$$

而更像：

$$
history(obs_{t-k:t}) \rightarrow latent / weights \rightarrow action
$$

### 13.3 导出也不同

[legged_gym/utils/exporter.py](../legged_gym/utils/exporter.py) 里对多种模型结构做了专门适配：

- `student_encoder`
- `student_moe_encoder`
- `actor_mcp`
- `actor_moe`
- `actor`
- `student`

导出后的 TorchScript 在 MoE 场景下可能返回：

- `action`
- `weights`
- `latent`

所以部署脚本中会先判断 `policy(obs)` 返回的是不是 `tuple`。

---

## 14. `play.py` 不只是“看看效果”，它还负责导出模型

[legged_gym/scripts/play.py](../legged_gym/scripts/play.py) 有三个很关键的行为。

### 14.1 它会强制关闭大部分训练扰动

例如：

- 关闭噪声；
- 关闭 push；
- 关闭摩擦随机化；
- 关闭 base mass / link mass / COM 随机化；
- 关闭 PD 与零偏随机化；
- `env_cfg.env.test = True`。

这很好理解：play 主要是为了看效果，不是继续做强随机训练。

### 14.2 它默认会恢复已有模型

`play.py` 内部会把：

```python
train_cfg.runner.resume = True
```

强制打开。

所以如果你没有训练结果，或者 `experiment_name` / `checkpoint` 没对上，就会出现“找不到模型”的问题。

### 14.3 它会导出三种模型文件

默认导出到：

- `logs/{experiment_name}/exported/policies/policy.pt`
- `logs/{experiment_name}/exported/policies/policy.onnx`
- `logs/{experiment_name}/exported/policies/policy.pkl`

一般可这样理解：

- `policy.pt`：TorchScript，常用于 Python / Sim2Sim / 真机 Python 部署；
- `policy.onnx`：常用于跨框架或 C++ 部署；
- `policy.pkl`：权重存档。

---

## 15. 日志、模型、resume 是怎么工作的

### 15.1 训练输出目录

默认日志根目录是：

```text
logs/{experiment_name}/{date}_{run_name}
```

这里面通常会有：

- `model_*.pt`
- `config.yaml`
- TensorBoard 日志
- 可选的 RoboGauge 结果

### 15.2 `config.yaml` 很重要

runner 会把本次训练用到的：

- `train_cfg`
- `env_cfg`

一起存进 `config.yaml`。

这对复现实验特别重要。你以后想知道某个模型当时到底用了什么 reward、什么 terrain、什么 domain random，第一件事不是猜，而是回头看这个配置快照。

### 15.3 自动恢复 checkpoint 的一个坑

[legged_gym/utils/helpers.py](../legged_gym/utils/helpers.py) 的 `get_load_path()` 会自动选择最后一个 run 和最后一个 model，但源码里自己写了一个注释：

> `TODO sort by date to handle change of month`

意思是它当前的排序逻辑并不绝对稳妥。  
所以如果你实验很多，**最好手动指定 `--load_run` 和 `--checkpoint`**。

---

## 16. Mujoco 部署：它和训练端如何对齐

### 16.1 入口与配置

- 主脚本：[deploy/deploy_mujoco/deploy_go2.py](../deploy/deploy_mujoco/deploy_go2.py)
- 配置文件：[deploy/deploy_mujoco/configs/go2.yaml](../deploy/deploy_mujoco/configs/go2.yaml)

### 16.2 Mujoco 脚本主要做什么

它会：

1. 读取 YAML；
2. 加载 TorchScript 策略；
3. 初始化 Mujoco 模型；
4. 每隔 `control_decimation` 构造一次 obs；
5. 推理出 action；
6. 转换为 `target_dof_pos`；
7. 用 PD 控制驱动 Mujoco。

### 16.3 Mujoco 里的 obs 结构

Mujoco 端手工构造的 obs 也是：

1. ang vel（3）
2. gravity orientation（3）
3. cmd（3）
4. qj（12）
5. dqj（12）
6. last action（12）

这与 Go2 训练端 actor obs 对齐。

### 16.4 一个非常重要的现实问题

Mujoco YAML 中的缩放不一定和训练端一模一样，比如：

- 训练端 `commands_scale` 来自环境配置；
- Mujoco 端使用 `cmd_scale`；
- 真机端使用 `command_scale`。

因此你在二开时，一定要做“对齐检查”，而不是只改一处。

建议你每次改 obs/action 后，都列一个检查表：

- 维度对了吗？
- 顺序对了吗？
- scale 对了吗？
- 默认关节角对了吗？
- joint name / index 对了吗？

---

## 17. 真机部署：它和 Mujoco 最大的不同是什么

### 17.1 入口与配置

- 主脚本：[deploy/deploy_real/deploy_real_go2.py](../deploy/deploy_real/deploy_real_go2.py)
- 配置加载器：[deploy/deploy_real/config_go2.py](../deploy/deploy_real/config_go2.py)
- 配置文件：[deploy/deploy_real/configs/go2.yaml](../deploy/deploy_real/configs/go2.yaml)

### 17.2 真机控制流程

真机端不是“直接起 policy 就跑”，而是有状态机：

1. `zero_torque_state()`：等待遥控器 `start`；
2. `move_to_default_pos()`：平滑移动到默认姿态；
3. `default_pos_state()`：等待遥控器 `A`；
4. `run()`：循环构造 obs，推理策略，发送 low-level command；
5. `select`：退出。

### 17.3 真机 obs 也是手工拼的

真机端在 `run()` 中同样手工构造 45 维 obs：

- imu 角速度；
- 重力方向；
- 遥控器命令；
- 关节角；
- 关节速度；
- 上一步 action。

### 17.4 真机和仿真最重要的区别

真机多了这些你在仿真里通常不用担心的问题：

- 网卡与 DDS 通信；
- 遥控器状态；
- 上下位机服务开关；
- 实际电机控制风险；
- 硬件故障与人身安全。

所以对新手而言，**在你完全确定 obs/action/默认姿态/关节索引都正确前，不建议直接上真机。**

---

## 18. 训练端与部署端必须保持一致的项目

这是整份文档里最值得你贴在桌面前的一节。

只要改了下面任何一项，都要同步检查 Mujoco 和真机部署端：

### 18.1 观测维度与顺序

如果你改了：

- `Go2Robot.compute_observations()`
- `num_observations`

就要同步改：

- [deploy/deploy_mujoco/deploy_go2.py](../deploy/deploy_mujoco/deploy_go2.py)
- [deploy/deploy_real/deploy_real_go2.py](../deploy/deploy_real/deploy_real_go2.py)
- [deploy/deploy_mujoco/configs/go2.yaml](../deploy/deploy_mujoco/configs/go2.yaml)
- [deploy/deploy_real/configs/go2.yaml](../deploy/deploy_real/configs/go2.yaml)

### 18.2 动作缩放

如果你改了：

- `GO2Cfg.control.action_scale`

就要同步改部署 YAML 里的 `action_scale`。

### 18.3 默认关节角

如果你改了：

- `GO2Cfg.init_state.default_joint_angles`

就要同步改：

- Mujoco 的 `default_angles`
- 真机的 `default_angles`

### 18.4 PD 参数

如果你改了训练端的 `stiffness` / `damping`，部署端也要考虑同步调整 `kps` / `kds`。

### 18.5 关节顺序

训练端和 Mujoco / 真机的关节索引映射不一定天然完全相同，尤其真机端要依赖 `joint2motor_idx`。  
这也是部署时常见 bug 来源。

---

## 19. 二次开发从哪里下手最合适

下面给你一个很实用的“改动类型 → 入口位置”表。

### 19.1 我想改奖励

优先改：

1. [legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py) 的 `rewards.scales`
2. [legged_gym/envs/go2/go2_env.py](../legged_gym/envs/go2/go2_env.py) 或 [legged_gym/envs/base/legged_robot.py](../legged_gym/envs/base/legged_robot.py) 的 `_reward_xxx()`

这是最适合新手做的第一类修改。

### 19.2 我想改观测

优先改：

1. `Go2Robot.compute_observations()`
2. `GO2Cfg.env.num_observations`
3. `GO2Cfg.env.num_privileged_obs`
4. Mujoco / 真机 obs 拼接逻辑

这类修改要特别小心，因为会影响训练、导出、部署全链路。

### 19.3 我想改动作控制

优先看：

1. `GO2Cfg.control`
2. `LeggedRobot._compute_torques()`
3. 部署端 `action_scale`、`kps`、`kds`

### 19.4 我想改命令采样

优先看：

1. `GO2Cfg.commands`
2. `LeggedRobot._resample_commands()`

### 19.5 我想新增一个 task

一般步骤：

1. 在 [legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py) 新增一个配置类；
2. 如果需要，新增环境类或继承 `Go2Robot`；
3. 在 [legged_gym/envs/__init__.py](../legged_gym/envs/__init__.py) 中调用 `task_registry.register()` 注册；
4. 再用新的 `--task=...` 启动。

### 19.6 我想改算法结构

优先看：

1. [rsl_rl/rsl_rl/modules](../rsl_rl/rsl_rl/modules)
2. [rsl_rl/rsl_rl/algorithms](../rsl_rl/rsl_rl/algorithms)
3. [rsl_rl/rsl_rl/runners/on_policy_runner_cts.py](../rsl_rl/rsl_rl/runners/on_policy_runner_cts.py)
4. [legged_gym/utils/exporter.py](../legged_gym/utils/exporter.py)

对新手来说，这一类是难度最高的，因为它会同时影响：

- 训练；
- 保存；
- 恢复；
- 导出；
- 部署推理接口。

---

## 20. 一个推荐的二开顺序

如果你打算长期基于本仓库开发，建议按下面顺序练手：

### 第 1 步：只改 reward scale

例如调大 `tracking_lin_vel`，调小 `torques`。  
这一步风险最低，适合建立直觉。

### 第 2 步：新增一个简单 reward

例如鼓励更平稳抬腿、惩罚更大 yaw 抖动。  
这一步能帮你熟悉 reward 自动注册机制。

### 第 3 步：改命令分布

例如加大横向速度范围，或者增加零命令比例。  
这一步能帮你理解 curriculum 与 task 难度控制。

### 第 4 步：改 terrain 比例或 domain random

这一步能帮助你理解 sim2real 风格训练的核心思想。

### 第 5 步：改观测

只有当前四步你已经比较熟了，再考虑改 obs。  
因为一旦改 obs，你就要同步改训练与部署端。

### 第 6 步：新增任务或修改模型结构

这时你才比较适合进入算法与结构层面的开发。

---

## 21. 从零跑通一次完整流程

下面是一条适合新手的建议实践路线。

### 21.1 第一步：先装环境

先按 [doc/setup_zh.md](setup_zh.md) 完成：

- Conda 环境；
- PyTorch；
- Isaac Gym；
- `rsl_rl`；
- 当前仓库本体。

### 21.2 第二步：先跑一条最熟悉的训练命令

建议先跑 README 中已有任务，而不是自己新造 task。

例如：

```bash
python legged_gym/scripts/train.py --task=go2_moe_cts
```

### 21.3 第三步：确认日志产物

检查：

- `logs/{experiment_name}/.../model_*.pt`
- `logs/{experiment_name}/.../config.yaml`

### 21.4 第四步：用 Play 检查模型

```bash
python legged_gym/scripts/play.py --task=go2_moe_cts
```

如果没有恢复成功，优先检查：

- `experiment_name`
- `load_run`
- `checkpoint`

### 21.5 第五步：确认导出文件

检查是否生成：

- `policy.pt`
- `policy.onnx`
- `policy.pkl`

### 21.6 第六步：跑 Mujoco

把 `deploy/deploy_mujoco/configs/go2.yaml` 的 `policy_path` 指向你导出的 `policy.pt`，再运行：

```bash
python deploy/deploy_mujoco/deploy_go2.py
```

### 21.7 第七步：再考虑真机

只有当下面几件事都确认没问题时，再考虑真机部署：

- play 效果正常；
- Mujoco 效果正常；
- obs 对齐正常；
- 默认姿态正常；
- 关节顺序确认无误；
- 遥控器状态机你已经看懂。

---

## 22. 常见坑点与排错建议

### 22.1 安装问题

#### 现象

- Isaac Gym 装不上；
- 导入 `isaacgym` 失败；
- `rsl_rl` 找不到。

#### 建议

- 先确认 Python 版本与 PyTorch 版本；
- 确认 `isaacgym/python` 目录已经 `pip install -e .`；
- 确认 [rsl_rl](../rsl_rl) 也已经单独安装；
- 最后再安装根目录的项目本体。

### 22.2 `play.py` 找不到模型

#### 原因

因为 `play.py` 默认 `resume=True`。

#### 建议

显式指定：

- `--experiment_name`
- `--load_run`
- `--checkpoint`

### 22.3 改了 obs 后部署直接崩

#### 原因

训练端和部署端不一致。

#### 建议

逐项核对：

- 维度；
- 顺序；
- scale；
- last action 是否保留；
- joint 索引是否一致。

### 22.4 导出的策略接口和你想的不一样

#### 原因

MoE / CTS 导出的 TorchScript 可能返回 tuple，而不是单个 tensor。

#### 建议

参考部署脚本里对：

```python
if isinstance(result, tuple):
```

的处理逻辑。

### 22.5 真机不动或姿态奇怪

#### 优先检查

1. `default_angles` 是否一致；
2. `joint2motor_idx` 是否正确；
3. `action_scale` 是否一致；
4. IMU 与 obs 顺序是否一致；
5. Unitree App 中服务开关是否按 README 要求设置。

---

## 23. 你后面读源码时最值得盯住的函数/类

如果你时间有限，先读下面这些：

### 环境与任务

- `TaskRegistry.register()`
- `TaskRegistry.make_env()`
- `TaskRegistry.make_alg_runner()`
- `Go2Robot.compute_observations()`
- `LeggedRobot.step()`
- `LeggedRobot.post_physics_step()`
- `LeggedRobot._resample_commands()`
- `LeggedRobot._compute_torques()`
- `LeggedRobot.compute_reward()`
- `LeggedRobot._prepare_reward_function()`

### 训练与导出

- `OnPolicyRunnerCTS.learn()`
- `export_policy_as_jit()`
- `export_policy_as_onnx()`
- `export_policy_as_pkl()`

### 部署

- `Controller.run()`（真机）
- Mujoco 主循环中的 obs 拼接与 policy 推理逻辑

---

## 24. 给新手的最终建议

### 建议 1：不要一上来改算法

先把：

- 仓库结构；
- task 注册；
- obs；
- rewards；
- commands；
- deployment 对齐；

搞清楚，再去动网络结构和 loss。

### 建议 2：每次只改一类东西

不要一次同时改：

- reward
- obs
- action scale
- terrain
- domain random

否则你很难定位问题来源。

### 建议 3：改 obs 前，先列联动清单

你只要动了 obs，就把这四个位置列出来一起检查：

1. 训练环境；
2. 训练配置；
3. Mujoco 部署；
4. 真机部署。

### 建议 4：真机前一定先过 Sim2Sim

这不是形式主义，而是为了安全。  
Mujoco 先跑通，能帮你提前发现大量“并不是训练坏了，而是部署对齐错了”的问题。

---

## 25. 一页纸速记

如果你想快速记住这套框架，可以只记下面这几句：

1. **`--task` 决定 env + cfg + runner。**
2. **`go2_config.py` 是最常改的文件。**
3. **`go2_env.py` 决定 Go2 的 obs 和部分 reward。**
4. **`legged_robot.py` 是环境主循环核心。**
5. **`on_policy_runner_cts.py` 是 CTS / MoE 训练主入口。**
6. **`play.py` 会恢复模型并导出策略。**
7. **部署端 obs 是手工拼的，和训练端必须完全对齐。**
8. **新手最适合先改 reward，再改 commands，再改 obs。**

---

## 26. 后续你可以怎么继续学

如果你已经看完这份文档，建议下一步这样做：

1. 通读 [legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py)；
2. 再读 [legged_gym/envs/go2/go2_env.py](../legged_gym/envs/go2/go2_env.py)；
3. 再顺着 [legged_gym/envs/base/legged_robot.py](../legged_gym/envs/base/legged_robot.py) 的 `step()` 往下读；
4. 然后读 [legged_gym/scripts/train.py](../legged_gym/scripts/train.py) 和 [legged_gym/utils/task_registry.py](../legged_gym/utils/task_registry.py)；
5. 最后再去看 [rsl_rl/rsl_rl/runners/on_policy_runner_cts.py](../rsl_rl/rsl_rl/runners/on_policy_runner_cts.py) 与对应的 modules / algorithms。

如果你愿意，接下来最适合做的第一件实操是：

> 新建一个自己的 task，例如 `go2_myexp`，只修改一两个 reward scale，跑一小段训练，再用 play 和 Mujoco 检查是否正常。

这样你会最快地建立“从配置到运行结果”的直觉。

---

## 27. 常见问题速查：5 分钟先定位到哪一层出了问题

这一节不讲原理，只讲“出问题先看哪里”。如果你卡住了，优先对照这张表。

| 现象 | 大概率问题层 | 优先检查位置 | 典型原因 |
| --- | --- | --- | --- |
| 训练命令一启动就报错 | 安装/入口层 | [doc/setup_zh.md](setup_zh.md)、[legged_gym/scripts/train.py](../legged_gym/scripts/train.py)、[legged_gym/utils/helpers.py](../legged_gym/utils/helpers.py) | 依赖没装全、Isaac Gym 不可用、参数不对 |
| `--task` 报未注册 | 任务注册层 | [legged_gym/envs/__init__.py](../legged_gym/envs/__init__.py) | task 名称没注册或拼错 |
| 训练能启动但很快 NaN / 发散 | 环境/奖励/控制层 | [legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py)、[legged_gym/envs/base/legged_robot.py](../legged_gym/envs/base/legged_robot.py) | reward 过激、PD 太大、action scale 不合理 |
| `play.py` 找不到模型 | 日志恢复层 | [legged_gym/scripts/play.py](../legged_gym/scripts/play.py)、[legged_gym/utils/helpers.py](../legged_gym/utils/helpers.py) | `resume=True`、experiment/checkpoint 不匹配 |
| Mujoco 能启动但机器人不动 | 部署配置层 | [deploy/deploy_mujoco/deploy_go2.py](../deploy/deploy_mujoco/deploy_go2.py)、[deploy/deploy_mujoco/configs/go2.yaml](../deploy/deploy_mujoco/configs/go2.yaml) | `policy_path` 错、obs 不对、action scale 不一致 |
| 真机站不稳或姿态异常 | 真机映射层 | [deploy/deploy_real/deploy_real_go2.py](../deploy/deploy_real/deploy_real_go2.py)、[deploy/deploy_real/configs/go2.yaml](../deploy/deploy_real/configs/go2.yaml) | 默认角、关节索引、obs 顺序不一致 |
| 改完 obs 后维度报错 | 训练/部署对齐层 | [legged_gym/envs/go2/go2_env.py](../legged_gym/envs/go2/go2_env.py)、[legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py)、部署脚本两端 | `num_observations`、拼接顺序、YAML 未同步 |
| 导出后策略接口不兼容 | 导出层 | [legged_gym/utils/exporter.py](../legged_gym/utils/exporter.py) | MoE/CTS 返回 `tuple`，部署代码按普通 PPO 处理了 |

### 27.1 最常用的定位顺序

建议按下面顺序排查，而不是来回乱翻：

1. 先看是不是**安装/路径**问题；
2. 再看是不是**task 注册**问题；
3. 再看是不是**配置维度**问题；
4. 再看是不是**训练端与部署端不一致**；
5. 最后才怀疑**算法本身**。

对新手来说，绝大多数问题都不是“算法错了”，而是“联动没改全”。

---

## 28. 第一次二次开发实战：新增一个自己的 `go2_myexp`

这一节给一个最小实战路线。目标不是立刻做论文创新，而是帮你第一次完整地改通：

- 新建自己的 task；
- 调一两个 reward；
- 能训练；
- 能 play；
- 能在 Mujoco 检查。

### 28.1 为什么先做这个实战

因为这是“最小闭环”：

1. 改动小；
2. 风险低；
3. 能很快建立配置、注册、训练、导出、部署之间的联系。

### 28.2 第一步：复制一个最接近的训练配置类

建议从 [legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py) 里复制 `GO2CfgMoECTS` 或 `GO2CfgCTS`，新建一个配置类，例如：

- `GO2CfgMyExp`

第一版只改两类内容：

1. `runner.experiment_name`
2. `GO2Cfg.rewards.scales` 中的一两个奖励项

例如你可以把：

- `tracking_lin_vel` 稍微调大；
- `torques` 稍微调小惩罚；
- 或单独新增一个更保守的 `hip_to_default` 权重。

### 28.3 第二步：如果只是改训练配置，环境类可以先不动

如果你没有修改 obs 和 reward 函数实现，而只是改 scale，那么环境类仍然可以继续使用：

- `Go2Robot`

这时你不需要新建 `go2_env_myexp.py`。

### 28.4 第三步：注册新 task

到 [legged_gym/envs/__init__.py](../legged_gym/envs/__init__.py) 中新增一条注册：

- task 名称：`go2_myexp`
- env 类：`Go2Robot`
- env cfg：`GO2Cfg()` 或你自己的环境配置类
- train cfg：`GO2CfgMyExp()`

你可以把这一步理解为：**让命令行知道你的实验名字应该映射到哪套实现。**

### 28.5 第四步：启动一次最小训练

建议第一轮不要直接大规模跑，把问题收敛到最小：

```bash
python legged_gym/scripts/train.py --task=go2_myexp --num_envs 512 --max_iterations 50
```

这里不是为了训出最强策略，而是为了确认：

- task 能找到；
- env 能创建；
- runner 能启动；
- 日志能生成；
- reward 没有明显爆炸。

### 28.6 第五步：检查训练产物

确认 [logs](../logs) 下是否生成：

- `model_*.pt`
- `config.yaml`

如果这一步没问题，说明“训练链路”已经基本打通。

### 28.7 第六步：用 Play 验证

然后运行：

```bash
python legged_gym/scripts/play.py --task=go2_myexp --load_run <你的run> --checkpoint <你的checkpoint>
```

重点观察：

- 机器人是否能稳定站立；
- 速度跟踪是否明显偏离；
- 是否出现高频抖动；
- 是否成功导出 `policy.pt`。

### 28.8 第七步：再去 Mujoco 检查

把 [deploy/deploy_mujoco/configs/go2.yaml](../deploy/deploy_mujoco/configs/go2.yaml) 的 `policy_path` 指到新导出的策略，再运行 [deploy/deploy_mujoco/deploy_go2.py](../deploy/deploy_mujoco/deploy_go2.py)。

如果你这一轮**没有改 obs 和动作定义**，那么 Mujoco 一般只需要换模型路径。  
如果你改了 obs 或动作缩放，那就要进入下一节的“配置联动表”逐项核对。

### 28.9 第一轮二开最推荐改什么

推荐按下面优先级来：

1. reward scale
2. command range
3. terrain proportion
4. domain random

不推荐第一轮就改：

1. obs 维度
2. actor/critic 网络结构
3. 导出接口
4. 真机控制逻辑

---

## 29. 训练产物说明：`logs` 目录里到底哪些文件最关键

训练一段时间后，很多新手会打开 [logs](../logs) 然后不知道该看什么。这里做一个速查。

### 29.1 常见目录结构

通常会是：

```text
logs/
	<experiment_name>/
		<date>_<run_name>/
			model_*.pt
			config.yaml
			events.out.tfevents.*
			robogauge_results/
```

以及：

```text
logs/<experiment_name>/exported/policies/
	policy.pt
	policy.onnx
	policy.pkl
```

### 29.2 每类文件是什么意思

| 文件/目录 | 用途 | 新手最常见用法 |
| --- | --- | --- |
| `model_*.pt` | 训练过程中的 checkpoint | 继续训练、Play 恢复、导出策略 |
| `config.yaml` | 当次训练的配置快照 | 复现实验、回查 reward/obs/terrain 参数 |
| `events.out.tfevents.*` | TensorBoard 日志 | 看 reward 曲线、速度跟踪、loss |
| `robogauge_results/` | RoboGauge 评估输出 | 对比不同 checkpoint 的 sim2sim 结果 |
| `exported/policies/policy.pt` | TorchScript 策略 | Mujoco 或真机 Python 部署 |
| `exported/policies/policy.onnx` | ONNX 模型 | C++ 或跨框架部署 |
| `exported/policies/policy.pkl` | 权重字典 | 自己做分析或二次处理 |

### 29.3 新手最容易搞混的两件事

#### A. `model_*.pt` 和 `policy.pt` 不是一回事

- `model_*.pt` 更像训练 checkpoint；
- `policy.pt` 是导出后的推理模型。

#### B. `config.yaml` 很值得保存

很多时候你以为自己“只改了一点点参数”，但过几天就忘了。  
这时 [logs](../logs) 中的 `config.yaml` 才是最可靠的实验记录。

---

## 30. 训练端 / 部署端配置联动表

这是二次开发最重要的一张表。只要你改了训练端的这些项，就要同步核对部署端。

| 训练端位置 | Mujoco 对应位置 | 真机对应位置 | 是否必须同步 | 说明 |
| --- | --- | --- | --- | --- |
| [legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py) 中 `env.num_observations` | [deploy/deploy_mujoco/configs/go2.yaml](../deploy/deploy_mujoco/configs/go2.yaml) 中 `num_obs` | [deploy/deploy_real/configs/go2.yaml](../deploy/deploy_real/configs/go2.yaml) 中 `num_obs` | 是 | 维度不一致会直接报错或推理错位 |
| `Go2Robot.compute_observations()` | [deploy/deploy_mujoco/deploy_go2.py](../deploy/deploy_mujoco/deploy_go2.py) 的 obs 拼接 | [deploy/deploy_real/deploy_real_go2.py](../deploy/deploy_real/deploy_real_go2.py) 的 obs 拼接 | 是 | 顺序、scale、是否保留 last action 都要对齐 |
| `control.action_scale` | Mujoco YAML 的 `action_scale` | 真机 YAML 的 `action_scale` | 是 | 动作幅度不一致会导致姿态完全不同 |
| `init_state.default_joint_angles` | Mujoco YAML 的 `default_angles` | 真机 YAML 的 `default_angles` | 是 | 默认站姿不一致会让 policy 输入分布错位 |
| `control.stiffness` / `control.damping` | Mujoco YAML 的 `kps` / `kds` | 真机 YAML 的 `kps` / `kds` | 强烈建议 | 控制手感与稳定性直接受影响 |
| `commands` 范围与缩放 | Mujoco YAML 的 `cmd_scale` / `max_cmd` / `cmd_init` | 真机 YAML 的 `command_scale` | 视改动而定 | 改命令分布后，部署端最好一起检查 |
| `asset` 中关节定义 | Mujoco YAML 的 `mujoco_joint_names` / `model_joint_names` | 真机 YAML 的 `joint2motor_idx` | 是 | 顺序映射错一个关节都可能姿态异常 |

### 30.1 最建议保存的联动检查清单

每次你改完训练端，建议把下面 7 项过一遍：

1. obs 维度是否变了；
2. obs 顺序是否变了；
3. obs scale 是否变了；
4. `action_scale` 是否变了；
5. `default_angles` 是否变了；
6. `kps/kds` 是否需要联动；
7. joint 顺序和索引是否仍一致。

如果这 7 项你都检查过，部署出问题的概率会低很多。

---

## 31. 真机部署前 Checklist

这一节建议你在每次上真机前都看一遍。

### 31.1 模型与任务检查

- [ ] 当前 `policy.pt` 来自正确的 task，而不是旧实验；
- [ ] 训练时使用的 obs 结构与你当前部署代码一致；
- [ ] 没有把普通 PPO 的部署假设错误套到 CTS / MoE 模型上；
- [ ] 如果策略返回 `tuple`，部署端已正确取出 action。

### 31.2 参数一致性检查

- [ ] [legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py) 的 `default_joint_angles` 与真机 YAML 的 `default_angles` 一致；
- [ ] `action_scale` 一致；
- [ ] `num_obs` 一致；
- [ ] obs 顺序一致；
- [ ] `joint2motor_idx` 已核对过；
- [ ] `kps` / `kds` 与当前策略风格匹配。

### 31.3 通信与服务检查

- [ ] 网卡名称正确；
- [ ] DDS 通信能正常建立；
- [ ] Unitree App 中已按 README 要求关闭 `mcf/*`；
- [ ] Unitree App 中已打开 `ota_box`；
- [ ] 遥控器连接正常。

### 31.4 安全流程检查

- [ ] 已先在 Isaac Gym Play 中看过效果；
- [ ] 已先在 Mujoco 中验证过；
- [ ] 已清楚 `start`、`A`、`select` 的状态机流程；
- [ ] 已预留安全空间并准备随时停止；
- [ ] 第一次测试时使用更保守的姿态和速度命令。

### 31.5 一个实用原则

如果你对某一项没有把握，先不要上真机，先回到 [deploy/deploy_mujoco/deploy_go2.py](../deploy/deploy_mujoco/deploy_go2.py) 和 [deploy/deploy_real/deploy_real_go2.py](../deploy/deploy_real/deploy_real_go2.py) 对照 obs/action 拼接逻辑。

对这个仓库来说，**真机问题经常不是“模型不行”，而是“部署前提不一致”。**

---

如需继续深入，建议你把这份文档和 [README_zh.md](../README_zh.md)、[doc/setup_zh.md](setup_zh.md) 一起看：前者解决“怎么用”，这份文档解决“为什么这么组织、应该去哪里改”。
