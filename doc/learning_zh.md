# Go2 RL GYM 学习与二次开发手册
> 面向对象：会基本 Python、能读一点代码，但对强化学习、Isaac Gym、四足机器人训练框架还不熟的开发者。  
> 目标：帮你先建立整体认识，再带你顺着“训练 → Play → 导出 → Mujoco → 真机 → 二次开发”这条主线读懂本仓库。

> 如果你的重点不是 Go2 本身，而是“如何把其他四足机器人接入本框架并跑通 Sim2Sim / 部署”，建议优先阅读：[新机器人接入详解：从训练到 Sim2Sim 与部署](./new_robot_zh.md)



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

### 8.3 奖励系统为什么会让新手觉得“很绕”

很多人第一次看 reward，会以为它只是：

1. 在配置里写几个系数；
2. 在代码里写几个 `_reward_xxx()`；
3. 然后求和。

但当前仓库里的奖励系统其实已经有 **4 层结构**：

1. **配置层**：在 [legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py) 或 [legged_gym/envs/base/legged_robot_config.py](../legged_gym/envs/base/legged_robot_config.py) 里定义 `rewards.scales`；
2. **绑定层**：在 [legged_gym/envs/base/legged_robot.py](../legged_gym/envs/base/legged_robot.py) 的 `_prepare_reward_function()` 中，把配置名自动映射到 `_reward_xxx()`；
3. **运行层**：每一步在 `compute_reward()` 里读取当前物理状态、命令状态、动作历史，真正算出每项 reward；
4. **课程层**：reward 还可能继续受到 `curriculum_rewards`、`turn_over_scales`、`dynamic_sigma` 的二次调制。

所以它复杂，不是因为“reward 名字多”，而是因为它已经从一个静态打分器，变成了一个**会随着训练阶段、地形类型、命令难度、机器人姿态切换规则的动态打分系统**。

### 8.4 读 reward 时，建议按“数据流”而不是按“函数顺序”理解

可以把当前 reward 系统理解成下面这条链：

1. `cfg.rewards.scales.xxx` 决定某个 reward 是否启用；
2. `_prepare_reward_function()` 把所有非零项收集起来；
3. 同时把 scale 乘上 `dt`；
4. `compute_reward()` 每个 step 调用对应的 `_reward_xxx()`；
5. 原始 reward 值再乘以 scale / curriculum scale / turn-over scale；
6. 所有项累加成 `self.rew_buf`；
7. 每项还会单独累计进 `self.episode_sums[name]`，供日志统计。

这里有一个新手很容易忽略的点：

> 配置里的 reward scale 会在 `_prepare_reward_function()` 中乘上 `dt`。

这意味着配置中的系数，更接近“每秒量级”的直觉；真正到每个仿真 step 使用时，已经被折算成单步权重了。

所以如果你改了 `decimation` 或底层仿真 `dt`，reward 的体感也可能跟着变化，不一定只是控制频率在变。

### 8.5 当前 reward 可以分成哪几类

如果直接从 [legged_gym/envs/base/legged_robot.py](../legged_gym/envs/base/legged_robot.py) 往下看 `_reward_*()`，会觉得很多；但按设计目标分组后就清楚很多。

#### A. 任务跟踪类：机器人有没有完成命令

- `_reward_tracking_lin_vel()`
- `_reward_tracking_ang_vel()`

这两项本质上在回答：

- 你给的 `x/y/yaw` 命令，机器人有没有跟上？
- 速度误差越小，reward 越接近 1；误差越大，reward 越接近 0。

这也是为什么 **命令系统和奖励系统必须一起看**：

- 命令采样太保守，tracking reward 会很好看，但任务可能太简单；
- 命令采样太激进，tracking reward 会变差，但不一定是策略退化，也可能只是任务更难了。

#### B. 姿态稳定类：机器人有没有站稳

- `_reward_orientation()`
- `_reward_upright()`
- `_reward_lin_vel_z()`
- `_reward_ang_vel_xy()`

这一组主要抑制：

- 机身左右乱晃；
- 上下乱弹；
- pitch / roll 抖动过大；
- 翻倒或明显倾斜。

其中 `upright` 是当前版本里比较“恢复训练导向”的奖励，尤其配合 turn-over 训练时更重要；而 `orientation` 更像传统 locomotion 里的“尽量保持平稳底盘”。

#### C. 几何与高度类：身体和地面的相对关系对不对

- `_reward_base_height()`
- `_reward_correct_base_height()`
- `_reward_legs_distance()`
- `Go2Robot._reward_hip_to_default()`

这组 reward 关注的是：

- 机身离地高度是否合理；
- 双腿/髋关节姿态是否太奇怪；
- 步态是否偏离默认站姿太多。

这里当前版本和 [legged_gym/envs/base/original.py](../legged_gym/envs/base/original.py) 的差异很大。

`original.py` 里的高度奖励更接近“直接拿 base 高度或高度图均值来算”，而当前版会结合：

- 足端接触；
- 刚体状态；
- 局部高度扫描；

去估计“机器人相对当前地面的有效高度”。这对台阶、斜坡、障碍物环境更合理，但也让 reward 变得更难一眼看懂。

#### D. 平滑与能耗类：动作是不是太猛、太抖、太费电

- `_reward_torques()`
- `_reward_dof_acc()`
- `_reward_action_rate()`
- `_reward_action_smoothness()`
- `_reward_dof_power()`

这一组通常不是为了“让机器人走起来”，而是为了避免它：

- 用很暴力的控制硬顶出一个步态；
- 每一帧都大幅切动作；
- 力矩和速度都过大，导致高能耗、难迁移到真机。

对真机迁移来说，这一组往往非常重要。因为很多仿真里能跑的动作，真正上机时会因为：

- 电机发热；
- 接触冲击；
- 摩擦不一致；
- 动作不连续；

而表现很差。

#### E. 接触与步态类：脚落地得像不像“正常走路”

- `_reward_feet_air_time()`
- `_reward_feet_contact_forces()`
- `_reward_feet_regulation()`
- `_reward_stumble()`

这组主要在塑造步态细节：

- 脚是不是有明确抬起和落下；
- 落地接触力是不是太大；
- 摆腿快的时候，有没有把脚适当抬高；
- 是否频繁踢到台阶立面或障碍物侧面。

如果你后面发现机器人“能动，但动作很脏”，问题经常就在这组 reward 上，而不是 tracking reward 本身。

#### F. 安全边界类：别把关节、电机和碰撞推到极限

- `_reward_collision()`
- `_reward_dof_pos_limits()`
- `_reward_dof_vel_limits()`
- `_reward_torque_limits()`
- `_reward_similar_to_default()`

这组 reward 可以理解为“防止策略钻仿真漏洞”。

没有这些约束时，策略可能会为了多拿一点 tracking reward，去使用：

- 贴着关节限位抖动；
- 高频大力矩抽动；
- 让不该碰地的 body 碰地；
- 长期保持一种不适合真实机器人的怪姿态。

### 8.6 当前版比 `original.py` 多复杂在哪里

如果和 [legged_gym/envs/base/original.py](../legged_gym/envs/base/original.py) 对比，当前 reward 不是简单“多几个名字”，而是多了几种复杂性来源：

1. **更多状态依赖**：现在 reward 会更频繁读取 `rigid_body_states`、`contact_forces`、动作历史、脚接触历史；
2. **更多训练阶段切换**：turn-over 模式下，同一个 reward 会在“翻身阶段”和“正常行走阶段”切换不同 scale；
3. **更多课程机制**：`curriculum_rewards` 可以让 reward 权重随训练迭代线性变化；
4. **更多难度自适应**：`dynamic_sigma` 会根据命令速度和 terrain 难度改变 tracking reward 的容忍度。

所以当前 reward 更像“任务设计系统”，而 `original.py` 更接近“传统 locomotion 打分器”。

### 8.7 `dynamic_sigma` 到底在解决什么问题

这是当前版本 reward 里最容易被忽略、但很关键的一层。

直觉上，固定的 tracking sigma 有个问题：

- 对低速、平地命令来说，它可能刚好；
- 但对高速、难地形命令来说，要求就会显得过严。

于是当前版会根据：

- 当前命令速度大小；
- 当前 terrain 类型；
- 当前 terrain level；

动态放宽 tracking reward 的“容错半径”。

你可以把它理解成：

> 在容易任务上，老师打分严格；在高速、难地形任务上，老师适当放宽标准。

这样训练出来的策略，通常更容易在“既要高速、又要复杂地形”的混合任务下稳定学习。

### 8.8 reward curriculum 该怎么理解

当前版除了 command curriculum，还有 reward curriculum。

这意味着某些 reward 项不是从训练一开始就固定权重，而是会按照配置中的区间，例如：

- 从第 0 次迭代到第 1500 次迭代；
- 从 `start_value` 线性过渡到 `end_value`。

这类设计常用于：

- 训练前期先更重视姿态或高度；
- 训练后期再逐渐减弱某些辅助 reward；
- 避免一开始约束太多，策略连基本移动都学不会。

### 8.9 新手二开 reward，建议按这 4 层改

如果你想改 reward，不要一上来就重写半个环境，建议按风险从低到高排序：

1. **最低风险**：只改 `rewards.scales`；
2. **中低风险**：新增一个简单的 `_reward_xxx()`；
3. **中风险**：修改现有 reward 的公式或依赖 buffer；
4. **高风险**：同时改 reward、obs、command 三者的耦合关系。

最常见的错误不是公式写错，而是：

- reward 名字和函数名字对不上；
- 改了 reward，但没有意识到它还受 curriculum / turn-over scale 影响；
- 只看总 reward，不看 `episode_sums` 里的单项统计。

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

### 10.5 `legged_robot.py` 和 `original.py` 的关系是什么

如果你在目录里看到 [legged_gym/envs/base/original.py](../legged_gym/envs/base/original.py)，最容易产生两个误解：

1. 它是不是当前真正被调用的环境；
2. 当前 [legged_gym/envs/base/legged_robot.py](../legged_gym/envs/base/legged_robot.py) 只是加了一点注释。

都不是。

当前实际生效的是：

- [legged_gym/envs/base/legged_robot.py](../legged_gym/envs/base/legged_robot.py)
- [legged_gym/envs/go2/go2_env.py](../legged_gym/envs/go2/go2_env.py)
- [legged_gym/envs/__init__.py](../legged_gym/envs/__init__.py)

而 [legged_gym/envs/base/original.py](../legged_gym/envs/base/original.py) 更像仓库里保留的一份“旧版/原始参考实现”。

### 10.6 当前版和 `original.py` 的主循环差异

如果只看大框架，两者都还是：

1. 接收 action；
2. 推进仿真；
3. 刷新状态；
4. 计算 reward；
5. reset；
6. 输出 obs。

但当前版已经在几个关键位置明显扩展了：

#### A. `step()` 更复杂

`original.py` 的 `step()` 更接近“动作直接进 `_compute_torques()`，然后推进物理”。

当前版多了：

- action delay 域随机化；
- motor strength 域随机化；
- test 模式下按仿真时间节奏 sleep；
- 更强的和 reset / curriculum 配套的状态维护。

#### B. `post_physics_step()` 更像总调度器

当前版这里不只是在算 reward，还会维护：

- `base_pos`、`rpy`、`base_lin_vel`、`base_ang_vel`；
- `commands_resampling_step` 倒计时；
- reward curriculum；
- turn-over 计时器；
- `max_move_distance`；
- push robots；
- `rigid_body_states` 相关逻辑。

这就是为什么你会觉得当前 `legged_robot.py` 逻辑“很厚”——它已经不是一个单纯的环境外壳，而是把训练难度控制、恢复训练、真机鲁棒性增强都塞进来了。

#### C. `reset_idx()` 不再只是“回到初始状态”

在 `original.py` 中，reset 逻辑相对更直接。

当前版 reset 已经承担：

- reset 级别的 domain randomization；
- terrain curriculum 更新；
- command 重采样；
- 日志统计写回；
- 起身/翻倒训练的状态清理。

所以现在的 reset 更接近一次“局部重新开局”，而不只是清 buffer。

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

### 11.1 命令系统里最重要的 4 个张量/变量

要读懂命令系统，先不要一上来扎进 `_resample_commands()`，先记住下面几个核心对象：

1. `self.commands`：当前环境真正持有的命令张量；
2. `self.command_ranges`：全局命令范围，会被训练进度更新；
3. `self.env_command_ranges`：按地形裁剪后的每个 env 专属范围；
4. `self.commands_resampling_step`：距离下次重采样还有多少 step。

其中最容易误解的是 `self.commands`。

当前主线里它的内部语义是：

- 第 1 维：`lin_vel_x`
- 第 2 维：`lin_vel_y`
- 第 3 维：`ang_vel_yaw`
- 第 4 维：`heading`

但是：

> actor observation 当前只显式使用前 3 维。

也就是说，**内部命令维度** 和 **策略显式看到的命令维度** 不是一回事。

这正是很多新手第一次改 command 时最容易踩坑的地方。

### 11.2 命令的完整生命周期

可以把一条 command 的生命周期理解成：

1. `_parse_cfg()` 先把配置里的 `commands.ranges` 读到运行时；
2. 每个 env 在 reset 时会先采一次 command；
3. rollout 过程中，`post_physics_step()` 会让 `commands_resampling_step` 递减；
4. 倒计时归零时，在 `_post_physics_step_callback()` 里触发 `_resample_commands()`；
5. 如有 `heading_command`，第 4 维 heading 会被转换成第 3 维 yaw 角速度目标；
6. 新命令再进入：
	 - observation；
	 - tracking reward；
	 - terrain curriculum 距离统计。

你可以把它理解成：

> command 不是一个静态标签，而是环境每隔一段时间重新发布给机器人的“短期任务单”。

### 11.3 为什么当前版不是简单“固定时间随机采样一下速度”

在 [legged_gym/envs/base/original.py](../legged_gym/envs/base/original.py) 里，命令逻辑相对简单：

- 从全局范围里采样；
- 小速度命令清零；
- 到了重采样时刻再重新给一组。

当前版则额外加入了几层控制：

1. **训练进度课程**：训练到一定迭代数后，自动放大命令范围；
2. **地形约束**：同样是全局速度范围，不同地形还能再做一次上限裁剪；
3. **动态下界**：剩余时间越少、剩余目标距离越大，下一次命令采样的速度下界越高；
4. **特殊模板**：命令有概率被替换成边界速度、零命令、原地转向等特殊模式；
5. **turn-over 保护**：翻身/起身阶段强制发零命令。

所以它本质上已经不只是一个“速度采样器”，而是一个**任务调度器**。

### 11.4 `heading_command` 到底在干什么

当 `heading_command=True` 时，环境并不是让策略直接跟踪“一个给定 yaw 速度”，而是：

1. 在 `self.commands[:, 3]` 里存目标 heading；
2. 在 `_post_physics_step_callback()` 中拿当前朝向和目标 heading 做差；
3. 把这个 heading error 转成 `self.commands[:, 2]` 的 yaw 角速度目标。

也就是说：

- 第 4 维更像“朝哪里看”；
- 第 3 维更像“现在该以多快角速度去转过去”。

这是一种很常见的两层控制思想：

- 高层给方向目标；
- 底层再变成速度目标。

但本仓库当前 Go2 主线默认把它关掉，因为真机和任务目标上，更偏向直接给速度命令。

### 11.5 `limit_vel` 为什么很重要

`limit_vel` 是当前版命令系统里非常工程化、也非常实用的一层。

它不是继续从连续区间采样，而是直接从：

- 最小值；
- 0；
- 最大值；

这些离散边界里组合出命令。

这样做的好处是，机器人会更频繁地遇到一些真实控制里很关键的边界场景，比如：

- 纯前进；
- 纯后退；
- 纯横移；
- 纯原地左转/右转；
- 从一个极限方向突然反向。

如果没有这一层，连续随机采样虽然“平均”，但反而可能不容易覆盖这些极端但重要的动作模式。

### 11.6 `zero_command_curriculum` 不只是“让机器人停一下”

零命令课程的直观作用，是让机器人学会：

- 不该动的时候站稳；
- 命令从运动切回静止时不要乱抖。

但它更深一层的作用是：

- 帮助策略建立“移动”和“稳定站立”这两种模式；
- 避免策略在训练中永远都只会持续移动，不会停止。

当前版还允许在零命令时，按概率附加边界 yaw 角速度，于是机器人还要学会“原地旋转但不平移”这种控制模式。

### 11.7 `dynamic_resample_commands` 在命令层解决了什么问题

如果 command 每次都允许采到很小速度，就会有一种情况：

- episode 已经过了大半；
- 机器人理论上还应该完成一段较远的路径；
- 但环境却还可能继续采到很慢的命令；
- 这会让任务本身变得“理论上不容易完成”。

所以当前版会根据：

- 当前还剩多少 episode 时间；
- 当前累计命令大概还差多少距离；

动态计算一个速度下界。

通俗讲，它在做的是：

> 既然你后面还得走这么远，那我接下来就别再给你一个慢得离谱的命令了。

这会让 terrain curriculum 和 tracking 任务之间的逻辑更一致。

### 11.8 `commands_xy_accumulation` 为什么会影响地形课程

当前版里，terrain curriculum 不一定只看“机器人最后离出生点有多远”，还可能看：

- 这一整局里，环境总共要求机器人在 xy 平面上完成多少命令位移。

于是 `commands_xy_accumulation` 的作用就出来了：

- 它近似记录“这局任务布置了多少路程”；
- reset 时，`_update_terrain_curriculum()` 可以拿它来判断：
	- 是机器人没走好；
	- 还是命令本身就要求得不多。

这比只看最终位移更公平，尤其在命令会多次切换的情况下。

### 11.9 turn-over 模式为什么要强制清命令

在翻倒恢复训练里，如果机器人刚 reset 成侧翻或仰翻状态，就立刻收到一个普通高速命令，训练信号会很乱：

- 一方面 reward 希望它先起身；
- 另一方面 command 又在要求它立刻前进或转向。

所以当前版会在一段保护时间内，把命令强制清零。

这段时间里，环境的主要目标就从“跟踪速度任务”切换成了“先恢复站立姿态”。

### 11.10 新手二开 command，最容易漏掉哪几处

如果你以后想给这个框架增加新的命令，比如：

- 跳跃强度；
- 目标高度；
- 特殊 gait phase；

通常至少要同步检查下面几层：

1. [legged_gym/envs/base/legged_robot_config.py](../legged_gym/envs/base/legged_robot_config.py) 里的 `num_commands` 与范围配置；
2. [legged_gym/envs/base/legged_robot.py](../legged_gym/envs/base/legged_robot.py) 的 `_resample_commands()`；
3. [legged_gym/envs/base/legged_robot.py](../legged_gym/envs/base/legged_robot.py) 或 [legged_gym/envs/go2/go2_env.py](../legged_gym/envs/go2/go2_env.py) 的 `compute_observations()`；
4. `_reward_tracking_*()` 或新增 reward；
5. 部署端 obs / command 对齐逻辑。

也就是说，**命令系统不是一个独立模块，而是贯穿配置、环境、奖励、观测、部署的横向主线。**

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
