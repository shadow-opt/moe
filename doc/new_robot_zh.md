# 新机器人接入详解：从训练到 Sim2Sim 与部署

> 面向对象：Python 基础尚可、第一次接触本仓库、希望把“不是 Go2 的其他四足机器人”接入本框架的开发者。  
> 目标：让你从“已有 URDF/MJCF，但不知道从哪里改”走到“能把新机器人跑通训练、导出、Mujoco Sim2Sim，并理解真机部署该怎么迁移”。

[TOC]

---

## 0. 先说结论：这个仓库里哪些是通用的，哪些是 Go2 专用的

如果你是第一次做迁移，最重要的不是先改代码，而是先分清楚：

### 0.1 通用部分

下面这些能力，本质上都可以迁移到其他四足机器人：

- 用 Isaac Gym 训练策略；
- 用 `task_registry` 管理任务；
- 用 `play.py` 加载模型和导出策略；
- 用 TorchScript 策略在 Mujoco 里做 Sim2Sim；
- 用“观测 → 策略 → 动作 → 目标关节角”的控制闭环做部署。

对应入口主要是：

- [legged_gym/utils/task_registry.py](../legged_gym/utils/task_registry.py)
- [legged_gym/scripts/train.py](../legged_gym/scripts/train.py)
- [legged_gym/scripts/play.py](../legged_gym/scripts/play.py)
- [legged_gym/utils/exporter.py](../legged_gym/utils/exporter.py)
- [deploy/deploy_mujoco/deploy_go2.py](../deploy/deploy_mujoco/deploy_go2.py)

### 0.2 Go2 专用部分

下面这些实现明显是围绕 Go2 写死或半写死的：

- Go2 的训练配置和奖励设计；
- Go2 的 URDF / Mujoco XML / 关节名字；
- Go2 的默认站立位、PD 参数、关节顺序；
- Go2 的真机通信接口；
- 基于 `unitree_sdk2py` 的低层控制与遥控器链路。

对应文件主要是：

- [legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py)
- [legged_gym/envs/go2/go2_env.py](../legged_gym/envs/go2/go2_env.py)
- [deploy/deploy_mujoco/configs/go2.yaml](../deploy/deploy_mujoco/configs/go2.yaml)
- [deploy/deploy_real/configs/go2.yaml](../deploy/deploy_real/configs/go2.yaml)
- [deploy/deploy_real/deploy_real_go2.py](../deploy/deploy_real/deploy_real_go2.py)

### 0.3 对“其他品牌四足”最现实的预期

如果你的机器人不是 Unitree 系列，那么通常应这样理解本仓库：

1. **训练链路可以复用**；
2. **Sim2Sim 链路大概率可以复用**；
3. **真机部署思路可以复用，但通信层通常必须重写**。

所以，本教程会把重点放在：

- 如何让新机器人先在训练端站起来；
- 如何导出策略；
- 如何在 Mujoco 中验证观测、动作、关节顺序都一致；
- 如何把 Go2 的真机控制器改造成“适配你自己机器人 SDK 的控制器”。

---

## 1. 推荐学习顺序

如果你是 noob，建议不要一上来就改十几个文件。

### 第一阶段：先看懂调用链

先读：

1. [README_zh.md](../README_zh.md)
2. [doc/learning_zh.md](./learning_zh.md)
3. [legged_gym/envs/__init__.py](../legged_gym/envs/__init__.py)
4. [legged_gym/utils/task_registry.py](../legged_gym/utils/task_registry.py)
5. [legged_gym/scripts/play.py](../legged_gym/scripts/play.py)

目标只有一个：

> 你必须知道 `--task=xxx` 最终会绑定到哪个环境类、哪份环境配置、哪份训练配置。

### 第二阶段：再看“Go2 是怎么被接进来的”

重点读：

1. [legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py)
2. [legged_gym/envs/go2/go2_env.py](../legged_gym/envs/go2/go2_env.py)
3. [resources/robots/go2/urdf/go2.urdf](../resources/robots/go2/urdf/go2.urdf)
4. [deploy/deploy_mujoco/configs/go2.yaml](../deploy/deploy_mujoco/configs/go2.yaml)
5. [deploy/deploy_real/configs/go2.yaml](../deploy/deploy_real/configs/go2.yaml)

目标是弄明白：

- 训练端吃什么 obs；
- 策略输出什么 action；
- action 怎样变成目标关节角；
- Sim2Sim 和真机为什么都要手动重建同样的 obs。

### 第三阶段：最后再迁移你的机器人

如果你的新机器人已经有 URDF / MJCF，那么建议迁移顺序一定是：

1. 先做训练接入；
2. 再做 `play.py` 验证；
3. 再导出 `policy.pt`；
4. 再做 Mujoco Sim2Sim；
5. 最后才做真机部署。

不要跳步骤。因为真机问题最复杂，而 Sim2Sim 正是你排查“模型、obs、动作、关节映射”是否一致的中间层。

---

## 2. 你要先建立的整体心智模型

这个仓库的主链路可以简化成下面这条：

**机器人资源** → **环境配置** → **任务注册** → **训练** → **Play/导出** → **Sim2Sim** → **真机部署**

更具体一点：

1. 机器人资源文件告诉仿真器“机器人长什么样”；
2. 环境配置告诉训练器“观测多少维、奖励怎么配、控制参数是多少”；
3. 任务注册把 `task` 名字和环境/训练配置绑定起来；
4. 训练输出 checkpoint；
5. `play.py` 会加载 checkpoint 并导出 `policy.pt` / `policy.onnx`；
6. Mujoco 和真机部署都会重新构建 obs，并把策略输出转成目标关节角；
7. 只要 obs 定义、action 含义、关节顺序三者一致，同一策略才可能跨端工作。

对新手来说，最关键的一句话是：

> **迁移新机器人，本质上不是“换一个模型文件”，而是让训练端、导出端、Sim2Sim、部署端对同一份策略的输入输出解释完全一致。**

---

## 3. 先看已有任务是怎么注册的

任务注册在 [legged_gym/envs/__init__.py](../legged_gym/envs/__init__.py) 中。

目前仓库注册的全是 Go2 任务，例如：

- `go2`
- `go2_cts`
- `go2_moe_cts`
- `go2_moe_ng_cts`
- `go2_mcp_cts`
- `go2_ac_moe_cts`
- `go2_dual_moe_cts`

这里有两个新手必须知道的事实：

### 3.1 `task` 不是随便起个名字

你在命令行里运行：

```bash
python legged_gym/scripts/train.py --task=go2_moe_cts
```

并不是直接去找某个同名文件，而是去任务注册表里查这个名字。

### 3.2 同一个环境类可以绑定多种训练配置

当前注册关系本质上是：

- 环境类：`Go2Robot`
- 环境配置：`GO2Cfg`
- 训练配置：`GO2CfgPPO` / `GO2CfgCTS` / `GO2CfgMoECTS` 等

也就是说：

- “机器人本体怎么仿真”可以不变；
- “算法怎么训练”可以变。

这对新机器人迁移很重要：

- 如果你只是换机器人本体，通常要先搞定新的环境类/配置类；
- 如果你只是换算法，很多时候只要增一个训练配置类即可。

---

## 4. 新机器人迁移时，最少要改哪些东西

下面是最实用的一张表。

| 层级 | 你要改什么 | 对应位置 | 不改会怎样 |
| --- | --- | --- | --- |
| 机器人资源 | URDF / MJCF / XML | `resources/robots/<your_robot>/` | 仿真都起不来 |
| 环境配置 | 默认关节角、资产路径、PD、obs/action 维度 | `legged_gym/envs/<your_robot>/...` | 训练动作语义错 |
| 环境类 | 观测、奖励、特殊逻辑 | `legged_gym/envs/<your_robot>/<your_robot>_env.py` | 模型输入不对 |
| 任务注册 | `task_registry.register(...)` | [legged_gym/envs/__init__.py](../legged_gym/envs/__init__.py) | `--task` 不能用 |
| 导出验证 | `play.py` 导出策略 | [legged_gym/scripts/play.py](../legged_gym/scripts/play.py) | 无法给 sim2sim / 部署用 |
| Sim2Sim | YAML + 关节映射 + XML | `deploy/deploy_mujoco/` | 模型能跑但动作错位 |
| 真机部署 | 状态订阅、命令发送、关节索引 | `deploy/deploy_real/` 或新目录 | 上机后高风险 |

如果你现在只有精力做一件事，请先做：

> **把训练端观测定义、Mujoco 端观测定义、真机端观测定义一项一项对齐。**

---

## 5. 训练端：先从 Go2 配置里抄出你的第一个机器人版本

### 5.1 Go2 配置为什么值得先读

[legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py) 几乎就是“这个机器人如何被训练”的总开关。

这里至少定义了：

- 初始位姿；
- 默认关节角；
- obs 维度；
- privileged obs 维度；
- 随机化策略；
- 控制类型与 PD 参数；
- 地形课程；
- 命令采样范围；
- 奖励项；
- 资产路径；
- PPO / CTS / MoE 等训练配置。

### 5.2 你应该如何新建自己的配置

推荐做法不是直接在 Go2 文件里硬改，而是：

1. 新建目录 `legged_gym/envs/<your_robot>/`；
2. 新建 `<your_robot>_config.py`；
3. 从 `GO2Cfg` 开始复制一份最小可跑版本；
4. 先删掉你看不懂但明显不是必须的高级项，保留最基本结构。

最开始你至少要改这些：

#### A. 资产路径

Go2 当前使用：

- `asset.file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'`

你必须换成自己的 URDF。

#### B. 默认关节角

也就是 `init_state.default_joint_angles`。

这组值极其关键，因为部署与 Sim2Sim 最终都会使用：

$$
q_{target} = q_{default} + action \times action\_scale
$$

如果默认站姿不对：

- 训练难以收敛；
- Mujoco 站不住；
- 真机上很危险。

#### C. 控制参数

至少包括：

- `control_type`
- `stiffness`
- `damping`
- `action_scale`
- `decimation`

这些值会直接影响策略输出到底代表多大幅度的关节偏移。

#### D. 观测/动作维度

Go2 当前是：

- `num_actions = 12`（基类默认）
- `num_observations = 45`

如果你的机器人不是 12 关节四足，这里一定要改。

如果你的机器人仍然是 12 关节四足，但你修改了 obs 项，也要同步改 `num_observations`。

#### E. 足端/body 名称

至少包括：

- `asset.foot_name`
- `asset.penalize_contacts_on`
- `asset.terminate_after_contacts_on`

因为接触检测、奖励、终止条件都依赖这些 body 名称。

### 5.3 如果你的机器人和 Go2 差很多怎么办

如果只是“别的四足，但同样 12 关节、关节控制逻辑相近”，可以从 Go2 改。

如果差异很大，例如：

- 关节数量不同；
- 观测项大改；
- 需要不同的动作解释；
- 控制接口不是目标关节角；

那你最好把 Go2 当作参考样例，而不是直接改皮肤。

---

## 6. 环境类：决定你的策略到底看见什么

### 6.1 先看 Go2 观测长什么样

Go2 在 [legged_gym/envs/go2/go2_env.py](../legged_gym/envs/go2/go2_env.py) 中重写了 `compute_observations()`。

当前普通 obs 的组成是：

1. 机体角速度 3 维；
2. 重力方向 3 维；
3. 命令 3 维；
4. 关节位置误差 12 维；
5. 关节速度 12 维；
6. 上一时刻动作 12 维。

总计：

$$
3 + 3 + 3 + 12 + 12 + 12 = 45
$$

这和配置里的 `num_observations = 45` 一一对应。

### 6.2 这件事为什么是迁移的第一优先级

因为下面三端都必须认同同一份 obs 定义：

- 训练端：[legged_gym/envs/go2/go2_env.py](../legged_gym/envs/go2/go2_env.py)
- Mujoco 端：[deploy/deploy_mujoco/deploy_go2.py](../deploy/deploy_mujoco/deploy_go2.py)
- 真机端：[deploy/deploy_real/deploy_real_go2.py](../deploy/deploy_real/deploy_real_go2.py)

如果你在训练端改了 obs，但没有同步改 sim2sim / 部署端，典型后果就是：

- 模型能加载，但动作完全异常；
- 看起来像“策略没学会”；
- 实际上是“输入维度或输入语义错了”。

### 6.3 迁移时怎么判断要不要重写环境类

你可以按下面标准判断：

#### 情况 A：结构基本不变

如果你的机器人仍然是标准四足，而且你想沿用：

- 角速度；
- 重力投影；
- 命令；
- 关节位置/速度；
- 上一动作；

那么你可以先保留和 Go2 类似的 obs 结构，只是替换：

- 关节数量；
- 默认角；
- 接触逻辑；
- 奖励。

#### 情况 B：结构明显不同

如果你的机器人需要额外信息，例如：

- 足端状态；
- 电机温度；
- 轮速；
- 特定传感器；

那就必须重写 `compute_observations()`，并同步更新：

- `num_observations`
- Mujoco 端 obs 重建
- 真机端 obs 重建

### 6.4 初学者强烈建议：先别改 obs 结构

如果你的目标只是“先把新机器人跑起来”，建议第一版尽量保持 Go2 风格 obs：

- 先让维度和语义简单一致；
- 等最小系统跑通，再增加传感器或高级信息。

这是因为 obs 每加一项，至少要在三处同步一次。

---

## 7. 奖励和课程：先保守，再逐步贴近你的机器人

### 7.1 Go2 奖励不等于你的机器人奖励

Go2 的奖励项和超参是为 Go2 调过的，尤其这些内容都很“机器人特定”：

- 站立高度；
- 关节极限；
- 躯干姿态；
- 脚间距；
- 接触惩罚；
- terrain curriculum；
- command curriculum。

因此，你不能简单认为“Go2 能用，你的机器人也一定能用”。

### 7.2 但第一版不要改太多

对新手更好的策略是：

1. 第一版尽量继承 Go2 奖励结构；
2. 只改明显不适配的项；
3. 等机器人能稳定站立/前进后，再逐步细调。

优先检查：

- `base_height_target`
- `max_contact_force`
- `soft_dof_pos_limit`
- 接触相关 body 名称
- `action_scale`
- `commands.ranges`

### 7.3 什么叫“明显不适配”

例如：

- 你的机器人比 Go2 高很多，`base_height_target` 不能照抄；
- 你的机器人扭矩能力更大，`max_contact_force` 不能照抄；
- 你的默认关节角完全不同，脚间距类奖励可能失真；
- 你的膝关节/髋关节顺序不同，关节限位奖励会错位。

---

## 8. 资源文件：URDF 和 MJCF 应该怎么放

### 8.1 推荐目录结构

建议参考 Go2 的组织方式：

```text
resources/
  robots/
    <your_robot>/
      urdf/
        <your_robot>.urdf
      assets/
      dae/
      <your_robot>.xml
      flat.xml
      stairs.xml
```

即使你现在没有全部文件，也建议从一开始就把目录分清：

- 训练端主要依赖 URDF；
- Mujoco 端主要依赖 XML/MJCF；
- 贴图/mesh 资源集中放到同一机器人目录下。

### 8.2 如果你只有 URDF 没有 MJCF

那你可以：

1. 先完成训练端；
2. 再补 Mujoco 模型；
3. 等 MJCF 准备好后再做 Sim2Sim。

但注意：

- 没有 MJCF，就没法复用当前的 `deploy/deploy_mujoco` 直接验证；
- 这样你会少掉一个很重要的中间排错层。

### 8.3 如果你已经有 MJCF

那很好。你最该做的是先确认：

- 关节名字是否和训练端一致；
- 关节顺序是否一致；
- 零位是否一致；
- 正方向是否一致；
- 躯干四元数定义是否一致。

这些比“地形多不多”更重要。

---

## 9. 任务注册：让 `--task=<your_robot>` 真正可用

### 9.1 任务工厂在做什么

[legged_gym/utils/task_registry.py](../legged_gym/utils/task_registry.py) 里主要做两件事：

- `make_env()`：根据 task 创建环境；
- `make_alg_runner()`：根据 task 创建训练器并处理 resume。

也就是说，只要你把机器人环境和配置写好了，但没有注册，那么命令行还是找不到它。

### 9.2 新机器人接入的最小注册步骤

你最终至少要做到：

1. 在 `legged_gym/envs/<your_robot>/` 放好环境类和配置类；
2. 在 [legged_gym/envs/__init__.py](../legged_gym/envs/__init__.py) 里导入它们；
3. 调用 `task_registry.register(...)` 注册名字。

推荐先只注册一个最小任务，例如：

- `<your_robot>` → 基础 PPO 或 CTS

先不要一口气注册五六个 MoE 变体。

原因很简单：

- 你现在最需要验证的是机器人建模和观测/动作一致性；
- 不是算法多样性。

---

## 10. 训练与 Play：先证明训练端是通的

### 10.1 先从最小训练开始

当你完成最小接入后，第一目标不是“高分”，而是：

- 环境可以正常 reset；
- 不会立刻发散；
- 机器人有基本站立趋势；
- reward 曲线不是完全异常。

### 10.2 为什么一定要跑 `play.py`

[legged_gym/scripts/play.py](../legged_gym/scripts/play.py) 不只是“看看效果”，它还是导出策略的入口。

当前 `play.py` 会做这些事：

1. 创建测试环境；
2. 恢复训练模型；
3. 取出推理策略；
4. 导出：
   - `policy.pt`
   - `policy.onnx`
   - `policy.pkl`

导出逻辑来自：

- [legged_gym/scripts/play.py](../legged_gym/scripts/play.py)
- [legged_gym/utils/exporter.py](../legged_gym/utils/exporter.py)

### 10.3 为什么 `policy.pt` 这么重要

因为当前仓库里：

- Mujoco Sim2Sim 用的是 `torch.jit.load(policy_path)`；
- Python 真机部署也用的是 `torch.jit.load(policy_path)`。

这意味着：

> 只要导出的 TorchScript 形式不变，Sim2Sim 和部署侧都可以复用相同的策略文件接口。

---

## 11. 观测、动作、默认角：三端必须一致

这是整篇教程最重要的一节。

### 11.1 训练端动作语义

当前 Go2 动作最终解释为：

$$
q_{target} = q_{default} + a \times s
$$

其中：

- $q_{target}$：目标关节角；
- $q_{default}$：默认关节角；
- $a$：策略输出动作；
- $s$：`action_scale`。

### 11.2 Mujoco 端动作语义

在 [deploy/deploy_mujoco/deploy_go2.py](../deploy/deploy_mujoco/deploy_go2.py) 中，同样使用：

- `action_scale`
- `default_angles`
- `target_dof_pos = action * action_scale + default_angles`

### 11.3 真机端动作语义

在 [deploy/deploy_real/deploy_real_go2.py](../deploy/deploy_real/deploy_real_go2.py) 中，同样使用：

- `self.config.default_angles`
- `self.config.action_scale`

因此你必须保证下面三份配置语义一致：

1. 训练端 `<your_robot>_config.py`
2. `deploy/deploy_mujoco/configs/<your_robot>.yaml`
3. `deploy/deploy_real/configs/<your_robot>.yaml`

如果这三处不一致，会出现非常典型的问题：

- Sim2Sim 里关节摆幅明显过大/过小；
- 真机站立时腿折叠或顶死；
- 模型在训练里看似正常，换端后完全异常。

### 11.4 一致性检查表

每次迁移都建议手工核对：

- 默认站姿是否同一套数值；
- `action_scale` 是否一致；
- 关节顺序是否一致；
- 关节正方向是否一致；
- obs 中关节位置是否都减去了同样的默认角；
- obs 中上一时刻动作是否按同样顺序填入。

---

## 12. Sim2Sim：这是新机器人迁移的核心验证层

### 12.1 为什么先做 Sim2Sim 而不是先做真机

因为 Sim2Sim 能帮你排除大量问题，而成本远低于真机：

- 模型文件是否正确；
- 观测维度是否对齐；
- 关节顺序是否对齐；
- 默认角是否对齐；
- `action_scale` 是否对齐；
- PD 是否大致合理；
- 命令输入是否生效。

如果 Sim2Sim 都跑不通，直接上真机几乎等于盲飞。

### 12.2 Go2 的 Sim2Sim 入口有哪些

相关文件是：

- [deploy/deploy_mujoco/deploy_go2.py](../deploy/deploy_mujoco/deploy_go2.py)
- [deploy/deploy_mujoco/deploy_go2_moe.py](../deploy/deploy_mujoco/deploy_go2_moe.py)
- [deploy/deploy_mujoco/configs/go2.yaml](../deploy/deploy_mujoco/configs/go2.yaml)

其中 YAML 配置最关键。

### 12.3 迁移到新机器人时，YAML 至少要改哪些项

假设你新增 `deploy/deploy_mujoco/configs/<your_robot>.yaml`，至少要逐项确认：

#### A. 模型路径

- `policy_path`

应指向你导出的 `policy.pt`。

#### B. Mujoco 模型路径

- `xml_path`

应指向你自己的 MJCF/XML。

#### C. 控制频率

- `simulation_dt`
- `control_decimation`

需要满足：

$$
control\_dt = simulation\_dt \times control\_decimation
$$

并尽量和训练侧策略频率一致。

#### D. 控制参数

- `kps`
- `kds`
- `default_angles`
- `action_scale`

#### E. 观测相关参数

- `lin_vel_scale`
- `ang_vel_scale`
- `dof_pos_scale`
- `dof_vel_scale`
- `cmd_scale`
- `num_actions`
- `num_obs`

#### F. 关节映射

- `mujoco_joint_names`
- `model_joint_names`

这是最容易被忽略、但最要命的一项。

### 12.4 关节映射为什么这么重要

在 [deploy/deploy_mujoco/deploy_go2.py](../deploy/deploy_mujoco/deploy_go2.py) 中，代码会根据：

- `mujoco_joint_names`
- `model_joint_names`

构造两个索引：

- `idx_model2mj`
- `idx_mj2model`

用途分别是：

- 把模型输出动作顺序映射到 Mujoco 顺序；
- 把 Mujoco 当前关节状态映射回模型训练顺序。

如果你的训练端和 Mujoco 端关节顺序不一致，但你没有显式映射，那么模型输入输出都会错位。

这类错误的症状通常是：

- 看起来所有数值都正常；
- 机器人却做出完全莫名其妙的动作。

### 12.5 新机器人 Sim2Sim 的推荐调试顺序

建议不要一上来就开复杂地形。

#### 第一步：平地 + 固定零命令

先确认：

- 不会炸飞；
- 能基本站住；
- 姿态不明显扭曲。

#### 第二步：平地 + 固定前进命令

再确认：

- 前进方向是否正确；
- 左右/前后腿是否错位；
- 摆腿频率是否异常。

#### 第三步：手柄遥控

再检查：

- 线速度命令方向；
- 横向命令方向；
- 偏航命令方向。

#### 第四步：复杂地形

最后再上：

- stairs
- slope
- race track
- 你自己的 terrain

### 12.6 如果 Sim2Sim 站不住，先查什么

优先级建议按这个顺序：

1. `default_angles`
2. `action_scale`
3. 关节顺序映射
4. 关节正方向
5. `kps/kds`
6. obs 拼接顺序
7. IMU / 四元数定义
8. 控制频率

---

## 13. 真机部署：你应该把 Go2 部署脚本当成“结构模板”

### 13.1 当前仓库的真机部署现状

当前 Python 真机部署主要在：

- [deploy/deploy_real/deploy_real_go2.py](../deploy/deploy_real/deploy_real_go2.py)
- [deploy/deploy_real/config_go2.py](../deploy/deploy_real/config_go2.py)
- [deploy/deploy_real/configs/go2.yaml](../deploy/deploy_real/configs/go2.yaml)

但这套实现明显依赖：

- `unitree_sdk2py`
- Go2 的消息类型
- Go2 的 topic
- Go2 的遥控器输入
- Go2 的电机索引顺序

也就是说：

> 如果你的机器人不是 Unitree Go2，那么这部分不能直接拿来跑，但它非常适合作为“自己部署控制器的蓝本”。

### 13.2 你真正该复用的是什么

不是通信细节，而是控制器主循环结构。

当前 Go2 控制器的通用逻辑可以概括为：

1. 读取机器人低层状态；
2. 提取 IMU 和关节状态；
3. 构建 obs；
4. 用 `torch.jit.load()` 加载的策略推理；
5. 把动作变成目标关节角；
6. 写入电机命令并发送；
7. 在安全状态下循环执行。

这个结构本身对任何四足都是成立的。

### 13.3 非 Unitree 机器人最少要自己实现什么

如果你要迁移到其他品牌四足，至少要自己搞定以下接口：

#### A. 状态输入接口

你需要能够获得：

- IMU 四元数；
- IMU 角速度；
- 每个关节位置；
- 每个关节速度；
- 可选：遥控或速度命令。

#### B. 命令输出接口

你需要能够发送：

- 目标关节角；
- 或等价的低层控制指令；
- 可配置的 `kp` / `kd`；
- 安全停止/阻尼模式。

#### C. 索引与映射

你需要明确：

- SDK 返回的电机顺序；
- 模型训练使用的关节顺序；
- 两者是否需要映射。

#### D. 安全流程

至少包括：

- 零力矩态；
- 缓慢 move-to-default；
- 进入策略控制；
- 紧急退出。

### 13.4 为什么不建议直接照抄 Go2 的通信层

因为 `deploy_real_go2.py` 里很多内容和 Go2 硬绑定，例如：

- `unitree_go_msg_dds__LowCmd_`
- `unitree_go_msg_dds__LowState_`
- `LowCmdGo`
- `LowStateGo`
- `wireless_remote`
- Go2 的 `joint2motor_idx`

这些不是“通用四足部署接口”，而是“Go2 的具体落地实现”。

### 13.5 你可以怎么拆分自己的部署实现

如果你要写自己的部署侧，建议按下面拆：

#### 第一层：硬件通信适配层

职责：

- 连接机器人 SDK；
- 读状态；
- 发命令；
- 处理安全退出。

#### 第二层：策略控制层

职责：

- 维护 `obs` buffer；
- 调用 `policy(obs)`；
- 输出 `target_dof_pos`；
- 不直接关心底层通信协议。

#### 第三层：配置层

职责：

- 默认角；
- `action_scale`；
- `kp/kd`；
- 关节映射；
- 控制周期。

这样一来，以后你换机器人时，通常主要改第一层和第三层，第二层尽量不动。

---

## 14. 其他品牌四足做部署时，最关键的四个对齐

### 14.1 四元数顺序

不同 SDK 里四元数顺序可能是：

- `(w, x, y, z)`
- `(x, y, z, w)`

如果顺序错了，重力方向会直接错，obs 前 6 维基本报废。

### 14.2 角速度坐标系

你要确认：

- 角速度是否在机体系；
- 轴定义是否和训练一致；
- 是否需要符号翻转。

### 14.3 关节零位

“看起来差一点点”的零位偏差，最终都会体现在：

- obs 中关节位置偏移；
- 动作中心偏移；
- 站姿与训练不一致。

### 14.4 控制周期

如果训练是 50Hz 策略更新，但你真机部署成了 100Hz 或 20Hz，策略表现可能明显变差。

你至少要保证：

- 真机主循环频率和训练策略频率尽量接近；
- `control_dt` 与 `decimation * sim.dt` 逻辑一致。

---

## 15. 从已有 URDF/MJCF 出发的推荐实操路线

你现在的前提是：**已经有 URDF/MJCF**。

那么最推荐的实际顺序如下。

### 第 1 步：先整理资源与命名

目标：

- 确认 URDF 能被 Isaac Gym 正常加载；
- 确认 MJCF 能被 Mujoco 正常加载；
- 列出训练顺序关节名；
- 列出 Mujoco 顺序关节名；
- 列出真实电机顺序。

建议你先手工做一张三列表：

| 训练顺序 | Mujoco 顺序 | 真机电机顺序 |
| --- | --- | --- |
| joint_1 | joint_1 | motor_7 |
| joint_2 | joint_2 | motor_3 |

这张表会在整个迁移过程中反复用到。

### 第 2 步：做最小训练接入

目标：

- 新建机器人配置；
- 新建环境类；
- 注册一个最小 task；
- 能开始训练。

### 第 3 步：用 `play.py` 验证并导出

目标：

- 能正常恢复模型；
- 能在 Isaac Gym 里看见基本运动；
- 导出 `policy.pt`。

### 第 4 步：做平地 Sim2Sim

目标：

- 在 Mujoco 平地站稳；
- 前进命令正确；
- 关节顺序正确。

### 第 5 步：做复杂地形 Sim2Sim

目标：

- 再验证 stairs / slope / rough terrain；
- 观察是否出现和训练端完全不同的失稳模式。

### 第 6 步：最后做真机部署

目标：

- 先完成安全上电和回默认位；
- 再接策略；
- 从很小命令开始。

---

## 16. 新机器人迁移时最容易踩的坑

### 坑 1：只改了训练端，没改部署端 obs

症状：

- 模型能加载；
- 推理也没报错；
- 但动作完全离谱。

### 坑 2：`num_obs` 改了，但 YAML 里没改

症状：

- 直接维度错误；
- 或脚本里 obs buffer 分段错位。

### 坑 3：默认角不一致

症状：

- 训练端没问题；
- Sim2Sim/真机站姿严重不对。

### 坑 4：关节顺序一致，但关节方向相反

症状：

- 表面看起来“映射是对的”；
- 实际上腿会朝完全相反的方向摆。

### 坑 5：MJCF 关节名和 URDF 关节名不同

症状：

- 你以为是同一关节；
- 代码里其实无法正确建立映射。

### 坑 6：部署控制频率和训练不一致

症状：

- 仿真里稳，真机上颤；
- 或者动作迟钝、跟不上命令。

### 坑 7：直接上真机

症状：

- 排错成本极高；
- 很难分辨是模型问题、obs 问题还是 SDK 问题。

---

## 17. 给 noob 的最小迁移清单

如果你只想知道“我现在到底该先干什么”，按下面清单走。

### 训练接入清单

- [ ] 在 `resources/robots/<your_robot>/` 放好资源文件
- [ ] 新建 `legged_gym/envs/<your_robot>/`
- [ ] 新建 `<your_robot>_config.py`
- [ ] 新建 `<your_robot>_env.py`
- [ ] 配好 `asset.file`
- [ ] 配好 `default_joint_angles`
- [ ] 配好 `action_scale`
- [ ] 配好 `num_observations`
- [ ] 在 [legged_gym/envs/__init__.py](../legged_gym/envs/__init__.py) 注册 task
- [ ] 先跑最小训练

### 导出与 Sim2Sim 清单

- [ ] 用 [legged_gym/scripts/play.py](../legged_gym/scripts/play.py) 导出 `policy.pt`
- [ ] 新建 `deploy/deploy_mujoco/configs/<your_robot>.yaml`
- [ ] 配好 `xml_path`
- [ ] 配好 `policy_path`
- [ ] 配好 `default_angles`
- [ ] 配好 `kps/kds`
- [ ] 配好 `num_obs`
- [ ] 配好 `num_actions`
- [ ] 配好 `mujoco_joint_names` / `model_joint_names`
- [ ] 先平地站立
- [ ] 再前进

### 真机部署清单

- [ ] 明确 SDK 提供的关节/IMU/命令接口
- [ ] 明确真机电机顺序
- [ ] 明确四元数顺序
- [ ] 明确控制周期
- [ ] 先做零力矩与回默认位
- [ ] 再接策略推理
- [ ] 先用很小速度命令测试

---

## 18. 建议你重点阅读的文件

下面这些文件非常值得按顺序读。

### 一级必读

- [legged_gym/envs/__init__.py](../legged_gym/envs/__init__.py)  
  任务注册入口，决定 `--task` 最终绑定什么。

- [legged_gym/utils/task_registry.py](../legged_gym/utils/task_registry.py)  
  环境和训练器工厂，是整个训练系统的中枢。

- [legged_gym/envs/go2/go2_config.py](../legged_gym/envs/go2/go2_config.py)  
  Go2 如何接入训练的最核心样板。

- [legged_gym/envs/go2/go2_env.py](../legged_gym/envs/go2/go2_env.py)  
  Go2 的 obs / privileged obs / 自定义奖励实现。

- [legged_gym/scripts/play.py](../legged_gym/scripts/play.py)  
  可视化与策略导出入口。

### 二级必读

- [legged_gym/envs/base/legged_robot_config.py](../legged_gym/envs/base/legged_robot_config.py)  
  所有配置类的基类结构。

- [legged_gym/envs/base/legged_robot.py](../legged_gym/envs/base/legged_robot.py)  
  通用四足环境逻辑，包括 step、reset、reward、terrain 等。

- [deploy/deploy_mujoco/deploy_go2.py](../deploy/deploy_mujoco/deploy_go2.py)  
  Sim2Sim 的主循环与 obs 重建逻辑。

- [deploy/deploy_mujoco/configs/go2.yaml](../deploy/deploy_mujoco/configs/go2.yaml)  
  Sim2Sim 迁移时最常改的一份配置。

- [deploy/deploy_real/deploy_real_go2.py](../deploy/deploy_real/deploy_real_go2.py)  
  真机部署控制器结构模板。

- [deploy/deploy_real/configs/go2.yaml](../deploy/deploy_real/configs/go2.yaml)  
  真机部署参数模板。

---

## 19. 一句最重要的迁移原则

最后再强调一次。

如果你正在把“其他品牌四足机器人”接入本仓库，请永远围绕下面这个原则检查：

> **训练端、Sim2Sim 端、真机端，必须对同一份策略的 obs 含义、action 含义、默认角、关节顺序、控制频率有完全一致的解释。**

只要这件事没对齐：

- 换算法没有意义；
- 调奖励帮助有限；
- 直接上真机风险很高。

只要这件事对齐了：

- 你就已经走完了迁移中最难的一半。

---

## 20. 你读完本文后，下一步最建议做什么

如果你现在准备真正开工，建议马上做下面三件事：

1. 为你的机器人列出一张“训练顺序 / Mujoco 顺序 / 真机电机顺序”的关节映射表；
2. 复制 Go2 的配置和环境文件，做一个最小 `<your_robot>` 版本；
3. 把目标定为“先平地 Sim2Sim 站稳”，而不是“先真机跑酷”。

如果这三步你都做到了，后面很多问题都会变成“可调试的工程问题”，而不是“毫无头绪的黑盒问题”。
