import mujoco
import numpy as np

# 1. 加载你的机器人XML模型 (MJCF或URDF格式)
model_path = "/home/sha-opt/moe/resources/robots/qianxihouzhouzu/urdf/qianxihouzhouzu.xml"
model = mujoco.MjModel.from_xml_path(model_path)

# ---------------- 方法 A：获取整个场景中所有物体的总质量 ----------------
# 注意：索引 0 通常是 worldbody，其质量始终为 0。
total_scene_mass = np.sum(model.body_mass)
print(f"场景所有活动刚体总质量: {total_scene_mass:.4f} kg")

# ---------------- 方法 B：获取特定机器人的总质量（推荐） ----------------
# 在有多个物体（如地面、桌子、其他机器人）的场景中，这种方法最严谨。
# 假设你的机器人根节点（比如 URDF 中的 base_link 或躯干）的名称为 "base_link"
robot_root_name = "trunk" 

try:
    # 获取该节点在模型中的 ID
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_root_name)
    
    # 提取以该节点为根的子树总质量
    robot_mass = model.body_subtreemass[body_id]
    print(f"机器人 '{robot_root_name}' 的总质量: {robot_mass:.4f} kg")
    
except KeyError:
    print(f"错误: 找不到名称为 '{robot_root_name}' 的刚体，请检查XML文件中的命名。")

# ---------------- 方法 C：查看每个连杆（Link/Body）的具体质量 ----------------
print("\n各连杆详细质量:")
for i in range(model.nbody):
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    mass = model.body_mass[i]
    if body_name: # 跳过没有名称的节点
        print(f"  - {body_name}: {mass:.4f} kg")