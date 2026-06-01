import mujoco

# 直接加载URDF文件
model = mujoco.MjModel.from_xml_path("/home/sha-opt/moe/resources/robots/aaaaa_fixed/aaa_fixed/urdf/z2.urdf")

# 或者保存为MuJoCo XML格式
mujoco.mj_saveLastXML("z2.xml", model)