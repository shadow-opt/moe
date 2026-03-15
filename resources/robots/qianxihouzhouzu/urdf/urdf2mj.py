import mujoco

# 1. 指定你的 URDF 文件路径和期望生成的 XML 文件路径
urdf_file = "/home/sha-opt/moe/resources/robots/qianxihouzhouzu/urdf/qianxihouzhouzu.urdf"  # 请替换为实际的 URDF 文件名
xml_file = "/home/sha-opt/moe/resources/robots/qianxihouzhouzu/urdf/qianxihouzhouzu.xml"    # 生成的 MuJoCo 规范化文件

try:
    # 将 URDF 解析至内存，构建为底层的 mjModel 结构
    # MuJoCo 会自动识别 .urdf 后缀并调用相应的解析器
    model = mujoco.MjModel.from_xml_path(urdf_file)
    
    # 将内存中的 mjModel 重新导出为 MuJoCo 原生的规范化 XML (MJCF 格式)
    mujoco.mj_saveLastXML(xml_file, model)
    
    print(f"转换成功！文件已保存至: {xml_file}")
    
except Exception as e:
    print(f"转换失败，请检查 URDF 语法或文件路径。错误详情: {e}")