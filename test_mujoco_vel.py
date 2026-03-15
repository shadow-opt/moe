import mujoco
m = mujoco.MjModel.from_xml_string('''
<mujoco>
  <worldbody>
    <body pos="0 0 1">
      <freejoint/>
      <geom size="0.1"/>
      <site name="imu_site"/>
    </body>
  </worldbody>
  <sensor>
    <gyro site="imu_site"/>
    <velocimeter site="imu_site"/>
  </sensor>
</mujoco>
''')
d = mujoco.MjData(m)
d.qpos[3:7] = [0.7071068, 0, 0, 0.7071068]
d.qvel[3:6] = [1, 0, 0] # say we put 1 in qvel[3]
# set translation vel: global X = 1
d.qvel[:3] = [1, 0, 0]
mujoco.mj_step(m, d)
print("qvel translation:", d.qvel[:3])
print("qvel rotation:", d.qvel[3:6])
print("gyro (local rot):", d.sensordata[0:3])
print("velocimeter (local trans):", d.sensordata[3:6])
