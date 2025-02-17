import os
import subprocess
import time
import itertools
import numpy as np
import mujoco
import mujoco.viewer
import mediapy as media
import matplotlib.pyplot as plt

MJCF = """
<mujoco model="inverted pendulum">
	<compiler inertiafromgeom="true"/>
	<default>
		<joint armature="0" damping="1" limited="true"/>
		<geom contype="0" friction="0.1 0.1 0.1" rgba="0.7 0.7 0 1"/>
		<tendon/>
		<motor ctrlrange="-1000 1000"/>
	</default>
	<option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
	<size nstack="3000"/>
	<worldbody>
		<!--geom name="ground" type="plane" pos="0 0 0" /-->
		<geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 10" type="capsule"/>
		<body name="cart" pos="0 0 0">
			<joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-10 10" type="slide"/>
			<geom name="cart" mass="1.0" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
			<body name="pole" pos="0 0 0">
				<joint axis="0 1 0" name="hinge" pos="0 0 0" limited="false" range="-90 90" type="hinge"/>
				<geom fromto="0 0 0 0.0001 0 1.0" name="cpole" mass="1.0" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
				<!--                 <body name="pole2" pos="0.001 0 0.6"><joint name="hinge2" type="hinge" pos="0 0 0" axis="0 1 0"/><geom name="cpole2" type="capsule" fromto="0 0 0 0 0 0.6" size="0.05 0.3" rgba="0.7 0 0.7 1"/><site name="tip2" pos="0 0 .6"/></body>-->
			</body>
		</body>
	</worldbody>
	<actuator>
		<motor gear="1" joint="slider" name="slide"/>
	</actuator>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(MJCF)
data = mujoco.MjData(model)





# model.opt.gravity[2] = -0.01
print("Running simulation with gravity:", model.opt.gravity)

duration = 10000  # (seconds)
framerate = 60  # (Hz)


with mujoco.viewer.launch_passive(model, data) as viewer:
  
  while True:
    frame = 0
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    while True:
      mujoco.mj_step(model, data)
      viewer.sync()
      time.sleep(0.001 / framerate)
      frame += 1