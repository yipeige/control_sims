import numpy as np
import mujoco
import mujoco.viewer
import time
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

# ========================
# 系统物理参数（需与MJCF模型匹配）
# ========================
M = 1.0      # 小车质量 (kg)
m = 1      # 摆杆质量 (kg)
l = 1.0      # 摆杆长度 (m)
l2 = l/2     # 摆杆质心到关节的距离 (m)
b = 1.0      # 小车阻尼系数 (N/m/s)
g = 9.81     # 重力加速度 (m/s²)
gear = 1   # 电机传动比（与MJCF文件一致）

# ========================
# LQR控制器参数
# ========================
Q = np.diag([10000, 500, 1000000, 100])
R = np.array([[10]])           

def compute_linear_model():

    denominator = I*(M+m) + M*m*l**2

    A = np.array([
        [0, 1, 0, 0],
        [0, -(I+m*l**2)*b/denominator, (m**2*g*l**2)/denominator, 0],
        [0, 0, 0, 1],
        [0, m*l*b/denominator, m*g*l*(M+m)/denominator, 0]
    ])

    B = np.array([
        [0],
        [(I+m*l**2)/denominator],
        [0],
        [-m*l/denominator]
    ])

    return A, B

# ========================
# LQR增益计算
# ========================
def compute_lqr_gain(A, B, Q, R):
    """求解连续时间代数Riccati方程"""
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K

# ========================
# 加载MJCF模型
# ========================
MJCF = """
<mujoco model="inverted pendulum">
	<compiler inertiafromgeom="true"/>
	<default>
		<joint armature="0" damping="0.1" limited="true"/>
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

# ========================
# 初始化控制器
# ========================
I = (1/3)*m*l**2  # 摆杆转动惯量
A, B = compute_linear_model()
K = compute_lqr_gain(A, B, Q, R)

print("LQR增益矩阵K:\n", K)

# ========================
# 仿真主循环
# ========================
frame = 0

initial_deg = 50

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 10
    viewer.cam.lookat[0] = 0

    # 初始扰动（倾斜5度）
    data.qpos[1] = np.deg2rad(0)
    
    # 记录状态历史
    state_history = []
    
    while viewer.is_running():
        
        if frame == 100:
            data.qpos[1] = np.deg2rad(initial_deg)
            initial_deg += 10

        # 获取当前状态 [位置, 速度, 角度, 角速度]
        x = np.array([
            data.qpos[0],   # 小车位置
            data.qvel[0],   # 小车速度
            data.qpos[1],   # 摆杆角度
            data.qvel[1]    # 摆杆角速度
        ])
        
        # LQR控制律计算
        u = -K @ x
        
        # 考虑传动比和电机限制
        u_clipped = np.clip(u[0]*gear, -1000.0, 1000.0)  # 匹配MJCF的ctrlrange
        
        # 施加控制
        data.ctrl[0] = u_clipped / gear
        
        # 记录状态
        state_history.append(x.copy())
        
        # 仿真步进
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01/60)  # 匹配0.02s的timestep
        frame += 1

# ========================
# 结果可视化（可选）
# ========================
t = np.arange(len(state_history)) * model.opt.timestep
states = np.array(state_history)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, states[:, 0], label='Cart Position')
plt.plot(t, states[:, 2], label='Pendulum Angle')
plt.legend()
plt.title("State Response")

plt.subplot(2, 1, 2)
plt.plot(t, states[:, 1], label='Cart Velocity')
plt.plot(t, states[:, 3], label='Pendulum Angular Velocity')
plt.legend()
plt.show()