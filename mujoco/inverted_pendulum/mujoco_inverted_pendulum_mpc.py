import numpy as np
import casadi as ca
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt

M = 1.0      # 小车质量 (kg)
m = 1      # 摆杆质量 (kg)
l = 1.0      # 摆杆长度 (m)
L = l
l2 = l/2     # 摆杆质心到关节的距离 (m)
b = 1.0      # 小车阻尼系数 (N/m/s)
g = 9.81     # 重力加速度 (m/s²)
I = (1/3)*m*l**2
g = 9.81
gear = 1

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

# 离散化参数
dt = 0.02    # 与模型timestep一致
I = np.eye(4)
A_d = I + A * dt  # 欧拉离散化
B_d = B * dt

# 模型定义
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

print("Running simulation with gravity:", model.opt.gravity)

# MPC参数
N = 20       # 增加预测步长

x_ref = np.array([0.0,  # desired cart position
                  0.0,  # desired cart velocity
                  0.0,  # desired pendulum angle (upright)
                  0.0]) # desired pendulum angular velocity

# 修改后的关键部分
def mpc_controller(x0):
    opti = ca.Opti()
    u_seq = opti.variable(N)
    x_seq = opti.variable(4, N+1)

    opti.subject_to(x_seq[:, 0] == x0)

    for i in range(N):
        # Correct matrix multiplication for CasADi
        state_curr = x_seq[:, i]
        input_curr = u_seq[i]
        
        # Convert numpy arrays to CasADi expressions
        A_d_cas = ca.DM(A_d)
        B_d_cas = ca.DM(B_d)
        
        # Compute next state without reshape
        next_state = ca.mtimes(A_d_cas, state_curr) + ca.mtimes(B_d_cas, input_curr)
        opti.subject_to(x_seq[:, i+1] == next_state)

    # 控制输入约束匹配 gear=10
    opti.subject_to(opti.bounded(-100/gear, u_seq, 100/gear))  # 实际输入范围为 [-10, 10] N
    # # # 控制位移范围
    opti.subject_to(opti.bounded(-10, x_seq[0, :], 10))  # 实际位移范围为 [-1, 1] m
    # # # 控制角度范围
    opti.subject_to(opti.bounded(-np.pi/3, x_seq[2, :], np.pi/3))  # 实际角度范围为 [-30, 30] deg

    # 调整权重矩阵
    # Q = np.diag([1000, 1000, 1000000, 100])  # 增大角度相关权重
    Q = np.diag([10000, 1000, 1000, 100])
    R = np.array([[10]])
    
    Q_terminal = Q * 1000  # Increased terminal cost for better convergence

    # Simplified cost function
    cost = 0
    for i in range(N):
        state_error = x_seq[:, i] - x_ref
        cost += ca.mtimes([state_error.T, Q, state_error]) + R[0,0] * u_seq[i]**2

    final_error = x_seq[:, -1] - x_ref
    cost += ca.mtimes([final_error.T, Q_terminal, final_error])

    opti.minimize(cost)

    # 启用 IPOPT 日志
    opts = {"ipopt.print_level": 5, "print_time": 1}
    opti.solver('ipopt', opts)
    sol = opti.solve()

    # 检查控制输入方向，必要时反转符号
    u = sol.value(u_seq[0])
    return u  # 若 MuJoCo 中方向相反，通过负号修正

frame = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 15
    viewer.cam.lookat[0] = 0
    history = []
    # 初始化状态（稍作扰动）
    data.qpos[1] = np.deg2rad(-40)  # 初始倾斜5度
    
    while viewer.is_running():

        # 获取当前状态 [位置, 角度, 速度, 角速度]
        x = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        
        # 计算控制输入
        u = mpc_controller(x)
        
        # 施加控制
        data.ctrl[0] = u * gear
        
        history.append(x.copy())

        # 仿真步进
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001 / 60)  # 约60Hz刷新率

# ========================
# 结果可视化（可选）
# ========================
t = np.arange(len(history)) * model.opt.timestep
states = np.array(history)

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