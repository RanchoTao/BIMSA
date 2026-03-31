import numpy as np

# 表1数据：加速阶段
v_accel = np.array([0, 10, 20, 30, 40])  # km/h
t_accel = np.array([0, 1.6, 3.0, 4.2, 5.0])  # s

# 表2数据：减速阶段
v_decel = np.array([40, 30, 20, 10, 0])  # km/h
t_decel = np.array([0, 2.2, 4.0, 5.5, 6.8])  # s

# 最小二乘法拟合线性函数 t = c*v + d
def linear_fit(v, t):
    n = len(v)
    c = (n * np.sum(v * t) - np.sum(v) * np.sum(t)) / (n * np.sum(v**2) - np.sum(v)**2)
    d = (np.sum(t) - c * np.sum(v)) / n
    return c, d

c1, d1 = linear_fit(v_accel, t_accel)
c2, d2 = linear_fit(v_decel, t_decel)

print(f"加速阶段: c1 = {c1:.4f}, d1 = {d1:.4f}")
print(f"减速阶段: c2 = {c2:.4f}, d2 = {d2:.4f}")

# 将速度转换为 m/s 后拟合
v_accel_mps = v_accel / 3.6
v_decel_mps = v_decel / 3.6
c1_m, d1_m = linear_fit(v_accel_mps, t_accel)
c2_m, d2_m = linear_fit(v_decel_mps, t_decel)
a1 = 1 / c1_m  # 加速度 m/s^2
a2 = -1 / c2_m  # 减速度 m/s^2（取正值）

v_max = 40 / 3.6  # 限速 40 km/h 转换为 m/s
s = (v_max**2 / 2) * (1/a1 + 1/a2)
print(f"加速度 a1 = {a1:.4f} m/s^2, 减速度 a2 = {a2:.4f} m/s^2")
print(f"路障间距 s = {s:.4f} m")