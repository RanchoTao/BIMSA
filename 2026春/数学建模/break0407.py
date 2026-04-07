import numpy as np
import matplotlib.pyplot as plt

# 1. 数据准备 (来自表1)
v_kmh = np.array([20, 40, 60, 80, 100, 120, 140])  # 车速，单位 km/h
d_m = np.array([6.5, 17.8, 33.6, 57.1, 83.4, 118.0, 153.5])  # 刹车距离，单位 m

# 2. 最小二乘法拟合模型 d = c1 * v + c2 * v^2
# 构建设计矩阵 X: 第一列为 v， 第二列为 v^2
X = np.column_stack((v_kmh, v_kmh**2))
# 使用最小二乘法求解参数 [c1, c2]
c1, c2 = np.linalg.lstsq(X, d_m, rcond=None)[0]

print("=== 题目1：刹车距离模型拟合结果 ===")
print(f"拟合得到的参数 c1 = {c1:.6f} m·h/km")
print(f"拟合得到的参数 c2 = {c2:.6f} m·h²/km²")
# 注意：c1的量纲可理解为 m/(km/h) = h/1000? 更准确说是：c1*v 得到米，所以c1单位是 m/(km/h)
# 实际上，c1是反应时间，单位应为 h（小时），但数值很小。1 h = 3600 s。
c1_hour = c1 / 1000  # 因为1 km = 1000 m，所以 c1 (m/(km/h)) 除以1000后，单位变为 h
print(f"司机反应时间 c1 ≈ {c1_hour*3600:.3f} s")  # 转换为秒

# 3. 估计刹车时的减速度 a (单位: m/s²)
# 公式: a = (1/(2*c2)) * (km/h -> m/s)^2
# (km/h 转换为 m/s 的系数: 1000/3600 = 5/18 ≈ 0.27778)
conversion_factor = (1000 / 3600)**2  # (5/18)^2
a_ms2 = 1 / (2 * c2) * conversion_factor
print(f"估计的刹车减速度 a ≈ {a_ms2:.3f} m/s²")
print("(注：普通家用车紧急刹车减速度约为 5-8 m/s²，此结果在合理范围内。)")

# 4. 对数据和拟合曲线作图
v_fit = np.linspace(min(v_kmh), max(v_kmh), 300)  # 生成平滑的拟合曲线车速点
d_fit = c1 * v_fit + c2 * (v_fit**2)  # 计算拟合的刹车距离

plt.figure(figsize=(8, 5))
plt.scatter(v_kmh, d_m, color='blue', label='原始实验数据', zorder=5)
plt.plot(v_fit, d_fit, color='red', linewidth=2, label=f'拟合曲线: d = {c1:.4f}v + {c2:.6f}v²')
plt.xlabel('车速 v (km/h)')
plt.ylabel('刹车距离 d (m)')
plt.title('刹车距离与车速的关系 (最小二乘法拟合)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# 输出拟合优度 R^2 作为参考
d_pred = c1 * v_kmh + c2 * (v_kmh**2)
ss_res = np.sum((d_m - d_pred)**2)
ss_tot = np.sum((d_m - np.mean(d_m))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"拟合优度 R² = {r_squared:.6f} (越接近1说明拟合越好)")

# 接题目1的代码，继续题目2的计算
print("\n=== 题目2：道路通行能力计算与分析 ===")
# 参数设置
c1 = 0.125201
c2 = 0.007714
d0 = 8.0  # 单位: 米， 取车身长度(约5米)的1.6倍

# 1. 计算不同车速下的通行能力 N
v_range = np.arange(20, 101, 10)  # 20, 30, ..., 100 km/h
N_values = 1000 * v_range / (c1 * v_range + c2 * v_range**2 + d0)

print("\n不同车速下的道路通行能力 N (辆/小时):")
for v, N in zip(v_range, N_values):
    print(f"  车速 v = {v:3d} km/h, 通行能力 N = {N:.1f} 辆/h")

# 2. 计算最大通行能力及其对应的最优车速
v_opt = np.sqrt(d0 / c2)  # 最优车速公式
N_max = 1000 / (c1 + 2 * np.sqrt(c2 * d0))  # 最大通行能力公式

print(f"\n理论计算（根据公式10）:")
print(f"  使通行能力最大的最优车速 v_opt = √(d0/c2) ≈ {v_opt:.2f} km/h")
print(f"  最大通行能力 N_max = {N_max:.1f} 辆/h")

# 3. 绘图展示 N-v 关系
v_plot = np.linspace(10, 120, 300)
N_plot = 1000 * v_plot / (c1 * v_plot + c2 * v_plot**2 + d0)

plt.figure(figsize=(8, 5))
plt.plot(v_plot, N_plot, linewidth=2, label=f'N(v)曲线 (d0={d0}m)')
plt.axvline(x=v_opt, color='red', linestyle='--', alpha=0.7, label=f'最优车速 v_opt={v_opt:.1f} km/h')
plt.axhline(y=N_max, color='green', linestyle='--', alpha=0.7, label=f'最大通行能力 N_max={N_max:.1f} 辆/h')
plt.scatter(v_range, N_values, color='blue', zorder=5, label='计算采样点')
plt.xlabel('车速 v (km/h)')
plt.ylabel('道路通行能力 N (辆/h)')
plt.title('道路通行能力 N 与车速 v 的关系')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# 4. 参数影响分析
print(f"\n参数对最大通行能力 N_max 的影响分析 (基于 N_max = 1000/(c1 + 2√(c2*d0)) ):")
print(f"  当前参数: c1={c1:.3f}, c2={c2:.4f}, d0={d0:.1f}m, N_max={N_max:.1f} 辆/h")
# 示例：改变d0
d0_new = 6.0
N_max_new = 1000 / (c1 + 2 * np.sqrt(c2 * d0_new))
print(f"  若安全距离 d0 减小至 {d0_new}m，则 N_max 增加至 {N_max_new:.1f} 辆/h")