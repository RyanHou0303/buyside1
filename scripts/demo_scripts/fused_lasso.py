import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

np.random.seed(42)
T = 300  # 300个时间步（天）

# 1. 造一个有 3 个阶段（2次突变）的真实规律 beta_true (台阶状)
beta_true = np.r_[np.full(100, 0.5), np.full(100, -0.5), np.full(100, 1.0)]

# 生成数据: y_t = x_t * beta_t + 噪音
x = np.random.randn(T)
y = x * beta_true + np.random.randn(T) * 0.4

# 2. 用 cvxpy 直接按公式写出 Fused Lasso 目标函数
beta = cp.Variable(T)  # 定义要优化的 T 个变量 (\beta_1, ..., \beta_T)
lam = 8.0              # 正则化强度 \lambda

# 完全照抄公式: sum((y_t - x_t * beta_t)^2) + lambda * sum(|beta_t - beta_{t-1}|)
loss = cp.sum_squares(y - cp.multiply(x, beta))
penalty = lam * cp.norm1(cp.diff(beta))  # cp.diff(beta) 就是 beta_t - beta_{t-1}

prob = cp.Problem(cp.Minimize(loss + penalty))
prob.solve()

# 3. 极简画图对比
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(beta_true, 'k--', lw=2, label='True Regime $\\beta_t$ (3 Regimes)')
ax.plot(beta.value, 'r-', lw=2.5, label=f'Fused Lasso $\\hat{{\\beta}}_t$ ($\\lambda={lam}$)')
ax.set(title='Fused Lasso: Detecting Structural Breaks', xlabel='Time Step $t$', ylabel='Coefficient $\\beta_t$')
ax.grid(True, ls='--', alpha=0.4)
ax.legend()
plt.show()