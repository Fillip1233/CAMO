import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义高精度函数 f_high
def f_high(x1, x2):
    return x1**2 + 2 * x2**2 - 0.3 * np.cos(3 * np.pi * x1) - 0.4 * np.cos(4 * np.pi * x2) + 0.7

# 定义低精度函数 f_low
def f_low(x1, x2):
    return f_high(0.7 * x1, x2) + x1 * x2 - 12

# 生成网格数据
x1 = np.linspace(-5, 5, 200)
x2 = np.linspace(-5, 5, 200)
X1, X2 = np.meshgrid(x1, x2)

# 计算函数值
Z_high = f_high(X1, X2)
Z_low = f_low(X1, X2)

# 创建图像
fig = plt.figure(figsize=(14, 6))

# 高精度图像
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X1, X2, Z_high, cmap='viridis')
ax1.set_title('High-Fidelity Function $f_{high}(x)$')
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_zlabel('$f_{high}$')

# 低精度图像
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X1, X2, Z_low, cmap='plasma')
ax2.set_title('Low-Fidelity Function $f_{low}(x)$')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_zlabel('$f_{low}$')

plt.tight_layout()
plt.savefig('bohachevsky_functions.png', dpi=300)
