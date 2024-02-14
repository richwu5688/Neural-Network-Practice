# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# 建立資料
x = np.arange(0, 6, 0.1) # 從0到6，以0.1為單位產生資料
y = np.sin(x)

# 繪製圖表
plt.plot(x, y)
plt.show()