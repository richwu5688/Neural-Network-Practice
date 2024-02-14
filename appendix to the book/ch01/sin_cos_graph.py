# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# 建立資料
x = np.arange(0, 6, 0.1) # 從0到6，以0.1為單位產生資料
y1 = np.sin(x)
y2 = np.cos(x)

# 繪製圖表
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label="cos")
plt.xlabel("x") # x軸標籤
plt.ylabel("y") # y軸標籤
plt.title('sin & cos')
plt.legend()
plt.show()