# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('../dataset/lena.png') # 載入影像
plt.imshow(img)

plt.show()