import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

lena = mpimg.imread('./data/lovelive_192_192/20-0.jpg') 
# 此时就已经是一个 np.array 了，可以对它进行任意处理
print(lena.shape) #(192, 192, 3)

plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
