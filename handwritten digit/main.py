import numpy as np
import os
import sys
import tools
from scipy.io import loadmat

data = loadmat(os.path.join('Data', 'data1.mat'))  # 读取手写数字数据
X, y = data['X'], data['y'].ravel()  # 预处理数据
y[y == 10] = 0
m = y.size  # 数据大小(5000)

# 随机选取100张图片展示
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

np.set_printoptions(threshold=sys.maxsize)
'''
tools.displaydata(sel[0], figsize=(4, 4))

for i in range(20):
    for j in range(20):
        print('{:f}'.format(sel[0, j * 20 + i]), end=' ')
    print()
'''

# 读取已训练好的节点权重值
weights = loadmat(os.path.join('Data', 'ex3weights.mat'))

# 利用已有权重值进行识别
Theta1, Theta2 = weights['Theta1'], weights['Theta2']
Theta2 = np.roll(Theta2, 1, axis=0)

print(tools.getresult(Theta1, Theta2, X))
