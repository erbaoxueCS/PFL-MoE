from datetime import datetime
print(datetime.now().strftime('%b.%d_%H.%M.%S'))
print(111)
# from torch.utils.tensorboard import SummaryWriter
# import torch
# import numpy as np
# import matplotlib
# import pandas as pd
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# TAG = 'histogram_test'
# logdir = f'runs/{TAG}'
# writer = SummaryWriter(logdir)
# writer.add_histogram_raw("haha",0, 500, 10, 500,)

# sigma = 1
# for step in range(1):
#     writer.add_histogram('hist-numpy2', np.array([0,1,2,3,4,4,2]), bins=np.array([0,1,2,3,4]))
#     sigma += 1
# sigma = 1
# for step in range(5):
#     torch_normal = torch.distributions.Normal(0, sigma)
#     writer.add_histogram('hist-torch', torch_normal.sample((1, 1000)), step)
#     sigma += 1
# writer.close()


# x = np.random.normal(0, 1, 100)
# # x = pd.Series(x)
# y = np.random.normal(1, 2, 100)
# # y = pd.Series(y)
#
# # plot loss curve
# plt.figure()
# plt.title('local train acc', fontsize=20)  # 标题，并设定字号大小
# labels = ['local', 'total']
# plt.boxplot([x, y], labels=labels)
# plt.ylabel('test acc')
# plt.savefig(f'{logdir}/local_train_acc.png')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create Map
cm = plt.get_cmap("Greens")

x = np.random.rand(30)
y = np.random.rand(30)
z = np.random.rand(30)
#col = [cm(float(i)/(29)) for i in xrange(29)] # BAD!!!
col = [cm(float(i)/(30)) for i in range(30)]

# 2D Plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x, y, s=100, c=col, marker='o')

# 3D Plot
fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
ax3D.scatter(x, y, z, s=10, c=col, marker='o')

plt.show()