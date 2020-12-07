import torch
import os
import matplotlib.pyplot as plt
import numpy as np
# import glob
import re
from tensorboard.backend.event_processing import event_accumulator
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

alpha = 0.9
dataset = 'cifar'
model = 'lenet'

rootpwd = "runs/exp/fed"
data_model = [['fmnist', 'lenet'],
             ['cifar', 'vggg'],
              ['cifar', 'lenet']]

regexs = []
for (data_set, model) in data_model:
    regexs.append(re.compile(r'{}_{}_1000_C0\.1_iidFalse_[0|2]\.[5|9|0]_user100_*'.format(data_set, model)))


def find_file(rootdir1, regex):
    ret = []
    for root, dirs, files in os.walk(rootdir1):
        for dir in dirs:
            # print(dir)
            if regex.match(dir):
                ret.append(dir)
                print(dir)
    return ret


def gety(file_names):
    res = []
    for f in file_names:
        event_file = ''
        pwd = os.path.join(rootpwd, f)
        for root, dirs, files in os.walk(pwd):
            assert len(files) == 1
            event_file = files[0]
        event_file = os.path.join(pwd, event_file)
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        vals = []
        for scalar in ea.Scalars('test_acc'):
            vals.append(scalar.value)
        assert len(vals) == 1000
        res.append(vals)
    return np.array(res)

axs = [plt]
file_names = [sorted(find_file(rootpwd, regex), reverse=True) for regex in regexs]

# for i in range(3):
print()


plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams['axes.grid'] = True
plt.grid(linestyle='-.')

line = ['-', '--', '-.']
handles = []
strid = 4
for i in range(3):
    y = gety(file_names[i])
    linewidth = np.array([1, 1.5, 1.25])

    l3, = axs[0].plot(range(0, 1000, strid), y[2][::strid], 'g' + line[i], linewidth=linewidth[i])
    l2, = axs[0].plot(range(0, 1000, strid), y[1][::strid], 'b' + line[i], linewidth=linewidth[i])
    l1, = axs[0].plot(range(0, 1000, strid), y[0][::strid], 'r' + line[i], linewidth=linewidth[i], )
    handles += l1, l2, l3
# axs[0].

axs[0].title('FedAvg')
axs[0].xlabel('Rounds')
axs[0].ylabel('Global Acc')

dict = [['Fashion-MNIST', 'LeNet-5'],
             ['CIFAR-10', 'VGG-16'],
              ['CIFAR-10', 'LeNet-5']]
alphas = [2.0, 0.9, 0.5]

Label_Com = [r'$\alpha={}$ {} {}'.format(alphas[j], dict[i][0], dict[i][1]) for i in range(3) for j in range(3)]
axs[0].legend(handles=handles, labels=Label_Com, loc='lower right', fontsize=8.8, ncol=2)

ax = axs[0].axes()
axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
                   bbox_to_anchor=(0.46, 0.31, 1, 1),
                   bbox_transform=ax.transAxes)
ys = []
for i in range(2):
    y = gety(file_names[i])
    # linewidth = 1
    l3, = axins.plot(range(0, 1000, strid), y[2][::strid], 'g' + line[i], linewidth=linewidth[i])
    l2, = axins.plot(range(0, 1000, strid), y[1][::strid], 'b' + line[i], linewidth=linewidth[i])
    l1, = axins.plot(range(0, 1000, strid), y[0][::strid], 'r' + line[i], linewidth=linewidth[i], )
    ys += [y[0][::strid],
           y[1][::strid],
           y[2][::strid]]
axins.grid(None)
axins._girdOn = False


# 设置放大区间
zone_left = int(805 / strid)
zone_right = int(999 / strid)

# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0  # x轴显示范围的扩展比例
y_ratio = 0.05  # y轴显示范围的扩展比例

for i, y in enumerate(ys):
    ys[i] = y

# X轴的显示范围
x = [i for i in range(0, 1000, strid)]
xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

# Y轴的显示范围
y = np.hstack((y[zone_left:zone_right] for y in ys))
# y = np.hstack((y0[zone_left:zone_right], y1[zone_left:zone_right], y2[zone_left:zone_right],
#                (y3[zone_left:zone_right], y4[zone_left:zone_right], y5[zone_left:zone_right])))
ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)
axins.axes.set_visible('off')
plt.xticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='k', lw=0.5, ls='--')
# plt.savefig("imgs/fed_acc.pdf", bbox_inches='tight', dpi=100, pad_inches=0.0)
plt.show()