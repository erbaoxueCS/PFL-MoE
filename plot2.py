import torch
import os
import matplotlib.pyplot as plt
import numpy as np
# import glob
import re
from tensorboard.backend.event_processing import event_accumulator
from mpl_toolkits.axes_grid.inset_locator import inset_axes

alpha = 0.9
# dataset = 'cifar'
# model = 'lenet'

rootpwd = "runs/exp/fed"
data_model = [['fmnist', 'lenet'],
             ['cifar', 'lenet'],
             ['cifar', 'vggg']]
regexs = []
for (data_set, model) in data_model:
    regexs.append(re.compile(r'{}_{}_1000_C0\.1_iidFalse_[0|2]\.[5|9|0]_user100_*'.format(data_set,model)))


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
file_names = [find_file(rootpwd, regex) for regex in regexs]

# for i in range(3):
print()


plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams['axes.grid'] = True
plt.grid(linestyle='-.')

line = ['-', '--', '-.']
for i in range(3):
    y = gety(file_names[i])
    linewidth = 1
    l1, = axs[0].plot(range(0, 1000, 5), y[0][::5], 'r' + line[i], linewidth=linewidth,)
    l2, = axs[0].plot(range(0, 1000, 5), y[1][::5], 'g' + line[i], linewidth=linewidth)
    l3, = axs[0].plot(range(0, 1000, 5), y[2][::5], 'b' + line[i], linewidth=linewidth)

# axs[0].

axs[0].title('FedAvg')
axs[0].xlabel('Rounds')
axs[0].ylabel('Global Acc')

Label_Com = [r'$CIFAR-10+LeNet5 \alpha=1$' for i in range(9)]
axs[0].legend([], labels=Label_Com, loc='lower right')

# ax = axs[0].axes()
# axins = inset_axes(ax, width="20%", height="10%", loc='lower left',
#                    bbox_to_anchor=(0.8, 0.1, 1, 1),
#                    bbox_transform=ax.transAxes)

plt.savefig("imgs/fed_acc.svg", bbox_inches='tight', dpi=100, pad_inches=0.0)