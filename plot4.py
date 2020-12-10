import torch
import os
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
import numpy as np
# import glob
import re
from tensorboard.backend.event_processing import event_accumulator
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

alpha = 0.9
dataset = 'cifar'
model = 'lenet'

rootpwd = "runs/exp/local"
rootpwd_fed = "runs/exp/fed"
data_model = [['fmnist', 'lenet'],
              ['cifar', 'lenet'],
              ['cifar', 'vgg'],]


data_model_fed = [['fmnist', 'lenet'],

              ['cifar', 'lenet'],
['cifar', 'vggg'],]

regexs = []
for (data_set, model) in data_model:
    regexs.append(re.compile(r'{}_{}_100_iidFalse_[0|2]\.[5|9|0]_user100_*'.format(data_set,model)))

regexs_fed = []
for (data_set, model) in data_model_fed:
    regexs_fed.append(re.compile(r'{}_{}_1000_C0\.1_iidFalse_[0|2]\.[5|9|0]_user100_*'.format(data_set, model)))


def find_file(rootdir1, regex):
    ret = []
    for root, dirs, files in os.walk(rootdir1):
        for dir in dirs:
            # print(dir)
            if regex.match(dir):
                ret.append(dir)
                print(dir)
    return ret


def gety(rootpwd, file_names):
    res = []
    for f in file_names:
        tensor_file = ''
        pwd = os.path.join(rootpwd, f)
        for root, dirs, files in os.walk(pwd):
            assert len(files) == 3
            files = sorted(files, key=lambda x: len(x))
            tensor_file = files[1]
        tensor_file = os.path.join(pwd, tensor_file)
        gate_data = torch.load(tensor_file)
        # line = gate_data['local_acc'].mean(0)
        acc_local = gate_data['local_acc'][:, -1]
        acc_global = gate_data['total_acc'][:, -1]
        # print(event_file)
        # ea = event_accumulator.EventAccumulator(event_file)
        # ea.Reload()
        # vals = []
        # for scalar in ea.Scalars('test/local/test_acc'):
        #     vals.append(scalar.value)
        # assert len(vals) == epochs
        # res.append(vals)
        res.append(list(acc_local))
        res.append(list(acc_global))
    return res

def gety_fed(rootpwd, file_names):
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
file_names = [sorted(find_file(rootpwd, regex), reverse=False) for regex in regexs]
file_names_fed = [sorted(find_file(rootpwd_fed, regex), reverse=False) for regex in regexs_fed]

# for i in range(3):
print()


plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams['axes.grid'] = True
plt.grid(linestyle='-.')
line = ['-.', '-', '--']
handles = []
strid = 1
epochs = 100
fig, axs2d = plt.subplots(1, 3, figsize=(15, 4))
fig.tight_layout()
axs = axs2d
titles = ['Fashion-MNIST+LeNet-5', 'CIFAR-10+LeNet-5', 'CIFAR-10+VGG-16']
colors = [ 'pink', 'lightblue' ]
for i in range(3):
    y = gety(rootpwd, file_names[i])
    y_fed = gety_fed(rootpwd_fed, file_names_fed[i])
    blank = 0.03
    # for j, line_y in enumerate(y_fed.max(1)):
    #     axs[i].axhline(line_y, xmin=j*(1/3)+blank, xmax=j*(1/3)+(1/3)-blank, color='b', linestyle='-', linewidth=0.8)
    linewidth = [1, 1, 1]
    labels = ['0.5', '0.5', '0.9', '0.9', '2.0', '2.0']
    labels = [r'$\alpha='+i+"$" for i in labels]

    bplot = axs[i].boxplot(y, labels=labels, patch_artist=True, notch=True, showmeans=True, )
    for patch, color in zip(bplot['boxes'], colors*3):
        patch.set_facecolor(color)
    # l1 = axs[2*i].boxplot(y[::2], labels=labels, notch=True, showmeans=True, )
    #
    # l1['boxes'][0].set(color='b')
    # axs[2*i].axhlin`e(y=93, xmin=0., xmax=0.33, color='b', linestyle='-', lw=1)
    # l2 = axs[2*i+1].boxplot(y[1::2], labels=labels, notch=True, showmeans=True)
    axs[i].set_ylabel('test acc')
    if i != 0:
        # axs[i].set_ylabel(False)
        # axs[i].y_label.set_visible(False)
        axs[i].yaxis.label.set_visible(False)
    else:
        axs[i].set_ylabel('Test Acc(%)')
    if i == 2:
        p3 = plt.scatter([], [], marker='s', color=colors[0])
        p4 = plt.scatter([], [], marker='s', color=colors[1])
        axs[i].legend(handles=[p3,p4], labels=['local test', 'global test'], loc='upper right', fontsize=11, bbox_to_anchor=(0, 0, 1.015, 1.11), ncol=2)

    # l1, = axs[i].plot(range(0, epochs, strid), y[0][::strid], 'r' + line[i], linewidth=linewidth[i],)
    # l2, = axs[i].plot(range(0, epochs, strid), y[1][::strid], 'b' + line[i], linewidth=linewidth[i])
    # l3, = axs[i].plot(range(0, epochs, strid), y[2][::strid], 'g' + line[i], linewidth=linewidth[i])
    # handles += l1, l2, l3
# axs[0].
    axs[i].grid(linestyle='--')
    if i==1:
        axs[i].set_title('Local Training', fontsize=13, pad=20)
    axs[i].set_xlabel(titles[i])


# dict = [['Fashion-MNIST', 'LeNet5'],
#              ['CIFAR-10', 'VGG-16'],
#               ['CIFAR-10', 'LeNet5']]
# alphas = [2.0, 0.9, 0.5]

# Label_Com = [r'$\alpha={}$ {} {}'.format(alphas[j], dict[i][0], dict[i][1]) for i in range(3) for j in range(3)]
# axs[0].legend([], labels=Label_Com, loc='lower right', fontsize=2, ncol=2 )
# fig.suptitle("Local Training", fontsize=13, weight='bold')
# fig.savefig("imgs/local_acc.pdf", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
fig.show()