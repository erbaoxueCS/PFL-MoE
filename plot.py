import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import re


def draw(dataset, model, alpha, f):

    epochs = 200 if model == 'vgg' else 100

    # pwd = 'runs/exp/{}/cifar_lenet_100_{}_user100'.format('gate2', '0.9')
    rootpwd = "runs/exp/{}/"
    rootdir1 = os.path.dirname(rootpwd.format('gate2'))
    rootdir2 = os.path.dirname(rootpwd.format('struct/gate2'))
    regex = re.compile(r'{}_{}_{}_{}_user100_*'.format(str(dataset), str(model), str(epochs), str(float(alpha))))


    def get3(pwd, pwd2, epochs=100, num_users=100):

        gate_pwd = pwd.format('gate2')
        per_pwd = pwd.format('per_fb')
        struct_pwd = pwd2.format('struct/gate2')
        per2_pwd = pwd2.format('struct/per_fb')
        file_name = 'local_train_epoch_acc'
        ret = np.zeros([10, num_users])
        gate_data = torch.load(os.path.join(gate_pwd, file_name))
        per_data = torch.load(os.path.join(per_pwd, file_name))
        struct_data = torch.load(os.path.join(struct_pwd, file_name))
        per2_data = torch.load(os.path.join(per2_pwd, file_name))

        for i in range(num_users):
            ks = ['local_acc', 'total_acc']
            for j, k in enumerate(ks):
                # gate = max(gate_data['local_acc'][i][epochs:-1])
                ret[5*j][i] = per_data[k][i][0]
                ret[5 *j + 1][i] = per_data[k][i][epochs]
                ret[5 *j + 2][i] = struct_data[k][i][-1]
                # ret[5 *j + 3][i] = gate_data[k][i][-1]
                if j == (0 if f else -1):
                    ret[5 *j + 3][i] = max(gate_data[k][i][epochs:-1])
                else:
                    ret[5 * j + 3][i] = gate_data[k][i][-1]
                ret[5 *j + 4][i] = per2_data[k][i][epochs]

            # x[i], y1[i], y2[i], y3[i], y4[i] = fed, per, gate, struct, per2
        return ret

    def getdirsize(dir):
        size = 0
        for root, dirs, files in os.walk(dir):
            size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
        return size

    def find_file(rootdir1, regex):
        maxdir = ''
        size_now = -1
        for root, dirs, files in os.walk(rootdir1):
            for dir in dirs:
                if regex.match(dir):
                    size = getdirsize(os.path.join(rootdir1, dir))
                    if size >= size_now:
                        maxdir = dir
                        size_now = size

        if maxdir is '':
            exit("no match")
        return maxdir




    pwd = os.path.join(rootpwd, find_file(rootdir1, regex))
    pwd2 = os.path.join(rootpwd, find_file(rootdir2, regex))
    fed, per, struct, gate, per2, g_fed, g_per, g_struct, g_gate, g_per2, = get3(pwd, pwd2, epochs)

    # plt.plot(fed, per - fed, 'c>')
    # plt.plot(fed, gate - fed, 'mp')
    # plt.plot(fed, struct - fed, 'gx')

    # plt.plot(fed, per-fed, 'co')

    # plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.labelsize"] = "large"
    plt.rcParams["axes.titleweight"] = "bold"
    # color_dot = "#0000FF"
    color_dot = "c"
    # cm_g = plt.get_cmap("Greens")
    # cm_r = plt.get_cmap("Reds")l1 = axs[0].scatter(fed, per-fed)
    #
    # len(r1)
    # col_g = [cm_g(float(i)/len(r1)) for i in range(len(r1))]
    #
    # c = (struct-per)[r1] / (max((struct-per)[r1]) )
    # col_r = [cm_r(i) for i in c]
    #
    # l2 = axs[0].scatter(fed[r1], (struct-fed)[r1], marker='>', c=col_r)
    # l3 = axs[0].scatter(fed[r2], (struct-fed)[r2], marker='>', c='g')

    r1 = (struct > per)
    r2 = ~ r1

    fig, axs2d = plt.subplots(2, 2, figsize=(10, 10))
    # axs = [axs]
    axs = [axs2d[0][0], axs2d[1][0], axs2d[0][1], axs2d[1][1]]
    l3, = axs[0].plot(fed[r2], (struct-fed)[r2], 'g>')
    l1, = axs[0].plot(fed, per-fed, '.', c=color_dot)
    l2, = axs[0].plot(fed[r1], (struct-fed)[r1], 'r>')

    axs[0].set_title('Local Test Acc of PFL',)
    axs[0].set_xlabel('FedAvg Local Acc(%)', )
    axs[0].set_ylabel('PFL Local Acc - FedAvg Local Acc(%)')

    Label_Com0 = ['PFL-FB', 'PFL-MF(>PFL-FB)', 'PFL-MF(<PFL-FB)']
    axs[0].legend(handles=[l1, l2, l3], labels=Label_Com0, loc='upper right')

    axs[0].axhline(y=0, color='b', linestyle='-.', lw=1)

    axs[1].axhline(y=0, color='b', linestyle='-.', lw=1)
    axs[1].set_title('Global Test Acc of PFL')
    axs[1].set_ylabel('PFL Global Acc - FedAvg Global Acc(%)')

    axs[1].set_xlabel('FedAvg Local Acc(%)', )

    # axs[1].set_xlabel('FedAvg')
    r1 = (g_struct > g_per)
    r2 = ~ r1
    l3, = axs[1].plot(fed[r2], (g_struct-np.mean(fed))[r2], 'g<')
    l1, = axs[1].plot(fed, g_per-np.mean(fed), '.', c=color_dot)
    l2, = axs[1].plot(fed[r1], (g_struct-np.mean(fed))[r1], 'r<')


    Label_Com1 = ['PFL-FB', 'PFL-MF(>PFL-FB)', 'PFL-MF(<PFL-FB)']
    # handles1, labels1 = axs[1].get_legend_handles_labels()
    axs[1].legend(handles=[l1, l2, l3], labels=Label_Com1, loc='lower right')
    r1 = (gate > per)
    r2 = ~ r1

    l3, = axs[2].plot(fed[r2], (gate-fed)[r2], 'g>')
    l1, = axs[2].plot(fed, per-fed, '.', c=color_dot)
    l2, = axs[2].plot(fed[r1], (gate-fed)[r1], 'r>')
    axs[2].set_title('Local Test Acc of PFL')
    axs[2].set_xlabel('FedAvg Local Acc(%)', )
    # axs[2].set_ylabel('Local Acc(PFL) - FedAvg Local Acc(%)')

    Label_Com0 = ['PFL-FB', 'PFL-MFE(>PFL-FB)', 'PFL-MFE(<PFL-FB)']
    axs[2].legend(handles=[l1, l2, l3], labels=Label_Com0, loc='upper right')

    axs[2].axhline(y=0, color='b', linestyle='-.', lw=1)

    axs[3].axhline(y=0, color='b', linestyle='-.', lw=1)
    axs[3].set_title('Global Test Acc of PFL')
    # axs[3].set_ylabel('Global Acc(PFL) - FedAvg Global Acc')

    axs[3].set_xlabel('FedAvg Local Acc(%)', )

    # axs[1].set_xlabel('FedAvg')
    r1 = (g_gate > g_per)
    r2 = ~ r1

    l3, = axs[3].plot(fed[r2], (g_gate-np.mean(fed))[r2], 'g<')
    l1, = axs[3].plot(fed, g_per-np.mean(fed), '.', c=color_dot)
    l2, = axs[3].plot(fed[r1], (g_gate-np.mean(fed))[r1], 'r<')


    Label_Com1 = ['PFL-FB', 'PFL-MFE(>PFL-FB)', 'PFL-MFE(<PFL-FB)']
    # handles1, labels1 = axs[1].get_legend_handles_labels()
    axs[3].legend(handles=[l1, l2, l3], labels=Label_Com1, loc='lower right')


    print("alpha={}, per={}".format(alpha, round(np.mean(per), 2)))
    print("alpha={}, per={}".format(alpha, round(np.mean(per2), 2)))
    print("alpha={}, struct={}".format(alpha, round(np.mean(struct), 2)))
    print("alpha={}, gate={}".format(alpha, round(np.mean(gate), 2)))
    print("alpha={}, g_per={}".format(alpha, round(np.mean(g_per), 2)))
    print("alpha={}, g_per2={}".format(alpha, round(np.mean(g_per2), 2)))
    print("alpha={}, g_struct={}".format(alpha, round(np.mean(g_struct), 2)))
    print("alpha={}, g_gate={}".format(alpha, round(np.mean(g_gate), 2)))

    # fig.suptitle('LeNet, CIFAR-10, Non-IID α=0.9, 100 Users')
    assert alpha == 0.9 or alpha == 2
    fig.savefig("imgs/{}{}_{}{}.pdf".format('09' if alpha==0.9 else '2', dataset, model, '_f'if f else ''), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    # fig.show()


if __name__ == '__main__':

    alpha = 0.9
    # alpha = 2
    fs = [True, False]
    # dataset = 'cifar'
    # model = 'vgg'
    dm = [['fmnist', 'lenet'],
          ['cifar', 'lenet'],
          ['cifar', 'vgg'],
          ]
    for f in fs:
        for d, m in dm:
            draw(d, m, alpha, f)