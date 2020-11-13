#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from utils.options import args_parser
from models.Nets import MLP, CNNMnist, CNNCifar, CNNGate
from utils.util import setup_seed
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils.sampling import cifar_noniid
import numpy as np
from models.Update import DatasetSplit
from models.test import user_test


def add_scalar(writer, test_result, epoch):
    test_loss, test_acc, user_loss, user_acc = test_result
    writer.add_scalar(f'test/global/test_loss', test_loss, epoch)
    writer.add_scalar(f'test/global/test_acc', test_acc, epoch)
    writer.add_scalar(f'test/local/test_loss', user_loss, epoch)
    writer.add_scalar(f'test/local/test_acc', user_acc, epoch)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    setup_seed(args.seed)

    # log
    current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
    TAG = 'exp/struct/gate2_{}_{}_{}_{}_{}'.format(args.dataset, args.model, args.epochs, args.alpha, current_time)
    logdir = f'runs/{TAG}'
    if args.debug:
        logdir = f'/tmp/runs/{TAG}'
    writer = SummaryWriter(logdir)

    # load dataset and split users
    train_loader, test_loader, class_weight = 1, 1, 1
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        # testing
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                      ]))

    elif args.dataset == 'cifar':
        save_dataset_path = f'./data/cifar_non_iid{args.alpha}_user{args.num_users}_fast_data'
        if args.rebuild:
            # training
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset_train = datasets.CIFAR10('../data/cifar', train=True, transform=transform_train, download=True)
            # testing
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset_test = datasets.CIFAR10('../data/cifar', train=False, transform=transform_test, download=True)
            # non_iid
            dict_users, _ = cifar_noniid(dataset_train, args.num_users, args.alpha)

            save_dataset = {
                "dataset_test": dataset_test,
                "dataset_train": dataset_train,
                "dict_users": dict_users
            }
            torch.save(save_dataset, save_dataset_path)
        else:
            save_dataset = torch.load(save_dataset_path)
            dataset_test = save_dataset['dataset_test']
            dataset_train = save_dataset['dataset_train']
            dict_users = save_dataset['dict_users']

        user0_train = DatasetSplit(dataset_train, dict_users[0])
        train_loader = DataLoader(user0_train, batch_size=64, shuffle=True)

        np.random.shuffle(dict_users[0])
        cut_point = len(dict_users[0]) // 10
        train_loader = DataLoader(DatasetSplit(dataset_train, dict_users[0][cut_point:]), batch_size=64, shuffle=True)
        gate_loader = DataLoader(DatasetSplit(dataset_train, dict_users[0][:cut_point]), batch_size=64, shuffle=True)

        class_weight = np.zeros(10)
        for image, label in user0_train:
            class_weight[label] += 1
        class_weight /= sum(class_weight)

        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
        for k, v in dict_users.items():
            writer.add_histogram(f'user_{k}/data_distribution',
                                 np.array(dataset_train.targets)[v],
                                 bins=np.arange(11))
            writer.add_histogram(f'all_user/data_distribution',
                                 np.array(dataset_train.targets)[v],
                                 bins=np.arange(11), global_step=k)
        img_size = dataset_train[0][0].shape
    else:
        exit('Error: unrecognized dataset')

    # build model
    net_glob = CNNGate(args=args).to(args.device)
    image, target = next(iter(train_loader))
    writer.add_graph(net_glob, image.to(args.device))

    # init
    global_weight = torch.load('./save/nn_cifar_cnn_120_Nov.03_22.31.53')['model']
    net_glob.load_state_dict(global_weight, False)
    net_glob.pfc1.load_state_dict({'weight': global_weight['fc1.weight'], 'bias': global_weight['fc1.bias']})
    net_glob.pfc2.load_state_dict({'weight': global_weight['fc2.weight'], 'bias': global_weight['fc2.bias']})
    net_glob.pfc3.load_state_dict({'weight': global_weight['fc3.weight'], 'bias': global_weight['fc3.bias']})

    # training
    optimizer = optim.SGD([
        {'params': net_glob.pfc1.parameters()},
        {'params': net_glob.pfc2.parameters()},
        {'params': net_glob.pfc3.parameters()},
        # {'params': net_glob.gate.parameters(),}
    ], lr=0.001, momentum=0.9, weight_decay=5e-4)

    optimizer_gate = optim.SGD([{'params': net_glob.gate.parameters()}], lr=0.001, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net_glob.parameters()), weight_decay=5e-4)
    creterion = nn.CrossEntropyLoss()

    list_loss = []
    test_result = user_test(net_glob, test_loader, class_weight)
    add_scalar(writer, test_result, 0)
    for epoch in range(1, args.epochs+1):
        net_glob.train()
        batch_loss = []
        gate_out = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output, g, z = net_glob(data)
            gate_out.append(g)
            loss = creterion(z, target)
            loss.backward()
            optimizer.step()
            # if batch_idx % 50 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train   _loader), loss.item()))
            batch_loss.append(loss.item())
        writer.add_histogram("local/gate_out", torch.cat(gate_out[0:-1], -1), epoch)
        writer.add_histogram("local/pfc1/weight", net_glob.pfc1.weight, epoch)
        writer.add_histogram("local/pfc2/weight", net_glob.pfc2.weight, epoch)
        writer.add_histogram("local/pfc3/weight", net_glob.pfc3.weight, epoch)
        loss_avg = sum(batch_loss) / len(batch_loss)
        print('\nTrain loss:', loss_avg)
        writer.add_scalar('train/train_loss', loss_avg, epoch)

        gate_epochs = 1
        for gate_epoch in range(gate_epochs):
            gate_epoch_loss = []
            for batch_idx, (data, target) in enumerate(gate_loader):
                data, target = data.to(args.device), target.to(args.device)
                optimizer_gate.zero_grad()
                output, g, z = net_glob(data)
                loss = creterion(output, target)
                loss.backward()
                optimizer_gate.step()
                gate_epoch_loss.append(loss.item())
            writer.add_histogram("gate/weight", net_glob.gate.weight)
            writer.add_histogram("gate/bais", net_glob.gate.bias)
            loss_avg = sum(gate_epoch_loss) / len(gate_epoch_loss)
            print('gate loss', loss_avg)
            writer.add_scalar('train/gate_train_loss', loss_avg, epoch*gate_epochs + gate_epoch)

        test_result = user_test(net_glob, test_loader, class_weight)
        add_scalar(writer, test_result, epoch)


    # save model weights
    # save_info = {
    #     "epochs": args.epochs,
    #     "optimizer": optimizer.state_dict(),
    #     "model": net_glob.state_dict()
    # }
    #
    # save_path = f'/tmp/runs/{TAG}' if args.debug else f'./save/{TAG}'
    # torch.save(save_info, save_path)
