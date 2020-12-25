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
import copy


def test(net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = []
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            test_loss.append(F.cross_entropy(log_probs, target).item())
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum().item()

        loss_avg = sum(test_loss)/len(test_loss)
        test_acc = 100. * correct / len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        loss_avg, correct, len(data_loader.dataset), test_acc))

    return test_acc, loss_avg


if __name__ == '__main__':

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    setup_seed(args.seed)

    # log
    current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
    TAG = 'nn_{}_{}_{}_{}'.format(args.dataset, args.model, args.epochs, current_time)
    logdir = f'runs/{TAG}'
    if args.debug:
        logdir = f'/tmp/runs/{TAG}'
    writer = SummaryWriter(logdir)

    # load dataset and split users
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
        rebuild_data = False
        save_dataset_path = './data/fast_data'
        if rebuild_data:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            # transform = transforms.Compose(
            #     [transforms.ToTensor(),
            #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR10('../data/cifar', train=True, transform=transform_train, download=True)
            img_size = dataset_train[0][0].shape
            # testing
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset_test = datasets.CIFAR10('../data/cifar', train=False, transform=transform_test, download=True)
            test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)

            dict_users, _ = cifar_noniid(dataset_train, 10, 0.9)
            for k, v in dict_users.items():
                writer.add_histogram(f'user_{k}/data_distribution',
                                     np.array(dataset_train.targets)[v],
                                     bins=np.arange(11))
                writer.add_histogram(f'all_user/data_distribution',
                                     np.array(dataset_train.targets)[v],
                                     bins=np.arange(11), global_step=k)

            dataset_train = DatasetSplit(dataset_train, dict_users[0])
            class_weight = np.zeros(10)
            for image, label in dataset_train:
                class_weight[label] += 1
            class_weight /= len(dataset_train)
            save_dataset = {
                "dataset_test": dataset_test,
                "dataset_train": dataset_train,
                "class_weight": class_weight
            }
            torch.save(save_dataset, save_dataset_path)
        else:
            save_dataset = torch.load(save_dataset_path)
            dataset_test = save_dataset['dataset_test']
            dataset_train = save_dataset['dataset_train']
            class_weight = save_dataset['class_weight']

        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
        img_size = dataset_train[0][0].shape
    else:
        exit('Error: unrecognized dataset')

    # build model
    net_glob = CNNGate(args=args).to(args.device)

    # init
    global_weight = torch.load('./save/nn_cifar_cnn_100_Oct.13_19.45.20')['model']
    net_glob.load_state_dict(global_weight, False)
    global_weight = copy.deepcopy(global_weight)
    net_glob.pfc1.load_state_dict({'weight': global_weight['fc1.weight'], 'bias': global_weight['fc1.bias']})
    net_glob.pfc2.load_state_dict({'weight': global_weight['fc2.weight'], 'bias': global_weight['fc2.bias']})
    net_glob.pfc3.load_state_dict({'weight': global_weight['fc3.weight'], 'bias': global_weight['fc3.bias']})


    # training
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net_glob.parameters()), lr=0.01, momentum=0.9, weight_decay=5e-4)

    creterion = nn.CrossEntropyLoss()

    list_loss = []
    test_acc, test_loss = test(net_glob, test_loader)
    writer.add_scalar('test_loss', test_loss, 0)
    writer.add_scalar('test_acc', test_acc, 0)

    for epoch in range(1, args.epochs+1):
        net_glob.train()
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = creterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            batch_loss.append(loss.item())
            writer.add_histogram("gate/weight", net_glob.gate.weight)
            writer.add_histogram("gate/bais", net_glob.gate.bias)
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)
        writer.add_scalar('train_loss', loss_avg, epoch)
        test_acc, test_loss = test(net_glob, test_loader)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

    # save model weights
    # save_info = {
    #     "epochs": args.epochs,
    #     "optimizer": optimizer.state_dict(),
    #     "model": net_glob.state_dict()
    # }
    #
    # save_path = f'/tmp/runs/{TAG}' if args.debug else f'./save/{TAG}'
    # torch.save(save_info, save_path)
