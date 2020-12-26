#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from utils.options import args_parser
from models.Nets import CNNCifar, vgg16
from utils.util import setup_seed
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from models.test import test


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
        logdir = f'runs2/{TAG}'
    writer = SummaryWriter(logdir)

    # load dataset and split users
    if args.dataset == 'cifar':
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
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
        img_size = dataset_train[0][0].shape
    elif args.dataset == 'fmnist':
        dataset_train = datasets.FashionMNIST('../data/fmnist/', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize((32, 32)),
                                           transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,)),
                                       ]))

        # testing
        dataset_test = datasets.FashionMNIST('../data/fmnist/', train=False, download=True,
                                      transform=transforms.Compose([
                                          transforms.Resize((32, 32)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                      ]))
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'lenet' and (args.dataset == 'cifar' or args.dataset == 'fmnist'):
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'vgg' and args.dataset == 'cifar':
        net_glob = vgg16().to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    img = dataset_train[0][0].unsqueeze(0).to(args.device)
    writer.add_graph(net_glob, img)

    # training
    creterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    # optimizer = optim.Adam(net_glob.parameters())
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    # # # scheduler.step()

    list_loss = []
    net_glob.train()
    for epoch in range(args.epochs):
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
        # scheduler.step()
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)
        writer.add_scalar('train_loss', loss_avg, epoch)
        test_acc, test_loss = test(args, net_glob, test_loader)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

    # save model weights
    save_info = {
        "epochs": args.epochs,
        "optimizer": optimizer.state_dict(),
        "model": net_glob.state_dict()
    }

    save_path = f'save2/{TAG}' if args.debug else f'save2/{TAG}'
    torch.save(save_info, save_path)
    writer.close()
