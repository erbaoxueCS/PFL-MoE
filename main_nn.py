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
from models.Nets import MLP, CNNMnist, CNNCifar
from utils.util import setup_seed
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


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
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)

        img_size = dataset_train[0][0].shape
    elif args.dataset == 'cifar':
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

        # testing
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, transform=transform_test, download=True)
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)

        img_size = dataset_train[0][0].shape
    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # training
    optimizer = optim.SGD(net_glob.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    creterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

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
        scheduler.step()
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)
        writer.add_scalar('train_loss', loss_avg, epoch)
        test_acc, test_loss = test(net_glob, test_loader)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

    # save model weights
    save_info = {
        "epochs": args.epochs,
        "optimizer": optimizer.state_dict(),
        "model": net_glob.state_dict()
    }

    save_path = f'/tmp/runs/{TAG}' if args.debug else f'./save/{TAG}'
    torch.save(save_info, save_path)
