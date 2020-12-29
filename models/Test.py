#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


def test(args, net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = []
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            test_loss.append(nn.CrossEntropyLoss()(log_probs, target).item())
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum().item()

        loss_avg = sum(test_loss)/len(test_loss)
        test_acc = 100. * correct / len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        loss_avg, correct, len(data_loader.dataset), test_acc))

    return test_acc, loss_avg


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.test_bs)
    l = len(data_loader)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct.item() / len(data_loader.dataset)
    # if args.verbose:
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def user_test(args, net_glob, data_loader, class_weight):
    #testing
    net_glob.eval()
    correct_class = np.zeros(10)
    class_loss = np.zeros(10)
    correct_class_acc = np.zeros(10)
    class_loss_avg = np.zeros(10)
    correct_class_size = np.zeros(10)
    correct = 0.0
    dataset_size = len(data_loader.dataset)
    total_loss = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            output, g, z = net_glob(data)
            # g = (g > 0.5).float()
            # output = y * g + z * (1-g)
            pred = output.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            loss = nn.CrossEntropyLoss(reduction='none')(output, target)
            total_loss += loss.sum().item()
            for i in range(10):
                class_ind = target.data.view_as(pred).eq(i * torch.ones_like(pred))
                correct_class_size[i] += class_ind.cpu().sum().item()
                correct_class[i] += (pred.eq(target.data.view_as(pred)) * class_ind).cpu().sum().item()
                class_loss[i] += (loss*class_ind.float()).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))
    total_l = total_loss / dataset_size
    for i in range(10):
        correct_class_acc[i] = (float(correct_class[i]) / float(correct_class_size[i]))
        class_loss_avg[i] = (float(class_loss[i]) / float(correct_class_size[i]))
    user_acc = correct_class_acc * class_weight
    user_loss = class_loss_avg * class_weight
    return total_l, acc, user_loss.sum(), 100*user_acc.sum()


def user_per_test(args, net_glob, data_loader, class_weight):
    #testing
    net_glob.eval()
    correct_class = np.zeros(10)
    class_loss = np.zeros(10)
    correct_class_acc = np.zeros(10)
    class_loss_avg = np.zeros(10)
    correct_class_size = np.zeros(10)
    correct = 0.0
    dataset_size = len(data_loader.dataset)
    total_loss = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            output, g, z = net_glob(data)
            pred = z.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            loss = nn.CrossEntropyLoss(reduction='none')(z, target)
            total_loss += loss.sum().item()
            for i in range(10):
                class_ind = target.data.view_as(pred).eq(i * torch.ones_like(pred))
                correct_class_size[i] += class_ind.cpu().sum().item()
                correct_class[i] += (pred.eq(target.data.view_as(pred)) * class_ind).cpu().sum().item()
                class_loss[i] += (loss*class_ind.float()).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))
    total_l = total_loss / dataset_size
    for i in range(10):
        correct_class_acc[i] = (float(correct_class[i]) / float(correct_class_size[i]))
        class_loss_avg[i] = (float(class_loss[i]) / float(correct_class_size[i]))
    user_acc = correct_class_acc * class_weight
    user_loss = class_loss_avg * class_weight
    return total_l, acc, user_loss.sum(), 100*user_acc.sum()


def local_test(args, net_glob, data_loader, class_weight):
    #testing
    net_glob.eval()
    correct_class = np.zeros(10)
    class_loss = np.zeros(10)
    correct_class_acc = np.zeros(10)
    class_loss_avg = np.zeros(10)
    correct_class_size = np.zeros(10)
    correct = 0.0
    dataset_size = len(data_loader.dataset)
    total_loss = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            output = net_glob(data)
            pred = output.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            loss = nn.CrossEntropyLoss(reduction='none')(output, target)
            total_loss += loss.sum().item()
            for i in range(10):
                class_ind = target.data.view_as(pred).eq(i * torch.ones_like(pred))
                correct_class_size[i] += class_ind.cpu().sum().item()
                correct_class[i] += (pred.eq(target.data.view_as(pred)) * class_ind).cpu().sum().item()
                class_loss[i] += (loss*class_ind.float()).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))
    total_l = total_loss / dataset_size
    for i in range(10):
        correct_class_acc[i] = (float(correct_class[i]) / float(correct_class_size[i]))
        class_loss_avg[i] = (float(class_loss[i]) / float(correct_class_size[i]))
    user_acc = correct_class_acc * class_weight
    user_loss = class_loss_avg * class_weight
    return total_l, acc, user_loss.sum(), 100*user_acc.sum()
