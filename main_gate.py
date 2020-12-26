#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from utils.options import args_parser
from models.Nets import CNNGate, gate_vgg16
from utils.util import setup_seed, add_scalar
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils.sampling import cifar_noniid
import numpy as np
from models.Update import DatasetSplit
from models.test import user_test, user_per_test


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    setup_seed(args.seed)

    # log
    current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
    TAG = 'exp/{}gate2/{}_{}_{}_{}_user{}_{}'.format('struct/' if args.struct else '', args.dataset, args.model, args.epochs,
                                              args.alpha, args.num_users, current_time)
    TAG2 = 'exp/{}per_fb/{}_{}_{}_{}_user{}_{}'.format('struct/' if args.struct else '', args.dataset, args.model, args.epochs,
                                              args.alpha, args.num_users, current_time)
    logdir = f'runs/{TAG}'
    logdir2 = f'runs/{TAG2}'
    if args.debug:
        logdir = f'runs2/{TAG}'
        logdir2 = f'runs2/{TAG2}'
    writer = SummaryWriter(logdir)
    writer2 = SummaryWriter(logdir2)

    # load dataset and split users
    train_loader, test_loader, class_weight = 1, 1, 1

    save_dataset_path = f'./data/{args.dataset}_non_iid{args.alpha}_user{args.num_users}_fast_data'
    # global_weight = torch.load('./save/exp/fed/cifar_resnet_1000_C0.1_iidFalse_2.0_user100_Nov.28_01.41.16_bst')['model']
    global_weight = torch.load(
        f'./save/exp/fed/{args.dataset}_{args.model}_1000_C0.1_iidFalse_{args.alpha}_user{args.num_users}_bst')[
        'model']
    if 'gate.weight' in global_weight:
        del (global_weight['gate.weight'])
        del (global_weight['gate.bias'])
    if args.rebuild:
        if args.dataset == "cifar":
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
        elif args.dataset == "fmnist":
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
        else:
            exit('Error: unrecognized dataset')
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

    test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    for k, v in dict_users.items():
        writer.add_histogram(f'user_{k}/data_distribution',
                             np.array(dataset_train.targets)[v],
                             bins=np.arange(11))
        writer.add_histogram(f'all_user/data_distribution',
                             np.array(dataset_train.targets)[v],
                          bins = np.arange(11), global_step = k)

    # build model
    if args.model == 'lenet' and (args.dataset == 'cifar' or args.dataset == 'fmnist'):
        net_glob = CNNGate(args=args).to(args.device)
    elif args.model == 'vgg' and args.dataset == 'cifar':
        net_glob = gate_vgg16(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    image, target = next(iter(test_loader))
    writer.add_graph(net_glob, image.to(args.device))

    gate_epochs = 200

    local_acc = np.zeros([args.num_users, args.epochs + gate_epochs + 1])
    total_acc = np.zeros([args.num_users, args.epochs + gate_epochs + 1])
    local_acc2 = np.zeros([args.num_users, args.epochs + gate_epochs + 1])
    total_acc2 = np.zeros([args.num_users, args.epochs + gate_epochs + 1])

    for user_num in range(len(dict_users)):
        # user data
        user_train = DatasetSplit(dataset_train, dict_users[user_num])

        np.random.shuffle(dict_users[user_num])
        cut_point = len(dict_users[user_num]) // 4
        train_loader = DataLoader(DatasetSplit(dataset_train, dict_users[user_num][cut_point:]),
                                  batch_size=64, shuffle=True)
        gate_loader = DataLoader(DatasetSplit(dataset_train, dict_users[user_num][:cut_point]),
                                 batch_size=64, shuffle=True)

        class_weight = np.zeros(10)
        for image, label in user_train:
            class_weight[label] += 1
        class_weight /= sum(class_weight)

        # init

        net_glob.load_state_dict(global_weight, False)

        if args.model == 'lenet':
            keys_ind = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
            net_glob.load_state_dict({'p' + k: global_weight[k] for k in keys_ind}, strict=False)
        elif args.model == 'vgg':
            keys_ind = ['classifier.1.weight', 'classifier.1.bias', 'classifier.4.weight', 'classifier.4.bias', 'classifier.6.weight', 'classifier.6.bias']
            net_glob.load_state_dict({'p' + k: global_weight[k] for k in keys_ind}, strict=False)
        else:
            exit("Error: unrecognized model")
        net_glob.gate.reset_parameters()

        # training
        if args.model == 'lenet':
            layer_set = {'p' + k[:k.rindex('.')] for k in keys_ind}
            optimizer = optim.SGD([{'params': getattr(net_glob, l).parameters()} for l in layer_set],
                                  lr=0.001, momentum=0.9, weight_decay=5e-4)
        elif args.model == 'vgg':
            layer_set = {k[len('pclassifier'):k.rindex('.')] for k in keys_ind}
            optimizer = optim.SGD([{'params': net_glob.pclassifier.parameters()}],
                                  lr=0.005, momentum=0.9, weight_decay=5e-4)
        else:
            exit('Error: unrecognized model')
        # optimizer_gate = optim.SGD([{'params': net_glob.gate.parameters()}], lr=0.001, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        test_result = user_per_test(args, net_glob, test_loader, class_weight)
        add_scalar(writer2, user_num, test_result, 0)
        total_acc2[user_num][0] = test_result[1]
        local_acc2[user_num][0] = test_result[3]

        for epoch in range(1, args.epochs+1):
            net_glob.train()
            batch_loss = []
            gate_out = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(args.device), target.to(args.device)
                optimizer.zero_grad()
                output, g, z = net_glob(data)
                gate_out.append(g)
                loss = criterion(z, target)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            if epoch % 10 == 1:
                # writer.add_histogram(f"user_{user_num}/gate_out", torch.cat(gate_out[0:-1], -1), epoch)
                if args.model == 'lenet':
                    for layer in layer_set:
                        writer.add_histogram(f"user_{user_num}/{layer}/weight", getattr(net_glob, layer).weight, epoch)
                elif args.model == 'vgg':
                    for layer in layer_set:
                        writer.add_histogram(f"user_{user_num}/pclassifier.{layer}/weight", getattr(net_glob.pclassifier, layer).weight, epoch)
            loss_avg = sum(batch_loss) / len(batch_loss)
            print(f'User {user_num} train loss:', loss_avg)
            writer2.add_scalar(f'user_{user_num}/pfc_train_loss', loss_avg, epoch)

            test_result = user_per_test(args, net_glob, test_loader, class_weight)
            print(f'global test acc:', test_result[1])
            add_scalar(writer2, user_num, test_result, epoch)
            total_acc2[user_num][epoch] = test_result[1]
            local_acc2[user_num][epoch] = test_result[3]

        test_result = user_test(args, net_glob, test_loader, class_weight)
        add_scalar(writer, user_num, test_result, args.epochs)
        total_acc[user_num][args.epochs] = test_result[1]
        local_acc[user_num][args.epochs] = test_result[3]

        optimizer_gate = optim.Adam([{'params': net_glob.gate.parameters()}], weight_decay=5e-4)

        for gate_epoch in range(1, 1 + gate_epochs):
            net_glob.train()
            gate_epoch_loss = []
            gate_out = torch.tensor([], device=args.device)
            for batch_idx, (data, target) in enumerate(gate_loader):
                data, target = data.to(args.device), target.to(args.device)
                optimizer_gate.zero_grad()
                output, g, z = net_glob(data)
                gate_out = torch.cat((gate_out, g.view(-1)))
                loss = criterion(output, target)
                loss.backward()
                optimizer_gate.step()
                gate_epoch_loss.append(loss.item())
            if gate_epoch % 10 == 1:
                writer.add_histogram(f"user_{user_num}/gate_out", gate_out)
                writer.add_histogram(f"user_{user_num}/gate/weight", net_glob.gate.weight)
                writer.add_histogram(f"user_{user_num}/gate/bais", net_glob.gate.bias)
            loss_avg = sum(gate_epoch_loss) / len(gate_epoch_loss)
            print(f'User {user_num} gate loss', loss_avg)
            writer.add_scalar(f'user_{user_num}/gate_train_loss', loss_avg, args.epochs + gate_epoch)

            test_result = user_test(args, net_glob, test_loader, class_weight)
            add_scalar(writer, user_num, test_result, args.epochs + gate_epoch)
            total_acc[user_num][args.epochs + gate_epoch] = test_result[1]
            local_acc[user_num][args.epochs + gate_epoch] = test_result[3]

    save_info = {
        "total_acc": total_acc,
        "local_acc": local_acc
    }
    save_info2 = {
        "total_acc": total_acc2,
        "local_acc": local_acc2
    }
    save_path = f'{logdir}/local_train_epoch_acc'
    save_path2 = f'{logdir2}/local_train_epoch_acc'
    torch.save(save_info, save_path)
    torch.save(save_info2, save_path2)

    total_acc = total_acc.mean(axis=0)
    local_acc = local_acc.mean(axis=0)
    total_acc2 = total_acc2.mean(axis=0)
    local_acc2 = local_acc2.mean(axis=0)
    for epoch, _ in enumerate(total_acc):
        if epoch >= args.epochs:
            writer.add_scalar('test/global/test_acc', total_acc[epoch], epoch)
            writer.add_scalar('test/local/test_acc', local_acc[epoch], epoch)
        if epoch <= args.epochs:
            writer2.add_scalar('test/global/test_acc', total_acc2[epoch], epoch)
            writer2.add_scalar('test/local/test_acc', local_acc2[epoch], epoch)
    writer.close()
    writer2.close()
