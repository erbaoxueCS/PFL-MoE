#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from utils.options import args_parser
from models.Nets import CNNGate
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils.sampling import cifar_noniid
from models.Update import DatasetSplit
from utils.util import *
from models.Test import user_per_test
import copy


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    setup_seed(args.seed)

    # log
    current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
    TAG = 'exp/non_iid/per_fb_{}_{}_{}_{}_user{}_{}'.format(args.dataset, args.model, args.epochs, args.alpha,
                                                            args.num_users, current_time)
    logdir = f'runs/{TAG}'
    if args.debug:
        logdir = f'runs2/{TAG}'
    writer = SummaryWriter(logdir)

    # load dataset and split users
    train_loader, test_loader, class_weight, dict_users, dataset_train = 1, 1, 1, 1, 1
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
        # global_weight = torch.load('./save/nn_cifar_cnn_100_Oct.13_19.45.20')['model']
        # global_weight = torch.load(f'./save/exp/fed/{args.dataset}_{args.model}_1000_C0.1_iidFalse_{args.alpha}_user{args.num_users}_1000es')['model']
        global_weight = torch.load(f'./save/exp/fed/cifar_lenet_1000_C0.1_iidFalse_0.2_user30*3_Nov.16_14.37.35_1000es')['model']
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

        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
        for k, v in dict_users.items():
            writer.add_histogram(f'user_{k}/data_distribution',
                                 np.array(dataset_train.targets)[v],
                                 bins=np.arange(11))
            writer.add_histogram(f'all_user/data_distribution',
                                 np.array(dataset_train.targets)[v],
                                 bins=np.arange(11), global_step=k)
        img_size = dataset_train[0][0].shape
    elif args.dataset == 'fmnist':
        pass
    else:
        exit('Error: unrecognized dataset')

    # build model
    net_glob = CNNGate(args=args).to(args.device)
    image, target = next(iter(test_loader))
    writer.add_graph(net_glob, image.to(args.device))

    local_acc = np.zeros([args.num_users, args.epochs + 1])
    total_acc = np.zeros([args.num_users, args.epochs + 1])

    for user_num in range(len(dict_users)):
        # user train data
        user_train = DatasetSplit(dataset_train, dict_users[user_num])
        train_loader = DataLoader(user_train, batch_size=64, shuffle=True)

        class_weight = np.zeros(10)
        for image, label in user_train:
            class_weight[label] += 1
        class_weight /= sum(class_weight)

        # init

        # global_weight = torch.load('./save/fed_cifar_cnn_1000_C0.1_iidFalse_0.9_Nov.05_09.31.38_500es')['model']
        net_glob.load_state_dict(global_weight, False)
        net_glob.pfc1.load_state_dict({'weight': global_weight['fc1.weight'], 'bias': global_weight['fc1.bias']})
        net_glob.pfc2.load_state_dict({'weight': global_weight['fc2.weight'], 'bias': global_weight['fc2.bias']})
        net_glob.pfc3.load_state_dict({'weight': global_weight['fc3.weight'], 'bias': global_weight['fc3.bias']})

        # training
        optimizer = optim.SGD([
            {'params': net_glob.pfc1.parameters()},
            {'params': net_glob.pfc2.parameters()},
            {'params': net_glob.pfc3.parameters()},
        ], lr=0.001, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        test_result = user_per_test(args, net_glob, test_loader, class_weight)
        add_scalar(writer, user_num, test_result, 0)
        total_acc[user_num][0] = test_result[1]
        local_acc[user_num][0] = test_result[3]

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
            writer.add_histogram(f"user_{user_num}/pfc1/weight", net_glob.pfc1.weight, epoch)
            writer.add_histogram(f"user_{user_num}/pfc2/weight", net_glob.pfc2.weight, epoch)
            writer.add_histogram(f"user_{user_num}/pfc3/weight", net_glob.pfc3.weight, epoch)
            loss_avg = sum(batch_loss) / len(batch_loss)
            print(f'User {user_num} Train loss:', loss_avg)
            writer.add_scalar(f'user_{user_num}/pfc_train_loss', loss_avg, epoch)

            test_result = user_per_test(args, net_glob, test_loader, class_weight)
            add_scalar(writer, user_num, test_result, epoch)
            total_acc[user_num][epoch] = test_result[1]
            local_acc[user_num][epoch] = test_result[3]

    save_info = {
        "total_acc": total_acc,
        "local_acc": local_acc
    }
    save_path = f'{logdir}/local_train_epoch_acc'
    torch.save(save_info, save_path)

    total_acc = total_acc.mean(axis=0)
    local_acc = local_acc.mean(axis=0)
    for epoch, _ in enumerate(total_acc):
        writer.add_scalar('test/global/test_acc', total_acc[epoch], epoch)
        writer.add_scalar('test/local/test_acc', local_acc[epoch], epoch)
    writer.close()
