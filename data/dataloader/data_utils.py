import numpy as np
import torch
from data.dataloader.sampler import CategoriesSampler

def set_up_datasets(args):
    if args.dataset == 'cifar100':
        import data.dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions=9
    if args.dataset =="manyshotcifar":
        import dataloader.cifar100.manyshot_cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = args.shot_num
        args.sessions = 9
    if args.dataset == 'cub200':
        import data.dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    
    if args.dataset == 'manyshotcub':
        import dataloader.cub200.manyshot_cub as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = args.shot_num
        args.sessions = 11

    if args.dataset == 'mini_imagenet':
        import data.dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == 'mini_imagenet_withpath':
        import dataloader.miniimagenet.miniimagenet_with_img as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    
    
    if args.dataset == 'manyshotmini':
        import dataloader.miniimagenet.manyshot_mini as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = args.shot_num
        args.sessions = 9
    
    if args.dataset == 'imagenet100':
        import data.dataloader.imagenet100.ImageNet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == 'imagenet1000':
        import data.dataloader.imagenet1000.ImageNet as Dataset
        args.base_class = 600
        args.num_classes=1000
        args.way = 50
        args.shot = 5
        args.sessions = 9

    return args, Dataset

def get_dataloader(args, session, dataset):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args, dataset)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args, dataset, session)
    return trainset, trainloader, testloader

def get_base_dataloader(args, dataset):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':

        trainset = dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = dataset.ImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = dataset.ImageNet(root=args.dataroot, train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.base_batch_size, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader



def get_base_dataloader_meta(args, dataset):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
        testset = dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_index)
    if args.dataset == 'mini_imagenet':
        trainset = dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index_path=txt_path)
        testset = dataset.MiniImageNet(root=args.dataroot, train=False,
                                            index=class_index)


    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(args, dataset, session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False)
    if args.dataset == 'cub200':
        trainset = dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'mini_imagenet':
        trainset = dataset.MiniImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = dataset.ImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)

    if args.incremental_batch_size == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=0, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.incremental_batch_size, shuffle=True,
                                                  num_workers=0, pin_memory=True)

    # test loader with classes from all previous sessions
    class_new = get_classes_up_to_session(args, session)

    if args.dataset == 'cifar100':
        testset = dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        testset = dataset.ImageNet(root=args.dataroot, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader


def get_single_class_dataloaders(args, dataset):
    sets, loaders = [], []
    for i in range(100):
        set = dataset.CIFAR100(root=args.dataroot, train=True, download=False, base_sess=False,
                                    train_tasks=100, shot=args.shot, few_shot=True,
                                    session_classes=[i])
        loader = torch.utils.data.DataLoader(dataset=set, batch_size=5, shuffle=True,
                                             num_workers=args.num_workers, pin_memory=True)

        sets.append(set)
        loaders.append(iter(loader))
    return sets, loaders


def get_classes_up_to_session(args, session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list


def get_train_session_classes(args, session):
    class_list=np.arange(args.base_class + (session - 1) * args.way, args.base_class + (session) * args.way)
    return class_list
