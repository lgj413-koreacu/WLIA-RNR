import argparse

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

RAW_DATA_PATH = "Cifar10/"

# Allocate data to users
def generate_Cifar10(dir_path, niid, balance, partition, num_clients=20, alpha=0.1, seed=1, aux=False, NUM_AUX=10):
    random.seed(seed)
    np.random.seed(seed)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition, alpha):
        return
        
    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root=RAW_DATA_PATH+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=RAW_DATA_PATH+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # ---------------- aux dataset separation ---------------- #
    aux_X, aux_Y = None, None
    if aux:
        aux_X, aux_Y = [], []
        remain_X, remain_Y = [], []

        for c in range(num_classes):
            idx_c = np.where(dataset_label == c)[0]
            np.random.shuffle(idx_c)

            aux_idx = idx_c[:NUM_AUX]
            aux_X.append(dataset_image[aux_idx])
            aux_Y.append(dataset_label[aux_idx])

            remain_idx = idx_c[NUM_AUX:]
            remain_X.append(dataset_image[remain_idx])
            remain_Y.append(dataset_label[remain_idx])

        aux_X = np.vstack(aux_X)
        aux_Y = np.hstack(aux_Y)

        aux_data_dict = {'x': aux_X, 'y': aux_Y}

        with open(dir_path + 'aux.npz', 'wb') as f:
            np.savez_compressed(f, data=aux_data_dict)

        dataset_image = np.vstack(remain_X)
        dataset_label = np.hstack(remain_Y)

        print(f"Auxiliary dataset size: {len(aux_X)}")
    # ------------------------------------------------ #

    class_per_client = 2

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client, alpha)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition, alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-seed', "--seed", type=int, default=0)
    parser.add_argument('-niid', "--niid", type=bool, default=True)
    parser.add_argument('-bal', "--balance", type=bool, default=True)
    parser.add_argument('-part', "--partition", type=str, default="dir")

    parser.add_argument('-nc', "--num_clients", type=int, default=10)
    parser.add_argument('-dir', "--alpha", type=float, default=0.5)
    parser.add_argument('-aux', "--aux", type=bool, default=True)
    parser.add_argument('-na', "--num_aux", type=int, default=10)

    args = parser.parse_args()

    if args.aux:
        dir_path = f"Cifar10_{args.partition}_N_{args.num_clients}_alpha_{args.alpha}_aux_{args.num_aux}_seed_{args.seed}/"
    else:
        dir_path = f"Cifar10_{args.partition}_N_{args.num_clients}_alpha_{args.alpha}/"

    print(f'save directory: {dir_path}')

    generate_Cifar10(
        dir_path,
        args.niid,
        args.balance,
        args.partition,
        args.num_clients,
        args.alpha,
        args.seed,
        args.aux,
        args.num_aux,
    )
