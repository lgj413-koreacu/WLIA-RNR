import random
import numpy as np
import os
import torch
from torch.utils.data import Subset, Dataset
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms


DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), 'dataset')


class IndexedCIFAR100(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.data = torchvision.datasets.CIFAR100(
            root=root, train=train, transform=transform, download=download)
        print('debug')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        return image, label, index

class IndexedTinyImagenet(Dataset):
    def __init__(self, root, transform=None):
        train_dataset = datasets.ImageFolder(root=root + '/train', transform=transform)
        all_classes = list(range(len(train_dataset)))
        selected_classes = random.sample(all_classes, 10)

        selected_indices = [i for i, (_, label) in enumerate(train_dataset) if label in selected_classes]
        self.data = Subset(train_dataset, selected_indices)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        return image, label, index

class IndexedSVHN(Dataset):
    def __init__(self, root, transform=None):
        if os.path.exists(root):
            self.data = datasets.SVHN(root=root, split='train', transform=transform, download=False)
        else:
            self.data = datasets.SVHN(root=root, split='train', transform=transform, download=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        return image, label, index

class IndexedSynthetic(Dataset):
    def __init__(self, root):
        cifar10_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        cifar10_std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        images = np.load(os.path.join(root, 'synthetic_images.npy'))
        images = images.astype(np.float32)  # Images are already in [0,1] range from generation
        images = (images - cifar10_mean[None, :, None, None]) / cifar10_std[None, :, None, None]

        self.image = torch.from_numpy(images).float()
        self.label = np.load(os.path.join(root, 'synthetic_labels.npy')).astype(int)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image = self.image[index]
        label = self.label[index]
        return image, label, index

def read_data(dataset, idx, is_train=True, is_attack=False):
    if is_train:
        train_data_dir = os.path.join(DATA_PATH, dataset, 'train/')

        if is_attack:
            train_file = train_data_dir + str(idx) + '_back_door.npz'
            if not os.path.exists(train_file):
                return None
        else:
            train_file = train_data_dir + str(idx) + '.npz'

        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join(DATA_PATH, dataset, 'test/')

        if is_attack:
            test_file = test_data_dir + str(idx) + '_back_door.npz'
            if not os.path.exists(test_file):
                return None
        else:
            test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data

def save_injected_data(dataset, idx, injected_train_data, injected_test_data):
    train_data_dir = os.path.join(DATA_PATH, dataset, 'train/')
    train_file = train_data_dir + str(idx) + '_back_door.npz'

    test_data_dir = os.path.join(DATA_PATH, dataset, 'test/')
    test_file = test_data_dir + str(idx) + '_back_door.npz'

    np.savez_compressed(train_file, data=np.array(injected_train_data, dtype=object))
    np.savez_compressed(test_file, data=np.array(injected_test_data, dtype=object))



def read_public_data(dataset, public_dataset, subset_idx, is_train=True):
    if dataset == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        dir_path = os.path.join('../dataset', 'mnist/')
        trainset = torchvision.datasets.MNIST(
            root=dir_path + "rawdata", train=True, download=False, transform=transform)

    elif dataset == 'Cifar10':
        if public_dataset == 'Cifar100':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dir_path = os.path.join('../dataset', 'Cifar100/')
            trainset = IndexedCIFAR100(root=dir_path + "rawdata", train=True, transform=transform, download=False)

        elif public_dataset == 'TinyImagenet':
            transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
            dir_path = os.path.join(DATA_PATH, 'TinyImagenet/rawdata/tiny-imagenet-200')
            trainset = IndexedTinyImagenet(root=dir_path, transform=transform)

        elif public_dataset == 'SVHN':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dir_path = os.path.join(DATA_PATH, 'SVHN')
            trainset = IndexedSVHN(root=dir_path, transform=transform)
            # data_num = len(trainset.data.dataset)
        elif public_dataset == 'Synthetic':
            dir_path = os.path.join(DATA_PATH, 'Synthetic')
            trainset = IndexedSynthetic(root=dir_path)
        else:
            raise NotImplementedError

    if subset_idx:
        subset_trainset = Subset(trainset, subset_idx)
        return subset_trainset
    else:
        return trainset


def read_aux_data(dataset):
    train_data_dir = os.path.join(DATA_PATH, dataset)

    train_file = train_data_dir + '/aux.npz'
    with open(train_file, 'rb') as f:
        train_data = np.load(f, allow_pickle=True)['data'].tolist()

    if dataset.lower().startswith(("agnews", "dbpedia")):
        X_train, X_train_lens = list(zip(*train_data['x']))

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
    else:
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
    return train_data


def read_client_data(dataset, idx, is_train=True, is_attack=False):
    if dataset.lower().startswith(("agnews", "dbpedia")):
        return read_client_data_text(dataset, idx, is_train)
    elif dataset[:2] == "Shakespeare":
        return read_client_data_Shakespeare(dataset, idx)

    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(dataset, idx, is_train)

    if is_train:
        train_data = read_data(dataset, idx, is_train, is_attack)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train, is_attack)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        # original code
        # train_data = [
        #     ((x, lens), y)
        #     for x, lens, y in zip(X_train, X_train_lens, y_train)
        # ]

        train_data = [
            (x, y)
            for x, y in zip(X_train, y_train)
        ]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        #### original code
        # test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        test_data = [
            (x, y)
            for x, y in zip(X_test, y_test)
        ]
        return test_data


def read_client_data_Shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

