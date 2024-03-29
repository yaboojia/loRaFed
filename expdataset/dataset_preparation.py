"""Dataset(CIFAR100) preparation for DepthFL."""

from typing import List, Optional, Tuple

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR100, CIFAR10, EMNIST

from datasets import load_dataset 

class TinyImagenet(Dataset):
    def __init__(self, path, split, transform=None):    #train or valid
        self.path = path
        self.split = split
        # self.subset = subset
        self.transform = transform
        self.dataset = load_dataset(self.path, split=self.split, trust_remote_code=True)
        self.classes = [i for i in range(200)]
        self.classes_size = len(self.classes)

    def __getitem__(self, index):
        index = int(index)
        img, target = self.dataset[index]['image'], self.dataset[index]['label']
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return (img, target)

    def __len__(self):
        return len(self.dataset)

class newEMNIST(Dataset):
    def __init__(self, train, transform):    #train or valid
        # self.path = path
        self.train = train
        # self.subset = subset
        self.transform = transform
        self.classes = [i for i in range(10)]
        self.dataset = EMNIST(
            "./dataset", split='mnist', train=self.train, download=True, transform=None
        )
    def __getitem__(self, index):
        index = int(index)
        # img, target = self.dataset[index]['image'], self.dataset[index]['label']
        img, target = self.dataset[index][0], self.dataset[index][1]
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return (img, target)

    def __len__(self):
        return len(self.dataset)
        




def _download_data(dataset) -> Tuple[Dataset, Dataset]:
    """Download (if necessary) and returns the CIFAR-100 dataset.

    Returns
    -------
    Tuple[CIFAR100, CIFAR100]
        The dataset for training and the dataset for testing CIFAR100.
    """
    if dataset=='cifar100':
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )

        trainset = CIFAR100(
            "./dataset", train=True, download=True, transform=transform_train
        )
        testset = CIFAR100(
            "./dataset", train=False, download=True, transform=transform_test
        )
    if dataset=='cifar10':
        transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
            #  transforms.Resize(224) if model_name == 'vgg16' else transforms.RandomHorizontalFlip(p=0),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        trainset = CIFAR10(
            "./dataset", train=True, download=True, transform=transform
        )
        testset = CIFAR10(
            "./dataset", train=False, download=True, transform=transform
        )
    if dataset=='emnist':
        transform = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainset = newEMNIST(True, transform=transform)
        testset = newEMNIST(False, transform=transform)
    if dataset=='tinyimagenet':
        transform = transforms.Compose([
        # transforms.Resize(224) if model_name == 'vgg16' else transforms.RandomHorizontalFlip(p=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        trainset = TinyImagenet('/home/jyb/flower/baselines/depthfl/tinyimagenet', 'train', transform)
        testset = TinyImagenet('/home/jyb/flower/baselines/depthfl/tinyimagenet', 'valid', transform)

    return trainset, testset


def _partition_data(
    num_clients,
    iid: Optional[bool] = True,
    beta=0.5,
    seed=41,
    dataset='cifar100'
) -> Tuple[List[Dataset], Dataset]:
    """Split training set to simulate the federated setting.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    iid : bool, optional
        Whether the data should be independent and identically distributed
        or if the data should first be sorted by labels and distributed by
        noniid manner to each client, by default true
    beta : hyperparameter for dirichlet distribution
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Dataset], Dataset]
        A list of dataset for each client and a
        single dataset to be use for testing the model.
    """
    trainset, testset = _download_data(dataset)

    datasets: List[Subset] = []

    if iid:
        distribute_iid(num_clients, seed, trainset, datasets)

    else:
        distribute_noniid(num_clients, beta, seed, trainset, datasets)

    return datasets, testset


def distribute_iid(num_clients, seed, trainset, datasets):
    """Distribute dataset in iid manner."""
    np.random.seed(seed)
    num_sample = int(len(trainset) / (num_clients))
    index = list(range(len(trainset)))
    for _ in range(num_clients):
        sample_idx = np.random.choice(index, num_sample, replace=False)
        index = list(set(index) - set(sample_idx))
        datasets.append(Subset(trainset, sample_idx))


def distribute_noniid(num_clients, beta, seed, trainset, datasets):
    """Distribute dataset in non-iid manner."""
    labels = np.array([label for _, label in trainset])
    min_size = 0
    np.random.seed(seed)

    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        # for each class in the dataset
        for k in range(np.max(labels) + 1):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
            # Balance
            proportions = np.array(
                [
                    p * (len(idx_j) < labels.shape[0] / num_clients)
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        # net_dataidx_map[j] = np.array(idx_batch[j])
        datasets.append(Subset(trainset, np.array(idx_batch[j])))

if __name__ == '__main__':
    a, b = _download_data('cifar10')
    print(a[0][0].size())

