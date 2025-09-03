import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10Dataset(Dataset):
    def __init__(self, root="/mnt/c/Users/HP/Downloads/cifar-10-python/cifar-10-batches-py", train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if train:
            files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            files = ["test_batch"]

        data_list, labels_list = [], []
        for file in files:
            batch = unpickle(os.path.join(root, file))
            data_list.append(batch[b'data'])
            labels_list.extend(batch[b'labels'])

        self.data = np.concatenate(data_list).reshape(-1, 3, 32, 32)
        self.labels = np.array(labels_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype("float32") / 255.0
        img = torch.tensor(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label


def get_cifar10_loader(root='/mnt/c/Users/HP/Downloads/cifar-10-python/cifar-10-batches-py', train=True, batch_size=1, shuffle=True):
    dataset = CIFAR10Dataset(root=root, train=train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_dataset(name="CIFAR10", root="/mnt/c/Users/HP/Downloads/cifar-10-python/cifar-10-batches-py", image_size=32, as_rgb=True):
    """
    Returns: (train_dataset, test_dataset, in_ch)
    Matches your autoencoder project call signature.
    """
    if name.lower() != "cifar10":
        raise ValueError("Only CIFAR10 is supported in this loader.")

    trainset = CIFAR10Dataset(root=root, train=True)
    testset = CIFAR10Dataset(root=root, train=False)
    in_ch = 3  # CIFAR10 is always RGB

    return trainset, testset, in_ch