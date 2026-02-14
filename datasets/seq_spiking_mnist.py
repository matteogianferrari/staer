from typing import Tuple

import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from PIL import Image
from argparse import Namespace
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order, store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args
from datasets.transforms.static_encoding import StaticEncoding
from datasets.seq_mnist import MyMNIST
import torch.nn.functional as F


class SeqSpikingMNIST(ContinualDataset):
    """The Sequential Spiking MNIST dataset.

    This version of MNIST is modified to be used in combination with Spiking Neural Networks.
    A 'StaticEncoding' transform is applied to the standard dataset, where the images are simply repeated
    T times along the temporal dimension. The resulting images after applying the transformations
    are tensors of shape [T, C, H, W] if single element, or [T, B, C, H, W] if working with batches.

    Attributes:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
    """
    NAME = 'seq-smnist'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (28, 28)
    TRANSFORM = transforms.ToTensor()

    def __init__(self, args: Namespace) -> None:
        super(SeqSpikingMNIST, self).__init__(args)

        self.T = args.T

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = transforms.Compose([
            self.TRANSFORM,
            StaticEncoding(T=self.T)
        ])

        train_dataset = MyMNIST(base_path() + 'MNIST',
                                train=True, download=True, transform=transform)
        test_dataset = MNIST(base_path() + 'MNIST',
                             train=False, download=True, transform=transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @set_default_from_args("backbone")
    def get_backbone():
        return "sresnet19-mnist"

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SeqSpikingMNIST.TRANSFORM])
        return transform

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 10

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 1

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = MNIST(base_path() + 'MNIST', train=True, download=True).classes
        classes = [c.split('-')[1].strip() for c in classes]
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names
