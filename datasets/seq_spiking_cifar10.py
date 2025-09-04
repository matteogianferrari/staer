import logging
from typing import Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10

from utils.conf import base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from datasets.utils import set_default_from_args
from datasets.transforms.static_encoding import StaticEncoding
from datasets.seq_cifar10 import TCIFAR10, MyCIFAR10
from models.spiking_er.losses import TSCELoss


class SeqSpikingCIFAR10(ContinualDataset):
    """Sequential Spiking CIFAR10 Dataset.

    This version of CIFAR10 is modified to be used in combination with Spiking Neural Networks.
    A 'Static encoding' is applied to the standard dataset, where the images are simply repeated
    T times along the temporal dimension. The resulting images after applying the transformations
    are tensors of shape [T, C, H, W] if single element, or [T, B, C, H, W] if working with batches.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformations to apply to the dataset.
    """
    NAME = 'seq-scifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (32, 32)
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    def __init__(self, args, transform_type: str = 'weak'):
        super(SeqSpikingCIFAR10, self).__init__(args)

        self.T = args.T

        assert transform_type in ['weak', 'strong'], "Transform type must be either 'weak' or 'strong'."

        if transform_type == 'strong':
            logging.info("Using strong augmentation for SCIFAR10")
            self.TRAIN_TRANSFORM = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 transforms.ToTensor(),
                 transforms.Normalize(SeqSpikingCIFAR10.MEAN, SeqSpikingCIFAR10.STD)])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Class method that returns the train and test loaders."""
        train_transform = transforms.Compose([
            self.TRAIN_TRANSFORM,
            StaticEncoding(T=self.T)
        ])

        test_transform = transforms.Compose([
            self.TEST_TRANSFORM,
            StaticEncoding(T=self.T)
        ])

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=train_transform)
        test_dataset = TCIFAR10(base_path() + 'CIFAR10', train=False,
                                download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @set_default_from_args("backbone")
    def get_backbone():
        return "sresnet19"

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss():
        return TSCELoss()

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SeqSpikingCIFAR10.MEAN, SeqSpikingCIFAR10.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SeqSpikingCIFAR10.MEAN, SeqSpikingCIFAR10.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = CIFAR10(base_path() + 'CIFAR10', train=True, download=True).classes
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names
