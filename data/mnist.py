import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def get_mnist_dataloaders(
    batch_size: int,
    data_root: str,
    rank: int,
    world: int,
    num_workers: int = 0,
    download: bool = True,
    mirror_url: Optional[str] = None,
):
    """
    Build per-rank train/test DataLoaders for MNIST.
    Each rank gets disjoint samples via simple striding.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_root, train=True, download=download, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_root, train=False, download=download, transform=transform
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=False
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world, rank=rank, shuffle=False, drop_last=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, test_loader
