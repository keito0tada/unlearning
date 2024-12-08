import torch
import torch.utils.data.dataloader
from torchvision import datasets, transforms
import copy, random
import numpy as np
import medmnist
from src.log.logger import logger_regular

# Transform image to tensor and normalize features from [0,255] to [0,1]
TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,), (0.5)),
    ]
)


def get_MNIST_dataset():
    train_dataset = datasets.MNIST(
        "data", download=True, train=True, transform=TRANSFORM
    )
    test_dataset = datasets.MNIST(
        "data", download=True, train=False, transform=TRANSFORM
    )
    logger_regular.info("get MNIST dataset")
    return train_dataset, test_dataset


def get_MNIST_dataloader(batch_size: int = 60):
    train_dataset, test_dataset = get_MNIST_dataset()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )
    logger_regular.info(f"get MNIST dataloader | batch size: {batch_size}")
    return train_loader, test_loader


def get_CIFAR100_dataset():
    train_dataset = datasets.CIFAR100(
        "data", download=True, train=True, transform=TRANSFORM
    )
    test_dataset = datasets.CIFAR100(
        "data", download=True, train=False, transform=TRANSFORM
    )
    logger_regular.info("get CIFAR100 dataset")
    return train_dataset, test_dataset


def get_CIFAR100_dataloader(batch_size: int = 64):
    train_dataset, test_dataset = get_CIFAR100_dataset()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )
    logger_regular.info(f"get CIFAR100 dataloader | batch size: {batch_size}")
    return train_loader, test_loader


def get_MedMNIST_dataset(data_flag: str):
    info = medmnist.INFO[data_flag]
    task = info["task"]
    num_channels = info["n_channels"]
    num_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])

    train_dataset = DataClass(split="train", transform=TRANSFORM, download=True)
    test_dataset = DataClass(split="test", transform=TRANSFORM, download=True)

    logger_regular.info(
        f"Loaded MedMNIST dataset. channels: {num_channels}, classes: {num_classes}, task: {task}"
    )
    return train_dataset, test_dataset


def get_MedMNIST_dataloader(
    data_flag: str = "pathmnist", batch_size: int = 64
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, int]:
    info = medmnist.INFO[data_flag]
    task = info["task"]
    num_channels = info["n_channels"]
    num_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])

    train_dataset = DataClass(split="train", transform=TRANSFORM, download=True)
    test_dataset = DataClass(split="test", transform=TRANSFORM, download=True)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=2 * batch_size, shuffle=True
    )

    return train_dataloader, test_dataloader, task, num_channels, num_classes
