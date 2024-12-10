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


def to_tensor(y):
    return torch.tensor(y, dtype=torch.int64)


def get_MNIST_dataset():
    train_dataset = datasets.MNIST(
        "data",
        download=True,
        train=True,
        transform=TRANSFORM,
        target_transform=to_tensor,
    )
    test_dataset = datasets.MNIST(
        "data",
        download=True,
        train=False,
        transform=TRANSFORM,
        target_transform=to_tensor,
    )
    logger_regular.info("get MNIST dataset")
    return train_dataset, test_dataset


def get_MNIST_dataloader(batch_size: int):
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
        "data",
        download=True,
        train=True,
        transform=TRANSFORM,
        target_transform=to_tensor,
    )
    test_dataset = datasets.CIFAR100(
        "data",
        download=True,
        train=False,
        transform=TRANSFORM,
        target_transform=to_tensor,
    )
    logger_regular.info("get CIFAR100 dataset")
    return train_dataset, test_dataset


def get_CIFAR100_dataloader(batch_size: int):
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

    train_data_x = []
    train_data_y = []
    for X, y in train_dataset:
        train_data_x.append(X)
        train_data_y.append(y[0])

    test_data_x = []
    test_data_y = []
    for X, y in test_dataset:
        test_data_x.append(X)
        test_data_y.append(y[0])

    logger_regular.info(
        f"Loaded MedMNIST dataset ({data_flag}) | channels: {num_channels}, classes: {num_classes}, task: {task}"
    )
    return torch.utils.data.TensorDataset(
        torch.stack(train_data_x), torch.tensor(train_data_y, dtype=torch.int64)
    ), torch.utils.data.TensorDataset(
        torch.stack(test_data_x), torch.tensor(test_data_y, dtype=torch.int64)
    )


def get_MedMNIST_dataloader(
    data_flag: str, batch_size: int
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, int]:
    train_dataset, test_dataset = get_MedMNIST_dataset(data_flag)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )

    logger_regular.info(
        f"get MedMNIST dataloader ({data_flag}) | batch size: {batch_size}"
    )
    return train_dataloader, test_dataloader
