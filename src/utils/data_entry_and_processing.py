import torch
from torchvision import datasets, transforms
import copy, random
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


def get_mnist_unlearning_threes_dataloader():
    # Using MNIST
    train_data = datasets.MNIST("data", download=True, train=True, transform=TRANSFORM)
    test_data = datasets.MNIST("data", download=True, train=False, transform=TRANSFORM)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=60, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=60, shuffle=True)

    # Test dataloader with 3's only
    threes_index = []
    nonthrees_index = []
    for i in range(0, len(test_data)):
        if test_data[i][1] == 3:
            threes_index.append(i)
        else:
            nonthrees_index.append(i)
    three_test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=64,
        sampler=torch.utils.data.SubsetRandomSampler(threes_index),
    )
    nonthree_test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=64,
        sampler=torch.utils.data.SubsetRandomSampler(nonthrees_index),
    )

    # Train dataloaders with limited 3s
    threes_index = []
    nonthrees_index = []
    count = 0
    for i in range(0, len(train_data)):
        if train_data[i][1] != 3:
            nonthrees_index.append(i)
            threes_index.append(i)
        if train_data[i][1] == 3 and count < 100:
            count += 1
            threes_index.append(i)
    nonthree_train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=64,
        sampler=torch.utils.data.SubsetRandomSampler(nonthrees_index),
    )
    three_train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=64,
        sampler=torch.utils.data.SubsetRandomSampler(threes_index),
    )

    # Unlearning dataset with all "3" labels randomly assigned
    unlearning_data = copy.deepcopy(train_data)
    unlearninglabels = list(range(10))
    unlearninglabels.remove(3)
    for i in range(len(unlearning_data)):
        if unlearning_data.targets[i] == 3:
            unlearning_data.targets[i] = random.choice(unlearninglabels)
    unlearning_train_loader = torch.utils.data.DataLoader(
        unlearning_data, batch_size=64, shuffle=True
    )

    return (
        train_loader,
        test_loader,
        three_train_loader,
        nonthree_train_loader,
        three_test_loader,
        nonthree_test_loader,
        unlearning_train_loader,
    )
