import torch
from torchvision import datasets, transforms
import copy, random

# Transform image to tensor and normalize features from [0,255] to [0,1]
TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,), (0.5)),
    ]
)


def get_MNIST_dataloader():
    # Using MNIST
    train_data = datasets.MNIST("data", download=True, train=True, transform=TRANSFORM)
    test_data = datasets.MNIST("data", download=True, train=False, transform=TRANSFORM)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=60, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=60, shuffle=True)
    return train_loader, test_loader


def get_CIFAR100_dataloader():
    # Using MNIST
    train_data = datasets.CIFAR100(
        "data", download=True, train=True, transform=TRANSFORM
    )
    test_data = datasets.CIFAR100(
        "data", download=True, train=False, transform=TRANSFORM
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    return train_loader, test_loader


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
