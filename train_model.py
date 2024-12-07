import torch
import torchvision

from src.utils.data_entry_and_processing import (
    get_CIFAR100_dataloader,
    get_MNIST_dataloader,
)
from src.model_trainer.cifar100 import CIFAR100ModelTrainer
from src.utils.misc import now

DEVICE = "cuda:0"


def train_resnet18_on_MNIST():
    NUM_CLASSES = 10
    NOW = now()
    PATH_MODEL = f"model/resnet18_on_mnist_{NOW}.pt"

    train_dataloader, test_dataloader = get_MNIST_dataloader()

    model = torchvision.models.resnet18()
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(512, NUM_CLASSES), torch.nn.LogSoftmax(dim=1)
    )
    optimizer = torch.optim.Adam(model.parameters())

    model_trainer = CIFAR100ModelTrainer(
        model=model.to(DEVICE),
        optimizer=optimizer,
        criterion=torch.nn.NLLLoss(),
        device=DEVICE,
    )
    model_trainer.iterate_train(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        log_label="Train Resnet18",
    )
    model_trainer.save(path=PATH_MODEL)


def train_resnet18_on_CIFAR100():
    NUM_CLASSES = 100
    NOW = now()
    PATH_MODEL = f"model/resnet18_on_cifar100_{NOW}.pt"

    train_dataloader, test_dataloader = get_CIFAR100_dataloader()

    model = torchvision.models.resnet18()
    model.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(512, NUM_CLASSES), torch.nn.LogSoftmax(dim=1)
    )
    optimizer = torch.optim.Adam(model.parameters())

    model_trainer = CIFAR100ModelTrainer(
        model=model.to(DEVICE),
        optimizer=optimizer,
        criterion=torch.nn.NLLLoss(),
        device=DEVICE,
    )
    model_trainer.iterate_train(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        training_epochs=200,
        log_label="Train Resnet18",
    )
    model_trainer.save(path=PATH_MODEL)

