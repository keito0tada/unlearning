import os
import torch
import torchvision

from data_entry_and_processing import (
    get_MNIST_dataloader,
    get_mnist_unlearning_threes_dataloader,
)
from model import ModelTrainer
from logger import logger_regular, logger_overwrite


DEVICE = "cuda"

PATH_RESNET18_ON_MNIST = "model/rsenet18_on_mnist.pt"
PATH_RESNET18_ON_NONTHREE_MNIST = "model/resnet18_on_nonthree_mnist.pt"
PATH_RESNET18_ON_RELABELED_MNIST = "model/resnet18_on_relabeled_mnist.pt"


def train_resnet18_on_MNIST(
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    path: str,
) -> ModelTrainer:
    NUM_CLASSES = 10

    model = torchvision.models.resnet18()
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(512, NUM_CLASSES), torch.nn.LogSoftmax(dim=1)
    )
    optimizer = torch.optim.Adam(model.parameters())

    model_trainer = ModelTrainer(
        model=model.to(device=DEVICE),
        optimizer=optimizer,
        device=DEVICE,
    )

    if os.path.isfile(path):
        model_trainer.load(path=path)
    else:
        model_trainer.iterate_train(
            train_loader=train_dataloader, test_loader=test_dataloader
        )
        model_trainer.save(path=path)

    return model_trainer


def three_unlearning():
    NUM_CLASSES = 10

    (
        train_loader,
        test_loader,
        three_train_loader,
        nonthree_train_loader,
        three_test_loader,
        nonthree_test_loader,
        unlearning_train_loader,
    ) = get_mnist_unlearning_threes_dataloader()

    learned_model_trainer = train_resnet18_on_MNIST(
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        path=PATH_RESNET18_ON_MNIST,
    )

    naive_unlearned_model_trainer = train_resnet18_on_MNIST(
        train_dataloader=nonthree_train_loader,
        test_dataloader=nonthree_test_loader,
        path=PATH_RESNET18_ON_NONTHREE_MNIST,
    )

    unlearned_model_trainer = train_resnet18_on_MNIST(
        train_dataloader=unlearning_train_loader,
        test_dataloader=nonthree_test_loader,
        path=PATH_RESNET18_ON_RELABELED_MNIST,
    )

    learned_model_trainer.test(test_loader=test_loader, dname="learned model")
    naive_unlearned_model_trainer.test(test_loader=test_loader, dname="naive model")
    unlearned_model_trainer.test(test_loader=test_loader, dname="unlearned model")

    learned_model_trainer.test(test_loader=three_test_loader, dname="learned model")
    naive_unlearned_model_trainer.test(
        test_loader=three_test_loader, dname="naive model"
    )
    unlearned_model_trainer.test(test_loader=three_test_loader, dname="unlearned model")

    learned_model_trainer.test(test_loader=nonthree_test_loader, dname="learned model")
    naive_unlearned_model_trainer.test(
        test_loader=nonthree_test_loader, dname="naive model"
    )
    unlearned_model_trainer.test(
        test_loader=nonthree_test_loader, dname="unlearned model"
    )


logger_regular.debug("hellooooo")
