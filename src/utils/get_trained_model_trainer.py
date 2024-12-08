from typing import Optional
import os
import torch
import torchvision
from model_trainer.cifar100 import CIFAR100ModelTrainer
from utils.data_entry import get_MNIST_dataloader


def get_MNIST_model_trainer(
    path: str, device: Optional[str] = None
) -> CIFAR100ModelTrainer:
    NUM_CLASSES = 10
    BATCH_SIZE = 64
    NUM_EPOCHS = 10

    model = torchvision.models.resnet18()
    model.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.fc = torch.nn.Sequential(torch.nn.Linear(512, NUM_CLASSES))
    optimizer = torch.optim.Adam(model.parameters())

    model_trainer = CIFAR100ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
    )

    if os.path.isfile(path):
        model_trainer.load(path)
    else:
        train_dataloader, test_dataloader = get_MNIST_dataloader(BATCH_SIZE)
        model_trainer.iterate_train(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            training_epochs=NUM_EPOCHS,
            log_label="Train Resnet18",
        )
        model_trainer.save(path=path)

    return model_trainer
