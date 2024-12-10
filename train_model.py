import torch
import torchvision

from src.utils.data_entry import (
    get_CIFAR100_dataloader,
    get_MNIST_dataloader,
)
from src.model_trainer.model_trainer import ModelTrainer
from src.utils.model_trainer_templates import get_resnet18_trainer

DEVICE = "cuda:0"


def test_training_cifar100():
    NUM_CHANNELS = 3
    NUM_CLASSES = 100
    BATCH_SIZE = 16

    model_trainer = get_resnet18_trainer(NUM_CHANNELS, NUM_CLASSES, DEVICE)
    train_dataloader, test_dataloader = get_CIFAR100_dataloader(BATCH_SIZE)
    model_trainer.iterate_train(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        training_epochs=100,
        log_label="resnet18 on CIFAR100",
    )


test_training_cifar100()
