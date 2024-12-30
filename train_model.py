import torch
import torchvision

from src.utils.data_entry import (
    get_CIFAR100_dataloader,
    get_num_channels_and_classes_of_dataset,
    get_MedMNIST_dataloader,
)
from src.model_trainer.model_trainer import ModelTrainer
from src.log.logger import NOW
from src.utils.model_trainer_templates import (
    get_resnet18_trainer,
    get_resnet18_trainer_with_scheduler,
)

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


def train_resnet18(DATETIME=NOW):
    DATASET = "PathMNIST"
    # DATASET = "TissueMNIST"
    NUM_CHANNELS, NUM_CLASSES = get_num_channels_and_classes_of_dataset(DATASET)

    BATCH_SIZE = 128
    NUM_EPOCHS = 150

    if DATASET == "PathMNIST":
        train_dataloader, test_dataloader = get_MedMNIST_dataloader(
            "pathmnist", BATCH_SIZE
        )
    elif DATASET == "TissueMNIST":
        train_dataloader, test_dataloader = get_MedMNIST_dataloader(
            "tissuemnist", BATCH_SIZE
        )
    else:
        raise Exception

    model_trainer = get_resnet18_trainer_with_scheduler(
        NUM_CHANNELS, NUM_CLASSES, DEVICE, lambda epoch: 0.001 if epoch < 100 else 0.1
    )
    model_trainer.iterate_train(
        train_dataloader, test_dataloader, training_epochs=NUM_EPOCHS, log_label=DATASET
    )
    model_trainer.save(
        f"model/resnet18_trained_on_{DATASET}_{BATCH_SIZE}_{NUM_EPOCHS}_{DATETIME}.pt"
    )


train_resnet18()
