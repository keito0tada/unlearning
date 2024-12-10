import torch
import torchvision
from src.model_trainer.model_trainer import ModelTrainer

from src.log.logger import logger_regular


def get_resnet50_trainer(
    num_channels: int, num_classes: int, device: str
) -> ModelTrainer:
    model = torchvision.models.resnet50()
    model.conv1 = torch.nn.Conv2d(
        num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, num_classes))
    optimizer = torch.optim.Adam(model.parameters())

    logger_regular.info("get resnet50 trainer")
    return ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
    )


def get_resnet18_trainer(
    num_channels: int, num_classes: int, device: str
) -> ModelTrainer:
    model = torchvision.models.resnet18()
    model.conv1 = torch.nn.Conv2d(
        num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, num_classes))
    optimizer = torch.optim.Adam(model.parameters())

    logger_regular.info("get resnet18 trainer")
    return ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
    )
