from collections.abc import Callable
import torch
import torchvision
from torch import nn
from typing import Any
import time

from src.log.logger import logger_regular, logger_overwrite
from src.model_trainer.model_trainer import ModelTrainer
from src.utils.misc import now


class AttackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, training=self.training)
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc3(x)
        return torch.sigmoid(x)
