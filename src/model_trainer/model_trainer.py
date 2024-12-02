import torch
import time
from typing import Callable

from src.log.logger import logger_regular, logger_overwrite


class ModelTrainer:
    def __init__(
        self,
        model: torch.nn.modules.module.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[], torch.Tensor],
        device: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        epoch: int,
        log_label: str,
        log_interval: int = 10,
    ):
        self.model.train()

        for index, (X, y) in enumerate(train_dataloader):
            # move data to device
            X = X.to(self.device)
            y = y.to(self.device)

            # train
            self.optimizer.zero_grad()
            pred_y = self.model(X)
            y = y.squeeze()
            loss = self.criterion(pred_y, y)
            loss.backward()
            self.optimizer.step()

            # output states
            if index % log_interval == 0:
                logger_overwrite.info(
                    f"{log_label} | Epoch: {epoch} [{index * len(X):6d}] Loss: {loss.item():.6f}"
                )

    def test(self, test_dataloader: torch.utils.data.DataLoader, log_label: str):
        self.model.eval()

        test_loss = 0
        total_num_example = 0
        correct_num = 0

        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(self.device)
                y = y.to(self.device)

                pred_y = self.model(X)
                total_num_example += y.size()[0]
                y = y.squeeze()
                test_loss += self.criterion(pred_y, y).item()
                _, pred_class = torch.topk(pred_y, 1, dim=1, largest=True, sorted=True)
                for index, target_class in enumerate(y):
                    if target_class in pred_class[index]:
                        correct_num += 1

        logger_regular.info(
            f"Mean loss: {test_loss / len(test_dataloader.dataset):.4f}, Accuracy: {correct_num}/{total_num_example} ({100 * correct_num / total_num_example:.0f}%)"
        )
        return correct_num / total_num_example

    def get_confusion_matrix(
        self, test_dataloader: torch.utils.data.DataLoader, log_label: str
    ):
        pass

    def iterate_train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        training_epochs: int,
        log_label: str = "train",
    ):
        self.model = self.model.to(self.device)
        self.optimizer_to(self.device)

        for epoch in range(training_epochs):
            start_time = time.process_time()
            self.train(
                epoch=epoch, train_dataloader=train_dataloader, log_label=log_label
            )
            self.test(test_dataloader=test_dataloader, log_label=log_label)
            logger_regular.info(
                f"{log_label} | Time taken: {time.process_time() - start_time}"
            )

        self.model = self.model.to("cpu")
        self.optimizer_to("cpu")

    def optimizer_to(self, device: str):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                state[k] = v.to(device)

    def save(self, path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        logger_regular.info(f"Model was saved at {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger_regular.info(f"Model loaded from {path}")
