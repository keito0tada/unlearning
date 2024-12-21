import torch
from torch.utils.data.dataloader import DataLoader
import time
import datetime
from src.model_trainer.model_trainer import ModelTrainer
from src.log.logger import logger_overwrite, logger_regular
from src.utils.binary_metrics import calc_metrics


class AttackModelTrainer(ModelTrainer):
    def train(
        self,
        train_dataloader: DataLoader,
        epoch: int,
        log_label: str,
        log_interval: int = 10,
    ):
        self.model.train()

        for index, (X, y) in enumerate(train_dataloader):
            # move data to device
            X = X.to(self.device)
            y = y.to(self.device).to(torch.float64)

            # train
            self.optimizer.zero_grad()
            pred_y = torch.flatten(self.model(X))
            loss = self.criterion(pred_y, y.to(torch.float32))
            loss.backward()
            self.optimizer.step()

            # output states
            if index % log_interval == 0:
                logger_overwrite.debug(
                    f"{log_label} | Epoch: {epoch} [{index * len(X):6d}] Loss: {loss.item():.6f}"
                )

    def test(self, test_dataloader: DataLoader, log_label: str):
        self.model.eval()

        test_loss = 0
        total_num_example = 0
        correct_num = 0

        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(self.device)
                y: torch.Tensor = y.to(self.device)

                pred_y = torch.flatten(self.model(X))
                total_num_example += y.size()[0]
                test_loss += self.criterion(pred_y, y.to(torch.float32)).item()
                pred = torch.round(pred_y)
                correct_num += pred.eq(y.data.view_as(pred)).sum()

        logger_regular.info(
            f"{log_label} | Mean loss: {test_loss / len(test_dataloader.dataset):.4f}, Accuracy: {correct_num}/{total_num_example} ({100 * correct_num / total_num_example:.0f}%)"
        )
        return correct_num / total_num_example

    def iterate_train(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        training_epochs: int,
        log_label: str = "train",
    ):
        self.model = self.model.to(self.device)
        self.optimizer_to(self.device)

        metrics = []

        for epoch in range(training_epochs):
            start_time = time.perf_counter()

            self.train(
                epoch=epoch, train_dataloader=train_dataloader, log_label=log_label
            )
            self.test(test_dataloader=test_dataloader, log_label=log_label)

            output, target = self.get_prediction_and_target(test_dataloader)
            accuracy, precision, recall, f1_score, confusion_matrix, auroc = (
                calc_metrics(output, target)
            )
            logger_regular.info(
                f"Epoch {epoch} | Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, f1_score: {f1_score}, AUROC: {auroc}"
            )
            logger_regular.info(
                f"Epoch {epoch} | tp: {confusion_matrix[1][1]}, fn: {confusion_matrix[1][0]}, fp: {confusion_matrix[0][1]}, tn: {confusion_matrix[0][0]}"
            )
            metrics.append(
                (accuracy, precision, recall, f1_score, confusion_matrix, auroc)
            )

            logger_regular.info(
                f"{log_label} | Time taken: {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
            )

        self.model = self.model.to("cpu")
        self.optimizer_to("cpu")

        return metrics

    def get_prediction_and_target(
        self, test_dataloader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()

        list_pred_y = []
        list_target_y = []
        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(self.device)
                list_pred_y.append(torch.flatten(self.model(X)).cpu())
                list_target_y.append(y)

        return torch.cat(list_pred_y), torch.cat(list_target_y)

    def attack(
        self, target_model: torch.nn.Module, dataloader: torch.utils.data.DataLoader
    ):
        pass
