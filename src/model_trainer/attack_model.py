import torch
from torch.utils.data.dataloader import DataLoader
import time
from src.model_trainer.model_trainer import ModelTrainer
from src.log.logger import logger_overwrite, logger_regular


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
            y = y.to(self.device)

            # train
            self.optimizer.zero_grad()
            pred_y = torch.flatten(self.model(X))
            loss = self.criterion(pred_y, y)
            loss.backward()
            self.optimizer.step()

            # output states
            if index % log_interval == 0:
                logger_overwrite.info(
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
                y = y.to(self.device)

                pred_y = torch.flatten(self.model(X))
                total_num_example += y.size()[0]
                test_loss += self.criterion(pred_y, y).item()
                pred = torch.round(pred_y)
                correct_num += pred.eq(y.data.view_as(pred)).sum()

        logger_regular.info(
            f"Mean loss: {test_loss / len(test_dataloader.dataset):.4f}, Accuracy: {correct_num}/{total_num_example} ({100 * correct_num / total_num_example:.0f}%)"
        )
        return correct_num / total_num_example

    def iterate_train(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        training_epochs=10,
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
            tp, tn, fp, fn = self.get_confusion_matrix(test_dataloader, log_label)
            logger_regular.info(
                f"Epoch {epoch} | tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}"
            )
            logger_regular.info(
                f"{log_label} | Time taken: {time.process_time() - start_time}"
            )

        self.model = self.model.to("cpu")
        self.optimizer_to("cpu")

    def get_confusion_matrix(self, test_dataloader: DataLoader, log_label: str):
        self.model.eval()

        tp, tn, fp, fn = (0, 0, 0, 0)
        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(self.device)

                pred_y = self.model(X)
                pred_y = torch.flatten(pred_y)
                pred_y = torch.round(pred_y)
                for index, pred in enumerate(pred_y):
                    if pred == y[index] == 1:
                        tp += 1
                    if pred == y[index] == 0:
                        tn += 1
                    if pred == 1 and y[index] == 0:
                        fp += 1
                    if pred == 0 and y[index] == 1:
                        fn += 1
        return tp, tn, fp, fn
