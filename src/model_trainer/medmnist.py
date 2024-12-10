import torch

from torch.utils.data.dataloader import DataLoader
from src.model_trainer.model_trainer import ModelTrainer
from src.log.logger import logger_overwrite, logger_regular


class MedMNISTModelTrainer(ModelTrainer):
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
            pred_y = self.model(X)
            y = y.squeeze()
            loss = self.criterion(pred_y, y)
            loss.backward()
            self.optimizer.step()

            # output states
            if index % log_interval == 0:
                logger_overwrite.debug(
                    f"{log_label} | Epoch: {epoch} [{index * len(X):6d}] Loss: {loss.item():.6f}"
                )

    def amnesiac_train_by_target_classes(
        self,
        train_dataloader: DataLoader,
        epoch: int,
        target_class: int,
        log_label: str,
        log_interval: int = 10,
    ):
        self.model.train()

        deltas = []
        for _ in range(len(train_dataloader)):
            delta = {}
            for param_tensor in self.model.state_dict():
                if "weight" in param_tensor or "bias" in param_tensor:
                    delta[param_tensor] = 0
            deltas.append(delta)

        for index, (X, y) in enumerate(train_dataloader):
            X = X.to(self.device)
            y = y.to(self.device)

            before = {}
            for param_tensor in self.model.state_dict():
                if "weight" in param_tensor or "bias" in param_tensor:
                    before[param_tensor] = (
                        self.model.state_dict()[param_tensor].clone().cpu()
                    )
                    before[param_tensor].requires_grad = False

            self.optimizer.zero_grad()
            pred_y = self.model(X)
            y = y.squeeze()
            loss = self.criterion(pred_y, y)
            loss.backward()
            self.optimizer.step()

            if target_class in y:
                after = {}
                for param_tensor in self.model.state_dict():
                    if "weight" in param_tensor or "bias" in param_tensor:
                        after[param_tensor] = (
                            self.model.state_dict()[param_tensor].clone().cpu()
                        )
                        after[param_tensor].requires_grad = False
                for key in before:
                    deltas[index][key] = after[key] - before[key]

            if index % log_interval == 0:
                logger_overwrite.debug(
                    f"{log_label} | Epoch: {epoch} [{index * len(X):6d}] Loss: {loss.item():.6f}"
                )
        return deltas

    def test(self, test_dataloader: DataLoader, log_label: str):
        self.model.eval()

        test_loss = 0
        total_num_example = 0
        correct_num = 0

        top_3_correct_num = 0
        top_5_correct_num = 0

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
                _, pred_class = torch.topk(pred_y, 3, dim=1, largest=True, sorted=True)
                for index, target_class in enumerate(y):
                    if target_class in pred_class[index]:
                        top_3_correct_num += 1
                _, pred_class = torch.topk(pred_y, 5, dim=1, largest=True, sorted=True)
                for index, target_class in enumerate(y):
                    if target_class in pred_class[index]:
                        top_5_correct_num += 1

        logger_regular.info(
            f"{log_label} | Mean loss: {test_loss / len(test_dataloader.dataset):.4f}, Accuracy: {correct_num}/{total_num_example} ({100 * correct_num / total_num_example:.0f}%)"
        )
        logger_regular.info(
            f"{log_label} | top-3 accuracy: {top_3_correct_num}/{total_num_example} ({100 * top_3_correct_num / total_num_example:.1f}%), top-5 accuracy: {top_5_correct_num}/{total_num_example} ({100 * top_5_correct_num/total_num_example:.1f}%)"
        )
        return correct_num / total_num_example
