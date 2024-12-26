import torch
from torch import nn
from typing import Callable, Optional
from src.log.logger import logger_regular, logger_overwrite
from src.utils.binary_metrics import calc_metrics
from src.model_trainer.attack_model import AttackModelTrainer


class AttackModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(num_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, training=self.training)
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc3(x)
        return torch.sigmoid(x)


def get_attack_model_trainer(num_classes: int, device: str) -> AttackModelTrainer:
    attack_model = AttackModel(num_classes)
    attack_optimizer = torch.optim.Adam(attack_model.parameters())
    attack_model_trainer = AttackModelTrainer(
        attack_model, attack_optimizer, nn.functional.binary_cross_entropy, device
    )
    return attack_model_trainer


def generate_attack_datasets(
    num_classes: int,
    target_model: torch.nn.Module,
    in_dataset: Optional[torch.utils.data.Dataset],
    out_dataset: Optional[torch.utils.data.Dataset],
    device: str,
):
    if in_dataset is None and out_dataset is None:
        raise Exception

    if in_dataset is not None:
        in_dataloader = torch.utils.data.DataLoader(
            in_dataset, batch_size=1, shuffle=True
        )
    if out_dataset is not None:
        out_dataloader = torch.utils.data.DataLoader(
            out_dataset, batch_size=1, shuffle=True
        )

    # generate attack datasets
    predictions = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    with torch.no_grad():
        target_model = target_model.to(device)
        target_model.eval()

        if in_dataset is not None:
            logger_regular.debug(f"Generating an attack dataset from the in loader.")
            for i, (X, y) in enumerate(in_dataloader):
                X = X.to(device)
                pred_y = target_model(X).view(num_classes)
                if torch.argmax(pred_y).item() == y:
                    predictions[y].append(nn.functional.softmax(pred_y, dim=0))
                    labels[y].append(1)

                if i % 100 == 0:
                    logger_overwrite.debug(f"{i} / {len(in_dataloader)}")

        if out_dataset is not None:
            logger_regular.debug(f"Generating an attack dataset from the out loader.")
            for i, (X, y) in enumerate(out_dataloader):
                X = X.to(device)
                pred_y = target_model(X).view(num_classes)
                predictions[y].append(nn.functional.softmax(pred_y, dim=0))
                labels[y].append(0)

                if i % 100 == 0:
                    logger_overwrite.debug(f"{i} / {len(out_dataloader)}")

    return predictions, labels


def attack(
    num_classes: int,
    batch_size: int,
    path_attack_models: str,
    predictions: list[list],
    labels: list[list],
    device: str,
):
    attack_prediction_and_target_datasets = []
    for target_class in range(num_classes):
        logger_regular.info(f"Class {target_class}")

        if len(predictions[target_class]) == 0:
            attack_prediction_and_target_datasets.append((None, None))
            logger_regular.info(f"Class: {target_class} | No data.")
            continue

        attack_dataset = torch.utils.data.TensorDataset(
            torch.stack(predictions[target_class]),
            torch.tensor(labels[target_class], dtype=torch.int64),
        )
        attack_dataloader = torch.utils.data.DataLoader(
            attack_dataset, batch_size=batch_size, shuffle=True
        )

        attack_model_trainer = get_attack_model_trainer(num_classes, device)
        attack_model_trainer.load(path_attack_models.format(target_class))

        attack_model_trainer.model = attack_model_trainer.model.to(device)
        prediction, target = attack_model_trainer.get_prediction_and_target(
            attack_dataloader
        )
        attack_model_trainer.model = attack_model_trainer.model.to("cpu")

        metrics = calc_metrics(prediction, target)
        logger_regular.info(
            f"Whole | Accuracy: {metrics['accuracy']}, Precision: {metrics['precision']}, Recall: {metrics['recall']}, f1_score: {metrics['f1_score']}, AUROC: {metrics['auroc']}"
        )
        logger_regular.info(
            f"Whole | tp: {metrics['confusion_matrix'][0][0]}, fn: {metrics['confusion_matrix'][0][1]}, fp: {metrics['confusion_matrix'][1][0]}, tn: {metrics['confusion_matrix'][1][1]}"
        )

        attack_prediction_and_target_datasets.append((prediction, target))

    return attack_prediction_and_target_datasets


def membership_inference_attack(
    num_classes: int,
    batch_size: int,
    target_model: torch.nn.Module,
    in_dataset: Optional[torch.utils.data.Dataset],
    out_dataset: Optional[torch.utils.data.Dataset],
    path_attack_models: str,
    device: str,
):
    predictions, labels = generate_attack_datasets(
        num_classes, target_model, in_dataset, out_dataset, device
    )

    attack_prediction_and_target_datasets = attack(
        num_classes, batch_size, path_attack_models, predictions, labels
    )

    accuracy, precision, recall, f1_score, confusion_matrix, auroc = calc_metrics(
        torch.cat(
            [
                output
                for output, _ in attack_prediction_and_target_datasets
                if output is not None
            ]
        ),
        torch.cat(
            [
                target
                for _, target in attack_prediction_and_target_datasets
                if target is not None
            ]
        ),
    )
    logger_regular.info(
        f"Whole | Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, f1_score: {f1_score}, AUROC: {auroc}"
    )
    logger_regular.info(
        f"Whole | tp: {confusion_matrix[1][1]}, fn: {confusion_matrix[1][0]}, fp: {confusion_matrix[0][1]}, tn: {confusion_matrix[0][0]}"
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": confusion_matrix,
        "auroc": auroc,
    }
