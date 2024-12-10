import torch
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_auroc,
    multiclass_confusion_matrix,
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
)


def calc_metrics(prediction: torch.Tensor, target: torch.Tensor, num_classes: int):
    return {
        "accuracy": multiclass_accuracy(prediction, target, num_classes=num_classes),
        "auroc": multiclass_auroc(prediction, target, num_classes=num_classes),
        "confusion_matrix": multiclass_confusion_matrix(
            prediction, target, num_classes=num_classes
        ),
        "f1_score": multiclass_f1_score(prediction, target, num_classes=num_classes),
        "precision": multiclass_precision(prediction, target, num_classes=num_classes),
        "recall": multiclass_recall(prediction, target, num_classes=num_classes),
    }
