import torch
from torcheval.metrics.functional import (
    binary_accuracy,
    binary_precision,
    binary_recall,
    binary_f1_score,
    binary_confusion_matrix,
    binary_auroc,
)


def calc_metrics(output: torch.Tensor, target: torch.Tensor):
    return {
        "accuracy": binary_accuracy(output, target).item(),
        "precision": binary_precision(output, target).item(),
        "recall": binary_recall(output, target).item(),
        "f1_score": binary_f1_score(output, target).item(),
        "confusion_matrix": torch.flip(
            binary_confusion_matrix(output, target), dims=[0, 1]
        ),
        "auroc": binary_auroc(output, target).item(),
    }


class BinaryMetrics:
    def __init__(self, tp, fn, fp, tn):
        self.tp = tp
        self.fn = fn
        self.fp = fp
        self.tn = tn

    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fn + self.fp + self.tn)

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def f1_score(self):
        return 2 / (1 / self.precision() + 1 / self.recall())
