import torch
import torcheval


def calc_metrics(output: torch.Tensor, target: torch.Tensor):
    accuracy = torcheval.metrics.functional.binary_accuracy(output, target).item()
    precision = torcheval.metrics.functional.binary_precision(output, target).item()
    recall = torcheval.metrics.functional.binary_recall(output, target).item()
    f1_score = torcheval.metrics.functional.binary_f1_score(output, target).item()
    confusion_matrix = torcheval.metrics.functional.binary_confusion_matrix(
        output, target
    )
    auroc = torcheval.metrics.functional.binary_auroc(output, target).item()
    return accuracy, precision, recall, f1_score, confusion_matrix, auroc


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
