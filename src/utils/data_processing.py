import torch
import copy
import random
import numpy as np


def split_dataset_by_target_classes(
    dataset: torch.utils.data.Dataset, target_classes: list[int]
):
    unlearn_indexes = []
    retain_indexes = []
    for index, data in enumerate(dataset):
        if data[1] in target_classes:
            unlearn_indexes.append(index)
        else:
            retain_indexes.append(index)
    return torch.utils.data.Subset(dataset, unlearn_indexes), torch.utils.data.Subset(
        dataset, retain_indexes
    )


def relabel_dataset(
    dataset,
    target_classes: list[int],
    num_classes: int,
):
    dataset = copy.deepcopy(dataset)
    labels = list(range(num_classes))
    for target_class in target_classes:
        labels.remove(target_class)
    for i in range(len(dataset)):
        if dataset.targets[i] in target_classes:
            dataset.targets[i] = random.choice(labels)
    return dataset


def relabel_dataset_with_target_classes(
    dataset: torch.utils.data.Dataset,
    target_classes: list[int],
    num_classes: int,
):
    data_x = []
    data_y = []
    labels = list(range(num_classes))
    for target_class in target_classes:
        labels.remove(target_class)
    for X, y in copy.deepcopy(dataset):
        data_x.append(X)
        if y in target_classes:
            data_y.append([random.choice(labels)])
        else:
            data_y.append(y)
    return torch.utils.data.TensorDataset(
        torch.stack(data_x),
        torch.tensor(np.array(data_y, dtype=int), dtype=torch.int64),
    )


def relabel_all_dataset(dataset: torch.utils.data.Dataset, num_classes: int):
    data_x = []
    data_y = []
    labels = list(range(num_classes))
    for X, y in copy.deepcopy(dataset):
        data_x.append(X)
        data_y.append([random.choice([label for label in labels if label != y])])
    return torch.utils.data.TensorDataset(
        torch.stack(data_x), torch.tensor(data_y, dtype=torch.int64)
    )
