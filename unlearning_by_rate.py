import matplotlib.axes
import torch
import torchvision
import time
import datetime
import matplotlib
from matplotlib import pyplot as plt
from src.model_trainer.cifar100 import CIFAR100ModelTrainer
from src.model_trainer.medmnist import MedMNISTModelTrainer
from src.utils.data_entry_and_processing import (
    get_CIFAR100_dataset,
    get_MNIST_dataset,
    get_MedMNIST_dataset,
    split_dataset_by_target_classes,
    relabel_dataset,
    relabel_dataset_with_target_classes,
    relabel_all_dataset,
)
from src.log.logger import logger_overwrite, logger_regular, NOW

# matplotlib.use("tkagg")

CUDA_INDEX = 0
DEVICE = f"cuda:{CUDA_INDEX}"

NUM_CHANNELS = 3
NUM_CLASSES = 9
BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_EPOCHS_UNLEARN = 5
KEEP_INTERVAL = 10


def get_resnet50_trainer():
    model = torchvision.models.resnet18()
    model.conv1 = torch.nn.Conv2d(
        NUM_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, NUM_CLASSES))
    optimizer = torch.optim.Adam(model.parameters())

    return MedMNISTModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        device=DEVICE,
    )


def get_resnet18_trainer():
    model = torchvision.models.resnet18()
    model.conv1 = torch.nn.Conv2d(
        NUM_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = torch.nn.Sequential(torch.nn.Linear(512, NUM_CLASSES))
    optimizer = torch.optim.Adam(model.parameters())

    return CIFAR100ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        device=DEVICE,
    )


def get_model_trainer():
    # return get_resnet18_trainer()
    return get_resnet50_trainer()


def train_target_model(
    path_unlearning_datasets: str, path_target_model: str, path_target_metrics: str
):
    LOG_LABEL = "Target model"

    (
        train_dataset,
        forget_train_dataset,
        retain_train_dataset,
        relabeled_train_dataset,
        test_dataset,
        rate,
    ) = torch.load(path_unlearning_datasets)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True
    )
    unlearn_train_dataloader = torch.utils.data.DataLoader(
        forget_train_dataset, BATCH_SIZE, shuffle=True
    )
    retain_train_dataloader = torch.utils.data.DataLoader(
        retain_train_dataset, BATCH_SIZE, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, shuffle=True
    )
    model_trainer = get_model_trainer()
    model_trainer.model = model_trainer.model.to(DEVICE)
    model_trainer.optimizer_to(DEVICE)

    accuracies = []
    for epoch in range(NUM_EPOCHS):
        start_time = time.perf_counter()
        model_trainer.train(
            epoch=epoch, train_dataloader=train_dataloader, log_label=LOG_LABEL
        )
        accuracies.append(
            (
                model_trainer.test(
                    test_dataloader=train_dataloader, log_label="Train dataset"
                ),
                model_trainer.test(
                    test_dataloader=unlearn_train_dataloader,
                    log_label="Forget train dataset",
                ),
                model_trainer.test(
                    test_dataloader=retain_train_dataloader,
                    log_label="Retain train dataset",
                ),
                model_trainer.test(
                    test_dataloader=test_dataloader,
                    log_label="Test dataset",
                ),
            )
        )
        logger_regular.info(
            f"{LOG_LABEL} | Time taken: {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
        )

    model_trainer.model = model_trainer.model.to("cpu")
    model_trainer.optimizer_to("cpu")

    model_trainer.save(path_target_model)
    torch.save(accuracies, path_target_metrics)
    logger_regular.info(f"Accuracies is saved at {path_target_metrics}")


def retain(
    path_unlearning_datasets: str, path_retain_model: str, path_retain_metrics: str
):
    LOG_LABEL = "Retain model"
    (
        train_dataset,
        forget_train_dataset,
        retain_train_dataset,
        relabeled_train_dataset,
        test_dataset,
        rate,
    ) = torch.load(path_unlearning_datasets)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True
    )
    forget_train_dataloader = torch.utils.data.DataLoader(
        forget_train_dataset, BATCH_SIZE, shuffle=True
    )
    retain_train_dataloader = torch.utils.data.DataLoader(
        retain_train_dataset, BATCH_SIZE, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, shuffle=True
    )
    model_trainer = get_model_trainer()
    model_trainer.model = model_trainer.model.to(DEVICE)
    model_trainer.optimizer_to(DEVICE)

    accuracies = []
    for epoch in range(NUM_EPOCHS):
        start_time = time.perf_counter()
        model_trainer.train(
            epoch=epoch, train_dataloader=retain_train_dataloader, log_label=LOG_LABEL
        )
        accuracies.append(
            (
                model_trainer.test(
                    test_dataloader=train_dataloader, log_label="Train dataset"
                ),
                model_trainer.test(
                    test_dataloader=forget_train_dataloader,
                    log_label="Forget train dataset",
                ),
                model_trainer.test(
                    test_dataloader=retain_train_dataloader,
                    log_label="Retain train dataset",
                ),
                model_trainer.test(
                    test_dataloader=test_dataloader, log_label="Test dataset"
                ),
            )
        )
        logger_regular.info(
            f"{LOG_LABEL} | Time taken: {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
        )

    model_trainer.model = model_trainer.model.to("cpu")
    model_trainer.optimizer_to("cpu")

    model_trainer.save(path_retain_model)
    torch.save(accuracies, path_retain_metrics)
    logger_regular.info(f"Accuracies is saved at {path_retain_metrics}")


def catastrophic_unlearn(
    path_unlearning_datasets: str,
    path_target_model: str,
    path_catastrophic_unlearn_model: str,
    path_catastrophic_metrics: str,
):
    LOG_LABEL = "Catastrophic model"

    (
        train_dataset,
        forget_train_dataset,
        retain_train_dataset,
        relabeled_train_dataset,
        test_dataset,
        rate,
    ) = torch.load(path_unlearning_datasets)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True
    )
    forget_train_dataloader = torch.utils.data.DataLoader(
        forget_train_dataset, BATCH_SIZE, shuffle=True
    )
    retain_train_dataloader = torch.utils.data.DataLoader(
        retain_train_dataset, BATCH_SIZE, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, shuffle=True
    )
    model_trainer = get_model_trainer()
    model_trainer.load(path_target_model)

    model_trainer.model = model_trainer.model.to(DEVICE)
    model_trainer.optimizer_to(DEVICE)

    accuracies = []
    for epoch in range(NUM_EPOCHS_UNLEARN):
        start_time = time.perf_counter()
        model_trainer.train(
            epoch=epoch,
            train_dataloader=forget_train_dataloader,
            log_label=LOG_LABEL,
        )
        accuracies.append(
            (
                model_trainer.test(
                    test_dataloader=train_dataloader, log_label="Train dataset"
                ),
                model_trainer.test(
                    test_dataloader=forget_train_dataloader,
                    log_label="Forget train dataset",
                ),
                model_trainer.test(
                    test_dataloader=retain_train_dataloader,
                    log_label="Retain train dataset",
                ),
                model_trainer.test(
                    test_dataloader=test_dataloader, log_label="Test dataset"
                ),
            )
        )
        logger_regular.info(
            f"{LOG_LABEL} | Time taken: {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
        )

    model_trainer.model = model_trainer.model.to("cpu")
    model_trainer.optimizer_to("cpu")

    model_trainer.save(path_catastrophic_unlearn_model)
    torch.save(accuracies, path_catastrophic_metrics)
    logger_regular.info(f"Accuracies is saved at {path_catastrophic_metrics}")


def relabeling_unlearn(
    path_unlearning_datasets: str,
    path_target_model: str,
    path_relabeling_unlearn_model: str,
    path_relabeling_metrics: str,
):
    LOG_LABEL = "Relabeling model"

    (
        train_dataset,
        forget_train_dataset,
        retain_train_dataset,
        relabeled_train_dataset,
        test_dataset,
        rate,
    ) = torch.load(path_unlearning_datasets)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True
    )
    forget_train_dataloader = torch.utils.data.DataLoader(
        forget_train_dataset, BATCH_SIZE, shuffle=True
    )
    retain_train_dataloader = torch.utils.data.DataLoader(
        retain_train_dataset, BATCH_SIZE, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, shuffle=True
    )

    model_trainer = get_model_trainer()
    model_trainer.load(path_target_model)

    model_trainer.model = model_trainer.model.to(DEVICE)
    model_trainer.optimizer_to(DEVICE)

    accuracies = []
    for epoch in range(NUM_EPOCHS_UNLEARN):
        start_time = time.perf_counter()
        model_trainer.train(
            epoch=epoch, train_dataloader=train_dataloader, log_label=LOG_LABEL
        )
        accuracies.append(
            (
                model_trainer.test(
                    test_dataloader=train_dataloader, log_label="Train dataset"
                ),
                model_trainer.test(
                    test_dataloader=forget_train_dataloader,
                    log_label="Forget train dataset",
                ),
                model_trainer.test(
                    test_dataloader=retain_train_dataloader,
                    log_label="Retain train dataset",
                ),
                model_trainer.test(
                    test_dataloader=test_dataloader, log_label="Test dataset"
                ),
            )
        )
        logger_regular.info(
            f"{LOG_LABEL} | Time taken: {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
        )

    model_trainer.model = model_trainer.model.to("cpu")
    model_trainer.optimizer_to("cpu")

    model_trainer.save(path_relabeling_unlearn_model)
    torch.save(accuracies, path_relabeling_metrics)
    logger_regular.info(f"Accuracies is saved at {path_relabeling_metrics}")


def training_of_amnesiac_unlearning(path_unlearning_datasets: str):
    LOG_LABEL = "Training of amnesiac unlearning"
    (
        train_dataset,
        forget_train_dataset,
        retain_train_dataset,
        relabeled_train_dataset,
        test_dataset,
        rate,
    ) = torch.load(path_unlearning_datasets)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True
    )
    forget_train_dataloader = torch.utils.data.DataLoader(
        forget_train_dataset, BATCH_SIZE, shuffle=True
    )
    retain_train_dataloader = torch.utils.data.DataLoader(
        retain_train_dataset, BATCH_SIZE, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, shuffle=True
    )
    model_trainer = get_model_trainer()
    model_trainer.model = model_trainer.model.to(DEVICE)
    model_trainer.optimizer_to(DEVICE)

    deltas = []
    for _ in range(len(train_dataloader)):
        delta = {}
        for param_tensor in model_trainer.model.state_dict():
            if "weight" in param_tensor or "bias" in param_tensor:
                delta[param_tensor] = 0
            deltas.append(delta)
    for epoch in range(NUM_EPOCHS):
        start_time = time.perf_counter()
        epoch_deltas = model_trainer.amnesiac_train(
            train_dataloader, epoch, BATCH_SIZE, KEEP_INTERVAL, LOG_LABEL
        )
        for batch in range(len(train_dataloader)):
            for key in deltas[batch]:
                deltas[batch][key] = epoch_deltas[batch][key] + deltas[batch][key]
        model_trainer.test(test_dataloader, log_label=LOG_LABEL)
        logger_regular.info(
            f"{LOG_LABEL} | Time taken: {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
        )


def amnesiac_unlearning(
    path_unlearning_datasets: str,
    path_amnesiac_target_model: str,
    path_amnesiac_deltas: str,
):
    LOG_LABEL = "Training of amnesiac unlearning"
    (
        train_dataset,
        forget_train_dataset,
        retain_train_dataset,
        relabeled_train_dataset,
        test_dataset,
        rate,
    ) = torch.load(path_unlearning_datasets)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True
    )
    forget_train_dataloader = torch.utils.data.DataLoader(
        forget_train_dataset, BATCH_SIZE, shuffle=True
    )
    retain_train_dataloader = torch.utils.data.DataLoader(
        retain_train_dataset, BATCH_SIZE, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, shuffle=True
    )

    model_trainer = get_model_trainer()
    model_trainer.load(path_amnesiac_target_model)
    deltas = torch.load(path_amnesiac_deltas)
    logger_regular.info(f"Amnesiac deltas is loaded from {path_amnesiac_deltas}.")

    model_trainer.model = model_trainer.model.to(DEVICE)
    model_trainer.optimizer_to(DEVICE)


def split_dataset_by_rate(path_unlearning_datasets: str, rate: float):
    train_dataset, test_dataset = get_MedMNIST_dataset("pathmnist")

    forget_train_dataset, retain_train_dataset = torch.utils.data.random_split(
        train_dataset,
        [
            int(len(train_dataset) * rate),
            len(train_dataset) - int(len(train_dataset) * rate),
        ],
    )
    relabeled_train_dataset = relabel_all_dataset(forget_train_dataset, NUM_CLASSES)

    torch.save(
        (
            train_dataset,
            forget_train_dataset,
            retain_train_dataset,
            relabeled_train_dataset,
            test_dataset,
            rate,
        ),
        path_unlearning_datasets,
    )
    logger_regular.info(f"Unlearning dataset is saved at {path_unlearning_datasets}")


def main():
    DATETIME = NOW

    PATH_UNLEARNING_DATASETS = f"data/unlearning_datasets_by_rate_{DATETIME}.pt"

    PATH_TARGET_MODEL = f"data/target_model_{DATETIME}.pt"
    PATH_TARGET_METRICS = f"data/target_metrics_{DATETIME}.pt"
    PATH_RETAIN_MODEL = f"data/retain_model_{DATETIME}.pt"
    PATH_RETAIN_METRICS = f"data/retain_metrics_{DATETIME}.pt"
    PATH_CATASTROPHIC_MODEL = f"data/catastrophic_model_{DATETIME}.pt"
    PATH_CATASTROPHIC_METRICS = f"data/catastrophic_metrics_{DATETIME}.pt"
    PATH_RELABELING_MODEL = f"data/relabeling_model_{DATETIME}.pt"
    PATH_RELABELING_METRICS = f"data/relabeling_metrics_{DATETIME}.pt"

    # PATH_TARGET_MODEL = f"data/target_model_2024-12-06-07:30:33.pt"
    # PATH_TARGET_METRICS = f"data/target_metrics_{DATETIME}.pt"
    # PATH_RETAIN_MODEL = f"data/retain_model_{DATETIME}.pt"
    # PATH_RETAIN_METRICS = f"data/retain_metrics_{DATETIME}.pt"
    # PATH_CATASTROPHIC_MODEL = f"data/catastrophic_model_{DATETIME}.pt"
    # PATH_CATASTROPHIC_METRICS = f"data/catastrophic_result_{DATETIME}.pt"
    # PATH_RELABELING_MODEL = f"data/relabeling_model_2024-12-06-07:30:33.pt"
    # PATH_RELABELING_METRICS = f"data/relabeling_metrics_2024-12-06-07:30:33.pt"

    # PATH_UNLEARNING_DATASET = f"data/unlearning_datasets_2024-12-06-07:30:33.pt"

    split_dataset_by_rate(PATH_UNLEARNING_DATASETS, 0.05)
    train_target_model(PATH_UNLEARNING_DATASETS, PATH_TARGET_MODEL, PATH_TARGET_METRICS)
    retain(PATH_UNLEARNING_DATASETS, PATH_RETAIN_MODEL, PATH_RETAIN_METRICS)
    catastrophic_unlearn(
        PATH_UNLEARNING_DATASETS,
        PATH_TARGET_MODEL,
        PATH_CATASTROPHIC_MODEL,
        PATH_CATASTROPHIC_METRICS,
    )
    relabeling_unlearn(
        PATH_UNLEARNING_DATASETS,
        PATH_TARGET_MODEL,
        PATH_RELABELING_MODEL,
        PATH_RELABELING_METRICS,
    )


def show_metrics():
    DATETIME = "2024-12-07-09:41:17"
    PATH_TARGET_METRICS = f"data/target_metrics_{DATETIME}.pt"
    PATH_RETAIN_METRICS = f"data/retain_metrics_{DATETIME}.pt"
    PATH_CATASTROPHIC_METRICS = f"data/catastrophic_metrics_{DATETIME}.pt"
    PATH_RELABELING_METRICS = f"data/relabeling_metrics_{DATETIME}.pt"

    NUM_EPOCHS = 10
    NUM_EPOCHS_UNLEARN = 5

    accuracies_target = torch.load(PATH_TARGET_METRICS)
    accuracies_retain = torch.load(PATH_RETAIN_METRICS)
    accuracies_catastrophic = torch.load(PATH_CATASTROPHIC_METRICS)
    accuracies_relabeling = torch.load(PATH_RELABELING_METRICS)

    plots = plt.subplots(1, 2, sharex="all", sharey="all")
    fig = plots[0]
    axes: list[matplotlib.axes.Axes] = plots[1]

    axes[0].set_xlim([-1, NUM_EPOCHS + NUM_EPOCHS_UNLEARN])
    axes[0].set_ylim([0, 1])

    axes[0].axhline(
        accuracies_retain[0][-1],
        c="yellow",
        linestyle="dashed",
        label="Train dataset of a retain model (10 epochs)",
    )
    axes[0].axhline(
        accuracies_retain[1][-1],
        c="red",
        linestyle="dashed",
        label="Forget dataset of a retain model (10 epochs)",
    )
    axes[0].axhline(
        accuracies_relabeling[2][-1],
        c="blue",
        linestyle="dashed",
        label="Retain dataset of a retain model (10 epochs)",
    )
    axes[0].axhline(
        accuracies_relabeling[3][-1],
        c="green",
        linestyle="dashed",
        label="Test dataset of a retain model (10 epochs)",
    )
    axes[0].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[0] for accs in accuracies_target]
        + [accs[0] for accs in accuracies_catastrophic],
        label="Train dataset",
    )
    axes[0].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[1] for accs in accuracies_target]
        + [accs[1] for accs in accuracies_catastrophic],
        label="Forget dataset",
    )
    axes[0].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[2] for accs in accuracies_target]
        + [accs[2] for accs in accuracies_catastrophic],
        label="Retain dataset",
    )
    axes[0].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[3] for accs in accuracies_target]
        + [accs[3] for accs in accuracies_catastrophic],
        label="Test dataset",
    )
    axes[0].set_title("Catastrophic")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[0] for accs in accuracies_target]
        + [accs[0] for accs in accuracies_relabeling],
        label="Train dataset",
    )
    axes[1].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[1] for accs in accuracies_target]
        + [accs[1] for accs in accuracies_relabeling],
        label="Forget dataset",
    )
    axes[1].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[2] for accs in accuracies_target]
        + [accs[2] for accs in accuracies_relabeling],
        label="Retain dataset",
    )
    axes[1].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[3] for accs in accuracies_target]
        + [accs[3] for accs in accuracies_relabeling],
        label="Test dataset",
    )
    axes[1].set_title("Relabeling")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.suptitle("Unlearning on resnet50 learning PathMNIST")
    plt.show()


def test():
    train_dataset, test_dataset = get_MedMNIST_dataset("pathmnist")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, shuffle=True
    )

    model_trainer = get_resnet50_trainer()

    model_trainer.iterate_train(train_dataloader, test_dataloader, 30)


show_metrics()
