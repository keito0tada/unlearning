import torch
import torchvision
import time
import datetime
import matplotlib
from matplotlib import pyplot as plt
from src.model_trainer.model_trainer import ModelTrainer
from src.utils.data_entry import (
    get_CIFAR100_dataset,
    get_MNIST_dataset,
    get_MedMNIST_dataset,
)
from src.utils.data_processing import (
    split_dataset_by_target_classes,
    relabel_dataset,
    relabel_dataset_with_target_classes,
)
from src.log.logger import logger_overwrite, logger_regular, NOW

# matplotlib.use("tkagg")

CUDA_INDEX = 1
DEVICE = f"cuda:{CUDA_INDEX}"

NUM_CHANNELS = 3
NUM_CLASSES = 100
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_EPOCHS_UNLEARN = 5
AMNESIAC_RATE = 1

TARGET_CLASSES = [3]


def get_resnet50_trainer():
    model = torchvision.models.resnet18()
    model.conv1 = torch.nn.Conv2d(
        NUM_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, NUM_CLASSES))
    optimizer = torch.optim.Adam(model.parameters())

    return ModelTrainer(
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
    model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, NUM_CLASSES))
    optimizer = torch.optim.Adam(model.parameters())

    return ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        device=DEVICE,
    )


def get_model_trainer():
    return get_resnet18_trainer()
    # return get_resnet50_trainer()


def train_target_model(
    path_unlearning_datasets: str, path_target_model: str, path_target_metrics: str
):
    LOG_LABEL = "Target model"

    (
        train_dataset,
        forget_train_dataset,
        retain_train_dataset,
        test_dataset,
        forget_test_dataset,
        retain_test_dataset,
        relabeled_train_dataset,
    ) = torch.load(path_unlearning_datasets)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True
    )
    forget_train_dataloader = torch.utils.data.DataLoader(
        forget_train_dataset, BATCH_SIZE, shuffle=True
    )
    retain_train_dataloder = torch.utils.data.DataLoader(
        retain_train_dataset, BATCH_SIZE, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, shuffle=True
    )
    forget_test_dataloader = torch.utils.data.DataLoader(
        forget_test_dataset, BATCH_SIZE, shuffle=True
    )
    retain_test_dataloader = torch.utils.data.DataLoader(
        retain_test_dataset, BATCH_SIZE, shuffle=True
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
                    test_dataloader=train_dataloader, log_label="All train dataset"
                ),
                model_trainer.test(
                    test_dataloader=forget_train_dataloader,
                    log_label="Forget train dataset",
                ),
                model_trainer.test(
                    test_dataloader=retain_train_dataloder,
                    log_label="Retain train dataset",
                ),
                model_trainer.test(
                    test_dataloader=test_dataloader, log_label="All test dataset"
                ),
                model_trainer.test(
                    test_dataloader=forget_test_dataloader,
                    log_label="Forget test dataset",
                ),
                model_trainer.test(
                    test_dataloader=retain_test_dataloader,
                    log_label="Retain test dataset",
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
        test_dataset,
        forget_test_dataset,
        retain_test_dataset,
        relabeled_train_dataset,
    ) = torch.load(path_unlearning_datasets)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True
    )
    forget_train_dataloader = torch.utils.data.DataLoader(
        forget_train_dataset, BATCH_SIZE, shuffle=True
    )
    retain_train_dataloder = torch.utils.data.DataLoader(
        retain_train_dataset, BATCH_SIZE, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, shuffle=True
    )
    forget_test_dataloader = torch.utils.data.DataLoader(
        forget_test_dataset, BATCH_SIZE, shuffle=True
    )
    retain_test_dataloader = torch.utils.data.DataLoader(
        retain_test_dataset, BATCH_SIZE, shuffle=True
    )
    model_trainer = get_model_trainer()
    model_trainer.model = model_trainer.model.to(DEVICE)
    model_trainer.optimizer_to(DEVICE)

    accuracies = []
    for epoch in range(NUM_EPOCHS):
        start_time = time.perf_counter()
        model_trainer.train(
            epoch=epoch, train_dataloader=retain_train_dataloder, log_label=LOG_LABEL
        )
        accuracies.append(
            (
                model_trainer.test(
                    test_dataloader=train_dataloader, log_label="All train dataset"
                ),
                model_trainer.test(
                    test_dataloader=forget_train_dataloader,
                    log_label="Forget train dataset",
                ),
                model_trainer.test(
                    test_dataloader=retain_train_dataloder,
                    log_label="Retain train dataset",
                ),
                model_trainer.test(
                    test_dataloader=test_dataloader, log_label="All test dataset"
                ),
                model_trainer.test(
                    test_dataloader=forget_test_dataloader,
                    log_label="Forget test dataset",
                ),
                model_trainer.test(
                    test_dataloader=retain_test_dataloader,
                    log_label="Retain test dataset",
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
        test_dataset,
        forget_test_dataset,
        retain_test_dataset,
        relabeled_train_dataset,
    ) = torch.load(path_unlearning_datasets)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True
    )
    forget_train_dataloader = torch.utils.data.DataLoader(
        forget_train_dataset, BATCH_SIZE, shuffle=True
    )
    retain_train_dataloder = torch.utils.data.DataLoader(
        retain_train_dataset, BATCH_SIZE, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, shuffle=True
    )
    forget_test_dataloader = torch.utils.data.DataLoader(
        forget_test_dataset, BATCH_SIZE, shuffle=True
    )
    retain_test_dataloader = torch.utils.data.DataLoader(
        retain_test_dataset, BATCH_SIZE, shuffle=True
    )
    model_trainer = get_model_trainer()
    model_trainer.load(path_target_model)

    model_trainer.model = model_trainer.model.to(DEVICE)
    model_trainer.optimizer_to(DEVICE)

    accuracies = []
    for epoch in range(NUM_EPOCHS_UNLEARN):
        start_time = time.perf_counter()
        model_trainer.train(
            epoch=epoch, train_dataloader=retain_train_dataloder, log_label=LOG_LABEL
        )
        accuracies.append(
            (
                model_trainer.test(
                    test_dataloader=train_dataloader, log_label="All train dataset"
                ),
                model_trainer.test(
                    test_dataloader=forget_train_dataloader,
                    log_label="Forget train dataset",
                ),
                model_trainer.test(
                    test_dataloader=retain_train_dataloder,
                    log_label="Retain train dataset",
                ),
                model_trainer.test(
                    test_dataloader=test_dataloader, log_label="All test dataset"
                ),
                model_trainer.test(
                    test_dataloader=forget_test_dataloader,
                    log_label="Forget test dataset",
                ),
                model_trainer.test(
                    test_dataloader=retain_test_dataloader,
                    log_label="Retain test dataset",
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
        test_dataset,
        forget_test_dataset,
        retain_test_dataset,
        relabeled_train_dataset,
    ) = torch.load(path_unlearning_datasets)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True
    )
    forget_train_dataloader = torch.utils.data.DataLoader(
        forget_train_dataset, BATCH_SIZE, shuffle=True
    )
    retain_train_dataloder = torch.utils.data.DataLoader(
        retain_train_dataset, BATCH_SIZE, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, shuffle=True
    )
    forget_test_dataloader = torch.utils.data.DataLoader(
        forget_test_dataset, BATCH_SIZE, shuffle=True
    )
    retain_test_dataloader = torch.utils.data.DataLoader(
        retain_test_dataset, BATCH_SIZE, shuffle=True
    )
    relabeling_train_dataloader = torch.utils.data.DataLoader(
        relabeled_train_dataset, BATCH_SIZE, shuffle=True
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
            train_dataloader=relabeling_train_dataloader,
            log_label=LOG_LABEL,
        )
        accuracies.append(
            (
                model_trainer.test(
                    test_dataloader=train_dataloader, log_label="All train dataset"
                ),
                model_trainer.test(
                    test_dataloader=forget_train_dataloader,
                    log_label="Forget train dataset",
                ),
                model_trainer.test(
                    test_dataloader=retain_train_dataloder,
                    log_label="Retain train dataset",
                ),
                model_trainer.test(
                    test_dataloader=test_dataloader, log_label="All test dataset"
                ),
                model_trainer.test(
                    test_dataloader=forget_test_dataloader,
                    log_label="Forget test dataset",
                ),
                model_trainer.test(
                    test_dataloader=retain_test_dataloader,
                    log_label="Retain test dataset",
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


def training_of_amnesiac_unlearning(
    path_unlearning_datasets: str,
    path_amnesiac_trained_model: str,
    path_amnesiac_deltas: str,
    path_amnesiac_model_training_metrics: str,
):
    LOG_LABEL = "Training of amnesiac unlearning"
    (
        train_dataset,
        forget_train_dataset,
        retain_train_dataset,
        test_dataset,
        forget_test_dataset,
        retain_test_dataset,
        relabeled_train_dataset,
    ) = torch.load(path_unlearning_datasets)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True
    )
    forget_train_dataloader = torch.utils.data.DataLoader(
        forget_train_dataset, BATCH_SIZE, shuffle=True
    )
    retain_train_dataloder = torch.utils.data.DataLoader(
        retain_train_dataset, BATCH_SIZE, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, shuffle=True
    )
    forget_test_dataloader = torch.utils.data.DataLoader(
        forget_test_dataset, BATCH_SIZE, shuffle=True
    )
    retain_test_dataloader = torch.utils.data.DataLoader(
        retain_test_dataset, BATCH_SIZE, shuffle=True
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
    accuracies = []
    for epoch in range(NUM_EPOCHS):
        start_time = time.perf_counter()
        epoch_deltas = model_trainer.amnesiac_train(
            train_dataloader, epoch, TARGET_CLASSES[0], LOG_LABEL
        )
        for batch in range(len(train_dataloader)):
            for key in deltas[batch]:
                deltas[batch][key] = epoch_deltas[batch][key] + deltas[batch][key]
        accuracies.append(
            (
                model_trainer.test(
                    test_dataloader=train_dataloader, log_label="All train dataset"
                ),
                model_trainer.test(
                    test_dataloader=forget_train_dataloader,
                    log_label="Forget train dataset",
                ),
                model_trainer.test(
                    test_dataloader=retain_train_dataloder,
                    log_label="Retain train dataset",
                ),
                model_trainer.test(
                    test_dataloader=test_dataloader, log_label="All test dataset"
                ),
                model_trainer.test(
                    test_dataloader=forget_test_dataloader,
                    log_label="Forget test dataset",
                ),
                model_trainer.test(
                    test_dataloader=retain_test_dataloader,
                    log_label="Retain test dataset",
                ),
            )
        )
        logger_regular.info(
            f"{LOG_LABEL} | Time taken: {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
        )

    model_trainer.save(path_amnesiac_trained_model)
    torch.save(deltas, path_amnesiac_deltas)
    logger_regular.info(f"Amnesiac deltas is saved at {path_amnesiac_deltas}.")
    torch.save(accuracies, path_amnesiac_model_training_metrics)
    logger_regular.info(f"Metrics is saved at {path_amnesiac_model_training_metrics}")


def amnesiac_unlearning(
    path_unlearning_datasets: str,
    path_amnesiac_target_model: str,
    path_amnesiac_deltas: str,
    path_amnesiac_unlearning_accuracies: str,
    batch_index: int,
):
    LOG_LABEL = "Training of amnesiac unlearning"
    (
        train_dataset,
        forget_train_dataset,
        retain_train_dataset,
        test_dataset,
        forget_test_dataset,
        retain_test_dataset,
        relabeled_train_dataset,
    ) = torch.load(path_unlearning_datasets)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True
    )
    forget_train_dataloader = torch.utils.data.DataLoader(
        forget_train_dataset, BATCH_SIZE, shuffle=True
    )
    retain_train_dataloder = torch.utils.data.DataLoader(
        retain_train_dataset, BATCH_SIZE, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, shuffle=True
    )
    forget_test_dataloader = torch.utils.data.DataLoader(
        forget_test_dataset, BATCH_SIZE, shuffle=True
    )
    retain_test_dataloader = torch.utils.data.DataLoader(
        retain_test_dataset, BATCH_SIZE, shuffle=True
    )

    model_trainer = get_model_trainer()
    model_trainer.load(path_amnesiac_target_model)
    deltas = torch.load(path_amnesiac_deltas)
    logger_regular.info(f"Amnesiac deltas is loaded from {path_amnesiac_deltas}.")

    with torch.no_grad():
        state = model_trainer.model.state_dict()
        for param_tensor in state:
            if "weight" in param_tensor or "bias" in param_tensor:
                state[param_tensor] = (
                    state[param_tensor]
                    - AMNESIAC_RATE * deltas[batch_index][param_tensor]
                )
        model_trainer.model.load_state_dict(state)

    model_trainer.model = model_trainer.model.to(DEVICE)
    model_trainer.optimizer_to(DEVICE)

    accuracies = []
    for epoch in range(NUM_EPOCHS_UNLEARN):
        start_time = time.perf_counter()
        model_trainer.train(
            epoch=epoch, train_dataloader=retain_train_dataloder, log_label=LOG_LABEL
        )
        accuracies.append(
            (
                model_trainer.test(
                    test_dataloader=train_dataloader, log_label="All train dataset"
                ),
                model_trainer.test(
                    test_dataloader=forget_train_dataloader,
                    log_label="Forget train dataset",
                ),
                model_trainer.test(
                    test_dataloader=retain_train_dataloder,
                    log_label="Retain train dataset",
                ),
                model_trainer.test(
                    test_dataloader=test_dataloader, log_label="All test dataset"
                ),
                model_trainer.test(
                    test_dataloader=forget_test_dataloader,
                    log_label="Forget test dataset",
                ),
                model_trainer.test(
                    test_dataloader=retain_test_dataloader,
                    log_label="Retain test dataset",
                ),
            )
        )
        logger_regular.info(
            f"{LOG_LABEL} | Time taken: {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
        )

    model_trainer.model = model_trainer.model.to("cpu")
    model_trainer.optimizer_to("cpu")

    model_trainer.save(path_amnesiac_target_model)
    torch.save(accuracies, path_amnesiac_unlearning_accuracies)
    logger_regular.info(f"Accuracies is saved at {path_amnesiac_unlearning_accuracies}")


def split_unlearning_datasets(path_unlearning_datasets: str):
    # train_dataset, test_dataset = get_MedMNIST_dataset("pathmnist")
    train_dataset, test_dataset = get_CIFAR100_dataset()

    forget_train_dataset, retain_train_dataset = split_dataset_by_target_classes(
        train_dataset, target_classes=TARGET_CLASSES
    )

    forget_test_dataset, retain_test_dataset = split_dataset_by_target_classes(
        test_dataset, target_classes=TARGET_CLASSES
    )

    relabeled_train_dataset = relabel_dataset_with_target_classes(
        train_dataset, TARGET_CLASSES, NUM_CLASSES
    )

    torch.save(
        (
            train_dataset,
            forget_train_dataset,
            retain_train_dataset,
            test_dataset,
            forget_test_dataset,
            retain_test_dataset,
            relabeled_train_dataset,
        ),
        path_unlearning_datasets,
    )
    logger_regular.info(f"Unlearning dataset is saved at {path_unlearning_datasets}")


def main():
    DATETIME = NOW

    PATH_UNLEARNING_DATASETS = f"data/unlearning_datasets_{DATETIME}.pt"

    PATH_TARGET_MODEL = f"data/target_model_{DATETIME}.pt"
    PATH_TARGET_METRICS = f"data/target_metrics_{DATETIME}.pt"

    PATH_RETAIN_MODEL = f"data/retain_model_{DATETIME}.pt"
    PATH_RETAIN_METRICS = f"data/retain_metrics_{DATETIME}.pt"

    PATH_CATASTROPHIC_MODEL = f"data/catastrophic_model_{DATETIME}.pt"
    PATH_CATASTROPHIC_METRICS = f"data/catastrophic_metrics_{DATETIME}.pt"

    PATH_RELABELING_MODEL = f"data/relabeling_model_{DATETIME}.pt"
    PATH_RELABELING_METRICS = f"data/relabeling_metrics_{DATETIME}.pt"

    PATH_AMNESIAC_MODEL = f"data/amnesiac_model_{DATETIME}.pt"
    PATH_AMNESIAC_TRAINING_METRICS = f"data/amnesiac_training_metrics_{DATETIME}.pt"
    PATH_AMNESIAC_DELTAS = f"data/amnesiac_deltas_{DATETIME}.pt"
    PATH_AMNESIAC_UNLEARNING_METRICS = f"data/amnesiac_unlearning_metrics_{DATETIME}.pt"

    # ----
    # PATH_UNLEARNING_DATASETS = f"data/unlearning_datasets_{DATETIME}.pt"

    # PATH_TARGET_MODEL = f"data/target_model_{DATETIME}.pt"
    # PATH_TARGET_METRICS = f"data/target_metrics_{DATETIME}.pt"

    # PATH_RETAIN_MODEL = f"data/retain_model_{DATETIME}.pt"
    # PATH_RETAIN_METRICS = f"data/retain_metrics_{DATETIME}.pt"

    # PATH_CATASTROPHIC_MODEL = f"data/catastrophic_model_{DATETIME}.pt"
    # PATH_CATASTROPHIC_METRICS = f"data/catastrophic_metrics_{DATETIME}.pt"

    # PATH_RELABELING_MODEL = f"data/relabeling_model_{DATETIME}.pt"
    # PATH_RELABELING_METRICS = f"data/relabeling_metrics_{DATETIME}.pt"

    # PATH_AMNESIAC_MODEL = f"data/amnesiac_model_{DATETIME}.pt"
    # PATH_AMNESIAC_TRAINING_METRICS = f"data/amnesiac_training_metrics_{DATETIME}.pt"
    # PATH_AMNESIAC_DELTAS = f"data/amnesiac_deltas_{DATETIME}.pt"
    # PATH_AMNESIAC_UNLEARNING_METRICS = f"data/amnesiac_unlearning_metrics_{DATETIME}.pt"

    split_unlearning_datasets(PATH_UNLEARNING_DATASETS)
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
    training_of_amnesiac_unlearning(
        PATH_UNLEARNING_DATASETS,
        PATH_AMNESIAC_MODEL,
        PATH_AMNESIAC_DELTAS,
        PATH_AMNESIAC_TRAINING_METRICS,
    )
    amnesiac_unlearning(
        PATH_UNLEARNING_DATASETS,
        PATH_AMNESIAC_MODEL,
        PATH_AMNESIAC_DELTAS,
        PATH_AMNESIAC_UNLEARNING_METRICS,
        1,
    )


def show_metrics():
    DATETIME = "2024-12-08-10:11:22"
    PATH_TARGET_METRICS = f"data/target_metrics_{DATETIME}.pt"
    PATH_RETAIN_METRICS = f"data/retain_metrics_{DATETIME}.pt"
    PATH_CATASTROPHIC_METRICS = f"data/catastrophic_metrics_{DATETIME}.pt"
    PATH_RELABELING_METRICS = f"data/relabeling_metrics_{DATETIME}.pt"
    PATH_AMNESIAC_TRAINING_METRICS = f"data/amnesiac_training_metrics_{DATETIME}.pt"
    PATH_AMNESIAC_UNLEARNING_METRICS = f"data/amnesiac_unlearning_metrics_{DATETIME}.pt"

    NUM_EPOCHS = 10
    NUM_EPOCHS_UNLEARN = 5

    accuracies_target = torch.load(PATH_TARGET_METRICS)
    accuracies_retain = torch.load(PATH_RETAIN_METRICS)
    accuracies_catastrophic = torch.load(PATH_CATASTROPHIC_METRICS)
    accuracies_relabeling = torch.load(PATH_RELABELING_METRICS)
    accuracies_amnesiac_training = torch.load(PATH_AMNESIAC_TRAINING_METRICS)
    accuracies_amnesiac_unlearning = torch.load(PATH_AMNESIAC_UNLEARNING_METRICS)

    fig, axes = plt.subplots(1, 4, sharex="all", sharey="all")

    axes[0].set_xlim([-1, NUM_EPOCHS + NUM_EPOCHS_UNLEARN])
    axes[0].set_ylim([0, 1])

    axes[0].plot(
        range(NUM_EPOCHS),
        [accs[0] for accs in accuracies_retain],
        label="All train dataset",
    )
    axes[0].plot(
        range(NUM_EPOCHS),
        [accs[1] for accs in accuracies_retain],
        label="Forget train dataset",
    )
    axes[0].plot(
        range(NUM_EPOCHS),
        [accs[2] for accs in accuracies_retain],
        label="Retain train dataset",
    )
    axes[0].plot(
        range(NUM_EPOCHS),
        [accs[3] for accs in accuracies_retain],
        label="All test dataset",
    )
    axes[0].plot(
        range(NUM_EPOCHS),
        [accs[4] for accs in accuracies_retain],
        label="Forget test dataset",
    )
    axes[0].plot(
        range(NUM_EPOCHS),
        [accs[5] for accs in accuracies_retain],
        label="Retain test dataset",
    )
    axes[0].set_title("Retain")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[0] for accs in accuracies_target]
        + [accs[0] for accs in accuracies_catastrophic],
        label="All train dataset",
    )
    axes[1].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[1] for accs in accuracies_target]
        + [accs[1] for accs in accuracies_catastrophic],
        label="Forget train dataset",
    )
    axes[1].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[2] for accs in accuracies_target]
        + [accs[2] for accs in accuracies_catastrophic],
        label="Retain train dataset",
    )
    axes[1].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[3] for accs in accuracies_target]
        + [accs[3] for accs in accuracies_catastrophic],
        label="All test dataset",
    )
    axes[1].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[4] for accs in accuracies_target]
        + [accs[4] for accs in accuracies_catastrophic],
        label="Forget test dataset",
    )
    axes[1].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[5] for accs in accuracies_target]
        + [accs[5] for accs in accuracies_catastrophic],
        label="Retain test dataset",
    )
    axes[1].set_title("Catastrophic")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    axes[2].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[0] for accs in accuracies_target]
        + [accs[0] for accs in accuracies_relabeling],
        label="All train dataset",
    )
    axes[2].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[1] for accs in accuracies_target]
        + [accs[1] for accs in accuracies_relabeling],
        label="Forget train dataset",
    )
    axes[2].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[2] for accs in accuracies_target]
        + [accs[2] for accs in accuracies_relabeling],
        label="Retain train dataset",
    )
    axes[2].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[3] for accs in accuracies_target]
        + [accs[3] for accs in accuracies_relabeling],
        label="All test dataset",
    )
    axes[2].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[4] for accs in accuracies_target]
        + [accs[4] for accs in accuracies_relabeling],
        label="Forget test dataset",
    )
    axes[2].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[5] for accs in accuracies_target]
        + [accs[5] for accs in accuracies_relabeling],
        label="Retain test dataset",
    )
    axes[2].set_title("Relabeling")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()

    axes[3].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[0] for accs in accuracies_amnesiac_training]
        + [accs[0] for accs in accuracies_amnesiac_unlearning],
        label="All train dataset",
    )
    axes[3].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[1] for accs in accuracies_amnesiac_training]
        + [accs[1] for accs in accuracies_amnesiac_unlearning],
        label="Forget train dataset",
    )
    axes[3].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[2] for accs in accuracies_amnesiac_training]
        + [accs[2] for accs in accuracies_amnesiac_unlearning],
        label="Retain train dataset",
    )
    axes[3].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[3] for accs in accuracies_amnesiac_training]
        + [accs[3] for accs in accuracies_amnesiac_unlearning],
        label="All test dataset",
    )
    axes[3].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[4] for accs in accuracies_amnesiac_training]
        + [accs[4] for accs in accuracies_amnesiac_unlearning],
        label="Forget test dataset",
    )
    axes[3].plot(
        range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
        [accs[5] for accs in accuracies_amnesiac_training]
        + [accs[5] for accs in accuracies_amnesiac_unlearning],
        label="Retain test dataset",
    )
    axes[3].set_title("Amnesiac")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Accuracy")
    axes[3].legend()

    plt.suptitle("Unlearning on resnet18 learning CIFAR100 (batch size: 64)")
    plt.show()


show_metrics()
