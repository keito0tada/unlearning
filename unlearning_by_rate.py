import matplotlib.axes
import torch
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
    split_dataset_by_rate,
    relabel_all_dataset,
)
from src.log.logger import logger_overwrite, logger_regular, NOW
from src.utils.model_trainer_templates import get_resnet18_trainer, get_resnet50_trainer
from src.utils.multiclass_metrics import calc_metrics
from src.attack.membership_inference_attack import (
    membership_inference_attack,
)

# matplotlib.use("tkagg")

CUDA_INDEX = 0
DEVICE = f"cuda:{CUDA_INDEX}"

NUM_CHANNELS = 3
NUM_CLASSES = 100
BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_EPOCHS_UNLEARN = 5

UNLEARNING_RATE = 0.05


def get_model_trainer():
    return get_resnet18_trainer(NUM_CHANNELS, NUM_CLASSES, DEVICE)
    # return get_resnet50_trainer(NUM_CHANNELS, NUM_CLASSES, DEVICE)


def get_metrics(
    model_trainer: ModelTrainer,
    all_train_dataloader: torch.utils.data.DataLoader,
    forget_train_dataloader: torch.utils.data.DataLoader,
    retain_train_dataloader: torch.utils.data.DataLoader,
    all_test_dataloader: torch.utils.data.DataLoader,
) -> dict:
    logger_regular.info("get metrics")
    with torch.no_grad():
        all_train_prediction, all_train_target = (
            model_trainer.get_prediction_and_target(
                test_dataloader=all_train_dataloader,
                log_label="All train dataset",
            )
        )
        forget_train_prediction, forget_train_target = (
            model_trainer.get_prediction_and_target(
                test_dataloader=forget_train_dataloader,
                log_label="Forget train dataset",
            )
        )
        retain_train_prediction, retain_train_target = (
            model_trainer.get_prediction_and_target(
                test_dataloader=retain_train_dataloader,
                log_label="Retain train dataset",
            )
        )
        all_test_prediction, all_test_target = model_trainer.get_prediction_and_target(
            test_dataloader=all_test_dataloader,
            log_label="All test dataset",
        )

        return {
            "all_train": calc_metrics(
                all_train_prediction,
                all_train_target,
                NUM_CLASSES,
            ),
            "forget_train": calc_metrics(
                forget_train_prediction,
                forget_train_target,
                NUM_CLASSES,
            ),
            "retain_train": calc_metrics(
                retain_train_prediction,
                retain_train_target,
                NUM_CLASSES,
            ),
            "all_test": calc_metrics(
                all_test_prediction,
                all_test_target,
                NUM_CLASSES,
            ),
        }


def train_target_model(
    path_unlearning_datasets: str,
    path_target_model: str,
    path_target_metrics: str,
    path_attack_models: str,
):
    LOG_LABEL = "Target model"

    (
        train_dataset,
        forget_train_dataset,
        retain_train_dataset,
        relabeled_train_dataset,
        test_dataset,
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

    train_time = datetime.timedelta(seconds=0)
    test_time = datetime.timedelta(seconds=0)
    metrics = []
    for epoch in range(NUM_EPOCHS):
        start_time = time.perf_counter()
        model_trainer.train(
            epoch=epoch, train_dataloader=train_dataloader, log_label=LOG_LABEL
        )
        train_end_time = time.perf_counter()
        metrics.append(
            dict(
                **get_metrics(
                    model_trainer,
                    train_dataloader,
                    forget_train_dataloader,
                    retain_train_dataloader,
                    test_dataloader,
                ),
                **{
                    "mia_retain": membership_inference_attack(
                        NUM_CLASSES,
                        BATCH_SIZE,
                        model_trainer.model,
                        retain_train_dataset,
                        None,
                        path_attack_models,
                        DEVICE,
                    ),
                    "mia_forget": membership_inference_attack(
                        NUM_CLASSES,
                        BATCH_SIZE,
                        model_trainer.model,
                        None,
                        forget_train_dataset,
                        path_attack_models,
                        DEVICE,
                    ),
                    "mia_test": membership_inference_attack(
                        NUM_CLASSES,
                        BATCH_SIZE,
                        model_trainer.model,
                        None,
                        test_dataset,
                        path_attack_models,
                        DEVICE,
                    ),
                },
            )
        )
        test_end_time = time.perf_counter()
        train_time += datetime.timedelta(seconds=train_end_time - start_time)
        test_time += datetime.timedelta(seconds=test_end_time - train_end_time)
    logger_regular.info(
        f"{LOG_LABEL} | Training costs {train_time}. Testing costs {test_time}."
    )

    model_trainer.model = model_trainer.model.to("cpu")
    model_trainer.optimizer_to("cpu")

    model_trainer.save(path_target_model)
    torch.save(metrics, path_target_metrics)
    logger_regular.info(f"Metrics is saved at {path_target_metrics}")


def retain(
    path_unlearning_datasets: str,
    path_retain_model: str,
    path_retain_metrics: str,
    path_attack_models: str,
):
    LOG_LABEL = "Retain model"
    (
        train_dataset,
        forget_train_dataset,
        retain_train_dataset,
        relabeled_train_dataset,
        test_dataset,
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

    train_time = datetime.timedelta(seconds=0)
    test_time = datetime.timedelta(seconds=0)
    metrics = []
    for epoch in range(NUM_EPOCHS):
        start_time = time.perf_counter()
        model_trainer.train(
            epoch=epoch, train_dataloader=retain_train_dataloader, log_label=LOG_LABEL
        )
        train_end_time = time.perf_counter()
        metrics.append(
            dict(
                **get_metrics(
                    model_trainer,
                    train_dataloader,
                    forget_train_dataloader,
                    retain_train_dataloader,
                    test_dataloader,
                ),
                **{
                    "mia_retain": membership_inference_attack(
                        NUM_CLASSES,
                        BATCH_SIZE,
                        model_trainer.model,
                        retain_train_dataset,
                        None,
                        path_attack_models,
                        DEVICE,
                    ),
                    "mia_forget": membership_inference_attack(
                        NUM_CLASSES,
                        BATCH_SIZE,
                        model_trainer.model,
                        None,
                        forget_train_dataset,
                        path_attack_models,
                        DEVICE,
                    ),
                    "mia_test": membership_inference_attack(
                        NUM_CLASSES,
                        BATCH_SIZE,
                        model_trainer.model,
                        None,
                        test_dataset,
                        path_attack_models,
                        DEVICE,
                    ),
                },
            )
        )
        test_end_time = time.perf_counter()
        train_time += datetime.timedelta(seconds=train_end_time - start_time)
        test_time += datetime.timedelta(seconds=test_end_time - train_end_time)
    logger_regular.info(
        f"{LOG_LABEL} | Training costs {train_time}. Testing costs {test_time}."
    )

    model_trainer.model = model_trainer.model.to("cpu")
    model_trainer.optimizer_to("cpu")

    model_trainer.save(path_retain_model)
    torch.save(metrics, path_retain_metrics)
    logger_regular.info(f"Accuracies is saved at {path_retain_metrics}")


def catastrophic_unlearn(
    path_unlearning_datasets: str,
    path_target_model: str,
    path_catastrophic_unlearn_model: str,
    path_catastrophic_metrics: str,
    path_attack_models: str,
):
    LOG_LABEL = "Catastrophic model"
    (
        train_dataset,
        forget_train_dataset,
        retain_train_dataset,
        relabeled_train_dataset,
        test_dataset,
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

    train_time = datetime.timedelta(seconds=0)
    test_time = datetime.timedelta(seconds=0)
    metrics = []
    for epoch in range(NUM_EPOCHS_UNLEARN):
        start_time = time.perf_counter()
        model_trainer.train(
            epoch=epoch, train_dataloader=retain_train_dataloader, log_label=LOG_LABEL
        )
        train_end_time = time.perf_counter()
        metrics.append(
            dict(
                **get_metrics(
                    model_trainer,
                    train_dataloader,
                    forget_train_dataloader,
                    retain_train_dataloader,
                    test_dataloader,
                ),
                **{
                    "mia_retain": membership_inference_attack(
                        NUM_CLASSES,
                        BATCH_SIZE,
                        model_trainer.model,
                        retain_train_dataset,
                        None,
                        path_attack_models,
                        DEVICE,
                    ),
                    "mia_forget": membership_inference_attack(
                        NUM_CLASSES,
                        BATCH_SIZE,
                        model_trainer.model,
                        None,
                        forget_train_dataset,
                        path_attack_models,
                        DEVICE,
                    ),
                    "mia_test": membership_inference_attack(
                        NUM_CLASSES,
                        BATCH_SIZE,
                        model_trainer.model,
                        None,
                        test_dataset,
                        path_attack_models,
                        DEVICE,
                    ),
                },
            )
        )
        test_end_time = time.perf_counter()
        train_time += datetime.timedelta(seconds=train_end_time - start_time)
        test_time += datetime.timedelta(seconds=test_end_time - train_end_time)
    logger_regular.info(
        f"{LOG_LABEL} | Training costs {train_time}. Testing costs {test_time}."
    )

    model_trainer.model = model_trainer.model.to("cpu")
    model_trainer.optimizer_to("cpu")

    model_trainer.save(path_catastrophic_unlearn_model)
    torch.save(metrics, path_catastrophic_metrics)
    logger_regular.info(f"Accuracies is saved at {path_catastrophic_metrics}")


def relabeling_unlearn(
    path_unlearning_datasets: str,
    path_target_model: str,
    path_relabeling_unlearn_model: str,
    path_relabeling_metrics: str,
    path_attack_models: str,
):
    LOG_LABEL = "Relabeling model"

    (
        train_dataset,
        forget_train_dataset,
        retain_train_dataset,
        relabeled_train_dataset,
        test_dataset,
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
    relabeling_train_dataloader = torch.utils.data.DataLoader(
        relabeled_train_dataset, BATCH_SIZE, shuffle=True
    )
    model_trainer = get_model_trainer()
    model_trainer.load(path_target_model)

    model_trainer.model = model_trainer.model.to(DEVICE)
    model_trainer.optimizer_to(DEVICE)

    train_time = datetime.timedelta(seconds=0)
    test_time = datetime.timedelta(seconds=0)
    metrics = []
    for epoch in range(NUM_EPOCHS_UNLEARN):
        start_time = time.perf_counter()
        model_trainer.train(
            epoch=epoch,
            train_dataloader=relabeling_train_dataloader,
            log_label=LOG_LABEL,
        )
        train_end_time = time.perf_counter()
        metrics.append(
            dict(
                **get_metrics(
                    model_trainer,
                    train_dataloader,
                    forget_train_dataloader,
                    retain_train_dataloader,
                    test_dataloader,
                ),
                **{
                    "mia_retain": membership_inference_attack(
                        NUM_CLASSES,
                        BATCH_SIZE,
                        model_trainer.model,
                        retain_train_dataset,
                        None,
                        path_attack_models,
                        DEVICE,
                    ),
                    "mia_forget": membership_inference_attack(
                        NUM_CLASSES,
                        BATCH_SIZE,
                        model_trainer.model,
                        None,
                        forget_train_dataset,
                        path_attack_models,
                        DEVICE,
                    ),
                    "mia_test": membership_inference_attack(
                        NUM_CLASSES,
                        BATCH_SIZE,
                        model_trainer.model,
                        None,
                        test_dataset,
                        path_attack_models,
                        DEVICE,
                    ),
                },
            )
        )
        test_end_time = time.perf_counter()
        train_time += datetime.timedelta(seconds=train_end_time - start_time)
        test_time += datetime.timedelta(seconds=test_end_time - train_end_time)
    logger_regular.info(
        f"{LOG_LABEL} | Training costs {train_time}. Testing costs {test_time}."
    )

    model_trainer.model = model_trainer.model.to("cpu")
    model_trainer.optimizer_to("cpu")

    model_trainer.save(path_relabeling_unlearn_model)
    torch.save(metrics, path_relabeling_metrics)
    logger_regular.info(f"Accuracies is saved at {path_relabeling_metrics}")


def split_unlearning_datasets(path_unlearning_datasets: str):
    # train_dataset, test_dataset = get_MedMNIST_dataset("pathmnist")
    train_dataset, test_dataset = get_CIFAR100_dataset()
    # train_dataset, test_dataset = get_MNIST_dataset()

    forget_train_dataset, retain_train_dataset = split_dataset_by_rate(
        train_dataset, UNLEARNING_RATE
    )

    relabeled_train_dataset = torch.utils.data.ConcatDataset(
        [retain_train_dataset, relabel_all_dataset(forget_train_dataset, NUM_CLASSES)]
    )

    torch.save(
        (
            train_dataset,
            forget_train_dataset,
            retain_train_dataset,
            relabeled_train_dataset,
            test_dataset,
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

    PATH_ATTACK_MODELS = f"model/attack_model_{{}}_2024-12-19-23:27:54.pt"

    # ----
    # PATH_UNLEARNING_DATASETS = f"data/unlearning_datasets_2024-12-10-11:39:54.pt"

    # PATH_TARGET_MODEL = f"data/target_model_{DATETIME}.pt"
    # PATH_TARGET_METRICS = f"data/target_metrics_{DATETIME}.pt"

    # PATH_RETAIN_MODEL = f"data/retain_model_{DATETIME}.pt"
    # PATH_RETAIN_METRICS = f"data/retain_metrics_{DATETIME}.pt"

    # PATH_CATASTROPHIC_MODEL = f"data/catastrophic_model_{DATETIME}.pt"
    # PATH_CATASTROPHIC_METRICS = f"data/catastrophic_metrics_{DATETIME}.pt"

    # PATH_RELABELING_MODEL = f"data/relabeling_model_{DATETIME}.pt"
    # PATH_RELABELING_METRICS = f"data/relabeling_metrics_{DATETIME}.pt"

    # hyperparameter
    logger_regular.info(f"NUM_CHANNELS: {NUM_CHANNELS}, NUM_CLASSES: {NUM_CLASSES}")
    logger_regular.info(f"BATCH_SIZE: {BATCH_SIZE}, NUM_EPOCH: {NUM_EPOCHS}")
    logger_regular.info(f"NUM_EPOCHS_UNLEARN: {NUM_EPOCHS_UNLEARN}")
    logger_regular.info(f"UNLEARNING RATE: {UNLEARNING_RATE}")

    start_time = time.perf_counter()

    split_unlearning_datasets(PATH_UNLEARNING_DATASETS)
    train_target_model(
        PATH_UNLEARNING_DATASETS,
        PATH_TARGET_MODEL,
        PATH_TARGET_METRICS,
        PATH_ATTACK_MODELS,
    )
    retain(
        PATH_UNLEARNING_DATASETS,
        PATH_RETAIN_MODEL,
        PATH_RETAIN_METRICS,
        PATH_ATTACK_MODELS,
    )
    catastrophic_unlearn(
        PATH_UNLEARNING_DATASETS,
        PATH_TARGET_MODEL,
        PATH_CATASTROPHIC_MODEL,
        PATH_CATASTROPHIC_METRICS,
        PATH_ATTACK_MODELS,
    )
    relabeling_unlearn(
        PATH_UNLEARNING_DATASETS,
        PATH_TARGET_MODEL,
        PATH_RELABELING_MODEL,
        PATH_RELABELING_METRICS,
        PATH_ATTACK_MODELS,
    )

    logger_regular.info(
        f"Whole time taken: {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
    )


def plot_metrics(
    ax: matplotlib.axes.Axes,
    x: range,
    metrics: list[dict[str, dict]],
    dataset_types: list[str],
    metrics_type: str,
    title: str,
):
    for dataset_type in dataset_types:
        ax.plot(
            x,
            [data[dataset_type][metrics_type] for data in metrics],
            label=dataset_type,
        )
    ax.set_title(f"{metrics_type} on {title}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metrics_type)


def show_metrics():
    DATETIME = "2024-12-21-18:51:45"
    PATH_TARGET_METRICS = f"data/target_metrics_{DATETIME}.pt"
    PATH_RETAIN_METRICS = f"data/retain_metrics_{DATETIME}.pt"
    PATH_CATASTROPHIC_METRICS = f"data/catastrophic_metrics_{DATETIME}.pt"
    PATH_RELABELING_METRICS = f"data/relabeling_metrics_{DATETIME}.pt"

    # DATASET_TYPES = ["all_train", "forget_train", "retain_train", "all_test"]
    DATASET_TYPES = ["mia_retain", "mia_forget", "mia_test"]
    METRICS_TYPES = [
        "accuracy",
        "auroc",
        # "confusion matrix",
        "f1_score",
        "precision",
        "recall",
    ]

    metrics_target = torch.load(PATH_TARGET_METRICS)
    metrics_retain = torch.load(PATH_RETAIN_METRICS)
    metrics_catastrophic = torch.load(PATH_CATASTROPHIC_METRICS)
    metrics_relabeling = torch.load(PATH_RELABELING_METRICS)

    fig, axes = plt.subplots(5, 3, sharex="all", sharey="all")

    for index, metrics_type in enumerate(METRICS_TYPES):
        plot_metrics(
            axes[index][0],
            range(NUM_EPOCHS),
            metrics_retain,
            DATASET_TYPES,
            metrics_type,
            "Retain",
        )
        plot_metrics(
            axes[index][1],
            range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
            metrics_target + metrics_catastrophic,
            DATASET_TYPES,
            metrics_type,
            "Catastrophic",
        )
        plot_metrics(
            axes[index][2],
            range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
            metrics_target + metrics_relabeling,
            DATASET_TYPES,
            metrics_type,
            "Relabeling",
        )

    axes[0][0].set_xlim([-1, NUM_EPOCHS + NUM_EPOCHS_UNLEARN])
    axes[0][0].set_ylim([0, 1])
    axes[0][2].legend(
        loc="upper left",
        bbox_to_anchor=(
            1.02,
            1.0,
        ),
        borderaxespad=0,
    )

    plt.suptitle(f"MIA to unlearning on resnet18 learning CIFAR100 ({DATETIME})")
    plt.subplots_adjust(hspace=0.5)
    plt.show()


main()
UNLEARNING_RATE = 0.1
main()
UNLEARNING_RATE = 0.15
main()
UNLEARNING_RATE = 0.2
main()
