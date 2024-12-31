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
    get_num_channels_and_classes_of_dataset,
)
from src.utils.data_processing import (
    split_dataset_by_target_classes,
    relabel_dataset_with_target_classes,
)
from src.log.logger import logger_overwrite, logger_regular, NOW
from src.utils.model_trainer_templates import get_resnet18_trainer, get_resnet50_trainer
from src.utils.multiclass_metrics import calc_metrics
from src.attack.membership_inference_attack import (
    membership_inference_attack,
)


CUDA_INDEX = 0
DEVICE = f"cuda:{CUDA_INDEX}"

DATASET = "CIFAR100"
# DATASET = "PathMNIST"
# DATASET = "TissueMNIST"
# DATASET = "MNIST"
NUM_CHANNELS, NUM_CLASSES = get_num_channels_and_classes_of_dataset(DATASET)

BATCH_SIZE = 64
NUM_EPOCHS = 1
NUM_EPOCHS_UNLEARN = 1
AMNESIAC_RATE = 1

TARGET_CLASSES = [81]


def get_model_trainer():
    return get_resnet18_trainer(NUM_CHANNELS, NUM_CLASSES, DEVICE)
    # return get_resnet50_trainer(NUM_CHANNELS, NUM_CLASSES, DEVICE)


def get_model_performance_metrics(
    model_trainer: ModelTrainer,
    all_train_dataloader: torch.utils.data.DataLoader,
    forget_train_dataloader: torch.utils.data.DataLoader,
    retain_train_dataloader: torch.utils.data.DataLoader,
    all_test_dataloader: torch.utils.data.DataLoader,
    forget_test_dataloader: torch.utils.data.DataLoader,
    retain_test_dataloader: torch.utils.data.DataLoader,
):
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
        forget_test_prediction, forget_test_target = (
            model_trainer.get_prediction_and_target(
                test_dataloader=forget_test_dataloader,
                log_label="Forget test dataset",
            )
        )
        retain_test_prediction, retain_test_target = (
            model_trainer.get_prediction_and_target(
                test_dataloader=retain_test_dataloader,
                log_label="Retain test dataset",
            )
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
            "forget_test": calc_metrics(
                forget_test_prediction,
                forget_test_target,
                NUM_CLASSES,
            ),
            "retain_test": calc_metrics(
                retain_test_prediction,
                retain_test_target,
                NUM_CLASSES,
            ),
        }


def get_mia_metrics(
    model: torch.nn.Module,
    retain_train_dataset: torch.utils.data.Dataset,
    forget_train_dataset: torch.utils.data.Dataset,
    # test_dataset: torch.utils.data.Dataset,
    path_attack_models: str,
) -> dict:
    return {
        "mia_retain_as_positive": membership_inference_attack(
            NUM_CLASSES,
            BATCH_SIZE,
            model,
            retain_train_dataset,
            None,
            path_attack_models,
            DEVICE,
        ),
        "mia_forget_as_positive": membership_inference_attack(
            NUM_CLASSES,
            BATCH_SIZE,
            model,
            forget_train_dataset,
            None,
            path_attack_models,
            DEVICE,
        ),
        # "mia_test_as_positive": membership_inference_attack(
        #     NUM_CLASSES,
        #     BATCH_SIZE,
        #     model,
        #     test_dataset,
        #     None,
        #     path_attack_models,
        #     DEVICE,
        # ),
        # "mia_retain_as_negative": membership_inference_attack(
        #     NUM_CLASSES,
        #     BATCH_SIZE,
        #     model,
        #     None,
        #     retain_train_dataset,
        #     path_attack_models,
        #     DEVICE,
        # ),
        # "mia_forget_as_negative": membership_inference_attack(
        #     NUM_CLASSES,
        #     BATCH_SIZE,
        #     model,
        #     None,
        #     forget_train_dataset,
        #     path_attack_models,
        #     DEVICE,
        # ),
        # "mia_test_as_negative": membership_inference_attack(
        #     NUM_CLASSES,
        #     BATCH_SIZE,
        #     model,
        #     None,
        #     test_dataset,
        #     path_attack_models,
        #     DEVICE,
        # ),
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
    retain_train_dataloader = torch.utils.data.DataLoader(
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
                **get_model_performance_metrics(
                    model_trainer,
                    train_dataloader,
                    forget_train_dataloader,
                    retain_train_dataloader,
                    test_dataloader,
                    forget_test_dataloader,
                    retain_test_dataloader,
                ),
                **get_mia_metrics(
                    model_trainer.model,
                    retain_train_dataset,
                    forget_train_dataset,
                    path_attack_models,
                ),
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
    retain_train_dataloader = torch.utils.data.DataLoader(
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
                **get_model_performance_metrics(
                    model_trainer,
                    train_dataloader,
                    forget_train_dataloader,
                    retain_train_dataloader,
                    test_dataloader,
                    forget_test_dataloader,
                    retain_test_dataloader,
                ),
                **get_mia_metrics(
                    model_trainer.model,
                    retain_train_dataset,
                    forget_train_dataset,
                    path_attack_models,
                ),
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
    logger_regular.info(f"Metrics is saved at {path_retain_metrics}")


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
    retain_train_dataloader = torch.utils.data.DataLoader(
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
                **get_model_performance_metrics(
                    model_trainer,
                    train_dataloader,
                    forget_train_dataloader,
                    retain_train_dataloader,
                    test_dataloader,
                    forget_test_dataloader,
                    retain_test_dataloader,
                ),
                **get_mia_metrics(
                    model_trainer.model,
                    retain_train_dataset,
                    forget_train_dataset,
                    path_attack_models,
                ),
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
    logger_regular.info(f"Metrics is saved at {path_catastrophic_metrics}")


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
    retain_train_dataloader = torch.utils.data.DataLoader(
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
        dict(
            **get_model_performance_metrics(
                model_trainer,
                train_dataloader,
                forget_train_dataloader,
                retain_train_dataloader,
                test_dataloader,
                forget_test_dataloader,
                retain_test_dataloader,
            ),
            **get_mia_metrics(
                model_trainer.model,
                retain_train_dataset,
                forget_train_dataset,
                path_attack_models,
            ),
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
    logger_regular.info(f"Metrics is saved at {path_relabeling_metrics}")


def training_of_amnesiac_unlearning(
    path_unlearning_datasets: str,
    path_amnesiac_trained_model: str,
    path_amnesiac_deltas: str,
    path_amnesiac_training_metrics: str,
    path_attack_models: str,
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
    retain_train_dataloader = torch.utils.data.DataLoader(
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

    train_time = datetime.timedelta(seconds=0)
    test_time = datetime.timedelta(seconds=0)
    metrics = []
    for epoch in range(NUM_EPOCHS):
        start_time = time.perf_counter()
        print("start")
        epoch_deltas = model_trainer.amnesiac_train_by_target_classes(
            train_dataloader, epoch, TARGET_CLASSES[0], LOG_LABEL
        )
        print("end")
        for batch in range(len(train_dataloader)):
            for key in deltas[batch]:
                deltas[batch][key] = epoch_deltas[batch][key] + deltas[batch][key]
        train_end_time = time.perf_counter()
        metrics.append(
            dict(
                **get_model_performance_metrics(
                    model_trainer,
                    train_dataloader,
                    forget_train_dataloader,
                    retain_train_dataloader,
                    test_dataloader,
                    forget_test_dataloader,
                    retain_test_dataloader,
                ),
                **get_mia_metrics(
                    model_trainer.model,
                    retain_train_dataset,
                    forget_train_dataset,
                    path_attack_models,
                ),
            )
        )
        test_end_time = time.perf_counter()
        train_time += datetime.timedelta(seconds=train_end_time - start_time)
        test_time += datetime.timedelta(seconds=test_end_time - train_end_time)
    logger_regular.info(
        f"{LOG_LABEL} | Training costs {train_time}. Testing costs {test_time}."
    )

    model_trainer.save(path_amnesiac_trained_model)
    torch.save(deltas, path_amnesiac_deltas)
    logger_regular.info(f"Amnesiac deltas is saved at {path_amnesiac_deltas}.")
    torch.save(metrics, path_amnesiac_training_metrics)
    logger_regular.info(f"Metrics is saved at {path_amnesiac_training_metrics}")


def amnesiac_unlearning(
    path_unlearning_datasets: str,
    path_amnesiac_target_model: str,
    path_amnesiac_deltas: str,
    path_amnesiac_unlearning_accuracies: str,
    path_attack_models: str,
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
    retain_train_dataloader = torch.utils.data.DataLoader(
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
        for batch_index in range(len(deltas)):
            for param_tensor in state:
                if "weight" in param_tensor or "bias" in param_tensor:
                    state[param_tensor] = (
                        state[param_tensor]
                        - AMNESIAC_RATE * deltas[batch_index][param_tensor]
                    )
        model_trainer.model.load_state_dict(state)

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
                **get_model_performance_metrics(
                    model_trainer,
                    train_dataloader,
                    forget_train_dataloader,
                    retain_train_dataloader,
                    test_dataloader,
                    forget_test_dataloader,
                    retain_test_dataloader,
                ),
                **get_mia_metrics(
                    model_trainer.model,
                    retain_train_dataset,
                    forget_train_dataset,
                    path_attack_models,
                ),
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

    model_trainer.save(path_amnesiac_target_model)
    torch.save(metrics, path_amnesiac_unlearning_accuracies)
    logger_regular.info(f"Metrics is saved at {path_amnesiac_unlearning_accuracies}")


def split_unlearning_datasets(path_unlearning_datasets: str):
    if DATASET == "CIFAR100":
        train_dataset, test_dataset = get_CIFAR100_dataset()
    elif DATASET == "PathMNIST":
        train_dataset, test_dataset = get_MedMNIST_dataset("pathmnist")
    elif DATASET == "TissueMNIST":
        train_dataset, test_dataset = get_MedMNIST_dataset("tissuemnist")
    elif DATASET == "MNIST":
        train_dataset, test_dataset = get_MNIST_dataset()
    else:
        raise Exception

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

    PATH_ATTACK_MODELS = f"model/attack_model_{{}}_2024-12-27-05:38:57.pt"

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

    # PATH_AMNESIAC_MODEL = f"data/amnesiac_model_{DATETIME}.pt"
    # PATH_AMNESIAC_TRAINING_METRICS = f"data/amnesiac_training_metrics_{DATETIME}.pt"
    # PATH_AMNESIAC_DELTAS = f"data/amnesiac_deltas_{DATETIME}.pt"
    # PATH_AMNESIAC_UNLEARNING_METRICS = f"data/amnesiac_unlearning_metrics_{DATETIME}.pt"

    # hyperparameter
    logger_regular.info("=== TARGE CLASS UNLEARNING ===")
    logger_regular.info(f"DATASET: {DATASET}")
    logger_regular.info(f"NUM_CHANNELS: {NUM_CHANNELS}, NUM_CLASSES: {NUM_CLASSES}")
    logger_regular.info(f"BATCH_SIZE: {BATCH_SIZE}, NUM_EPOCH: {NUM_EPOCHS}")
    logger_regular.info(
        f"NUM_EPOCHS_UNLEARN: {NUM_EPOCHS_UNLEARN}, AMNESIAC_RATE: {AMNESIAC_RATE}"
    )
    logger_regular.info(f"FORGET CLASSES: {TARGET_CLASSES}")

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
    training_of_amnesiac_unlearning(
        PATH_UNLEARNING_DATASETS,
        PATH_AMNESIAC_MODEL,
        PATH_AMNESIAC_DELTAS,
        PATH_AMNESIAC_TRAINING_METRICS,
        PATH_ATTACK_MODELS,
    )
    amnesiac_unlearning(
        PATH_UNLEARNING_DATASETS,
        PATH_AMNESIAC_MODEL,
        PATH_AMNESIAC_DELTAS,
        PATH_AMNESIAC_UNLEARNING_METRICS,
        PATH_ATTACK_MODELS,
    )

    logger_regular.info(
        f"Whole time taken: {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
    )
    logger_regular.info("=== Completed ===")


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


def show_metrics(DATETIME=NOW, is_model_performance=True, is_save=True):
    PATH_TARGET_METRICS = f"data/target_metrics_{DATETIME}.pt"
    PATH_RETAIN_METRICS = f"data/retain_metrics_{DATETIME}.pt"
    PATH_CATASTROPHIC_METRICS = f"data/catastrophic_metrics_{DATETIME}.pt"
    PATH_RELABELING_METRICS = f"data/relabeling_metrics_{DATETIME}.pt"
    PATH_AMNESIAC_TRAINING_METRICS = f"data/amnesiac_training_metrics_{DATETIME}.pt"
    PATH_AMNESIAC_UNLEARNING_METRICS = f"data/amnesiac_unlearning_metrics_{DATETIME}.pt"

    if is_model_performance:
        DATASET_TYPES = [
            "all_train",
            "forget_train",
            "retain_train",
            "all_test",
            "forget_test",
            "retain_test",
        ]
    else:
        DATASET_TYPES = [
            "mia_retain_as_positive",
            "mia_forget_as_positive",
            # "mia_test_as_positive",
            # "mia_retain_as_negative",
            # "mia_forget_as_negative",
            # "mia_test_as_negative",
        ]

    METRICS_TYPES = [
        "accuracy",
        "auroc",
        # "confusion matrix",
        "f1_score",
        "precision",
        "recall",
    ]

    logger_regular.info(f"show metrics | {DATETIME}")

    metrics_target = torch.load(PATH_TARGET_METRICS)
    metrics_retain = torch.load(PATH_RETAIN_METRICS)
    metrics_catastrophic = torch.load(PATH_CATASTROPHIC_METRICS)
    metrics_relabeling = torch.load(PATH_RELABELING_METRICS)
    metrics_amnesiac_training = torch.load(PATH_AMNESIAC_TRAINING_METRICS)
    metrics_amnesiac_unlearning = torch.load(PATH_AMNESIAC_UNLEARNING_METRICS)

    fig, axes = plt.subplots(5, 4, sharex="all", sharey="all")

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
        plot_metrics(
            axes[index][3],
            range(NUM_EPOCHS + NUM_EPOCHS_UNLEARN),
            metrics_amnesiac_training + metrics_amnesiac_unlearning,
            DATASET_TYPES,
            metrics_type,
            "Amnesiac",
        )

    axes[0][0].set_xlim([-1, NUM_EPOCHS + NUM_EPOCHS_UNLEARN])
    axes[0][0].set_ylim([-0.1, 1.1])
    axes[0][3].legend(
        loc="upper left",
        bbox_to_anchor=(
            1.02,
            1.0,
        ),
        borderaxespad=0,
    )

    if is_model_performance:
        plt.suptitle(
            f"Model Performance | unlearning resnet18 trained on {DATASET} ({DATETIME})"
        )
        plt.subplots_adjust(hspace=0.5)
        if is_save:
            plt.savefig(
                f"image/performance_unlearning_resnet18_trained_on_{DATASET}_{DATETIME}.png"
            )
        else:
            plt.show()
    else:
        plt.suptitle(f"MIA | unlearning resnet18 trained on {DATASET} ({DATETIME})")
        plt.subplots_adjust(hspace=0.5)
        if is_save:
            plt.savefig(
                f"image/mia_unlearning_resnet18_trained_on_{DATASET}_{DATETIME}.png"
            )
        else:
            plt.show()


main()
show_metrics()
show_metrics(is_model_performance=False)
