import matplotlib.axes
import torch
import torchvision
import torcheval
from torch import nn
import time
import datetime
import matplotlib
import matplotlib.pyplot as plt

from src.model_trainer.model_trainer import ModelTrainer
from src.model_trainer.attack_model import AttackModelTrainer
from src.utils.data_entry import (
    get_CIFAR100_dataset,
    get_MNIST_dataset,
    get_MedMNIST_dataset,
    get_num_channels_and_classes_of_dataset,
)
from src.utils.model_trainer_templates import get_resnet50_trainer, get_resnet18_trainer

from src.log.logger import logger_overwrite, logger_regular, cuda_memory_usage, NOW, now
from src.utils.binary_metrics import calc_metrics
from src.attack.membership_inference_attack import generate_attack_datasets, attack


CUDA_INDEX = 0
DEVICE = f"cuda:{CUDA_INDEX}"

# DATASET = "CIFAR100"
# DATASET = "PathMNIST"
DATASET = "TissueMNIST"
# DATASET = "MNIST"
NUM_CHANNELS, NUM_CLASSES = get_num_channels_and_classes_of_dataset(DATASET)

NUM_SHADOW_MODELS = 20
BATCH_SIZE = 64
ATTACK_BATCH_SIZE = 8
NUM_EPOCHS = 10


class AttackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_CLASSES, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, training=self.training)
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc3(x)
        return torch.sigmoid(x)


def get_attack_model_trainer() -> AttackModelTrainer:
    attack_model = AttackModel()
    attack_optimizer = torch.optim.Adam(attack_model.parameters())
    attack_model_trainer = AttackModelTrainer(
        attack_model, attack_optimizer, nn.functional.binary_cross_entropy, DEVICE
    )
    return attack_model_trainer


def get_model_trainer() -> ModelTrainer:
    return get_resnet18_trainer(NUM_CHANNELS, NUM_CLASSES, DEVICE)
    # return get_resnet50_trainer(NUM_CHANNELS, NUM_CLASSES, DEVICE)


def get_dataset():
    if DATASET == "CIFAR100":
        return get_CIFAR100_dataset()
    elif DATASET == "PathMNIST":
        return get_MedMNIST_dataset("pathmnist")
    elif DATASET == "TissueMNIST":
        return get_MedMNIST_dataset("tissuemnist")
    elif DATASET == "MNIST":
        return get_MNIST_dataset()
    else:
        raise Exception


def generate_target_model(
    path_target_model: str, path_in_dataset: str, path_out_dataset: str
):
    LOG_LABEL = "Generating a target model"

    train_dataset, test_dataset = get_dataset()

    in_dataset, out_dataset = torch.utils.data.random_split(
        train_dataset,
        [
            int(len(train_dataset) / 2),
            len(train_dataset) - int(len(train_dataset) / 2),
        ],
    )
    torch.save(in_dataset, path_in_dataset)
    logger_regular.info(f"IN dataset is saved at {path_in_dataset}")
    torch.save(out_dataset, path_out_dataset)
    logger_regular.info(f"Out dataset is saved at {path_out_dataset}")

    in_dataloader = torch.utils.data.DataLoader(
        in_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, shuffle=True
    )

    # train target model
    target_model_trainer = get_model_trainer()
    target_model_trainer.iterate_train(
        in_dataloader, test_dataloader, NUM_EPOCHS, LOG_LABEL
    )
    target_model_trainer.save(path_target_model)


def generate_shadow_models(path_shadow_models: str, path_shadow_datasets: str):
    LOG_LABEL = "Generaing shadow models"
    # load train and test dataloaders
    train_dataset, test_dataset = get_dataset()

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, shuffle=True
    )

    # generate shadow datasets
    shadow_datasets = []
    for _ in range(NUM_SHADOW_MODELS):
        shadow_datasets.append(
            torch.utils.data.random_split(
                train_dataset,
                [
                    int(len(train_dataset) / 2),
                    len(train_dataset) - int(len(train_dataset) / 2),
                ],
            )
        )
    torch.save(shadow_datasets, path_shadow_datasets)
    logger_regular.info(f"Shadow datasets are saved at {path_shadow_datasets}")

    # generate shadow models
    for index in range(NUM_SHADOW_MODELS):
        start_time = time.perf_counter()

        shadow_model_trainer = get_model_trainer()
        in_dataloader = torch.utils.data.DataLoader(
            shadow_datasets[index][0], batch_size=BATCH_SIZE, shuffle=True
        )

        logger_regular.info(f"Training shadow model {index}")
        shadow_model_trainer.iterate_train(
            train_dataloader=in_dataloader,
            test_dataloader=test_dataloader,
            training_epochs=NUM_EPOCHS,
            log_label=LOG_LABEL,
        )
        logger_regular.info(
            f"Time taken: {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
        )
        shadow_model_trainer.save(path_shadow_models.format(index))
        logger_regular.info(
            f"Shadow model {index} is saved at {path_shadow_models.format(index)}"
        )


def generate_attack_datasets_and_save(
    path_attack_datasets: str,
    path_shadow_model_trainers: str,
    path_shadow_datasets: str,
):
    shadow_model_trainers = []
    for index in range(NUM_SHADOW_MODELS):
        shadow_model_trainer = get_model_trainer()
        shadow_model_trainer.load(path_shadow_model_trainers.format(index))
        shadow_model_trainers.append(shadow_model_trainer)
    shadow_datasets = torch.load(path_shadow_datasets)

    all_predictions = [[] for _ in range(NUM_CLASSES)]
    all_labels = [[] for _ in range(NUM_CLASSES)]
    for index, shadow_model_trainer in enumerate(shadow_model_trainers):
        predictions, labels = generate_attack_datasets(
            NUM_CLASSES,
            shadow_model_trainer.model,
            shadow_datasets[index][0],
            shadow_datasets[index][1],
            DEVICE,
        )
        for target_class in range(NUM_CLASSES):
            logger_regular.debug(
                f"Class {target_class}: {len(predictions[target_class])}"
            )
            all_predictions[target_class].extend(predictions[target_class])
            all_labels[target_class].extend(labels[target_class])

    for target_class in range(NUM_CLASSES):
        torch.save(
            torch.utils.data.TensorDataset(
                torch.stack(all_predictions[target_class]),
                torch.tensor(all_labels[target_class], dtype=torch.int64),
            ),
            path_attack_datasets.format(target_class),
        )
        logger_regular.info(
            f"Attack dataset {target_class} is saved at {path_attack_datasets.format(target_class)}"
        )


def generate_attack_models(
    path_attack_datasets: str,
    path_attack_models: str,
    path_metrics_training_attack_models: str,
):
    predictions_of_each_class = []
    targets_of_each_class = []

    for target_class in range(NUM_CLASSES):
        start_time = time.perf_counter()

        attack_model_trainer = get_attack_model_trainer()

        attack_dataset = torch.load(path_attack_datasets.format(target_class))
        attack_dataset_train, attack_dataset_test = torch.utils.data.random_split(
            attack_dataset,
            [
                int(0.9 * len(attack_dataset)),
                len(attack_dataset) - int(0.9 * len(attack_dataset)),
            ],
        )
        attack_train_dataloader = torch.utils.data.DataLoader(
            attack_dataset_train, batch_size=ATTACK_BATCH_SIZE, shuffle=True
        )
        attack_test_dataloader = torch.utils.data.DataLoader(
            attack_dataset_test, batch_size=ATTACK_BATCH_SIZE, shuffle=True
        )

        predictions, targets = attack_model_trainer.iterate_train(
            attack_train_dataloader,
            attack_test_dataloader,
            NUM_EPOCHS,
            log_label="Attack model training",
        )
        predictions_of_each_class.append(predictions)
        targets_of_each_class.append(targets)

        attack_model_trainer.save(path_attack_models.format(target_class))

        logger_regular.info(
            f"Time taken: {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
        )

    torch.save(
        (predictions_of_each_class, targets_of_each_class),
        path_metrics_training_attack_models,
    )
    logger_regular.info(
        f"Metrics when training attack models is saved at {path_metrics_training_attack_models}"
    )


def attack_and_save(
    path_target_model: str,
    path_in_dataset: str,
    path_out_dataset: str,
    path_attack_models: str,
    path_attack_prediction_and_target_datasets: str,
):
    # load target model
    target_model_trainer = get_model_trainer()
    target_model_trainer.load(path_target_model)

    # dataset used to train the target model
    in_dataset = torch.load(path_in_dataset)
    out_dataset = torch.load(path_out_dataset)

    predictions, labels = generate_attack_datasets(
        NUM_CLASSES, target_model_trainer.model, in_dataset, out_dataset, DEVICE
    )

    attack_prediction_and_target_datasets = attack(
        NUM_CLASSES, BATCH_SIZE, path_attack_models, predictions, labels, DEVICE
    )

    metrics = calc_metrics(
        torch.cat([output for output, _ in attack_prediction_and_target_datasets]),
        torch.cat([target for _, target in attack_prediction_and_target_datasets]),
    )
    logger_regular.info(
        f"Whole | Accuracy: {metrics['accuracy']}, Precision: {metrics['precision']}, Recall: {metrics['recall']}, f1_score: {metrics['f1_score']}, AUROC: {metrics['auroc']}"
    )
    logger_regular.info(
        f"Whole | tp: {metrics['confusion_matrix'][0][0]}, fn: {metrics['confusion_matrix'][0][1]}, fp: {metrics['confusion_matrix'][1][0]}, tn: {metrics['confusion_matrix'][1][1]}"
    )
    torch.save(
        attack_prediction_and_target_datasets,
        path_attack_prediction_and_target_datasets,
    )
    logger_regular.info(
        f"Attack prediction and target datasets are saved at {path_attack_prediction_and_target_datasets}"
    )


def membership_inference_attack():
    DATETIME = NOW
    PATH_TARGET_MODEL = f"model/target_model_{DATETIME}.pt"
    PATH_IN_TARGET_DATASET = f"data/in_dataset_{DATETIME}.pt"
    PATH_OUT_TARGET_DATASET = f"data/out_dataset_{DATETIME}.pt"

    PATH_SHADOW_MODELS = f"model/shadow_model_{{}}_{DATETIME}.pt"
    PATH_SHADOW_DATASETS = f"data/shadow_dataset_{DATETIME}.pt"

    PATH_ATACK_DATASETS = f"data/attack_dataset_{{}}_{DATETIME}.pt"

    PATH_ATTACK_MODELS = f"model/attack_model_{{}}_{DATETIME}.pt"
    PATH_METRICS_TRAINING_ATTACK_MODELS = (
        f"data/metrics_training_attack_models_{DATETIME}.pt"
    )

    PATH_ATTACK_PREDICTION_AND_TARGET_DATASETS = (
        f"data/attack_prediction_and_target_datasets_{DATETIME}.pt"
    )

    # ---
    # PATH_TARGET_MODEL = f"model/target_model_2024-12-05-07:32:48..pt"
    # PATH_IN_TARGET_DATASET = f"data/in_dataset_2024-12-05-07:32:48..pt"
    # PATH_OUT_TARGET_DATASET = f"data/out_dataset_2024-12-05-07:32:48..pt"

    # PATH_SHADOW_MODELS = f"model/shadow_model_{{}}_2024-12-05-07:32:48.pt"
    # PATH_SHADOW_DATASETS = f"data/shadow_dataset_2024-12-05-07:32:48.pt"

    # PATH_ATACK_DATASETS = f"data/attack_dataset_{{}}_2024-12-05-07:32:48.pt"

    # PATH_ATTACK_MODELS = f"model/attack_model_{{}}_2024-12-05-07:32:48.pt"

    # hyperparameter
    logger_regular.info(f"NUM_CHANNELS: {NUM_CHANNELS}, NUM_CLASSES: {NUM_CLASSES}")
    logger_regular.info(
        f"NUM_SHADOW_MODEL: {NUM_SHADOW_MODELS}, BATCH_SIZE: {BATCH_SIZE}, ATTACK_BATCH_SIZE: {ATTACK_BATCH_SIZE}, NUM_EPOCHS: {NUM_EPOCHS}"
    )

    # main
    whole_start_time = start_time = time.perf_counter()

    generate_target_model(
        PATH_TARGET_MODEL, PATH_IN_TARGET_DATASET, PATH_OUT_TARGET_DATASET
    )
    logger_regular.info(
        f"Generating a target model costs : {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
    )
    start_time = time.perf_counter()

    generate_shadow_models(PATH_SHADOW_MODELS, PATH_SHADOW_DATASETS)
    logger_regular.info(
        f"Generating a target model costs : {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
    )
    start_time = time.perf_counter()

    generate_attack_datasets_and_save(
        PATH_ATACK_DATASETS, PATH_SHADOW_MODELS, PATH_SHADOW_DATASETS
    )
    logger_regular.info(
        f"Generating a target model costs : {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
    )
    start_time = time.perf_counter()

    generate_attack_models(
        PATH_ATACK_DATASETS, PATH_ATTACK_MODELS, PATH_METRICS_TRAINING_ATTACK_MODELS
    )
    logger_regular.info(
        f"Generating a target model costs : {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
    )
    start_time = time.perf_counter()

    attack_and_save(
        PATH_TARGET_MODEL,
        PATH_IN_TARGET_DATASET,
        PATH_OUT_TARGET_DATASET,
        PATH_ATTACK_MODELS,
        PATH_ATTACK_PREDICTION_AND_TARGET_DATASETS,
    )
    logger_regular.info(
        f"Generating a target model costs : {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
    )
    start_time = time.perf_counter()

    logger_regular.info(
        f"Whole time taken: {datetime.timedelta(seconds=time.perf_counter() - whole_start_time)}"
    )


def show_metrics(DATETIME=NOW, is_save=True):
    PATH_PREDICTION_AND_TARGET_OF_TRAINING_ATTACK_MODELS = (
        f"data/metrics_training_attack_models_{DATETIME}.pt"
    )
    PATH_ATTACK_PREDICTION_AND_TARGET_DATASETS = (
        f"data/attack_prediction_and_target_datasets_{DATETIME}.pt"
    )

    METRICS_TYPES = ["accuracy", "precision", "recall", "f1_score", "auroc"]
    fig, axes = plt.subplots(3, 3, sharex="all", sharey="all", figsize=(18, 9))
    axes[0][0].set_xlim([-1, 10])
    axes[0][0].set_ylim([0, 1])

    ax_dict: dict[str, matplotlib.axes.Axes] = {
        "accuracy": axes[1][0],
        "precision": axes[1][1],
        "recall": axes[2][0],
        "f1_score": axes[2][1],
        "auroc": axes[2][2],
    }

    predictions_of_each_class, targets_of_each_class = torch.load(
        PATH_PREDICTION_AND_TARGET_OF_TRAINING_ATTACK_MODELS
    )

    for target_class in range(NUM_CLASSES):
        metrics_of_each_epoch = {metrics_type: [] for metrics_type in METRICS_TYPES}
        for epoch in range(NUM_EPOCHS):
            metrics = calc_metrics(
                predictions_of_each_class[target_class][epoch],
                targets_of_each_class[target_class][epoch],
            )
            for metrics_type in METRICS_TYPES:
                metrics_of_each_epoch[metrics_type].append(metrics[metrics_type])
        for metrics_type in METRICS_TYPES:
            ax_dict[metrics_type].plot(
                metrics_of_each_epoch[metrics_type], label=f"Class {target_class}"
            )
            ax_dict[metrics_type].set_title(f"{metrics_type} of each class")
    axes[2][2].legend(
        loc="upper left",
        bbox_to_anchor=(
            1.02,
            1.0,
        ),
        borderaxespad=0,
    )

    whole_metrics = {metrics_type: [] for metrics_type in METRICS_TYPES}
    for epoch in range(NUM_EPOCHS):
        predictions = []
        targets = []
        for target_class in range(NUM_CLASSES):
            predictions.append(predictions_of_each_class[target_class][epoch])
            targets.append(targets_of_each_class[target_class][epoch])
        metrics = calc_metrics(torch.cat(predictions), torch.cat(targets))
        for metrics_type in METRICS_TYPES:
            whole_metrics[metrics_type].append(metrics[metrics_type])
    for metrics_type in METRICS_TYPES:
        axes[0][0].plot(whole_metrics[metrics_type], label=metrics_type)
    axes[0][0].set_title("Metrics")
    axes[0][0].legend()

    attack_prediction_and_target_datasets = torch.load(
        PATH_ATTACK_PREDICTION_AND_TARGET_DATASETS
    )
    metrics = calc_metrics(
        torch.cat([output for output, _ in attack_prediction_and_target_datasets]),
        torch.cat([target for _, target in attack_prediction_and_target_datasets]),
    )
    axes[0][2].set_title("Result of attacking a target model")
    axes[0][2].axis("off")
    table = axes[0][2].table(
        cellText=[
            [
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1_score']:.4f}",
                f"{metrics['auroc']:.4f}",
            ]
        ],
        colLabels=["Accuracy", "Precision", "Recall", "F1 Score", "AUROC"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    axes[1][2].set_title("Result of attacking a target model")
    axes[1][2].axis("off")
    table = axes[1][2].table(
        cellText=[
            [
                metrics["confusion_matrix"][0][0].item(),
                metrics["confusion_matrix"][0][1].item(),
                metrics["confusion_matrix"][1][0].item(),
                metrics["confusion_matrix"][1][1].item(),
            ]
        ],
        colLabels=["tp", "fn", "fp", "tn"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    plt.suptitle(f"MIA to resnet18 trained on {DATASET} ({DATETIME})")
    if is_save:
        plt.savefig(f"image/mia_to_resnet18_trained_on_{DATASET}_{DATETIME}.png")
    else:
        matplotlib.use("tkagg")
        plt.show()


show_metrics("2024-12-26-17:05:16", False)
