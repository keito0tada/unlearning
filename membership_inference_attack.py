import torch
import torchvision
import torcheval
from torch import nn
import time
import datetime
import matplotlib
import matplotlib.pyplot as plt

from src.model_trainer.model_trainer import ModelTrainer
from src.model_trainer.cifar100 import CIFAR100ModelTrainer
from src.model_trainer.attack_model import AttackModelTrainer
from src.utils.data_entry import (
    get_CIFAR100_dataloader,
    get_CIFAR100_dataset,
    get_MNIST_dataloader,
    get_MNIST_dataset,
    get_MedMNIST_dataset,
)
from src.utils.model_trainer_templates import get_resnet50_trainer, get_resnet18_trainer

from src.log.logger import logger_overwrite, logger_regular, cuda_memory_usage, NOW
from src.utils.binary_metrics import calc_metrics, BinaryMetrics

matplotlib.use("tkagg")

CUDA_INDEX = 0
DEVICE = f"cuda:{CUDA_INDEX}"

NUM_SHADOW_MODELS = 20
NUM_CHANNELS = 3
NUM_CLASSES = 100
BATCH_SIZE = 16
ATTACK_BATCH_SIZE = 16
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
    return get_CIFAR100_dataset()
    # return get_MedMNIST_dataset("pathmnist")


def generate_target_model(
    path_target_model: str, path_in_dataset: str, path_out_dataset: str
):
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
        in_dataloader, test_dataloader, NUM_EPOCHS, "target model"
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


def generate_attack_datasets(
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

    for target_class in range(NUM_CLASSES):
        cuda_memory_usage(CUDA_INDEX)
        start_time = time.perf_counter()

        attack_dataset_x = []
        attack_dataset_y = []
        in_count = 0
        out_count = 0
        in_target_count = 0

        for index, shadow_model_trainer in enumerate(shadow_model_trainers):
            with torch.no_grad():
                cuda_memory_usage(CUDA_INDEX)
                logger_regular.debug(
                    f"Generating class {target_class} set from model {index}"
                )
                shadow_model = shadow_model_trainer.model.to(device=DEVICE)
                shadow_model.eval()

                in_dataloader = torch.utils.data.DataLoader(
                    shadow_datasets[index][0], batch_size=1
                )
                logger_regular.debug("Generating an attack dataset from an in dataset")
                for j, (X, y) in enumerate(in_dataloader):
                    if y.item() == target_class:
                        X = X.to(DEVICE)
                        pred_y = shadow_model(X).view(NUM_CLASSES)
                        in_target_count += 1
                        if torch.argmax(pred_y).item() == target_class:
                            attack_dataset_x.append(nn.Softmax(dim=0)(pred_y).cpu())
                            attack_dataset_y.append(0)
                            in_count += 1
                    if j % 100 == 0:
                        logger_overwrite.debug(
                            f"{j} | the size of the attack dataset is {len(attack_dataset_x)} now."
                        )

                out_dataloader = torch.utils.data.DataLoader(
                    shadow_datasets[index][1], batch_size=1
                )
                logger_regular.debug("Generating an attack dataset from an out dataset")
                for j, (X, y) in enumerate(out_dataloader):
                    if y == target_class:
                        X = X.to(DEVICE)
                        pred_y = shadow_model(X).view(NUM_CLASSES)
                        attack_dataset_x.append(nn.Softmax(dim=0)(pred_y).cpu())
                        attack_dataset_y.append(1)
                        out_count += 1
                    if j % 100 == 0:
                        logger_overwrite.debug(
                            f"{j} | the size of the attack dataset is {len(attack_dataset_x)} now."
                        )

                shadow_model = shadow_model.to(device="cpu")

        logger_regular.info(
            f"The size of the attack dataset is {len(attack_dataset_x)} (in: {in_count}, out: {out_count}).         "
        )
        logger_regular.info(
            f"predicted target class in in dataset is {in_target_count}"
        )
        logger_regular.info(
            f"Time taken: {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
        )

        torch.save(
            torch.utils.data.TensorDataset(
                torch.stack(attack_dataset_x),
                torch.tensor(attack_dataset_y, dtype=torch.int64),
            ),
            path_attack_datasets.format(target_class),
        )
        logger_regular.info(
            f"Attack dataset {target_class} is saved at {path_attack_datasets.format(target_class)}"
        )
        cuda_memory_usage(CUDA_INDEX)


def generate_attack_models(
    path_attack_datasets: str,
    path_attack_models: str,
    path_metrics_training_attack_models: str,
):
    metrics = []
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

        metrics.append(
            attack_model_trainer.iterate_train(
                attack_train_dataloader,
                attack_test_dataloader,
                NUM_EPOCHS,
                log_label="Attack model training",
            )
        )
        attack_model_trainer.save(path_attack_models.format(target_class))

        logger_regular.info(
            f"Time taken: {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
        )

    torch.save(metrics, path_metrics_training_attack_models)
    logger_regular.info(
        f"Metrics when training attack models is saved at {path_metrics_training_attack_models}"
    )


def attack(
    path_target_model: str,
    path_in_dataset: str,
    path_out_dataset: str,
    path_attack_models: str,
    path_attack_output_and_target_datasets: str,
):
    # load target model
    target_model_trainer = get_model_trainer()
    target_model_trainer.load(path_target_model)

    # dataset used to train the target model
    in_dataset = torch.load(path_in_dataset)
    out_dataset = torch.load(path_out_dataset)

    in_dataloader = torch.utils.data.DataLoader(in_dataset, batch_size=1, shuffle=True)
    out_dataloader = torch.utils.data.DataLoader(
        out_dataset, batch_size=1, shuffle=True
    )

    attack_output_and_target_datasets = []
    # attack
    for target_class in range(NUM_CLASSES):
        logger_regular.info(f"Class {target_class}")

        # load attack model
        attack_model_trainer = get_attack_model_trainer()
        attack_model_trainer.load(path_attack_models.format(target_class))

        attack_x_dataset = []
        attack_y_dataset = []
        with torch.no_grad():
            target_model = target_model_trainer.model.to(DEVICE)
            target_model.eval()

            logger_regular.debug(f"Generating an attack dataset from the in loader.")
            for i, (X, y) in enumerate(in_dataloader):
                X = X.to(DEVICE)
                if target_class == y.item():
                    pred_y = target_model(X).view(NUM_CLASSES)
                    if torch.argmax(pred_y).item() == target_class:
                        attack_x_dataset.append(nn.functional.softmax(pred_y, dim=0))
                        attack_y_dataset.append(0)
                if i % 100 == 0:
                    logger_overwrite.debug(
                        f"{i} | the size of the attack dataset is {len(attack_x_dataset)} now."
                    )

            logger_regular.debug(f"Generating an attack dataset from the out loader.")
            for i, (X, y) in enumerate(out_dataloader):
                X = X.to(DEVICE)
                if target_class == y.item():
                    pred_y = target_model(X).view(NUM_CLASSES)
                    attack_x_dataset.append(nn.functional.softmax(pred_y, dim=0))
                    attack_y_dataset.append(1)
                if i % 100 == 0:
                    logger_overwrite.debug(
                        f"{i} | the size of the attack dataset is {len(attack_x_dataset)} now."
                    )
        logger_regular.info(
            f"The size of the attack dataset is {len(attack_x_dataset)}."
        )

        attack_dataset = torch.utils.data.TensorDataset(
            torch.stack(attack_x_dataset),
            torch.tensor(attack_y_dataset, dtype=torch.int64),
        )
        attack_dataloader = torch.utils.data.DataLoader(
            attack_dataset, batch_size=BATCH_SIZE, shuffle=True
        )

        attack_model_trainer.model = attack_model_trainer.model.to(DEVICE)
        output, target = attack_model_trainer.get_output_and_target(attack_dataloader)
        accuracy, precision, recall, f1_score, confusion_matrix, auroc = calc_metrics(
            output, target
        )
        logger_regular.info(
            f"Class: {target_class} | Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, f1_score: {f1_score}, AUROC: {auroc}"
        )
        logger_regular.info(
            f"Class: {target_class} | tp: {confusion_matrix[1][1]}, fn: {confusion_matrix[1][0]}, fp: {confusion_matrix[0][1]}, tn: {confusion_matrix[0][0]}"
        )
        attack_output_and_target_datasets.append((output, target))

    accuracy, precision, recall, f1_score, confusion_matrix, auroc = calc_metrics(
        torch.cat([output for output, _ in attack_output_and_target_datasets]),
        torch.cat([target for _, target in attack_output_and_target_datasets]),
    )
    logger_regular.info(
        f"Whole | Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, f1_score: {f1_score}, AUROC: {auroc}"
    )
    logger_regular.info(
        f"Whole | tp: {confusion_matrix[1][1]}, fn: {confusion_matrix[1][0]}, fp: {confusion_matrix[0][1]}, tn: {confusion_matrix[0][0]}"
    )
    torch.save(
        attack_output_and_target_datasets, path_attack_output_and_target_datasets
    )
    logger_regular.info(
        f"Attack output and target datasets are saved at {path_attack_output_and_target_datasets}"
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

    PATH_ATTACK_OUTPUT_AND_TARGET_DATASETS = (
        f"data/attack_output_and_target_datasets_{DATETIME}.pt"
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

    start_time = time.perf_counter()

    generate_target_model(
        PATH_TARGET_MODEL, PATH_IN_TARGET_DATASET, PATH_OUT_TARGET_DATASET
    )
    generate_shadow_models(PATH_SHADOW_MODELS, PATH_SHADOW_DATASETS)
    generate_attack_datasets(
        PATH_ATACK_DATASETS, PATH_SHADOW_MODELS, PATH_SHADOW_DATASETS
    )
    generate_attack_models(
        PATH_ATACK_DATASETS, PATH_ATTACK_MODELS, PATH_METRICS_TRAINING_ATTACK_MODELS
    )
    attack(
        PATH_TARGET_MODEL,
        PATH_IN_TARGET_DATASET,
        PATH_OUT_TARGET_DATASET,
        PATH_ATTACK_MODELS,
        PATH_ATTACK_OUTPUT_AND_TARGET_DATASETS,
    )

    logger_regular.info(
        f"Whole time taken: {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
    )


def show_metrics_training_attack_model():
    DATETIME = "2024-12-09-15:07:22"
    PATH_METRICS_TRAINING_ATTACK_MODELS = (
        f"data/metrics_training_attack_models_{DATETIME}.pt"
    )
    PATH_ATTACK_OUTPUT_AND_TARGET_DATASETS = (
        f"data/attack_output_and_target_datasets_{DATETIME}.pt"
    )

    ig, axes = plt.subplots(3, 3, sharex="all")

    metrics = torch.load(PATH_METRICS_TRAINING_ATTACK_MODELS)
    confusion_matrixes = []

    for target_class in range(NUM_CLASSES):
        accuracy = [metric[0] for metric in metrics[target_class]]
        precision = [metric[1] for metric in metrics[target_class]]
        recall = [metric[2] for metric in metrics[target_class]]
        f1_score = [metric[3] for metric in metrics[target_class]]
        confusion_matrix = [metric[4] for metric in metrics[target_class]]
        auroc = [metric[5] for metric in metrics[target_class]]

        axes[1][0].plot(accuracy, label=f"Class {target_class}")
        axes[1][1].plot(precision, label=f"Class {target_class}")
        axes[2][2].plot(recall, label=f"Class {target_class}")
        axes[2][0].plot(f1_score, label=f"Class {target_class}")
        axes[2][1].plot(auroc, label=f"Class {target_class}")

        confusion_matrixes.append(confusion_matrix)

    axes[1][0].set_xlim([-1, 10])
    axes[2][2].legend(
        loc="upper left",
        bbox_to_anchor=(
            1.02,
            1.0,
        ),
        borderaxespad=0,
    )
    axes[1][0].set_title("Accuracy of each class")
    axes[1][0].set_xlabel("Epoch")
    axes[1][0].set_ylabel("Accuracy")
    axes[1][0].set_ylim([0, 1])
    axes[1][1].set_title("Precision of each class")
    axes[1][1].set_xlabel("Epoch")
    axes[1][1].set_ylabel("Precision")
    axes[1][1].set_ylim([0, 1])
    axes[2][2].set_title("Recall of each class")
    axes[2][2].set_xlabel("Epoch")
    axes[2][2].set_ylabel("Recall")
    axes[2][2].set_ylim([0, 1])
    axes[2][0].set_title("F1-Score of each class")
    axes[2][0].set_xlabel("Epoch")
    axes[2][0].set_ylabel("F1-Score")
    axes[2][0].set_ylim([0, 1])
    axes[2][1].set_title("Auroc of each class")
    axes[2][1].set_xlabel("Epoch")
    axes[2][1].set_ylabel("Auroc")
    axes[2][1].set_ylim([0, 1])

    whole_accuracy = []
    whole_precision = []
    whole_recall = []
    whole_f1_score = []
    whole_tp = []
    whole_tn = []
    whole_fp = []
    whole_fn = []
    for i in range(NUM_EPOCHS):
        tps, tns, fps, fns = 0, 0, 0, 0
        for j in range(NUM_CLASSES):
            tp = confusion_matrixes[j][i][1][1]
            fn = confusion_matrixes[j][i][1][0]
            fp = confusion_matrixes[j][i][0][1]
            tn = confusion_matrixes[j][i][0][0]
            tps += tp
            tns += tn
            fps += fp
            fns += fn
        binary_metrics = BinaryMetrics(tps, fns, fps, tns)
        whole_accuracy.append(binary_metrics.accuracy())
        whole_precision.append(binary_metrics.precision())
        whole_recall.append(binary_metrics.recall())
        whole_f1_score.append(binary_metrics.f1_score())
        whole_tp.append(tps)
        whole_tn.append(tns)
        whole_fp.append(fps)
        whole_fn.append(fns)

    axes[0][0].set_title("Metrics on each epoch")
    axes[0][0].set_xlabel("Epoch")
    axes[0][0].set_ylim([0, 1])
    axes[0][0].plot(whole_accuracy, label="Accuracy")
    axes[0][0].plot(whole_precision, label="Precision")
    axes[0][0].plot(whole_recall, label="Recall")
    axes[0][0].plot(whole_f1_score, label="F1 Score")
    axes[0][0].legend()

    axes[0][1].set_title("Confusion matrix on each epoch")
    axes[0][1].set_xlabel("Epoch")
    axes[0][1].plot(range(NUM_EPOCHS), whole_tp, label="tp")
    axes[0][1].plot(range(NUM_EPOCHS), whole_tn, label="tn")
    axes[0][1].plot(range(NUM_EPOCHS), whole_fp, label="fp")
    axes[0][1].plot(range(NUM_EPOCHS), whole_fn, label="fn")
    axes[0][1].legend()

    attack_output_and_target_datasets = torch.load(
        PATH_ATTACK_OUTPUT_AND_TARGET_DATASETS
    )
    accuracy, precision, recall, f1_score, confusion_matrix, auroc = calc_metrics(
        torch.cat([output for output, _ in attack_output_and_target_datasets]),
        torch.cat([target for _, target in attack_output_and_target_datasets]),
    )
    axes[0][2].set_title("Result of attacking a target model")
    axes[0][2].axis("off")
    table = axes[0][2].table(
        cellText=[
            [
                f"{accuracy:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1_score:.4f}",
                f"{auroc:.4f}",
            ]
        ],
        colLabels=["Accuracy", "Precision", "Recall", "F1 Score", "AUROC"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    axes[1][2].set_title("Result of attacking a target model")
    axes[1][2].axis("off")
    table = axes[1][2].table(
        cellText=[
            [
                confusion_matrix[1][1].item(),
                confusion_matrix[1][0].item(),
                confusion_matrix[0][1].item(),
                confusion_matrix[0][0].item(),
            ]
        ],
        colLabels=["tp", "fn", "fp", "tn"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    plt.suptitle("MIA to resnet50 on PathMNIST")
    plt.show()


def attack_result():
    PATH_ATTACK_OUTPUT_AND_TARGET_DATASETS = (
        f"data/attack_target_model_datasets_2024-12-04-06:15:30.pt"
    )

    datasets = torch.load(PATH_ATTACK_OUTPUT_AND_TARGET_DATASETS)

    start_time = time.perf_counter()
    for index, (output, target) in enumerate(datasets):
        accuracy, precision, recall, f1_score, confusion_matrix, auroc = calc_metrics(
            output, target.to(torch.int64)
        )
        logger_regular.info(
            f"Class: {index} | Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, f1_score: {f1_score}, AUROC: {auroc}"
        )
        print(confusion_matrix)
        logger_regular.info(
            f"tp: {confusion_matrix[1][1]}, fn: {confusion_matrix[1][0]}, fp: {confusion_matrix[0][1]}, tn: {confusion_matrix[0][0]}"
        )

    accuracy, precision, recall, f1_score, confusion_matrix, auroc = calc_metrics(
        torch.cat([output for output, _ in datasets]),
        torch.cat([target for _, target in datasets]).to(torch.int64),
    )
    logger_regular.info(
        f"Whole: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, f1_score: {f1_score}, AUROC: {auroc}"
    )
    logger_regular.info(
        f"tp: {confusion_matrix[1][1]}, fn: {confusion_matrix[1][0]}, fp: {confusion_matrix[0][1]}, tn: {confusion_matrix[0][0]}"
    )
    logger_regular.info(
        f"Whole time taken: {datetime.timedelta(seconds=time.perf_counter() - start_time)}"
    )


membership_inference_attack()
