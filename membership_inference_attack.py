import torch
import torchvision
from torch import nn
import time
import gc

from src.model_trainer.model_trainer import ModelTrainer
from src.model_trainer.cifar100 import CIFAR100ModelTrainer
from src.model_trainer.attack_model import AttackModelTrainer
from src.utils.data_entry_and_processing import (
    get_CIFAR100_dataloader,
    get_MNIST_dataloader,
)

from src.log.logger import logger_overwrite, logger_regular, cuda_memory_usage
from src.utils.misc import now

DEVICE = "cuda:1"

NUM_SHADOW_MODELS = 20
NUM_CHANNELS = 1
NUM_CLASSES = 10
BATCH_SIZE = 60
ATTACK_BATCH_SIZE = 8
NUM_EPOCHS = 10
NOW = now()


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


def get_model() -> tuple[torchvision.Module, torch.optim.Optimizer]:
    shadow_model = torchvision.models.resnet18()
    shadow_model.conv1 = nn.Conv2d(
        NUM_CHANNELS,
        64,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False,
    )
    shadow_model.fc = nn.Sequential(nn.Linear(512, NUM_CLASSES))
    shadow_optimizer = torch.optim.Adam(shadow_model.parameters())
    return shadow_model, shadow_optimizer


def get_model_trainer() -> ModelTrainer:
    model, optimizer = get_model()
    return CIFAR100ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        device=DEVICE,
    )


def generate_shadow_models(path_shadow_models: str, path_shadow_datasets: str):
    # load train and test dataloaders
    train_dataloader, test_dataloader = get_MNIST_dataloader()

    # generate shadow datasets
    shadow_datasets = []
    for _ in range(NUM_SHADOW_MODELS):
        shadow_datasets.append(
            torch.utils.data.random_split(
                train_dataloader.dataset,
                [
                    int(len(train_dataloader.dataset) / 2),
                    int(len(train_dataloader.dataset) / 2),
                ],
            )
        )
    torch.save(shadow_datasets, path_shadow_datasets)
    logger_regular.info(f"Shadow datasets are saved at {path_shadow_datasets}")

    # generate shadow models
    for index in range(NUM_SHADOW_MODELS):
        start_time = time.process_time()

        shadow_model_trainer = get_model_trainer()
        in_dataloader = torch.utils.data.DataLoader(
            shadow_datasets[index][0], batch_size=BATCH_SIZE, shuffle=True
        )

        logger_regular.info(f"Training shadow model {index}")
        shadow_model_trainer.iterate_train(
            train_dataloader=in_dataloader,
            test_dataloader=test_dataloader,
            training_epochs=NUM_EPOCHS,
        )
        logger_regular.info(f"Time taken: {time.process_time() - start_time}")
        shadow_model_trainer.save(path_shadow_models.format(index))


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
        cuda_memory_usage(1)
        start_time = time.process_time()
        attack_dataset_x = []
        attack_dataset_y = []
        for index, shadow_model_trainer in enumerate(shadow_model_trainers):
            with torch.no_grad():
                cuda_memory_usage(1)
                logger_regular.info(
                    f"Generating class {target_class} set from model {index}"
                )
                shadow_model = shadow_model_trainer.model.to(device=DEVICE)

                in_dataloader = torch.utils.data.DataLoader(
                    shadow_datasets[index][0], batch_size=1
                )
                logger_regular.info("Generating an attack dataset from an in dataset")
                for j, (X, y) in enumerate(in_dataloader):
                    if y == target_class:
                        X = X.to(DEVICE)
                        pred_y = shadow_model(X).view(NUM_CLASSES)
                        if torch.argmax(pred_y).item() == target_class:
                            attack_dataset_x.append(nn.Softmax(dim=0)(pred_y).cpu())
                            attack_dataset_y.append(1)
                    if j % 100 == 0:
                        logger_overwrite.info(
                            f"{j} | the size of the attack dataset is {len(attack_dataset_x)} now."
                        )

                out_dataloader = torch.utils.data.DataLoader(
                    shadow_datasets[index][1], batch_size=1
                )
                logger_regular.info("Generating an attack dataset from an out dataset")
                for j, (X, y) in enumerate(out_dataloader):
                    if y == target_class:
                        X = X.to(DEVICE)
                        pred_y = shadow_model(X).view(NUM_CLASSES)
                        attack_dataset_x.append(nn.Softmax(dim=0)(pred_y).cpu())
                        attack_dataset_y.append(0)
                    if j % 100 == 0:
                        logger_overwrite.info(
                            f"{j} | the size of the attack dataset is {len(attack_dataset_x)} now."
                        )

                shadow_model = shadow_model.to(device="cpu")

        logger_regular.info(
            f"The size of the attack dataset is {len(attack_dataset_x)}.         "
        )
        logger_regular.info(f"Time taken: {time.process_time() - start_time}")

        torch.save(
            torch.utils.data.TensorDataset(
                torch.stack(attack_dataset_x), torch.Tensor(attack_dataset_y)
            ),
            path_attack_datasets.format(target_class),
        )
        logger_regular.info(
            f"Attack dataset {target_class} is saved at {path_attack_datasets.format(target_class)}"
        )
        cuda_memory_usage(1)


def generate_attack_models(path_attack_datasets: str, path_attack_models: str):
    for target_class in range(NUM_CLASSES):
        start_time = time.process_time()

        attack_model = AttackModel()
        attack_optimizer = torch.optim.Adam(attack_model.parameters())
        attack_model_trainer = AttackModelTrainer(
            attack_model, attack_optimizer, nn.functional.binary_cross_entropy, DEVICE
        )

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

        attack_model_trainer.iterate_train(
            attack_train_dataloader,
            attack_test_dataloader,
            log_label="Attack model training",
        )
        attack_model_trainer.save(path_attack_models.format(target_class))

        logger_regular.info(f"Time taken: {time.process_time() - start_time}")


def generate_target_model(
    path_target_model: str, path_in_dataset: str, path_out_dataset: str
):
    # dataset
    train_dataloader, test_dataloader = get_MNIST_dataloader(batch_size=BATCH_SIZE)

    in_dataset, out_dataset = torch.utils.data.random_split(
        train_dataloader.dataset,
        [
            int(len(train_dataloader.dataset) / 2),
            int(len(train_dataloader.dataset) / 2),
        ],
    )
    torch.save(in_dataset, path_in_dataset)
    logger_regular.info(f"IN dataset is saved at {path_in_dataset}")
    torch.save(out_dataset, path_out_dataset)
    logger_regular.info(f"Out dataset is saved at {path_out_dataset}")

    in_dataloader = torch.utils.data.DataLoader(
        in_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # train target model
    target_model_trainer = get_model_trainer()
    target_model_trainer.iterate_train(
        in_dataloader, test_dataloader, NUM_EPOCHS, "target model"
    )
    target_model_trainer.save(path_target_model)


def attack(
    path_target_model: str,
    path_in_dataset: str,
    path_out_dataset: str,
    path_attack_models: str,
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

    tps, tns, fps, fns = (0, 0, 0, 0)
    # attack
    for target_class in range(NUM_CLASSES):
        logger_regular.info(f"Class {target_class}")

        # load attack model
        attack_model = AttackModel()
        attack_optimizer = torch.optim.Adam(attack_model.parameters())
        attack_model_trainer = AttackModelTrainer(
            attack_model, attack_optimizer, nn.functional.binary_cross_entropy, DEVICE
        )
        attack_model_trainer.load(path_attack_models.format(target_class))

        attack_x_dataset = []
        attack_y_dataset = []
        with torch.no_grad():
            target_model = target_model_trainer.model.to(DEVICE)
            target_model.eval()

            logger_regular.info(f"Generating an attack dataset from the in loader.")
            for i, (X, y) in enumerate(in_dataloader):
                X = X.to(DEVICE)
                if target_class == y.item():
                    pred_y = target_model(X).view(NUM_CLASSES)
                    if torch.argmax(pred_y).item() == target_class:
                        attack_x_dataset.append(nn.functional.softmax(pred_y, dim=0))
                        attack_y_dataset.append(1)
                if i % 100 == 0:
                    logger_overwrite.info(
                        f"{i} | the size of the attack dataset is {len(attack_x_dataset)} now."
                    )

            logger_regular.info(f"Generating an attack dataset from the out loader.")
            for i, (X, y) in enumerate(out_dataloader):
                X = X.to(DEVICE)
                if target_class == y.item():
                    pred_y = target_model(X).view(NUM_CLASSES)
                    attack_x_dataset.append(nn.functional.softmax(pred_y, dim=0))
                    attack_y_dataset.append(0)
                if i % 100 == 0:
                    logger_overwrite.info(
                        f"{i} | the size of the attack dataset is {len(attack_x_dataset)} now."
                    )
        logger_regular.debug(torch.stack(attack_x_dataset).shape)
        logger_regular.debug(torch.Tensor(attack_y_dataset).shape)
        logger_regular.info(
            f"The size of the attack dataset is {len(attack_x_dataset)}."
        )

        attack_dataset = torch.utils.data.TensorDataset(
            torch.stack(attack_x_dataset), torch.Tensor(attack_y_dataset)
        )
        attack_dataloader = torch.utils.data.DataLoader(
            attack_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        attack_model_trainer.model = attack_model_trainer.model.to(DEVICE)
        tp, tn, fp, fn = attack_model_trainer.get_confusion_matrix(
            attack_dataloader, log_label="confusion matrix"
        )
        tps += tp
        tns += tn
        fps += fp
        fns += fn
        logger_regular.info(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
        logger_regular.info(
            f"recall: {tp / (tp+fn)}, precision: {tp/(tp+fp)}, accuracy: {(tp+tn)/ (tp+fp+tn+fn)}"
        )
    logger_regular.info("Whole result")
    logger_regular.info(f"tp: {tps}, tn: {tns}, fp: {fps}, fn: {fns}")
    logger_regular.info(
        f"recall: {tps / (tps+fns)}, precision: {tps/(tps+fps)}, accuracy: {(tps+tns)/ (tps+fps+tns+fns)}"
    )


def membership_inference_attack():
    # PATH_TARGET_MODEL = f"model/target_model_{NOW}.pt"
    # PATH_IN_TARGET_DATASET = f"data/in_dataset_{NOW}.pt"
    # PATH_OUT_TARGET_DATASET = f"data/out_dataset_{NOW}.pt"

    # PATH_SHADOW_MODELS = f"model/shadow_model_{{}}_{NOW}.pt"
    # PATH_SHADOW_DATASETS = f"data/shadow_dataset_{NOW}.pt"

    # PATH_ATACK_DATASETS = f"data/attack_dataset_{{}}_{NOW}.pt"

    # PATH_ATTACK_MODELS = f"model/attack_model_{{}}_{NOW}.pt"

    PATH_TARGET_MODEL = f"model/target_model_2024-12-02-08:49:25.pt"
    PATH_IN_TARGET_DATASET = f"data/in_dataset_2024-12-02-08:49:25.pt"
    PATH_OUT_TARGET_DATASET = f"data/out_dataset_2024-12-02-08:49:25.pt"

    PATH_SHADOW_MODELS = f"model/shadow_model_{{}}_2024-12-02-08:49:25.pt"
    PATH_SHADOW_DATASETS = f"data/shadow_dataset_2024-12-02-08:49:25.pt"

    PATH_ATACK_DATASETS = f"data/attack_dataset_{{}}_2024-12-02-08:49:25.pt"

    PATH_ATTACK_MODELS = f"model/attack_model_{{}}_2024-12-02-08:49:25.pt"

    # generate_target_model(
    #     PATH_TARGET_MODEL, PATH_IN_TARGET_DATASET, PATH_OUT_TARGET_DATASET
    # )
    # generate_shadow_models(PATH_SHADOW_MODELS, PATH_SHADOW_DATASETS)
    # generate_attack_datasets(
    #     PATH_ATACK_DATASETS, PATH_SHADOW_MODELS, PATH_SHADOW_DATASETS
    # )
    # generate_attack_models(PATH_ATACK_DATASETS, PATH_ATTACK_MODELS)
    attack(
        PATH_TARGET_MODEL,
        PATH_IN_TARGET_DATASET,
        PATH_OUT_TARGET_DATASET,
        PATH_ATTACK_MODELS,
    )


def check_attack_dataset():
    PATH_ATACK_DATASETS = f"data/attack_dataset_{{}}_2024-12-02-08:49:25.pt"

    for target_class in range(NUM_CLASSES):
        attack_dataset = torch.load(PATH_ATACK_DATASETS.format(target_class))
        logger_regular.info(
            f"1: {len([1 for X, y in attack_dataset if y == 1])}, 0: {len([1 for X, y in attack_dataset if y == 0])}"
        )
