import os
import datetime
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import torcheval
from torcheval.metrics.functional import multiclass_accuracy

from utils.data_entry import (
    get_MNIST_dataloader,
    get_MNIST_dataset,
    get_mnist_unlearning_threes_dataloader,
    get_MedMNIST_dataloader,
)
from src.model_trainer.model_trainer import ModelTrainer
from src.attack.model_inversion_attack import ModelInversionAttack
from src.log.logger import logger_regular, logger_overwrite
from src.utils.misc import now

matplotlib.use("tkagg")

DEVICE = "cuda"

PATH_RESNET18_ON_MNIST = "model/rsenet18_on_mnist.pt"
PATH_RESNET18_ON_NONTHREE_MNIST = "model/resnet18_on_nonthree_mnist.pt"
PATH_RESNET18_ON_RELABELED_MNIST = "model/resnet18_on_relabeled_mnist.pt"


PATH_IMAGE = "image/testimg10.pt"


def import_model_for_MNIST(path: str):
    NUM_CLASSES = 10

    model = torchvision.models.resnet18()
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(512, NUM_CLASSES), torch.nn.LogSoftmax(dim=1)
    )

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def train_resnet18_on_MNIST(
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    path: str,
) -> ModelTrainer:
    NUM_CLASSES = 10

    model = torchvision.models.resnet18()
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(512, NUM_CLASSES), torch.nn.LogSoftmax(dim=1)
    )
    optimizer = torch.optim.Adam(model.parameters())

    model_trainer = ModelTrainer(
        model=model.to(device=DEVICE),
        optimizer=optimizer,
        device=DEVICE,
    )

    if os.path.isfile(path):
        model_trainer.load(path=path)
    else:
        model_trainer.iterate_train(
            train_dataloader=train_dataloader, test_dataloader=test_dataloader
        )
        model_trainer.save(path=path)

    return model_trainer


def three_unlearning():
    NUM_CLASSES = 10

    (
        train_loader,
        test_loader,
        three_train_loader,
        nonthree_train_loader,
        three_test_loader,
        nonthree_test_loader,
        unlearning_train_loader,
    ) = get_mnist_unlearning_threes_dataloader()

    learned_model_trainer = train_resnet18_on_MNIST(
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        path=PATH_RESNET18_ON_MNIST,
    )

    naive_unlearned_model_trainer = train_resnet18_on_MNIST(
        train_dataloader=nonthree_train_loader,
        test_dataloader=nonthree_test_loader,
        path=PATH_RESNET18_ON_NONTHREE_MNIST,
    )

    unlearned_model_trainer = train_resnet18_on_MNIST(
        train_dataloader=unlearning_train_loader,
        test_dataloader=nonthree_test_loader,
        path=PATH_RESNET18_ON_RELABELED_MNIST,
    )

    learned_model_trainer.test(test_dataloader=test_loader, dname="learned model")
    naive_unlearned_model_trainer.test(test_dataloader=test_loader, dname="naive model")
    unlearned_model_trainer.test(test_dataloader=test_loader, dname="unlearned model")

    learned_model_trainer.test(test_dataloader=three_test_loader, dname="learned model")
    naive_unlearned_model_trainer.test(
        test_dataloader=three_test_loader, dname="naive model"
    )
    unlearned_model_trainer.test(
        test_dataloader=three_test_loader, dname="unlearned model"
    )

    learned_model_trainer.test(
        test_dataloader=nonthree_test_loader, dname="learned model"
    )
    naive_unlearned_model_trainer.test(
        test_dataloader=nonthree_test_loader, dname="naive model"
    )
    unlearned_model_trainer.test(
        test_dataloader=nonthree_test_loader, dname="unlearned model"
    )


def test():
    model = import_model_for_MNIST(PATH_RESNET18_ON_MNIST)
    model_inversion_attack = ModelInversionAttack(model=model.to(DEVICE), device=DEVICE)
    images = model_inversion_attack.generate_images(target=3, num_examples=10)
    torch.save({"images": images}, PATH_IMAGE)


def normalize(img):
    pass


def test2():
    images = torch.load(PATH_IMAGE)["images"]
    plt.imshow(images[0][0].cpu()[0].permute(1, 2, 0).detach().numpy(), cmap="Greys")
    plt.show()
    images_inverted = torch.cat([image[0].cpu() / 2 + 0.5 for image in images])
    plt.imshow(
        images_inverted[0].permute(1, 2, 0).detach().numpy(),
        cmap="Greys_r",
    )
    plt.show()
    image_combined = torchvision.utils.make_grid(images_inverted)
    plt.imshow(image_combined.permute(1, 2, 0), vmin=0, vmax=1, cmap="Greys_r")
    plt.show()


def test3():
    data_loader, _ = get_MNIST_dataloader()
    imgs, _ = next(iter(data_loader))
    print(imgs[0])
    print(imgs.shape)

    img = torchvision.utils.make_grid(imgs)
    print(img.shape)

    plt.imshow(img.permute(1, 2, 0))
    plt.show()


def test4():
    shape = (1, 16, 16)
    # image = torch.rand(shape, dtype=torch.float, requires_grad=False)
    image0 = torch.zeros(shape, dtype=torch.float, requires_grad=False)
    image1 = torch.ones(shape, dtype=torch.float, requires_grad=False)
    imagem1 = torch.ones(shape, dtype=torch.float, requires_grad=False) * -1

    image = torchvision.utils.make_grid(
        [image0 / 2 + 0.5, image1 / 2 + 0.5, imagem1 / 2 + 0.5]
    )

    plt.imshow(image.permute(1, 2, 0), vmin=0, vmax=1, cmap="Greys")
    plt.show()


def test5():
    train, test = get_MNIST_dataloader()
    train_0 = next(iter(train))
    test_0 = next(iter(test))
    print(test_0[1].data)
    logger_regular.debug("hi")


def test6():
    PATH_MODEL = f"model/resnet18_on_tissue_mnist_{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.pt"

    train_dataloader, test_dataloader, task, num_channels, num_classes = (
        get_MedMNIST_dataloader(data_flag="octmnist")
    )
    print(num_classes)
    print(task)

    model = torchvision.models.resnet18()
    model.conv1 = torch.nn.Conv2d(
        num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = torch.nn.Sequential(torch.nn.Linear(512, num_classes))
    optimizer = torch.optim.Adam(model.parameters())

    model_trainer = ModelTrainer(
        model=model.to(device=DEVICE),
        optimizer=optimizer,
        criterion=(
            torch.nn.BCEWithLogitsLoss()
            if task == "multi-label, binary-class"
            else torch.nn.CrossEntropyLoss()
        ),
        device=DEVICE,
    )

    model_trainer.iterate_train(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        log_label="OCTMnist",
    )
    model_trainer.save(path=PATH_MODEL)


def test7():
    NOW = now()
    PATH_MODEL = f"model/test_{NOW}.pt"
    NUM_CHANNELS = 1
    NUM_CLASSES = 10

    train_dataloader, test_dataloader = get_MNIST_dataloader()

    model = torchvision.models.resnet18()
    model.conv1 = torch.nn.Conv2d(
        NUM_CHANNELS, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = torch.nn.Sequential(torch.nn.Linear(512, NUM_CLASSES))
    optimizer = torch.optim.Adam(model.parameters())

    model_trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        device=DEVICE,
    )
    model_trainer.iterate_train(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        log_label="MNIST",
    )
    model_trainer.save(path=PATH_MODEL)

    model_trainer.load(path=PATH_MODEL)
    model_trainer.iterate_train(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        log_label="MNIST",
    )


def test8():
    train_dataloader, test_dataloader = get_MNIST_dataloader()
    labels = torch.Tensor([data[1] for data in train_dataloader.dataset])
    accuracy = multiclass_accuracy(labels, labels, num_classes=100)
    print(accuracy.item())


def test9():
    logger_regular.debug("debug")
    logger_regular.info("info")
    for i in range(10):
        logger_overwrite.debug(i)
    logger_overwrite.info("o info")
    logger_regular.info("hello")


def test10():
    train_dataset, _ = get_MNIST_dataset()
    for i in range(len(train_dataset)):
        print(train_dataset[i])
        return
