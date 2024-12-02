from src.attack.model_inversion_attack import ModelInversionAttack
from src.utils.get_trained_model_trainer import get_MNIST_model_trainer
from src.utils.misc import now

DEVICE = "cuda:0"
NOW = now()


def test1():
    model_trainer = get_MNIST_model_trainer(f"model/resnet18_mnist_{NOW}.pt")
    model_inversion_attack = ModelInversionAttack(
        model=model_trainer.model, device=DEVICE, criterion=model_trainer.criterion
    )
