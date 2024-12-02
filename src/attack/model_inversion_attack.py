import torch
from scipy import ndimage

from src.log.logger import logger_regular, logger_overwrite


class ModelInversionAttack:
    def __init__(
        self,
        model: torch.nn.modules.module.Module,
        device: str,
        criterion=torch.nn.functional.nll_loss,
        shape: tuple = (1, 1, 28, 28),
    ):
        self.model = model
        self.criterion = criterion
        self.shape = shape
        self.device = device

    def invert(self, x: torch.Tensor, y: torch.Tensor, num_iters=5000, learning_rate=1):
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)

        loss = 1000000
        x_history = [x]

        for i in range(num_iters):
            self.model.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            x.retain_grad()
            loss.backward()
            x = x - learning_rate * x.grad

            if i % 100 == 0:
                logger_overwrite.info(f"Iteration: {i}, Loss: {loss}")

            # make an image clear
            if i % 500 == 0:
                x_median = ndimage.median_filter(x.cpu().detach(), size=2)
                x_gaussian = ndimage.gaussian_filter(x_median, sigma=2, truncate=-1 / 6)
                x_gaussian2 = ndimage.gaussian_filter(x_gaussian, sigma=1, truncate=-1)
                x = torch.from_numpy(x_gaussian + 80 * (x_gaussian - x_gaussian2)).to(
                    device=self.device
                )
                x.requires_grad = True
                x_history.append(x)

        return x, x_history

    def generate_images(
        self, target: int, num_examples=1, num_iters=8000, learning_rate=1, div=256
    ):
        images = []
        for i in range(num_examples):
            noise = torch.rand(self.shape, dtype=torch.float, requires_grad=False)
            noise *= 2
            noise -= 1
            # noise /= div
            # noise -= 1
            noise.requires_grad = True
            images.append(
                self.invert(
                    x=noise,
                    y=torch.tensor([target]),
                    num_iters=num_iters,
                    learning_rate=learning_rate,
                )
            )
        return images
