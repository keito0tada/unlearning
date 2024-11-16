import torch
import time


class ModelTrainer:
    def __init__(
        self,
        model: torch.nn.modules.module.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int,
        log_interval=10,
        criterion=torch.nn.functional.nll_loss,
    ):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # move data to device
            data = data.to(self.device)
            target = target.to(self.device)

            # train
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # output states
            if batch_idx % log_interval == 0:
                print(
                    "\rEpoch: {} [{:6d}]\tLoss: {:.6f}".format(
                        epoch, batch_idx * len(data), loss.item()
                    ),
                    end="",
                )

    def test(
        self,
        test_loader: torch.utils.data.DataLoader,
        dname="Test set",
        printable=True,
        criterion=torch.nn.functional.nll_loss,
    ):
        self.model.eval()
        test_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                total += target.size()[0]
                test_loss += criterion(output, target).item()
                _, pred = torch.topk(output, 1, dim=1, largest=True, sorted=True)
                for i, t in enumerate(target):
                    if t in pred[i]:
                        correct += 1
        test_loss /= len(test_loader.dataset)
        if printable:
            print(
                "{}: Mean loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                    dname, test_loss, correct, total, 100.0 * correct / total
                )
            )
        return 1.0 * correct / total

    def iterate_train(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        training_epochs=10,
    ):
        deltas = []
        for _ in range(50):
            delta = {}
            for param_tensor in self.model.state_dict():
                if "weight" in param_tensor or "bias" in param_tensor:
                    delta[param_tensor] = 0
            deltas.append(delta)
        for epoch in range(1, training_epochs + 1):
            starttime = time.process_time()
            self.train(epoch=epoch, train_loader=train_loader)
            self.test(test_loader=test_loader, dname="All data")
            print(f"Time taken: {time.process_time() - starttime}")

    def save(self, path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"model loaded from {path}")
