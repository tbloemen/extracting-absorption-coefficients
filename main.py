import os.path

import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

from constants import (
    BATCH_SIZE,
    NnStage,
    EPOCHS,
    LOW_DELTA,
    LEARNING_RATE,
    SAME_DELTA_EPOCHS,
    MODEL_PATH,
    MODEL_NAME,
)
from printing import print_datashape, print_model, print_done, print_model_found


def get_dataloaders() -> dict[NnStage, DataLoader]:
    # TODO change into actual data
    training_data = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=ToTensor()
    )

    # TODO change into actual data
    test_data = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    print_datashape(train_dataloader, NnStage.TRAINING)
    print_datashape(test_dataloader, NnStage.TEST)
    return {NnStage.TRAINING: train_dataloader, NnStage.TEST: test_dataloader}


def get_device() -> str:
    # if torch.cuda.is_available():
    #     return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        logits: Tensor = self.linear_relu_stack(x)
        return logits


def train(
    dataloader: DataLoader,
    model: NeuralNetwork,
    loss_fn: CrossEntropyLoss,
    optimizer: Optimizer,
    device: str,
    epoch: int,
):
    model.train()

    with tqdm(dataloader, unit="batch") as tepoch:
        X: Tensor
        Y: Tensor
        for X, Y in tepoch:
            tepoch.set_description(f"Epoch {epoch} - Training")

            X, Y = X.to(device), Y.to(device)

            # Compute prediction error
            pred: Tensor = model(X)
            loss: Tensor = loss_fn(pred, Y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(
    dataloader: DataLoader,
    model: NeuralNetwork,
    loss_fn: CrossEntropyLoss,
    device: str,
    epoch: int,
) -> float:
    size = len(dataloader.dataset)  # type: ignore
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            X: Tensor
            Y: Tensor
            for i, (X, Y) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch} - Testing")

                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, Y).item()
                correct += (pred.argmax(1) == Y).type(torch.float).sum().item()

                tepoch.set_postfix(
                    average_loss=test_loss / (i + 1),
                    accuracy=correct / min(size, (i + 1) * BATCH_SIZE) * 100.0,
                )
    return correct / size * 100.0


def run_epochs(
    dataloaders: dict[NnStage, DataLoader],
    model: NeuralNetwork,
    loss_fn: CrossEntropyLoss,
    device: str,
    optimizer: Optimizer,
) -> None:
    epoch, accuracy, accuracy_new, low_delta = 0, 0, 0, 0
    while epoch < EPOCHS and low_delta < SAME_DELTA_EPOCHS:
        epoch += 1
        train(
            dataloader=dataloaders[NnStage.TRAINING],
            model=model,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
        )
        accuracy_new = test(
            dataloader=dataloaders[NnStage.TEST],
            model=model,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
        )
        if abs(accuracy_new - accuracy) > LOW_DELTA:
            low_delta = 0
            accuracy = accuracy_new
        else:
            low_delta += 1

    print_done(epoch, accuracy_new)


def main():
    dataloaders = get_dataloaders()

    device = get_device()
    model = NeuralNetwork().to(device)
    if os.path.isfile(MODEL_PATH + MODEL_NAME):
        print_model_found(MODEL_PATH + MODEL_NAME)
        model.load_state_dict(torch.load(MODEL_PATH + MODEL_NAME))
    print_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    run_epochs(
        dataloaders=dataloaders,
        model=model,
        loss_fn=loss_fn,
        device=device,
        optimizer=optimizer,
    )
    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    torch.save(model.state_dict(), MODEL_PATH + MODEL_NAME)


if __name__ == "__main__":
    main()
