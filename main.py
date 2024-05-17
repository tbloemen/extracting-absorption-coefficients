import os.path
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
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
    GAMMA,
    OUT_FEATURES,
    POWER,
)
from loss import DareGramLoss
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

    print_datashape(train_dataloader, NnStage.SOURCE)
    print_datashape(test_dataloader, NnStage.TARGET)
    return {NnStage.SOURCE: train_dataloader, NnStage.TARGET: test_dataloader}


def get_device() -> str:
    # if torch.cuda.is_available():
    #     return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelRegression(nn.Module):
    def __init__(self):
        super(ModelRegression, self).__init__()
        self.audio_features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=33),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=17),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=9),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        # L_x, L_y, L_z, (x_speaker, y_speaker), (x_microphone, y_microphone)
        num_numerical_features = 3 + 2 + 2

        self.numeric_features = nn.Sequential(
            nn.Linear(in_features=num_numerical_features, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2000),
            nn.ReLU(),
        )

        self.combined_features = nn.Sequential(
            nn.Linear(in_features=2000 * 2, out_features=1000), nn.ReLU()
        )

        # TODO: change out_features to be the actual number of parameters

        foo = nn.Linear(2000, OUT_FEATURES)
        foo.weight.data.normal_(0, 0.01)
        foo.bias.data.fill_(0.0)

        self.classifier_layer = nn.Sequential(foo, nn.Sigmoid())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.audio_features(x)
        x = x.view(-1, 2000)
        y = self.numeric_features(x)
        z = torch.cat((x, y), 1)
        features = self.combined_features(z)
        outC = self.classifier_layer(features)
        return outC, features


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
    dataloaders: dict[NnStage.value, DataLoader],
    model: NeuralNetwork,
    loss_fn: CrossEntropyLoss,
    optimizer: Optimizer,
    device: str,
    epoch: int,
):
    model.train()

    daregram_loss_fn = DareGramLoss()

    source_iterator = iter(dataloaders[NnStage.SOURCE])

    with tqdm(dataloaders[NnStage.TARGET], unit="batch") as tepoch:
        X_target: Tensor
        Y_target: Tensor
        X_source: Tensor
        Y_source: Tensor
        total_daregram, total_total, total_classifier = 0, 0, 0
        for i, (X_target, Y_target) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch} - Training")

            try:
                X_source, Y_source = next(source_iterator)
            except StopIteration:
                source_iterator = iter(dataloaders[NnStage.SOURCE])
                X_source, Y_source = next(source_iterator)

            X_source, Y_source = X_source.to(device), Y_source.to(device)
            X_target, Y_target = X_target.to(device), Y_target.to(device)

            # Compute prediction error
            _, feature_t = model(X_target)
            outC_s, feature_s = model(X_source)
            daregram_loss: Tensor = daregram_loss_fn(feature_t, feature_s)
            classifier_loss: Tensor = loss_fn(outC_s, Y_source)
            total_loss = daregram_loss + classifier_loss

            # Backpropagation
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_daregram += daregram_loss.item()
            total_classifier += classifier_loss.item()
            total_loss += total_loss.item()

            tepoch.set_postfix(
                total_loss=total_total / (i + 1),
                daregram_loss=total_daregram / (i + 1),
                classifier_loss=total_classifier / (i + 1),
            )


def test(
    dataloaders: dict[NnStage, DataLoader],
    model: NeuralNetwork,
    device: str,
    epoch: int,
) -> float:

    size = 0
    for dataloader in dataloaders.values():
        size += len(dataloader.dataset)  # type: ignore
    model.eval()
    mse, mae = [], []
    with torch.no_grad():
        with tqdm(dataloaders[NnStage.TARGET], unit="batch") as tepoch:
            X_target: Tensor
            Y_target: Tensor
            for i, (X_target, Y_target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch} - Testing")

                X_target, Y_target = X_target.to(device), Y_target.to(device)

                outC, feature_t = model(X_target)
                mse.append(MSELoss()(outC, Y_target))
                mae.append(L1Loss()(outC, Y_target))

                # Not nice to print(), maybe file writing?
                for feature in range(OUT_FEATURES):
                    mse.append(MSELoss()(outC[:, feature], Y_target[:, feature]))
                    mae.append(L1Loss()(outC[:, feature], Y_target[:, feature]))
                tepoch.set_postfix(MSE=mse[0], MAE=mae[0])
    return 0


def increase_lr(optimizer: Optimizer, epoch: int, param_lr: list[float]) -> Optimizer:
    lr = LEARNING_RATE * (1 + GAMMA * epoch) ** (-POWER)
    i = 0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_lr[i]
    optimizer.zero_grad()
    return optimizer


def run_epochs(
    dataloaders: dict[NnStage, DataLoader],
    model: NeuralNetwork,
    device: str,
    optimizer: Optimizer,
) -> None:
    epoch, accuracy, accuracy_new, low_delta = 0, 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    while epoch < EPOCHS and low_delta < SAME_DELTA_EPOCHS:
        epoch += 1
        optimizer = increase_lr(optimizer=optimizer, epoch=epoch, param_lr=param_lr)
        train(
            dataloaders=dataloaders,
            model=model,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
        )
        accuracy_new = test(
            dataloaders=dataloaders,
            model=model,
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

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    run_epochs(
        dataloaders=dataloaders,
        model=model,
        device=device,
        optimizer=optimizer,
    )

    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    torch.save(model.state_dict(), MODEL_PATH + MODEL_NAME)


if __name__ == "__main__":
    main()
