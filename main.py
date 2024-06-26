"""
This file houses the main functionality of the machine learning model.
"""

import datetime
import logging
import os.path
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import MSELoss, L1Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import printing
from constants import (
    NnStage,
    EPOCHS,
    LOW_DELTA,
    LEARNING_RATE,
    SAME_DELTA_EPOCHS,
    MODEL_PATH,
    MODEL_NAME,
    GAMMA,
    POWER,
    DEVICE,
    NUM_SAMPLES,
    OUTPUT_DIR,
    CENTER_FREQUENCIES,
)
from data_gen import get_dataloaders
from loss import dare_gram_loss
from printing import print_model, print_done, print_model_found

logging.basicConfig(filename="rp.log", filemode="w")
logging.info(f"Logging has started at {datetime.datetime.now()}")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.audio_features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=33),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=17),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(4),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=9),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(4),
            nn.Linear(in_features=746, out_features=125),
        )

        # L_x, L_y, L_z, (x_speaker, y_speaker, z_speaker), (x_microphone, y_microphone, z_microphone)
        num_numerical_features = 9

        self.numeric_features = nn.Sequential(
            nn.Linear(in_features=num_numerical_features, out_features=64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
        )

        self.combined_features = nn.Sequential(
            nn.Linear(in_features=2500, out_features=500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
        )

        self.classifier_layer = nn.Sequential(
            nn.Linear(500, len(CENTER_FREQUENCIES)),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Feeds the data forward in the model.
        :param x: The input data from the dataloader.
        :return: A tuple of the estimated 10 absorption coefficients, and the feature space for the DARE-GRAM loss.
        """
        rir, numeric_data = (
            x[:, :NUM_SAMPLES].reshape((-1, 1, NUM_SAMPLES)),
            x[:, NUM_SAMPLES:],
        )
        x = self.audio_features(rir)
        x = x.reshape(rir.shape[0], -1)
        y = self.numeric_features(numeric_data)
        z = torch.cat((x, y), dim=1)
        features = self.combined_features(z)
        outC = self.classifier_layer(features)
        return outC.squeeze(), features.squeeze()


def train(
    dataloaders: dict[NnStage, DataLoader],
    model: NeuralNetwork,
    optimizer: Optimizer,
    epoch: int,
):
    """
    Trains the model.
    :param dataloaders: The dataloaders of the real and simulated dataset.
    :param model: The neural network model to be trained.
    :param optimizer: The optimizer which optimizes the model.
    :param epoch: The epoch (iteration) the model is on.
    """
    model.train()

    target_iterator = iter(dataloaders[NnStage.TARGET])

    with tqdm(dataloaders[NnStage.SOURCE], unit="batch") as tepoch:
        X_target: Tensor
        Y_target: Tensor
        X_source: Tensor
        Y_source: Tensor
        total_daregram, total_total, total_classifier_t, total_classifier_s = 0, 0, 0, 0
        for i, (X_source, Y_source) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch} - Training")

            try:
                X_target, Y_target = next(target_iterator)
            except StopIteration:
                target_iterator = iter(dataloaders[NnStage.TARGET])
                X_target, Y_target = next(target_iterator)

            if X_source.shape != X_target.shape:
                continue

            X_source, Y_source = X_source.to(DEVICE), Y_source.to(DEVICE)
            X_target, Y_target = X_target.to(DEVICE), Y_target.to(DEVICE)

            # Compute prediction error
            outC_t, feature_t = model(X_target)
            outC_s, feature_s = model(X_source)
            daregram_loss: Tensor = dare_gram_loss(feature_t, feature_s)
            classifier_loss_t: Tensor = nn.MSELoss()(outC_t, Y_target)
            classifier_loss_s: Tensor = nn.MSELoss()(outC_s, Y_source)

            total_loss = daregram_loss + classifier_loss_s

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            total_daregram += daregram_loss.item()
            total_classifier_t += classifier_loss_t.item()
            total_classifier_s += classifier_loss_s.item()
            total_total += total_loss.item()

            tepoch.set_postfix(
                total_loss=total_total / (i + 1),
                daregram_loss=total_daregram / (i + 1),
                classifier_s_loss=total_classifier_s / (i + 1),
                classifier_t_loss=total_classifier_t / (i + 1),
            )
    optimizer.zero_grad()


def test(
    model: NeuralNetwork, epoch: int, valid_loader: DataLoader = None
) -> Tuple[float, float]:

    model.eval()
    error_dfs = []
    target_dataloader_with_freq = get_dataloaders()[NnStage.TARGET]
    error_file = "error.csv"

    if valid_loader is not None:
        target_dataloader_with_freq = valid_loader
        error_file = "error_validation.csv"
    with torch.no_grad():
        with tqdm(target_dataloader_with_freq, unit="batch") as tepoch:
            X_target: Tensor
            Y_target: Tensor
            mse_loss_total, mae_loss_total = 0, 0
            for i, (X_target, Y_target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch} - Testing")

                X_target, Y_target = X_target.to(DEVICE), Y_target.to(DEVICE)

                outC, _ = model(X_target)
                mse_loss = MSELoss()(outC, Y_target)
                mae_loss = L1Loss()(outC, Y_target)

                mse_loss_total += mse_loss.item()
                mae_loss_total += mae_loss.item()

                tepoch.set_postfix(
                    MSE=mse_loss_total / (i + 1), MAE=mae_loss_total / (i + 1)
                )

                error = torch.abs(outC - Y_target)

                error_dfs.append(pd.DataFrame(error.cpu()))
    error_df = pd.concat(error_dfs)
    error_df.to_csv(Path.cwd() / OUTPUT_DIR / error_file)
    mae = error_df.mean(axis="rows")
    mse = error_df.pow(2).mean(axis="rows")
    mae.to_csv(Path.cwd() / OUTPUT_DIR / f"mae/mae_{epoch}.csv")
    mse.to_csv(Path.cwd() / OUTPUT_DIR / f"mse/mse_{epoch}.csv")
    num_batches = len(target_dataloader_with_freq)
    return mse_loss_total / num_batches, mae_loss_total / num_batches


def inv_lr_scheduler(
    optimizer: Optimizer, epoch: int, param_lr: list[float]
) -> Optimizer:
    """
    Decreases the learning rate over time.
    :param optimizer: The optimizer of which the learning rate needs to be adjusted.
    :param epoch: The current epoch number.
    :param param_lr: The parameters regarding the learning rate of the optimizer.
    :return: The updated optimizer.
    """
    lr = LEARNING_RATE * (1 + GAMMA * epoch) ** (-POWER)
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr * param_lr[i]
    optimizer.zero_grad()
    return optimizer


def run_epochs(
    dataloaders: dict[NnStage, DataLoader],
    model: NeuralNetwork,
    optimizer: Optimizer,
) -> None:
    """
    Runs all epochs.
    :param dataloaders: The data loaders for the real and simulated dataset.
    :param model: The machine learning model which needs to be trained and tested.
    :param optimizer: The optimizer to optimize the model.
    """
    epoch, mse, mse_new, low_delta = 0, 0, 0, 0
    epoch_mse_list: list[dict[str, float | int]] = []

    param_lr_list = [param_group["lr"] for param_group in optimizer.param_groups]

    while epoch < EPOCHS and low_delta < SAME_DELTA_EPOCHS:
        epoch += 1
        optimizer = inv_lr_scheduler(
            optimizer=optimizer, epoch=epoch, param_lr=param_lr_list
        )
        train(
            dataloaders=dataloaders,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
        )
        if epoch % 1 == 0:
            save_model(model)
            mse_new, mae_new = test(
                model=model,
                epoch=epoch,
            )

            epoch_mse_dict = {"epoch": epoch, "mse": mse_new, "mae": mae_new}
            epoch_mse_list.append(epoch_mse_dict)

            if abs(mse_new - mse) > LOW_DELTA:
                low_delta = 0
                mse = mse_new
            else:
                low_delta += 1
    df = pd.DataFrame(epoch_mse_list)
    df.to_csv(Path.cwd() / "errors.csv")
    print_done(epoch, mse)


def save_model(model: NeuralNetwork):
    """Saves the model to file."""
    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    torch.save(model.state_dict(), MODEL_PATH + MODEL_NAME)


def get_results():
    """
    Gets the result of a trained network.
    """
    torch.cuda.empty_cache()
    model = NeuralNetwork().to(DEVICE)
    if os.path.isfile(MODEL_PATH + MODEL_NAME):
        model.load_state_dict(torch.load(MODEL_PATH + MODEL_NAME))
    print_model(model)

    test(model, 0)

    printing.results()


def main():
    """Prepares the data and runs the model."""
    dataloaders = get_dataloaders()
    printing.main()
    torch.cuda.empty_cache()

    model = NeuralNetwork().to(DEVICE)
    model_name = "my_model_mse.pth"
    if os.path.isfile(MODEL_PATH + model_name):
        print_model_found(MODEL_PATH + model_name)
        model.load_state_dict(torch.load(MODEL_PATH + model_name))
    print_model(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    run_epochs(
        dataloaders=dataloaders,
        model=model,
        optimizer=optimizer,
    )

    save_model(model)
    printing.results()


if __name__ == "__main__":
    get_results()
