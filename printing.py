import numpy as np
import pandas as pd
from torch import Tensor, nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from constants import NnStage, NUM_SAMPLES, RIR_DURATION, SAMPLERATE


def print_datashape(dataloader: DataLoader, nn_stage: NnStage) -> None:
    X: Tensor
    Y: Tensor
    for X, Y in dataloader:
        input_rir = X[:, :NUM_SAMPLES]
        print(f"Shape of {nn_stage.value} input RIR: {input_rir.shape}")
        input_numerical = X[:, NUM_SAMPLES:]
        print(
            f"Shape of {nn_stage.value} input numerical data: {input_numerical.shape}"
        )
        print(f"Shape of {nn_stage.value} output data: {Y.shape}")
        return


def print_model(model: nn.Module) -> None:
    print("--------------------")
    print("This neural network model:")
    print(model)
    print("--------------------")


def print_test_results(correct: float, test_loss: float) -> None:
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def print_done(epoch: int, accuracy: float) -> None:
    print(f"Done! It took {epoch} epochs to obtain an accuracy of {accuracy}")


def print_model_found(model_str: str) -> None:
    print(f"Found the following model: {model_str}.\n" f"This model will be loaded in.")


def print_num_succesful_data_points(df: pd.DataFrame) -> None:
    print(f"Successfully extracted {df.shape[0]} data points.")


def print_rir(rir: Tensor) -> None:
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_title("RIR in the time domain")
    ax1.plot(
        np.arange(rir.shape[0]) / SAMPLERATE,
        rir,
    )
    ax1.set(xlabel="Time [seconds]", ylabel="RIR [amplitude]")

    frequency_domain = np.fft.fft(rir)
    k = np.arange(len(frequency_domain))
    T = len(frequency_domain) / SAMPLERATE
    frqLabel = k / T

    ax2.set_title("RIR in the frequency domain")
    ax2.plot(
        frqLabel[: int(len(frequency_domain) / 2)],
        np.abs(frequency_domain[: int(len(frequency_domain) / 2)]),
    )
    ax2.set(xlabel=r"Frequency [Hz]$", ylabel="RIR [magnitude]")

    plt.show()
