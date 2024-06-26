"""
Houses all the files which print something, either to a figure or to the standard output.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import seaborn as sns
import numpy as np
import pandas as pd
import pyroomacoustics as pra
import torchaudio
from matplotlib.axes import Axes
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data_gen import get_dataloaders, RirAndRoomLayoutDataset
from tol_colors import tol_cmap, tol_cset

from constants import (
    NnStage,
    NUM_SAMPLES,
    SAMPLERATE,
    CENTER_FREQUENCIES,
    OUTPUT_DIR,
    AGG_FILE,
    DataHeaders,
    LABEL_FILE_REAL,
    LABEL_FILE_SIMULATED,
    LABEL_FILE_VALID,
    RIR_DIR_VALID,
    BATCH_SIZE,
)
from estimator import (
    FloorEstimator,
    WallEstimator,
    CeilingEstimator,
    ReflectiveEstimator,
    AbsCoefEstimator,
    AbsCoefEstimatorAtFreq,
)

# This renders the figures with the LaTeX font instead of the regular matplotlib font
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.pyplot.rcParams["svg.fonttype"] = "none"
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")
cmap = tol_cset("bright")


def print_datashape(dataloader: DataLoader, nn_stage: NnStage) -> None:
    """
    Prints the shape of the dataloader.
    :param dataloader: The dataloader of which you want to know the shape.
    :param nn_stage: The type of dataloader: training, testing or validation.
    :return: None.
    """
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
    """
    Prints the model layout.
    :param model: the model to be printed.
    """
    print("--------------------")
    print("This neural network model:")
    print(model)
    print("--------------------")


def print_done(epoch: int, error: float) -> None:
    """
    Prints a message when the model is done with training.
    :param epoch: The final amount of epochs it took to complete the training.
    :param error: The final error of the model.
    """
    print(f"Done! It took {epoch} epochs to obtain an error of {error}")


def print_model_found(model_str: str) -> None:
    """
    Prints that the model is found.
    :param model_str: the model file found, as a string.
    """
    print(f"Found the following model: {model_str}.\n" f"This model will be loaded in.")


def print_num_successful_data_points(df: pd.DataFrame) -> None:
    """
    Prints how many data points were successfully extracted.
    :param df: the DataFrame of the extracted data.
    """
    print(f"Successfully extracted {df.shape[0]} data points.")


def print_rir(rirs: list[Tensor | np.ndarray]) -> None:
    """
    Helper function for print_rirs.
    Prints an example rir of each dataset.
    :param rirs: the rirs to be printed. Can be Tensors or numpy arrays.
    """
    fig, axs = plt.subplots(2, 3, figsize=(14, 5))
    fig.tight_layout()
    axs[0, 0].set_title("dEchorate")
    axs[0, 1].set_title("pyroomacoustics")
    axs[0, 2].set_title("inHouse")
    for i, rir in enumerate(rirs):
        if type(rir) is Tensor:
            rir = rir.numpy()
        decibels = 20 * np.log10(np.abs(rir))
        max_decibels = np.max(decibels)
        decibels = decibels - max_decibels
        xscale = np.arange(rir.shape[0]) / SAMPLERATE
        axs[0, i].plot(xscale, decibels, color=cmap[0])
        axs[0, i].grid(axis="y")

        frequency_domain = np.fft.fft(rir)
        k = np.arange(len(frequency_domain))
        T = len(frequency_domain) / SAMPLERATE
        frqLabel = k / T

        axs[1, i].plot(
            frqLabel[: int(len(frequency_domain) / 2)],
            np.abs(frequency_domain[: int(len(frequency_domain) / 2)]),
            color=cmap[1],
        )
        if i == 0:
            axs[0, i].set_ylabel("RIR [db]")
            axs[1, i].set_ylabel("RIR [magnitude]")

        axs[0, i].set_xlabel("Time [seconds]")
        axs[1, i].set_xlabel("Frequency [Hz]")

        axs[1, i].set_xscale("log")
        axs[1, i].set_xticks(
            CENTER_FREQUENCIES, labels=freqs_as_str(), rotation=45, ha="right"
        )
        axs[1, i].set_xlim(31.25 / np.sqrt(2), 16000 * np.sqrt(2))
        axs[1, i].minorticks_off()
        axs[1, i].grid(axis="y")

    plt.savefig(Path.cwd() / OUTPUT_DIR / "example_rirs.svg", bbox_inches="tight")


def freqs_as_str() -> list[str]:
    """
    Returns a list of center frequencies, formatted as strings.
    :return: A list of the center frequencies as strings.
    """
    labels: list[str] = []
    for freq in CENTER_FREQUENCIES:
        res_freq = freq
        if freq % 1 == 0:
            res_freq = int(freq)
        if res_freq % 1000 == 0:
            res_freq = f"{int(res_freq / 1000)}k"
        labels.append(str(res_freq))
    return labels


def draw_absorption_profiles() -> None:
    """
    Draws the profiles for the absorption coefficients, estimated with the material database.
    """
    axs: Axes
    fig, axs = plt.subplots(1, 4, figsize=(14, 3))
    axs[0].set_title("Reflective surface")
    axs[1].set_title("Floor profile")
    axs[2].set_title("Wall profile")
    axs[3].set_title("Ceiling profile")

    freq_start = CENTER_FREQUENCIES[0] / np.sqrt(2)
    freq_end = CENTER_FREQUENCIES[-1] * np.sqrt(2)
    freqs = np.linspace(start=freq_start, stop=freq_end, num=int(freq_end - freq_start))

    for i in range(4):

        estimator = ReflectiveEstimator()
        if i == 1:
            estimator = FloorEstimator()
        if i == 2:
            estimator = WallEstimator()
        if i == 3:
            estimator = CeilingEstimator()

        uppers = np.array(
            [estimator.get_estimator_at_freq(x).upper_bound for x in freqs]
        )
        lowers = np.array(
            [estimator.get_estimator_at_freq(x).lower_bound for x in freqs]
        )
        means = np.array([estimator.get_estimator_at_freq(x).mean for x in freqs])
        std_devs = np.array([estimator.get_estimator_at_freq(x).stdev for x in freqs])

        axs[i].plot(freqs, uppers, label="Upper", color=cmap[0])
        axs[i].plot(freqs, means, label="Mean", color=cmap[1])
        axs[i].plot(freqs, lowers, label="Lower", color=cmap[2])

        axs[i].fill_between(
            freqs, means + std_devs, means - std_devs, alpha=0.2, color=cmap[1]
        )

    labels = freqs_as_str()

    for i, ax in enumerate(axs.flat):
        ax.set(xlabel="Frequency [Hz]", ylabel=r"absorption coefficient")
        ax.set_xscale("log")
        ax.set_xticks(CENTER_FREQUENCIES, labels=labels, rotation=45, ha="right")
        ax.minorticks_off()
        ax.grid(axis="y")
        ax.set_ylim([0, 1.1])
        if i == 0:
            ax.legend()
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(
        Path.cwd() / OUTPUT_DIR / "simulated_absorption_profiles.svg",
        bbox_inches="tight",
    )


def abs_coef_dist(real: bool) -> None:
    """
    Draws a single distribution of absorption coefficients
    :param real: If true, draws the absorption coefficient distribution of the real (dEchorate) dataset,
    otherwise draws the absorption coefficient of the simulated dataset.
    """
    df = pd.read_csv(Path.cwd() / LABEL_FILE_SIMULATED)
    if real:
        df = pd.read_csv(Path.cwd() / LABEL_FILE_REAL)

    grouped = df[
        [
            f"{DataHeaders.ABS_COEF.value}_{center_freq}Hz"
            for center_freq in CENTER_FREQUENCIES
        ]
    ]

    labels = freqs_as_str()

    fig, ax = plt.subplots()
    ax.grid(axis="y")
    ax.boxplot(grouped, tick_labels=labels)
    state = "real" if real else "simulated"
    ax.set_title(f"Absorption Coefficient Distribution - {state.capitalize()} data")

    plt.savefig(
        Path.cwd() / OUTPUT_DIR / f"absorb_coef_dist_{state}.svg", bbox_inches="tight"
    )


def compare_abs_coef_dists() -> None:
    """
    Compares all absorption coefficient distributions of all datasets against each other.
    The data is plotted as a boxplot per frequency band.
    """
    sim_df = pd.read_csv(Path.cwd() / LABEL_FILE_SIMULATED)
    real_df = pd.read_csv(Path.cwd() / LABEL_FILE_REAL)
    valid_df = pd.read_csv(Path.cwd() / LABEL_FILE_VALID)

    abs_coef_headers = [
        f"{DataHeaders.ABS_COEF.value}_{center_freq}Hz"
        for center_freq in CENTER_FREQUENCIES
    ]

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Absorption Coefficient")
    ax.grid(axis="y")

    width = 0.17

    real_bp = ax.boxplot(
        real_df[abs_coef_headers],
        positions=np.arange(len(CENTER_FREQUENCIES)) - 0.25,
        widths=width,
        patch_artist=True,
    )

    sim_bp = ax.boxplot(
        sim_df[abs_coef_headers],
        positions=np.arange(len(CENTER_FREQUENCIES)),
        widths=width,
        patch_artist=True,
    )

    valid_bp = ax.boxplot(
        valid_df[abs_coef_headers],
        positions=np.arange(len(CENTER_FREQUENCIES)) + 0.25,
        widths=width,
        patch_artist=True,
    )

    for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(real_bp[element], color="k")
        plt.setp(sim_bp[element], color="k")
        plt.setp(valid_bp[element], color="k")

    for patch in real_bp["boxes"]:
        patch.set_facecolor(cmap[0])

    for patch in sim_bp["boxes"]:
        patch.set_facecolor(cmap[1])

    for patch in valid_bp["boxes"]:
        patch.set_facecolor(cmap[2])

    ax.set_title("Absorption Coefficient Distributions Compared")
    ax.legend(
        [real_bp["boxes"][0], sim_bp["boxes"][0], valid_bp["boxes"][0]],
        ["dEchorate", "pyroomacoustics", "inHouse"],
    )
    ax.set_xlim(-0.55, len(CENTER_FREQUENCIES) - 1 + 0.55)
    ax.set_xticks(np.arange(len(CENTER_FREQUENCIES)))
    ax.set_xticklabels(freqs_as_str())
    plt.savefig(Path.cwd() / OUTPUT_DIR / "both.svg", bbox_inches="tight")


def compare_abs_coef_dists_line() -> None:
    """
    Compares the distributions of absorption coefficients against each other.
    The data is plotted as lines with error bands, one standard deviation above and below each line.
    """
    sim_df = pd.read_csv(Path.cwd() / LABEL_FILE_SIMULATED)
    real_df = pd.read_csv(Path.cwd() / LABEL_FILE_REAL)
    valid_df = pd.read_csv(Path.cwd() / LABEL_FILE_VALID)

    abs_coef_headers = [
        f"{DataHeaders.ABS_COEF.value}_{center_freq}Hz"
        for center_freq in CENTER_FREQUENCIES
    ]

    fig, ax = plt.subplots(figsize=(5, 4))

    freq_start = CENTER_FREQUENCIES[0] / np.sqrt(2)
    freq_end = CENTER_FREQUENCIES[-1] * np.sqrt(2)
    freqs = np.linspace(start=freq_start, stop=freq_end, num=int(freq_end - freq_start))
    labels = ["dEchorate", "pyroomacoustics", "inHouse"]

    for i, df in enumerate(
        [
            real_df[abs_coef_headers],
            sim_df[abs_coef_headers],
            valid_df[abs_coef_headers],
        ]
    ):
        means = df.mean(axis=0).values
        std_devs = df.std(axis=0).values

        est_at_freqs = []

        for mean, std_dev, center_freq in zip(means, std_devs, CENTER_FREQUENCIES):
            est_at_freqs.append(
                AbsCoefEstimatorAtFreq(frequency=center_freq, mean=mean, stdev=std_dev)
            )

        estimator = AbsCoefEstimator(freq_dep_estimators=est_at_freqs)

        means = np.array([estimator.get_estimator_at_freq(x).mean for x in freqs])
        std_devs = np.array([estimator.get_estimator_at_freq(x).stdev for x in freqs])

        ax.plot(freqs, means, color=cmap[i], label=labels[i])
        ax.fill_between(
            freqs, means + std_devs, means - std_devs, alpha=0.2, color=cmap[i]
        )

    ax.set_title("Absorption Coefficient Distributions Compared")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Absorption Coefficient")
    ax.legend()
    ax.set_xscale("log")
    ax.set_xticks(CENTER_FREQUENCIES, labels=freqs_as_str(), rotation=45, ha="right")
    ax.minorticks_off()
    ax.grid(axis="y")
    plt.savefig(Path.cwd() / OUTPUT_DIR / "abs_coef_dist.svg", bbox_inches="tight")


def _get_t60s(dataloaders: list[DataLoader], labels: list[str]) -> None:
    all_t60s = []
    for dataloader in dataloaders:
        t60s = []
        for x, _ in dataloader:
            rirs = x[:, :NUM_SAMPLES].reshape((-1, NUM_SAMPLES))
            for rir in rirs:
                rt30 = 2 * pra.measure_rt60(rir, fs=SAMPLERATE, decay_db=30)
                t60s.append(rt30)
        all_t60s.append(t60s)

    plt.figure(figsize=(4, 3))
    plt.grid(visible=True, axis="both")
    bins = np.linspace(0, 1, 200)
    for i, t60s in enumerate(all_t60s):
        sns.histplot(
            t60s,
            bins=bins,
            alpha=1 / len(all_t60s),
            label=labels[i],
            color=cmap[i],
            stat="probability",
            element="step",
        )
    plt.ylabel(r"Relative occurrence of $T_{30}$ (%)")
    plt.xlabel("Time (seconds)")

    plt.legend()
    plt.savefig(Path.cwd() / OUTPUT_DIR / "t60s.svg", bbox_inches="tight")


def aggregate_heatmap_real() -> None:
    """
    Creates a heatmap of the absorption coefficient per room configuration in the dEchorate dataset.
    :return:
    """
    agg = pd.read_csv(Path.cwd() / AGG_FILE)
    room_strings = agg[DataHeaders.ROOM_ID.value].astype(str)
    config_to_number = {
        "0": 1,
        "1": 9,
        "10": 8,
        "100": 7,
        "1000": 6,
        "10000": 10,
        "11000": 2,
        "11100": 3,
        "11110": 4,
        "11111": 5,
        "20002": 11,
    }
    rooms_uniq_str = list(dict.fromkeys(room_strings))
    rooms = np.array(list(map(lambda x: config_to_number[x], rooms_uniq_str)))
    print(rooms)
    agg = agg.to_numpy()

    sorted_rooms = np.sort(rooms)

    # x axis: rooms
    # y axis: frequencies

    abs_coefs = agg[:, 1:].reshape((len(rooms), len(CENTER_FREQUENCIES)))
    abs_coefs = abs_coefs[rooms - 1, :]

    fig, ax = plt.subplots()
    ax.imshow(abs_coefs.T, cmap=tol_cmap("Sunset"))

    ax.set_xlabel("Room number")
    ax.set_ylabel("Frequency (Hz)")

    ax.set_xticks(np.arange(len(rooms)), labels=sorted_rooms)
    ax.set_yticks(np.arange(len(CENTER_FREQUENCIES)), labels=freqs_as_str())

    plt.setp(ax.get_xticklabels())

    for room in range(len(rooms)):
        for center_freq in range(len(CENTER_FREQUENCIES)):
            ax.text(
                room,
                center_freq,
                abs_coefs[room, center_freq].round(decimals=3),
                ha="center",
                va="center",
                color="w",
            )
    ax.set_title("Ground truth absorption coefficients")
    fig.tight_layout()
    plt.savefig(Path.cwd() / OUTPUT_DIR / "ground_truth.svg", bbox_inches="tight")


def results(stage: NnStage = None) -> None:
    """
    Prints the result of the model. This function presumes that the model has already been tested,
    as it relies on output files which are made by those functions.
    :param stage: If the stage is validation, then the corresponding files will be loaded for the final plot,
    otherwise the results of the dEchorate dataset will be plotted.
    """
    label_file = LABEL_FILE_REAL
    suffix = "real"
    dataset_name = "dEchorate"
    error_file = "error.csv"
    if stage == NnStage.VALIDATION:
        suffix = "validation"
        label_file = LABEL_FILE_VALID
        dataset_name = "inHouse"
        error_file = "error_validation.csv"
    df = pd.read_csv(Path.cwd() / label_file)

    error_headers = [
        f"{DataHeaders.ERROR.value}_{center_freq}Hz"
        for center_freq in CENTER_FREQUENCIES
    ]
    for center_freq in CENTER_FREQUENCIES:
        df[f"{DataHeaders.ERROR.value}_{center_freq}Hz"] = np.abs(
            df[f"{DataHeaders.EYRING.value}_{center_freq}Hz"]
            - df[f"{DataHeaders.ABS_COEF.value}_{center_freq}Hz"]
        )

    model_df = pd.read_csv(Path.cwd() / OUTPUT_DIR / error_file)
    model_df.drop(columns=model_df.columns[0], axis=1, inplace=True)
    model_df.drop(index=model_df.index[0], axis=0, inplace=True)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_ylabel("Absolute error")
    ax.set_xlabel("Frequency (Hz)")
    ax.grid(axis="y")

    width = 0.25

    eyring_bp = ax.boxplot(
        df[error_headers],
        positions=np.arange(len(CENTER_FREQUENCIES)) - 0.15,
        widths=width,
        patch_artist=True,
    )

    model_bp = ax.boxplot(
        model_df,
        positions=np.arange(len(CENTER_FREQUENCIES)) + 0.15,
        widths=width,
        patch_artist=True,
    )

    for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(eyring_bp[element], color="k")
        plt.setp(model_bp[element], color="k")

    for patch in eyring_bp["boxes"]:
        patch.set_facecolor(cmap[0])

    for patch in model_bp["boxes"]:
        patch.set_facecolor(cmap[1])

    ax.set_title(f"Results - {dataset_name}")
    ax.legend([eyring_bp["boxes"][0], model_bp["boxes"][0]], ["Eyring", "ML model"])
    ax.set_xlim(-0.55, len(CENTER_FREQUENCIES) - 1 + 0.55)
    if stage == NnStage.VALIDATION:
        ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(CENTER_FREQUENCIES)))
    ax.set_xticklabels(freqs_as_str())

    plt.savefig(Path.cwd() / OUTPUT_DIR / f"results_{suffix}.svg", bbox_inches="tight")


def print_rirs() -> None:
    """
    Prints some example rirs. The rirs are printed in dB against time, and also magnitude against frequency.
    """
    real_rir, _ = torchaudio.load(
        Path.cwd() / "data/rirs/real/rir_real_dEchorate_room000000_source7_mic0.wav"
    )

    sim_rir, _ = torchaudio.load(Path.cwd() / "data/rirs/simulated/rir_sim_3.wav")

    valid_rir, _ = torchaudio.load(
        Path.cwd() / "data/rirs/validation/rir_validation_inHouse_source1_mic1.wav"
    )

    rirs = [real_rir.squeeze(), sim_rir.squeeze(), valid_rir.squeeze()]
    print_rir(rirs)


def get_t60s() -> None:
    """
    Prints the distribution of decay times (t60s) of all datasets.
    :return:
    """
    dataloaders = list(get_dataloaders().values())
    validation_data = RirAndRoomLayoutDataset(
        label_file=Path.cwd() / LABEL_FILE_VALID, rir_dir=Path.cwd() / RIR_DIR_VALID
    )

    validation_dataloader = DataLoader(
        validation_data, batch_size=BATCH_SIZE, shuffle=True
    )
    dataloaders.append(validation_dataloader)

    labels = ["pyroomacoustics", "dEchorate", "inHouse"]

    _get_t60s(dataloaders, labels)


def main() -> None:
    """
    Main function. Is used to select what needs to be printed.
    """
    compare_abs_coef_dists()
    aggregate_heatmap_real()
    draw_absorption_profiles()
    get_t60s()
    compare_abs_coef_dists_line()


if __name__ == "__main__":
    main()
