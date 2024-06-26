"""
The file where the real and simulated dataset are generated.
"""

import logging
import os
from pathlib import Path


import h5py
import numpy as np
import pandas as pd
import pyroomacoustics as pra
import torch
import torchaudio
from pyroomacoustics import ShoeBox
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from constants import (
    NnStage,
    BATCH_SIZE,
    DataHeaders,
    Surface,
    LABEL_FILE_SIMULATED,
    RIR_DIR_SIMULATED,
    LABEL_FILE_REAL,
    RIR_DIR_REAL,
    NUM_SIM_RIRS,
    SAMPLERATE,
    CENTER_FREQUENCIES,
    OUTPUT_DIR,
    AGG_FILE,
    NUM_SAMPLES,
    MAX_ORDER,
    RIR_DIR_VALID,
)
from estimator import (
    WallEstimator,
    CeilingEstimator,
    FloorEstimator,
    ReflectiveEstimator,
)
from preprocessing import preprocess_rir, rir_in_octave_bands


class RirAndRoomLayoutDataset(Dataset):
    def __init__(self, label_file: str, rir_dir: str):
        self.labels: pd.DataFrame = pd.read_csv(label_file)
        self.rir_dir: str = rir_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        rir_path = os.path.join(
            self.rir_dir,
            self.labels.loc[self.labels.index[idx], DataHeaders.RIR_PATH.value],
        )
        rir, _ = torchaudio.load(rir_path)

        numeric_data_headers = [
            DataHeaders.LENGTH_X,
            DataHeaders.LENGTH_Y,
            DataHeaders.LENGTH_Z,
            DataHeaders.MICROPHONE_X,
            DataHeaders.MICROPHONE_Y,
            DataHeaders.MICROPHONE_Z,
            DataHeaders.SPEAKER_X,
            DataHeaders.SPEAKER_Y,
            DataHeaders.SPEAKER_Z,
        ]

        numeric_data = [
            self.labels.loc[self.labels.index[idx], header.value]
            for header in numeric_data_headers
        ]

        numeric_tensor = Tensor(numeric_data)

        X = torch.cat((rir.squeeze(), numeric_tensor))

        absorption_coefs = [
            self.labels.loc[
                self.labels.index[idx], f"{DataHeaders.ABS_COEF.value}_{center_freq}Hz"
            ]
            for center_freq in CENTER_FREQUENCIES
        ]

        return X, Tensor(absorption_coefs)


def get_source_and_receiver_positions(
    room_dims: list[float],
) -> tuple[list[float], list[float]]:
    """
    Get source positions and receiver in a room while maintaining the proper distance.
    If this is not possible, throws a ValueError.
    :param room_dims: The room dimensions of the room to place the source and receiver in.
    :return: The locations of the source and receiver.
    """
    if len(room_dims) != 3:
        raise ValueError(
            f"The room dimensions were not of size 3, but of size {len(room_dims)}"
        )

    min_distance = 0.5

    window_dims = [None, None, None]
    for i, room_dim in enumerate(room_dims):

        window_dim = [
            min_distance,
            room_dim - min_distance,
        ]

        window_dims[i] = window_dim
        if window_dim[1] - window_dim[0] <= 0:
            raise ValueError(
                "The room dimensions have made a window that is too small. "
                + "Either decrease the minimum distance or generate new dimensions."
            )

    source_pos = [None, None, None]
    receiver_pos = [None, None, None]

    for i, window_dim in enumerate(window_dims):
        source_pos[i] = np.random.uniform(low=window_dim[0], high=window_dim[1])
        receiver_pos[i] = np.random.uniform(low=window_dim[0], high=window_dim[1])

    if np.linalg.norm(np.array(source_pos) - np.array(receiver_pos)) < 1:
        raise ValueError("The source and receiver were too close to each other.")

    return source_pos, receiver_pos


def get_surface_material(surface: Surface) -> dict:
    """
    Gets the surface material in the format used by pyroomacoustics.
    :param surface: Which surface to estimate the profile for.
    :return: A surface material dict.
    """
    res = []
    estimator = ReflectiveEstimator()
    if np.random.random() < 0.5:
        estimator = WallEstimator()
        if surface == Surface.CEILING:
            estimator = CeilingEstimator()
        if surface == Surface.FLOOR:
            estimator = FloorEstimator()

    for center_freq in CENTER_FREQUENCIES:
        res.append(estimator.estimate_abs_coef(frequency=center_freq))
    return {
        "description": f"{surface.name.capitalize()} estimator material",
        "coeffs": res,
        "center_freqs": CENTER_FREQUENCIES,
    }


def get_materials() -> tuple[dict, dict[Surface, list[float]]]:
    """
    Makes the materials for the pyroomacoustics Room creator. Also returns the coefficients per surface.
    :return: The pyroomacoustics material, and the coefficients per surface.
    """
    materials = {}
    for surface in Surface:
        materials[surface] = get_surface_material(surface)

    foo = pra.make_materials(
        ceiling=materials[Surface.CEILING],
        floor=materials[Surface.FLOOR],
        east=materials[Surface.EAST],
        west=materials[Surface.WEST],
        north=materials[Surface.NORTH],
        south=materials[Surface.SOUTH],
    )

    coefficients_per_surface: dict[Surface, list[float]] = {}
    for surface in Surface:
        coefficients_per_surface[surface] = materials[surface]["coeffs"]
    return foo, coefficients_per_surface


def weighted_abs_coefs(
    coefficients_per_surface: dict[Surface, list[float]], room_dim: list[float]
) -> dict[float, float]:
    """
    Calculates the absorption coefficients per frequency using the surface areas of each surface in a room.
    :param coefficients_per_surface: The coefficients per surface.
    :param room_dim: The room dimensions.
    :return: the weighted absorption coefficients per frequency band.
    """

    area_per_surface = {
        Surface.EAST: room_dim[1] * room_dim[2],
        Surface.WEST: room_dim[1] * room_dim[2],
        Surface.NORTH: room_dim[0] * room_dim[2],
        Surface.SOUTH: room_dim[0] * room_dim[2],
        Surface.FLOOR: room_dim[1] * room_dim[0],
        Surface.CEILING: room_dim[1] * room_dim[0],
    }

    total_area = sum(area_per_surface.values())

    coef_per_freq = {}

    for k, freq_idx in enumerate(CENTER_FREQUENCIES):
        coef_per_freq[freq_idx] = 0
        for surface in Surface:
            coef_per_freq[freq_idx] += (
                coefficients_per_surface[surface][k]
                * area_per_surface[surface]
                / total_area
            )
    return coef_per_freq


def generate_room() -> (
    tuple[ShoeBox, list[float], list[float], list[float], dict[float, float]]
):
    """
    Generates a simulated room.
    :return: The room, the room dimensions, the source position, the receiver position,
    and the absorption coefficients per frequency band.
    """
    length_x = np.random.uniform(low=4, high=10)
    length_y = np.random.uniform(low=2, high=10)
    length_z = np.random.uniform(low=2.5, high=5)

    room_dim = [length_x, length_y, length_z]

    materials, coefficients_per_surface = get_materials()

    # If the real dataset has object in the room,
    # add scattering coefficient estimator

    room = pra.ShoeBox(
        p=room_dim, fs=SAMPLERATE, materials=materials, max_order=MAX_ORDER
    )

    my_weighted_abs_coefs = weighted_abs_coefs(coefficients_per_surface, room_dim)

    source_position, receiver_position = get_source_and_receiver_positions(room_dim)
    room.add_source(position=source_position)
    room.add_microphone(loc=receiver_position, fs=SAMPLERATE)

    return room, room_dim, source_position, receiver_position, my_weighted_abs_coefs


def get_res_dict(
    filename: str,
    rir: Tensor,
    samplerate: int,
    room_dim: list[float],
    source_pos: list[float],
    receiver_pos: list[float],
    stage: NnStage,
    coefficient_per_freq: dict[float, float] = None,
):
    """
    Creates a dictionary to load into a DataFrame.
    :param filename: the desired filename for the room impulse response.
    :param rir: (raw) the room impulse response.
    :param samplerate: the samplerate of the (raw) room impulse response.
    :param room_dim: the room dimensions.
    :param source_pos: the source position.
    :param receiver_pos: the receiver position.
    :param stage: the stage of the network.
    :param coefficient_per_freq: If the rir is simulated, the weighted coefficients per center frequency.
    :return: a dictionary with all the right data headers.
    """
    processed_rir = preprocess_rir(rir_raw=rir, sample_rate=samplerate)
    rir_dir = RIR_DIR_REAL
    if stage == NnStage.SOURCE:
        rir_dir = RIR_DIR_SIMULATED
    elif stage == NnStage.VALIDATION:
        rir_dir = RIR_DIR_VALID

    uri = Path.cwd() / rir_dir / filename

    res_dict: dict[str, any] = {DataHeaders.RIR_PATH.value: filename}

    (
        res_dict[DataHeaders.LENGTH_X.value],
        res_dict[DataHeaders.LENGTH_Y.value],
        res_dict[DataHeaders.LENGTH_Z.value],
    ) = room_dim[:3]
    (
        res_dict[DataHeaders.SPEAKER_X.value],
        res_dict[DataHeaders.SPEAKER_Y.value],
        res_dict[DataHeaders.SPEAKER_Z.value],
    ) = source_pos[:3]
    (
        res_dict[DataHeaders.MICROPHONE_X.value],
        res_dict[DataHeaders.MICROPHONE_Y.value],
        res_dict[DataHeaders.MICROPHONE_Z.value],
    ) = receiver_pos[:3]
    if stage == NnStage.SOURCE:
        for center_freq in CENTER_FREQUENCIES:
            res_dict[f"{DataHeaders.ABS_COEF.value}_{center_freq}Hz"] = (
                coefficient_per_freq[center_freq]
            )
    else:
        surface_area = (
            room_dim[0] * room_dim[1] * 2
            + room_dim[0] * room_dim[2] * 2
            + room_dim[1] * room_dim[2] * 2
        )
        volume = room_dim[0] * room_dim[1] * room_dim[2]
        octave_bands = rir_in_octave_bands(processed_rir)
        for octave_band, center_freq in zip(octave_bands, CENTER_FREQUENCIES):
            try:
                abs_coef = get_abs_coef_from_real(
                    surface_area=surface_area,
                    volume=volume,
                    filtered_rir=octave_band,
                )
                res_dict[f"{DataHeaders.EYRING.value}_{center_freq}Hz"] = abs_coef
            except ValueError:
                continue

    rir_np = processed_rir.numpy()
    processed_rir = rir_np.copy()
    processed_rir.resize((NUM_SAMPLES, 1), refcheck=False)
    torchaudio.save(
        uri=uri, src=Tensor(processed_rir), sample_rate=SAMPLERATE, channels_first=False
    )
    return res_dict


def generate_simulated_data():
    """
    Generates all the simulated data. Creates a label file and a folder of rirs.
    """
    res = []

    for i in trange(NUM_SIM_RIRS, unit="rooms"):
        room, room_dim, source_pos, receiver_pos, coefficient_per_freq = (
            None,
            None,
            None,
            None,
            None,
        )
        j = 0
        while room is None:
            try:
                (
                    room,
                    room_dim,
                    source_pos,
                    receiver_pos,
                    coefficient_per_freq,
                ) = generate_room()
                room.compute_rir()
            except ValueError as e:
                j += 1
                if j >= 20:
                    raise InterruptedError(
                        "Tried to simulate a room for more than 20 times unsuccessfully."
                        + "Please decrease the minimum distance."
                    )
                logging.error(msg=f"Error {j}: {e}")
                continue

        rir = room.rir[0][0]

        filename = f"rir_sim_{i}.wav"
        res.append(
            get_res_dict(
                filename=filename,
                rir=rir,
                samplerate=SAMPLERATE,
                room_dim=room_dim,
                source_pos=source_pos,
                receiver_pos=receiver_pos,
                coefficient_per_freq=coefficient_per_freq,
                stage=NnStage.SOURCE,
            )
        )

    df = pd.DataFrame(res)
    df.to_csv(Path.cwd() / LABEL_FILE_SIMULATED)


def get_abs_coef_from_real(
    volume: float, surface_area: float, filtered_rir: Tensor
) -> float:
    """
    Estimates the absorption coefficient from a filtered room impulse response.
    It inverts Eyring's formula to achieve this.
    :param volume: The volume of the room where the room impulse is taken from.
    :param surface_area: The surface area of the room where the room impulse is taken from.
    :param filtered_rir: The filtered room impulse response.
    :return: The estimated absorption coefficient.
    """
    #   Calculate RT60 or RT30
    dbs = 30
    rt60 = None
    i = 2
    # maximally go to T15 as an approximation
    while rt60 is None and i < 4:
        try:
            rt60 = (
                60 / dbs * pra.measure_rt60(h=filtered_rir, fs=SAMPLERATE, decay_db=dbs)
            )
        except ValueError:
            i += 1
            dbs = 60 / i

    if rt60 is None:
        raise ValueError("Energy level in signal was not enough for the T15.")

    # invert Eyring's formula to get average absorption coefficient at that band.

    average_abs_coef_eyring = 1 - np.exp(-0.163 * volume / (surface_area * rt60))

    return average_abs_coef_eyring


def generate_real_data_dEchorate():
    """
    Generates the real data from the dEchorate dataset.
    """
    rir_dset = h5py.File(Path.cwd() / Path("dEchorate/dEchorate_rir.h5"), mode="r")
    annotations = h5py.File(
        Path.cwd() / Path("dEchorate/dEchorate_annotations.h5"), mode="r"
    )

    fs = rir_dset.attrs["sampling_rate"]
    room_dim: list[float] = annotations["room_size"][:]

    room_codes = list(rir_dset["rir"].keys())

    # These sources correspond to omnidirectional sources
    sources = ["7", "8", "9"]

    src_positions: dict[str, tuple[float, float, float]] = {
        sources[0]: (3.651, 1.004, 1.38),
        sources[1]: (2.958, 4.558, 1.486),
        sources[2]: (0.892, 3.013, 1.403),
    }

    mics = annotations["microphones"][:]

    res = []

    for room_code in tqdm(room_codes, unit="rooms"):
        for source in sources:
            for mic in range(mics.shape[1] - 1):
                real_rir = Tensor(rir_dset["rir"][room_code][source][:, mic])

                filename = (
                    f"rir_real_dEchorate_room{room_code}_source{source}_mic{mic}.wav"
                )

                res_dict = get_res_dict(
                    filename=filename,
                    rir=real_rir,
                    samplerate=fs,
                    room_dim=room_dim,
                    source_pos=list(src_positions[source]),
                    receiver_pos=mics[:3, mic],
                    stage=NnStage.TARGET,
                )
                res_dict[DataHeaders.ROOM_ID.value] = room_code
                res.append(res_dict)

    df = pd.DataFrame(res)

    eyring_headers = [
        f"{DataHeaders.EYRING.value}_{center_freq}Hz"
        for center_freq in CENTER_FREQUENCIES
    ]
    abs_coef_headers = [
        f"{DataHeaders.ABS_COEF.value}_{center_freq}Hz"
        for center_freq in CENTER_FREQUENCIES
    ]
    print(df)
    convert_dict = {}

    for eyring_header, abs_coef_header in zip(eyring_headers, abs_coef_headers):
        convert_dict[eyring_header] = abs_coef_header

    grouped = df.groupby([DataHeaders.ROOM_ID.value])[eyring_headers].mean()

    grouped.rename(columns=convert_dict, inplace=True)
    grouped.to_csv(Path.cwd() / AGG_FILE)
    df = pd.merge(df, grouped, on=[DataHeaders.ROOM_ID.value])
    df.to_csv(Path.cwd() / LABEL_FILE_REAL)


def create_directories():
    """
    Creates all necessary directories for data generation.
    """
    dirs = [
        "data",
        "data/rirs",
        RIR_DIR_SIMULATED,
        RIR_DIR_REAL,
        RIR_DIR_VALID,
        OUTPUT_DIR,
        f"{OUTPUT_DIR}/mae",
        f"{OUTPUT_DIR}/mse",
    ]

    for my_dir in dirs:
        if not os.path.exists(Path.cwd() / my_dir):
            os.mkdir(Path.cwd() / my_dir)


def get_dataloaders() -> dict[NnStage, DataLoader]:
    """
    Creates dataloaders for the real and simulated data.
    :return: A dict of dataloaders, the key of which is the stage of the neural network.
    """
    create_directories()
    if not (os.path.isfile(Path.cwd() / LABEL_FILE_REAL)):
        generate_real_data_dEchorate()
    if not (os.path.isfile(Path.cwd() / LABEL_FILE_SIMULATED)):
        generate_simulated_data()

    simulated_data = RirAndRoomLayoutDataset(
        label_file=Path.cwd() / LABEL_FILE_SIMULATED,
        rir_dir=Path.cwd() / RIR_DIR_SIMULATED,
    )

    real_data = RirAndRoomLayoutDataset(
        label_file=Path.cwd() / LABEL_FILE_REAL, rir_dir=Path.cwd() / RIR_DIR_REAL
    )

    simulated_dataloader = DataLoader(
        simulated_data, batch_size=BATCH_SIZE, shuffle=True
    )
    real_dataloader = DataLoader(real_data, batch_size=BATCH_SIZE, shuffle=True)

    return {NnStage.SOURCE: simulated_dataloader, NnStage.TARGET: real_dataloader}
