import logging
import os
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import pandas as pd
import pyroomacoustics as pra
import torchaudio
from pyroomacoustics import ShoeBox
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

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
    FREQ_LOWER_BOUND,
    FREQ_UPPER_BOUND,
    RealDataHeaders,
)
from estimator import Wall_Estimator, Ceiling_Estimator, Floor_Estimator
from preprocessing import rir_in_octave_bands
from printing import print_datashape


class RIR_and_Numeric_Dataset(Dataset):
    def __init__(self, label_file: str, rir_dir: str):
        self.labels: pd.DataFrame = pd.read_csv(label_file)
        self.rir_dir: str = rir_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[
        Tuple[
            Tuple[Tensor, int],
            Tuple[Tuple[float, float, float], Tuple[float, float], Tuple[float, float]],
        ],
        dict[Surface, float],
    ]:
        rir_path = os.path.join(
            self.rir_dir, self.labels.loc[self.labels.index[idx], DataHeaders.RIR_PATH]
        )
        rir, sample_rate = torchaudio.load(rir_path)
        length_x = self.labels.loc[self.labels.index[idx], DataHeaders.LENGTH_X]
        length_y = self.labels.loc[self.labels.index[idx], DataHeaders.LENGTH_Y]
        length_z = self.labels.loc[self.labels.index[idx], DataHeaders.LENGTH_Z]

        length = (length_x, length_y, length_z)

        microphone_x = self.labels.loc[self.labels.index[idx], DataHeaders.MICROPHONE_X]
        microphone_y = self.labels.loc[self.labels.index[idx], DataHeaders.MICROPHONE_Y]

        microphone = (microphone_x, microphone_y)

        speaker_x = self.labels.loc[self.labels.index[idx], DataHeaders.SPEAKER_X]
        speaker_y = self.labels.loc[self.labels.index[idx], DataHeaders.SPEAKER_Y]

        speaker = (speaker_x, speaker_y)

        absorption_params = {}
        for surface in Surface:
            absorption_params[surface] = self.labels.loc[
                self.labels.index[idx],
                f"{DataHeaders.ABS_COEF}{surface.name.lower()}",
            ]

        return ((rir, sample_rate), (length, microphone, speaker)), absorption_params


def get_center_frequencies(center_freq=1000):
    lower_bound = center_freq / np.sqrt(2)
    while lower_bound / 2 >= FREQ_LOWER_BOUND:
        lower_bound /= 2

    higher_bound = center_freq * np.sqrt(2)
    while higher_bound * 2 <= FREQ_UPPER_BOUND:
        higher_bound *= 2

    res = []
    while lower_bound < higher_bound:
        res.append(lower_bound * np.sqrt(2))
        lower_bound *= 2

    return res


def get_source_and_receiver_positions(
    room_dims: list[float],
) -> tuple[list[float], list[float]]:
    if len(room_dims) != 3:
        raise ValueError(
            f"The room dimensions were not of size 3, but of size {len(room_dims)}"
        )

    min_distance = 1.0
    height = 1.5

    window_dims = [None, None]
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

    source_pos = [None, None, height]
    receiver_pos = [None, None, height]

    for i in range(2):
        source_pos[i] = np.random.uniform(window_dims[i][0], window_dims[i][1])
        receiver_pos[i] = np.random.uniform(window_dims[i][0], window_dims[i][1])

    return source_pos, receiver_pos


def get_surface_material(surface: Surface):
    center_freqs = get_center_frequencies()
    res = np.array([np.random.uniform(low=0.01, high=0.12)] * len(center_freqs))
    if np.random.random() < 0.5:
        estimator = Wall_Estimator()
        if surface == Surface.CEILING:
            estimator = Ceiling_Estimator()
        if surface == Surface.FLOOR:
            estimator = Floor_Estimator()

        for i, center_freq in enumerate(center_freqs):
            res[i] = estimator.estimate_abs_coef(frequency=center_freq)
    return {
        "description": f"{surface.name.capitalize()} estimator material",
        "coeffs": res,
        "center_freqs": center_freqs,
    }


def get_materials():
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

    for surface in Surface:
        materials[surface] = materials[surface]["coeffs"]
    return foo, materials


def generate_room() -> tuple[ShoeBox, list[float], list[float], list[float]]:
    length_x = np.random.uniform(low=4, high=10)
    length_y = np.random.uniform(low=2, high=10)
    length_z = np.random.uniform(low=2.5, high=5)

    room_dim = [length_x, length_y, length_z]

    materials, material_dict = get_materials()

    room = pra.ShoeBox(p=room_dim, fs=SAMPLERATE, materials=materials)

    source_position, receiver_position = get_source_and_receiver_positions(room_dim)
    room.add_source(position=source_position)
    room.add_microphone(loc=receiver_position, fs=SAMPLERATE)

    return room, room_dim, source_position, receiver_position


def generate_simulated_data():
    for i in range(NUM_SIM_RIRS):
        room, room_dim, source_pos, receiver_pos = None, None, None, None
        j = 0
        while room is None:
            try:
                room, room_dim, source_pos, receiver_pos = generate_room()
            except ValueError as e:
                j += 1
                if j >= 20:
                    raise InterruptedError(
                        "Tried to simulate a room for more than 20 times unsuccessfully."
                        + "Please decrease the minimum distance."
                    )
                logging.error(msg=f"Error {j}: {e}")
                continue

        room.compute_rir()
        rir = room.rir[0][0]

        # TODO: filewriting to RIR_DIR_SIMULATED and LABEL_FILE_SIMULATED
        # append a line to either a csv file directly or to a pandas file


def generate_real_data():
    # Pseudocode:

    # For all room impulse responses:
    #   Separate into octave bands
    #   Calculate RT60 or RT30
    #   (This probably has to be done by a Schroeder curve, evaluated at -5 and -35 for RT30)
    #   invert Eyring's formula to get average absorption coefficient at that band.
    #   if data about each wall, floor and ceiling is known: skew the mean towards the values of the walls.

    pass


def get_dataloaders() -> dict[NnStage, DataLoader]:
    for filepath in [
        LABEL_FILE_SIMULATED,
        RIR_DIR_SIMULATED,
        LABEL_FILE_REAL,
        RIR_DIR_REAL,
    ]:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File {filepath} was not found.")

    simulated_data = RIR_and_Numeric_Dataset(
        label_file=LABEL_FILE_SIMULATED, rir_dir=RIR_DIR_SIMULATED
    )

    real_data = RIR_and_Numeric_Dataset(
        label_file=LABEL_FILE_REAL, rir_dir=RIR_DIR_REAL
    )

    simulated_dataloader = DataLoader(
        simulated_data, batch_size=BATCH_SIZE, shuffle=True
    )
    real_dataloader = DataLoader(real_data, batch_size=BATCH_SIZE, shuffle=True)

    print_datashape(simulated_dataloader, NnStage.SOURCE)
    print_datashape(real_dataloader, NnStage.TARGET)
    return {NnStage.SOURCE: simulated_dataloader, NnStage.TARGET: real_dataloader}
