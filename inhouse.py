"""This file houses the methods needed for validating the machine learning model with the inHouse dataset,
which are recordings made by another researcher of the TU Delft."""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader

import printing
from constants import (
    NnStage,
    DataHeaders,
    CENTER_FREQUENCIES,
    AGG_FILE_VALID,
    LABEL_FILE_VALID,
    RIR_DIR_VALID,
    BATCH_SIZE,
    DEVICE,
    MODEL_PATH,
    MODEL_NAME,
)
from data_gen import get_res_dict, create_directories, RirAndRoomLayoutDataset
from main import NeuralNetwork, test


def generate() -> None:
    """
    Generates the inHouse dataset sound files and label file.
    """
    orig_loc_loudspeaker_3 = np.array([5.5600, 2.0190, 1.1750])

    new_room_dims = [6.32, 4.31, 3.08]  # This is with the curtains.
    new_loc_loudspeaker_3 = np.array(
        [
            new_room_dims[0] - 0.4,
            0.7,
            orig_loc_loudspeaker_3[2],
        ]
    )
    print(new_loc_loudspeaker_3)

    displacement = new_loc_loudspeaker_3 - orig_loc_loudspeaker_3

    print(displacement)

    res = []
    for orig_filename in (Path.cwd() / "InHouse" / "bigger" / "Closed Curtains").glob(
        "*.mat"
    ):
        source_channel = int(re.sub("[^0-9]", "", orig_filename.name))
        mat = loadmat(orig_filename)  # type: ignore
        fs = mat["Fs"][0]
        source_pos = mat["loc_loudspeaker"][0]
        for i, mic_pos in enumerate(mat["loc_microphone"]):
            filename = f"rir_validation_inHouse_source{source_channel}_mic{i+1}.wav"

            rir = mat["RIR"][:, i]

            res_dict = get_res_dict(
                filename=filename,
                rir=rir,
                samplerate=fs,
                room_dim=new_room_dims,
                source_pos=source_pos + displacement,
                receiver_pos=mic_pos + displacement,
                stage=NnStage.VALIDATION,
            )
            res.append(res_dict)
    df = pd.DataFrame(res)

    print(df)

    eyring_headers = [
        f"{DataHeaders.EYRING.value}_{center_freq}Hz"
        for center_freq in CENTER_FREQUENCIES
    ]
    abs_coef_headers = [
        f"{DataHeaders.ABS_COEF.value}_{center_freq}Hz"
        for center_freq in CENTER_FREQUENCIES
    ]
    convert_dict = {}

    for eyring_header, abs_coef_header in zip(eyring_headers, abs_coef_headers):
        convert_dict[eyring_header] = abs_coef_header

    grouped = df[eyring_headers].mean(axis="rows").to_frame()
    grouped = grouped.T
    grouped.rename(columns=convert_dict, inplace=True)
    grouped.to_csv(Path.cwd() / AGG_FILE_VALID)
    foo = np.resize(grouped, ((df.shape[0]), len(eyring_headers)))
    df[abs_coef_headers] = foo
    df.to_csv(Path.cwd() / LABEL_FILE_VALID)


def main() -> None:
    """
    Validates the model. It creates the dataset, loads it and tests it on the model. The results are plotted to a
    matplotlib figure in the output directory.
    """
    create_directories()
    if not (os.path.isfile(Path.cwd() / LABEL_FILE_VALID)):
        generate()

    validation_data = RirAndRoomLayoutDataset(
        label_file=Path.cwd() / LABEL_FILE_VALID, rir_dir=Path.cwd() / RIR_DIR_VALID
    )

    validation_dataloader = DataLoader(
        validation_data, batch_size=BATCH_SIZE, shuffle=True
    )

    torch.cuda.empty_cache()

    model = NeuralNetwork().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH + MODEL_NAME))
    model.eval()
    test(model=model, epoch=0, valid_loader=validation_dataloader)
    printing.results(NnStage.VALIDATION)


if __name__ == "__main__":
    main()
