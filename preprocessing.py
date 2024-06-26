"""
This file houses the preprocessing functions to standardize all datasets.
"""

import numpy as np
import torch
from numpy import ndarray
from scipy import signal
from torch import Tensor
from torchaudio.functional import resample

from constants import SAMPLERATE, CENTER_FREQUENCIES, NUM_SAMPLES


def preprocess_rir(rir_raw: ndarray | Tensor, sample_rate: int) -> Tensor:
    """
    This function preprocesses the raw room impulse response into a normalized signal.
    :param rir_raw: The raw room impulse response.
    :param sample_rate: the sample rate of the raw room impulse response.
    :return: a normalized room impulse response with a sample rate of `SAMPLERATE` in the constants file.
    """
    if sample_rate is not SAMPLERATE:
        rir_raw = resample(rir_raw, sample_rate, SAMPLERATE)

    if type(rir_raw) is Tensor:
        rir_raw = rir_raw.numpy()
    copied_rir = rir_raw.copy()

    rir: Tensor = Tensor(copied_rir)
    rir = rir / torch.linalg.vector_norm(rir)

    return rir


def get_octave_band(
    rir: ndarray, lower_bound: float, upper_bound: float, poles: int = 5
) -> Tensor:
    """
    Applies a bandpass to a room impulse response with a Butterworth filter,
    and resizes it to match the desired number of samples.
    :param rir: the room impulse response.
    :param lower_bound: the lower bound of the bandpass filter.
    :param upper_bound: the upper bound of the bandpass filter.
    :param poles: the amount of poles to apply to the butterworth filter. More poles mean a steeper curve
    from the butterworth filter.
    :return: the bandpassed room impulse response, resized to match the number of samples.
    """
    sos = signal.butter(
        N=poles,
        Wn=[lower_bound, upper_bound],
        fs=SAMPLERATE,
        btype="bandpass",
        output="sos",
    )
    filtered_rir: ndarray = signal.sosfilt(sos=sos, x=rir)
    foo = filtered_rir.copy()
    foo.resize((NUM_SAMPLES, 1), refcheck=False)
    return Tensor(foo)


def rir_in_octave_bands(rir: Tensor) -> list[Tensor]:
    """
    Returns the room impulse response bandpassed into all the octave bands with
    the center frequencies defined in the constants.
    :param rir: The room impulse response to be bandpassed.
    :return: a list of bandpassed rirs.
    """
    rir: ndarray = rir.numpy()
    octave_bands = []

    for center_freq in CENTER_FREQUENCIES:
        octave_bands.append(
            get_octave_band(rir, center_freq / np.sqrt(2), center_freq * np.sqrt(2))
        )

    return octave_bands
