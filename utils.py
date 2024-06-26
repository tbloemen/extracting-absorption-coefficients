"""
Some utility functions for general use in the project.
"""

import numpy as np
import torch

# Spectrum of hearing
FREQ_LOWER_BOUND = 20
FREQ_UPPER_BOUND = 20000


def get_center_frequencies(center_freq=1000) -> list[float]:
    """
    This creates the center frequencies in the constants.py file. It is based upon octave bands, as opposed to 1/3rd
    octave bands or another type of octave band. The center frequency used to calculate the centers of the octave
    bands is defaulted to 1000 Hz.
    :param center_freq: The center frequency of the octave bands.
    :return: an array with center frequencies within the range of human hearing.
    """
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


def get_device() -> str:
    """
    Returns the device used for torch, based on what is available on your hardware.
    :return: The device used for torch.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
