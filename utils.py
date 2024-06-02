import numpy as np
import torch

# Spectrum of hearing
FREQ_LOWER_BOUND = 20
FREQ_UPPER_BOUND = 20000


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


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
