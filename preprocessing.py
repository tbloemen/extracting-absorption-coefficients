import random
from math import sqrt

from numpy import ndarray
from scipy import signal
import torch
from torch import Tensor
from torchaudio.functional import add_noise, resample
from torchaudio.io import AudioEffector
from constants import SAMPLERATE, RIR_DURATION


def preprocess_rir(
    rir_raw: Tensor, sample_rate: int, is_simulated: bool
) -> list[Tensor]:
    if sample_rate is not SAMPLERATE:
        rir_raw = resample(rir_raw, sample_rate, SAMPLERATE)

    # Todo: select fragment which is actually useful
    start = int(SAMPLERATE * 1.01)
    rir = rir_raw[:, start : start + SAMPLERATE * RIR_DURATION]
    rir = rir / torch.linalg.vector_norm(rir)

    if is_simulated:
        # random number between 50 and 70
        noise_db = random.random() * 20 + 50
        noise_waveform = generate_white_noise(rir.shape[0])
        rir = add_noise(rir, noise_waveform, noise_db).squeeze()

    effector = AudioEffector(format="wav")
    codec_applied = effector.apply(waveform=rir, sample_rate=SAMPLERATE)
    return rir_in_octave_bands(rir=codec_applied, min_freq=60, max_freq=SAMPLERATE)


def generate_white_noise(num_frames: int):
    return torch.rand((num_frames,)) - 0.5


def get_octave_band(
    rir: ndarray, lower_bound: float, higher_bound: float, poles: int = 5
) -> Tensor:
    sos = signal.butter(
        N=poles,
        Wn=[lower_bound, higher_bound],
        fs=SAMPLERATE,
        btype="bandpass",
        output="sos",
    )
    filtered_rir = signal.sosfilt(sos=sos, x=rir)

    # normalize the filtered signal for uniformity
    filtered_rir = filtered_rir / torch.linalg.vector_norm(filtered_rir)
    return torch.from_numpy(filtered_rir)


def rir_in_octave_bands(
    rir: Tensor, min_freq: float, max_freq: float, center_freq: float = 1000.0
) -> list:

    if min_freq >= center_freq >= max_freq:
        raise ValueError(
            f"Please readjust the frequencies.\nLower bound: {min_freq}\nCenter freq: {center_freq}\nMax freq: {max_freq}"
        )

    rir: ndarray = rir.numpy()

    octave_bands = []
    lower_bound = center_freq / sqrt(2)
    while lower_bound / 2 >= min_freq:
        lower_bound /= 2

    higher_bound = center_freq * sqrt(2)
    while higher_bound * 2 <= max_freq:
        higher_bound *= 2

    while lower_bound < higher_bound:
        octave_bands.append(get_octave_band(rir, lower_bound, lower_bound * 2))
        lower_bound *= 2

    return octave_bands
