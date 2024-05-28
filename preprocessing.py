import numpy as np
import torch
from numpy import ndarray
from scipy import signal
from torch import Tensor
from torchaudio.functional import add_noise, resample
from torchaudio.io import AudioEffector

from constants import SAMPLERATE, RIR_DURATION
from data_gen import get_center_frequencies


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
        noise_db = np.random.uniform(low=50, high=70)
        noise_waveform = generate_white_noise(rir.shape[0])
        rir = add_noise(rir, noise_waveform, noise_db).squeeze()

    effector = AudioEffector(format="wav")
    codec_applied = effector.apply(waveform=rir, sample_rate=SAMPLERATE)
    return rir_in_octave_bands(rir=codec_applied)


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


def rir_in_octave_bands(rir: Tensor) -> list:

    rir: ndarray = rir.numpy()
    octave_bands = []

    for center_freq in get_center_frequencies():
        octave_bands.append(
            get_octave_band(rir, center_freq / np.sqrt(2), center_freq * np.sqrt(2))
        )

    return octave_bands
