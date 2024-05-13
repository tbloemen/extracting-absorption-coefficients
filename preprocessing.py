import random

import torch
from torch import Tensor
from torchaudio.functional import add_noise, resample
from torchaudio.io import AudioEffector
from constants import SAMPLERATE, RIR_DURATION


def preprocess_rir(rir_raw: Tensor, sample_rate: int, is_simulated: bool) -> Tensor:
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
    return codec_applied


def generate_white_noise(num_frames: int):
    return torch.rand((num_frames,)) - 0.5
