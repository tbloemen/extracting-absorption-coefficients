from enum import Enum

from utils import get_center_frequencies, get_device

DEVICE = get_device()

BATCH_SIZE = 64
EPOCHS = 250

# Learning rate
LEARNING_RATE = 1e-3
GAMMA = 1e-4
POWER = 0.75

# At how many percent do we say there is no significant change?
LOW_DELTA = 0.001

# How many iterations should this low change be after each other?
SAME_DELTA_EPOCHS = 5

THRESHOLD = 0.9
TRADEOFF_ANGLE = 0.5
TRADEOFF_SCALE = 0.01

SAMPLERATE = 48000
RIR_DURATION = 1.5
NUM_SAMPLES = int(SAMPLERATE * RIR_DURATION)


class NnStage(Enum):
    TARGET = "target"
    SOURCE = "source"


class Surface(Enum):
    FLOOR = 0
    CEILING = 1
    NORTH = 2
    EAST = 3
    SOUTH = 4
    WEST = 5


MODEL_PATH = "models/"
MODEL_NAME = "my_model.pth"

RIR_DIR_SIMULATED = "data/rirs/simulated/"
LABEL_FILE_SIMULATED = "data/labels_simulated.csv"

RIR_DIR_REAL = "data/rirs/real"
LABEL_FILE_REAL = "data/labels_real.csv"

NUM_SIM_RIRS = 4000


CENTER_FREQUENCIES = get_center_frequencies()


class DataHeaders(Enum):

    RIR_PATH = "rir_path"
    LENGTH_X = "length_x"
    LENGTH_Y = "length_y"
    LENGTH_Z = "length_z"
    MICROPHONE_X = "microphone_x"
    MICROPHONE_Y = "microphone_y"
    MICROPHONE_Z = "microphone_z"
    SPEAKER_X = "speaker_x"
    SPEAKER_Y = "speaker_y"
    SPEAKER_Z = "speaker_z"
    ABS_COEF = "abs_coef"
    FREQUENCY = "frequency"
