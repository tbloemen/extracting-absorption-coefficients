"""
The constants used throughout the project. These are used everywhere, so there is as little hard-coding as necessary.
"""

from enum import Enum

from utils import get_center_frequencies, get_device

DEVICE = get_device()

BATCH_SIZE = 32
EPOCHS = 30

# Learning rate
LEARNING_RATE = 1e-3
GAMMA = 1e-4
POWER = 0.75

# At how many percent do we say there is no significant change?
LOW_DELTA = 0.01

# How many iterations should this low change be after each other?
SAME_DELTA_EPOCHS = 30

THRESHOLD = 0.9
TRADEOFF_ANGLE = 0.05
TRADEOFF_SCALE = 0.001

SAMPLERATE = 48000
RIR_DURATION = 1
NUM_SAMPLES = int(SAMPLERATE * RIR_DURATION)

NUM_SIM_RIRS = 4000
MAX_ORDER = 100

CENTER_FREQUENCIES = get_center_frequencies()


class DataHeaders(Enum):
    """
    The headers used for in the csv files in this project.
    """

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
    EYRING = "eyring"
    FREQUENCY = "frequency"
    ERROR = "error"
    SQUARED_ERROR = "squared_error"
    ROOM_ID = "room_id"


class NnStage(Enum):
    """
    The stages of the machine learning algorithm.
    """

    TARGET = "target"
    SOURCE = "source"
    VALIDATION = "validation"


class Surface(Enum):
    """
    The different surfaces of a shoebox room.
    """

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
AGG_FILE = "data/agg_real.csv"

RIR_DIR_VALID = "data/rirs/validation"
LABEL_FILE_VALID = "data/labels_validation.csv"
AGG_FILE_VALID = "data/agg_valid.csv"

OUTPUT_DIR = "outputs"
