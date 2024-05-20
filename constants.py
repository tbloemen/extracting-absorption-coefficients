from enum import Enum

BATCH_SIZE = 64
EPOCHS = 250

# Learning rate
LEARNING_RATE = 1e-3
GAMMA = 1e-4
POWER = 0.75

# At how many percent do we say there is no significant change?
LOW_DELTA = 0.5

# How many iterations should this low change be after each other?
SAME_DELTA_EPOCHS = 10

THRESHOLD = 0.9
TRADEOFF_ANGLE = 0.05
TRADEOFF_SCALE = 0.001

SAMPLERATE = 48000
RIR_DURATION = 1.5

# Spectrum of hearing
FREQ_LOWER_BOUND = 60
FREQ_UPPER_BOUND = 20000


class NnStage(Enum):
    TARGET = "target"
    SOURCE = "source"


class Surface(Enum):
    FLOOR = 0
    CEILING = 1
    LEFT = 2
    FRONT = 3
    RIGHT = 4
    BACK = 5


# abs. coef. for floor, ceiling, wall_left, wall_front, wall_right, wall_south, it that order.
OUT_FEATURES = len(Surface)

MODEL_PATH = "models/"
MODEL_NAME = "my_model.pth"
