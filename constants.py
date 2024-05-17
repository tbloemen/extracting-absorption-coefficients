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

SAMPLERATE = 16000
RIR_DURATION = 1.5

OUT_FEATURES = 12


class NnStage(Enum):
    TARGET = "target"
    SOURCE = "source"


MODEL_PATH = "models/"
MODEL_NAME = "my_model.pth"
