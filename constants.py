from enum import Enum

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 250

# At how many percent do we say there is no significant change?
LOW_DELTA = 0.5

# How many iterations should this low change be after each other?
SAME_DELTA_EPOCHS = 10

THRESHOLD = 0.9
TRADEOFF_ANGLE = 0.05
TRADEOFF_SCALE = 0.001

SAMPLERATE = 16000
RIR_DURATION = 1.5


class NnStage(Enum):
    TRAINING = "training"
    TEST = "test"


MODEL_PATH = "models/"
MODEL_NAME = "my_model.pth"
