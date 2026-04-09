import os

# Dataset
DATASET      = "CIFAR10"
NUM_CLASSES  = 10
DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Split
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Training
BATCH_SIZE    = 64
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
OPTIMIZER     = "adam"
SEED          = 42

# Input
CIFAR_SIZE       = 32   # native CIFAR-10 resolution
MODEL_INPUT_SIZE = 224  # MobileNet expected input
