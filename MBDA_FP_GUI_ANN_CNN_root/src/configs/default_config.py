import os

# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

ANN_OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'ann')
CNN_OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'cnn')

# DEFAULT DATASET SETTINGS
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 512
NUM_CLASSES = 26

# DEFAULT GUI VALUES (ANN)
ANN_DEFAULT_LAYERS = "512, 256, 128"
ANN_DEFAULT_EPOCHS = "20"
ANN_DEFAULT_LR = "0.001"

# DEFAULT GUI VALUES (CNN)
CNN_DEFAULT_EPOCHS = "15"
CNN_DEFAULT_LR = "0.001"
CNN_DEFAULT_STEP = "5"
CNN_DEFAULT_GAMMA = "0.5"