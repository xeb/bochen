"""Configuration for Experiment 2: Object-Centric World Model."""

# Data collection
COLLECT_STEPS_PER_GAME = 50000
COLLECT_BATCH_FILE = "data/{game_id}_transitions.npz"

# Model architecture
SEGMENTER_CHANNELS = [16, 32, 64]
TRANSITION_HIDDEN = 256
TRANSITION_LAYERS = 3
OBJECT_EMBED_DIM = 64

# Training
TRAIN_EPOCHS = 50
TRAIN_BATCH_SIZE = 256
TRAIN_LR = 1e-3
VALIDATION_SPLIT = 0.1
MIN_SEGMENTATION_IOU = 0.95
MIN_TRANSITION_ACCURACY = 0.85

# Agent
MAX_STEPS = 100
NUM_CANDIDATE_ACTIONS = 100  # 6 simple + top ACTION6 coordinates
INFO_GAIN_ALPHA_START = 1.0  # pure exploration
INFO_GAIN_ALPHA_DECAY = 0.02  # per step

# Episode timeout in seconds
EPISODE_TIMEOUT = 120
