"""
Centralized configuration for the Cyberbullying Detection AI project.
All hyperparameters and paths are defined here.
"""
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TRAIN_IMDB_CSV = os.path.join(DATA_DIR, 'train_imdb.csv')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')

# --- Two-stage pipeline (methodology Figure 1) ---
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MESSAGES_CSV = os.path.join(PROCESSED_DIR, 'messages.csv')   # Stage 1: message-level aggression
PAIRS_CSV = os.path.join(PROCESSED_DIR, 'pairs.csv')         # Stage 2: user-pair cyberbullying

# --- Model Selection ---
MBERT_MODEL = 'bert-base-multilingual-cased'
MURIL_MODEL = 'google/muril-base-cased'

# Default text model (switch between MBERT_MODEL and MURIL_MODEL)
TEXT_MODEL = MBERT_MODEL

# --- Training Hyperparameters ---
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_LEN = 128
WARMUP_STEPS = 100

# --- Model Architecture ---
FUSION_DIM = 512
NUM_CLASSES = 3  # Aggression, Repetition, Intent
CONTEXT_FEATURES = 2  # user_age, past_flags
DROPOUT_RATE = 0.3

# --- Image Processing ---
IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
IMAGE_STD = [0.229, 0.224, 0.225]

# --- Data Split ---
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# --- Early Stopping ---
PATIENCE = 3

# --- Device ---
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
