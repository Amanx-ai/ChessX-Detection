# F:\ChessXDetection\src\config.py

import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT is defined by moving up one directory from src/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "piece_classifier.pth")
STOCKFISH_EXECUTABLE = os.path.join(PROJECT_ROOT, "engines", "stockfish.exe")

# --- IMAGE & VISION SETTINGS ---
# Standardizing the output size of the perspective-corrected board
BOARD_SIZE_PIXELS = 720
SQUARE_SIZE = BOARD_SIZE_PIXELS // 8

# --- RECOGNITION CLASSES ---
# The classes the CNN is trained to predict
PIECE_CLASSES = [
    "empty", "wP", "wR", "wN", "wB", "wQ", "wK",
    "bP", "bR", "bB", "bN", "bQ", "bK"
]