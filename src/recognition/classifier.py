import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import sys
import numpy as np
import chess # Required for FEN assembly logic

# Import configuration constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from config import MODELS_DIR, MODEL_SAVE_PATH, PIECE_CLASSES, SQUARE_SIZE

# --- MODEL AND TRANSFORM SETUP ---
TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

MODEL = None # Global variable to hold the loaded model

def load_recognition_model():
    """Initializes and loads the pre-trained ResNet18 model."""
    global MODEL
    if MODEL is not None:
        return True

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"ERROR: Model not found at {MODEL_SAVE_PATH}. Please train the model first.")
        return False
        
    try:
        # Initialize architecture (ResNet18)
        loaded_model = models.resnet18(weights=None)
        # Match the output layer to our 13 PIECE_CLASSES
        loaded_model.fc = nn.Linear(loaded_model.fc.in_features, len(PIECE_CLASSES))
        # Load the saved weights
        loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu')))
        loaded_model.eval()
        MODEL = loaded_model
        print("Model loaded successfully.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to load recognition model: {e}")
        return False

def get_fen_from_image(board_img: np.ndarray, is_flipped: bool = False) -> str or None:
    """
    Runs CNN inference on 64 squares and returns the FEN piece placement string.
    
    Args:
        board_img: The perspective-corrected board image.
        is_flipped: If True, reverses the board mapping (H1 is top-left).
        
    Returns:
        The FEN piece placement string (e.g., 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR').
    """
    if MODEL is None:
        if not load_recognition_model():
            return None
    
    try:
        squares_batch = []
        # 1. Segmentation and Batch Preparation
        for r in range(8):
            for c in range(8):
                y_start, y_end = r * SQUARE_SIZE, (r + 1) * SQUARE_SIZE
                x_start, x_end = c * SQUARE_SIZE, (c + 1) * SQUARE_SIZE
                square = board_img[y_start:y_end, x_start:x_end]
                squares_batch.append(TRANSFORMS(Image.fromarray(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))))

        # 2. Recognition (Inference) 
        with torch.no_grad():
            output = MODEL(torch.stack(squares_batch))
            # Get the index with the highest probability and map it back to the class name
            preds = [PIECE_CLASSES[idx] for idx in torch.max(output, 1)[1]]

        # 3. FEN Reconstruction
        board_array = [preds[i:i+8] for i in range(0, 64, 8)]

        # FEN generation logic (must handle empty square counts and slashes)
        fen_map = {
            'bR':'r','bN':'n','bB':'b','bQ':'q','bK':'k','bP':'p',
            'wR':'R','wN':'N','wB':'B','wQ':'Q','wK':'K','wP':'P',
            'empty':'1'
        }
        
        fen = ''
        for row in board_array:
            empty_count = 0
            for piece in row:
                if piece == 'empty':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen += str(empty_count)
                        empty_count = 0
                    fen += fen_map[piece]
            if empty_count > 0:
                fen += str(empty_count)
            fen += '/'
            
        return fen.rstrip('/')

    except Exception as e:
        print(f"Error during FEN generation: {e}")
        return None

# Attempt to load model on import (non-blocking for threads)
load_recognition_model()