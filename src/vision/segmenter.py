# F:\ChessXDetection\src\vision\segmenter.py

import cv2
import numpy as np
import sys
import os

# Import configuration constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from config import BOARD_SIZE_PIXELS, SQUARE_SIZE

def rc_to_algebraic(r: int, c: int) -> str:
    """
    Converts (row, col) index (0-7, 0-7) to standard algebraic notation (e.g., 0, 0 -> 'a8').
    Assumes A8 is Top-Left (r=0, c=0).
    """
    file_char = chr(ord('a') + c) 
    rank_num = BOARD_SIZE_PIXELS // SQUARE_SIZE - r
    
    return f"{file_char}{rank_num}"

def segment_board(warped_image_path: str) -> dict:
    """
    Loads the flattened board image and segments it into 64 labeled crops.
    """
    img = cv2.imread(warped_image_path)
    if img is None:
        raise FileNotFoundError(f"Segmenter: Could not load warped image at {warped_image_path}")
        
    if img.shape[0] != BOARD_SIZE_PIXELS or img.shape[1] != BOARD_SIZE_PIXELS:
        img = cv2.resize(img, (BOARD_SIZE_PIXELS, BOARD_SIZE_PIXELS)) 

    segmented_squares = {}
    board_size = BOARD_SIZE_PIXELS // SQUARE_SIZE # 8

    # Loop through the 8x8 grid: r (rows 0-7, top-to-bottom), c (cols 0-7, left-to-right)
    for r in range(board_size):
        for c in range(board_size):
            
            x_start = c * SQUARE_SIZE
            x_end = (c + 1) * SQUARE_SIZE
            y_start = r * SQUARE_SIZE
            y_end = (r + 1) * SQUARE_SIZE
            
            square_crop = img[y_start:y_end, x_start:x_end]
            label = rc_to_algebraic(r, c) # 
            
            segmented_squares[label] = square_crop

    return segmented_squares

if __name__ == '__main__':
    # ... (Test script code omitted for brevity but should exist in your actual file) ...
    pass