# F:\ChessXDetection\src\vision\segmenter.py

import cv2
import numpy as np
import sys
import os

# Import configuration constants from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from config import BOARD_SIZE_PIXELS, SQUARE_SIZE

def rc_to_algebraic(r: int, c: int) -> str:
    """
    Converts (row, col) index (0-7, 0-7) to standard algebraic notation (e.g., 0, 0 -> 'a8').
    Assumes A8 is Top-Left (r=0, c=0).
    """
    # 0, 1, 2, ... 7 maps to 'a', 'b', 'c', ... 'h'
    file_char = chr(ord('a') + c) 
    
    # 0 maps to rank 8, 7 maps to rank 1
    rank_num = BOARD_SIZE_PIXELS // SQUARE_SIZE - r
    
    return f"{file_char}{rank_num}"

def segment_board(warped_image_path: str) -> dict:
    """
    Loads the flattened board image and segments it into 64 labeled crops.
    
    Args:
        warped_image_path: Path to the perspective-corrected board image.
        
    Returns:
        A dictionary mapping square label ('a1') to the cropped image (np.array).
    """
    img = cv2.imread(warped_image_path)
    if img is None:
        raise FileNotFoundError(f"Segmenter: Could not load warped image at {warped_image_path}")
        
    # Resize check (safety measure)
    if img.shape[0] != BOARD_SIZE_PIXELS or img.shape[1] != BOARD_SIZE_PIXELS:
        img = cv2.resize(img, (BOARD_SIZE_PIXELS, BOARD_SIZE_PIXELS)) 

    segmented_squares = {}
    board_size = BOARD_SIZE_PIXELS // SQUARE_SIZE # Should be 8

    # Loop through the 8x8 grid: r (rows 0-7), c (cols 0-7)
    for r in range(board_size):
        for c in range(board_size):
            
            # Calculate pixel boundaries
            x_start = c * SQUARE_SIZE
            x_end = (c + 1) * SQUARE_SIZE
            y_start = r * SQUARE_SIZE
            y_end = (r + 1) * SQUARE_SIZE
            
            # Crop the square using NumPy array slicing 
            # Syntax: img[y_start:y_end, x_start:x_end]
            square_crop = img[y_start:y_end, x_start:x_end]
            
            # Get the algebraic label (a8, b8, ..., h1)
            label = rc_to_algebraic(r, c)
            
            segmented_squares[label] = square_crop

    return segmented_squares

if __name__ == '__main__':
    # Simple test to verify mapping and slicing
    test_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)), "data", "processed", "flattened_board.jpg")
    
    if not os.path.exists(test_path):
        print("ERROR: Please run detector.py first to create a flattened_board.jpg in data/processed/.")
    else:
        try:
            squares = segment_board(test_path)
            print(f"Segmented {len(squares)} squares. Sample keys: {list(squares.keys())[:5]}")
            
            # Display a sample square to confirm size and content
            cv2.imshow('Sample Square (a1)', squares['a1'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("Segmenter test passed.")
            
        except FileNotFoundError as e:
            print(e)