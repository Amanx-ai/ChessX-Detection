# F:\ChessXDetection\src\vision\detector.py

import cv2
import numpy as np
import math
import sys
import os

# Import configuration constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from config import BOARD_SIZE_PIXELS, SQUARE_SIZE

def detect_and_crop_board(image_path: str) -> np.ndarray or None:
    """
    Implements a robust, multi-check computer vision pipeline to find and
    perspective-correct a chessboard from an image.
    """
    try:
        img = cv2.imread(image_path)
        if img is None: return None

        # --- 1. Find the Brightest Object Candidate (Board Heuristic) ---
        # Isolates the board area based on brightness (L-channel) 
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)
        _, bright_mask = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = np.ones((15, 15), np.uint8)
        cleaned_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours: return None
        candidate_contour = max(contours, key=cv2.contourArea)
        candidate_mask = np.zeros_like(cleaned_mask)
        cv2.drawContours(candidate_mask, [candidate_contour], -1, 255, -1)
        x, y, w, h = cv2.boundingRect(candidate_contour)

        # --- 2. Grid and Square Validation (Justify it's a grid, not just a blob) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150) # Canny to find edges 
        masked_edges = cv2.bitwise_and(edges, edges, mask=candidate_mask)

        # Check for grid-like lines using Hough Transform
        lines = cv2.HoughLines(masked_edges[y:y+h, x:x+w], 1, np.pi / 180, threshold=int(min(w, h) * 0.2)) 
        grid_conf = min(1.0, (len(lines) / 30.0) if lines is not None else 0)

        # Check for density of contours (pieces/squares)
        sq_contours, _ = cv2.findContours(candidate_mask[y:y+h, x:x+w], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sq_conf = min(1.0, (len(sq_contours) - 20) / 45.0 if len(sq_contours) > 20 else 0)

        total_conf = (grid_conf + sq_conf) / 2
        if total_conf < 0.4: return None

        # --- 3. Precision Corner Detection using HoughLinesP ---
        final_lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 40, 50, 20)
        if final_lines is None: return None

        h_lines, v_lines = [], []
        for line in final_lines:
            x1,y1,x2,y2 = line[0]; angle = math.degrees(math.atan2(y2-y1,x2-x1))
            if -45 < angle < 45: h_lines.append(line)
            elif 45 < abs(angle) < 135: v_lines.append(line)

        if not h_lines or not v_lines: return None

        # Find boundary lines (Top, Bot, Left, Right)
        top = min(h_lines, key=lambda l:(l[0][1]+l[0][3])/2)
        bot = max(h_lines, key=lambda l:(l[0][1]+l[0][3])/2)
        left = min(v_lines, key=lambda l:(l[0][0]+l[0][2])/2)
        right = max(v_lines, key=lambda l:(l[0][0]+l[0][2])/2)

        def intersect(l1,l2):
            x1,y1,x2,y2 = l1[0]; x3,y3,x4,y4 = l2[0]
            d = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            if d == 0: return None
            t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/d
            return [int(x1+t*(x2-x1)), int(y1+t*(y2-y2))]

        corners = [p for p in [intersect(top,left),intersect(top,right),intersect(bot,right),intersect(bot,left)] if p]
        if len(corners) != 4: return None

        # --- 4. Perspective Correction ---
        corners = np.array(corners,"f")
        rect = np.zeros((4,2),"f")
        s = corners.sum(1); rect[0] = corners[np.argmin(s)]; rect[2] = corners[np.argmax(s)]
        d = np.diff(corners,1); rect[1] = corners[np.argmin(d)]; rect[3] = corners[np.argmax(d)]
        
        # Destination points for a perfect square 
        dst = np.array([[0,0],[BOARD_SIZE_PIXELS-1,0],[BOARD_SIZE_PIXELS-1,BOARD_SIZE_PIXELS-1],[0,BOARD_SIZE_PIXELS-1]],"f") 
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (BOARD_SIZE_PIXELS, BOARD_SIZE_PIXELS))
        
        return warped
        
    except Exception as e:
        print(f"ERROR in detect_and_crop_board: {e}")
        return None

if __name__ == '__main__':
    # ... (Test script code omitted for brevity but should exist in your actual file) ...
    pass