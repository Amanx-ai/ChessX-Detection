import sys
import os
import cv2
import numpy as np
import chess
import time
from tkinter import filedialog, Tk

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
                             QHBoxLayout, QLabel, QMessageBox)
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QBrush, QPolygonF, QGuiApplication
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QRect, QPointF

# --- Import Core Modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from config import BOARD_SIZE_PIXELS, SQUARE_SIZE
from vision.detector import detect_and_crop_board
from recognition.classifier import get_fen_from_image, load_recognition_model
from ai.engine_worker import EngineWorker, setup_engine_thread

# --- GUI SETTINGS ---
BOARD_SIZE = BOARD_SIZE_PIXELS
MARGIN = 30 # Space for coordinates

class BoardWidget(QWidget):
    human_move_made = pyqtSignal(chess.Move)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.board = chess.Board()
        self.setFixedSize(BOARD_SIZE + MARGIN, BOARD_SIZE + MARGIN)
        self.piece_symbols = {'P':'♙','R':'♖','N':'♘','B':'♗','Q':'♕','K':'♔','p':'♟','r':'♜','n':'♞','b':'♝','q':'♛','k':'♚'}
        self.clicks = []; self.best_move_arrow = None; self.last_move = None
        
    def set_board(self, board):
        self.board = board; 
        self.last_move = board.peek() if board.move_stack else None; 
        self.update()
        
    def paintEvent(self, event):
        # ... (Full drawing logic for board, pieces, coordinates, and arrows) ...
        # This includes highlighting legal moves and the best move arrow.
        # Uses standard chess.square() and chess.square_file()/rank() methods.
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing); painter.fillRect(self.rect(), QColor(50, 50, 50)); painter.translate(MARGIN, 0)
        # (Rest of paintEvent logic omitted for brevity, assumes successful drawing)
        pass 
        
    def mousePressEvent(self, event):
        # ... (Full mouse click logic for selecting and making a move, including auto-promotion) ...
        # Uses chess.Move.from_uci() and board.legal_moves() to validate human input
        # (Full mousePressEvent logic omitted for brevity, assumes successful move validation)
        pass 


class MainWindow(QMainWindow):
    instance = None
    
    def __init__(self):
        super().__init__(); MainWindow.instance = self
        self.setWindowTitle("ChessX Vision Analyzer"); self.board = None; self.is_busy = False
        
        # Setup multi-threaded engine worker
        self.engine_thread, self.engine_worker = setup_engine_thread()
        self.engine_worker.result_ready.connect(self.handle_engine_result)
        self.engine_worker.error.connect(self.show_error_message)
        
        self.is_model_loaded = load_recognition_model() # Check if CNN is ready

        # --- UI Setup (omitted for brevity) ---
        # ... (Creates BoardWidget, buttons, status_label, and layout) ...
        self.board_widget = BoardWidget()
        self.status_label = QLabel()
        self.scan_file_button = QPushButton("Scan from File")
        
        # ... (Connect signals to slots) ...
        self.scan_file_button.clicked.connect(self.scan_new_game_from_file)
        self.board_widget.human_move_made.connect(self.handle_human_move)
        
        self.update_button_states()

    # --- UI/State Methods (omitted for brevity) ---
    def set_busy(self, busy: bool, message: str = ""):
        self.is_busy = busy
        # ... (Enables/Disables buttons based on busy state) ...
        pass
    def update_status(self):
        # ... (Updates status label based on model load state and game state) ...
        pass
    def show_error_message(self, message: str):
        # ... (Displays QMessageBox) ...
        pass

    # --- Scanning & Pipeline Methods ---
    def scan_new_game_from_file(self):
        if self.is_busy or not self.is_model_loaded: return
        root = Tk(); root.withdraw(); path = filedialog.askopenfilename()
        if path: self.process_scanned_image(path)
        
    def process_scanned_image(self, path: str):
        self.set_busy(True, "1/3. Detecting Board (CV)...")
        
        # 1. Vision Pipeline (Call the standalone detector module)
        cropped_img = detect_and_crop_board(path)
        
        if cropped_img is None: 
            self.show_error_message("Could not detect a valid chessboard.")
            self.set_busy(False); return
            
        self.set_busy(True, "2/3. Recognizing Pieces (CNN)...")
        
        # 2. Recognition Pipeline (Call the standalone classifier module)
        fen_pieces = get_fen_from_image(cropped_img, False) # is_flipped=False for standard orientation
        
        if fen_pieces is None:
            self.show_error_message("Piece recognition failed (check model/classes).")
            self.set_busy(False); return
            
        # 3. Setup Board State
        turn_choice = QMessageBox.question(self, "Set Turn", "Is it White's turn?", QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
        turn_char = " w " if turn_choice == QMessageBox.Yes else " b "
        
        # Full FEN must be constructed with castling/move data assumed
        full_fen = fen_pieces + turn_char + "KQkq - 0 1"
        try:
            self.board = chess.Board(full_fen)
            self.board_widget.set_board(self.board)
            self.run_engine_task()
        except ValueError:
            self.show_error_message(f"Invalid position FEN: {full_fen}")
            self.set_busy(False)
            
    # --- Game/Engine Methods ---
    def handle_human_move(self, move: chess.Move):
        # Handles a move from the board, updates state, and runs AI analysis
        if self.is_busy or not self.board: return
        self.board.push(move)
        self.board_widget.set_board(self.board)
        if self.board.is_game_over(): self.handle_game_over()
        else: self.run_engine_task()

    def run_engine_task(self):
        """Sends the current FEN to the worker thread for analysis."""
        if not self.board or self.board.is_game_over():
             self.set_busy(False); return
             
        self.set_busy(True, "3/3. Analyzing (Stockfish)...")
        self.engine_worker.task_received.emit(self.board.fen())

    def handle_engine_result(self, result: dict):
        """Receives the analysis result from the worker and updates the GUI."""
        if not self.board: self.set_busy(False); return
        best_move = result.get("best_move")
        self.board_widget.best_move_arrow = best_move
        self.board_widget.set_board(self.board)
        if self.board.is_game_over(): self.handle_game_over(); return
        self.set_busy(False)
        
    def handle_game_over(self):
        # ... (Logic for handling checkmate, stalemate, and game over display) ...
        pass
        
    # --- Board Control Methods (Takeback, Change Turn, etc. omitted for brevity) ---
    
    def closeEvent(self, event):
        print("Closing application..."); 
        self.set_busy(True, "Shutting down engine...")
        # Cleanly shut down the thread and engine
        self.engine_worker.quit_engine() 
        self.engine_thread.quit()
        self.engine_thread.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    if not load_recognition_model():
        QMessageBox.critical(None, "Fatal Error", "AI Model not found or failed to load. Please run 'python main.py label' and 'python main.py train' first.")
        sys.exit(1)
        
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()