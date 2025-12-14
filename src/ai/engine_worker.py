import chess
import chess.engine
import sys
import os
from PyQt5.QtCore import QThread, pyqtSignal, QObject

# Import configuration constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from config import STOCKFISH_EXECUTABLE

class EngineWorker(QObject):
    """
    QObject worker to run the Stockfish chess engine persistently in a separate thread.
    Uses Signals/Slots for safe communication with the main GUI thread.
    """
    result_ready = pyqtSignal(dict) # Sends best move back
    error = pyqtSignal(str)         # Sends error messages back
    task_received = pyqtSignal(str) # Receives new FEN to analyze

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = None
        # Connect the signal from the main thread to the worker's method
        self.task_received.connect(self.run_analysis)

    def start_engine(self):
        """Initializes the Stockfish engine process."""
        if not os.path.exists(STOCKFISH_EXECUTABLE):
            self.error.emit(f"Stockfish not found at {STOCKFISH_EXECUTABLE}")
            return
            
        try:
            # Pop open the Stockfish executable using the UCI protocol
            self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_EXECUTABLE)
            print("Stockfish engine started persistently.")
        except Exception as e:
            self.error.emit(f"Failed to start Stockfish engine: {e}")

    def run_analysis(self, fen: str):
        """Analyzes the given FEN and finds the best move (executed in the worker thread)."""
        if not self.engine:
            self.error.emit("Engine not running.")
            self.result_ready.emit({})
            return

        try:
            board = chess.Board(fen) # Load the position
        except ValueError:
            self.error.emit("Invalid FEN received.")
            self.result_ready.emit({})
            return

        if board.is_game_over():
            self.result_ready.emit({})
            return

        try:
            # Set a time limit for analysis to control responsiveness
            limit = chess.engine.Limit(time=0.5) 
            info = self.engine.analyse(board, limit)
            
            # Extract the Principal Variation (best move)
            best_move = info.get("pv", [None])[0] 
            
            # Send result back to the main thread
            self.result_ready.emit({"best_move": best_move})

        except chess.engine.EngineError as e:
            self.error.emit(f"Engine analysis error: {e}")
            self.result_ready.emit({})
            
    def quit_engine(self):
        """Properly shuts down the Stockfish process."""
        if self.engine:
            try:
                self.engine.quit()
            except Exception as e:
                print(f"Error quitting engine: {e}")

# Helper function for setting up the worker and thread
def setup_engine_thread():
    engine_thread = QThread()
    engine_worker = EngineWorker()
    engine_worker.moveToThread(engine_thread)
    engine_thread.started.connect(engine_worker.start_engine)
    engine_thread.start()
    return engine_thread, engine_worker