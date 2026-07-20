"""
Non-blocking Text-To-Speech System using pyttsx3 and Queue Threading.
Prevents main UI frame lag and handles cooldown logic.
"""

import pyttsx3
import threading
import queue
import time
import logging

from config import TTS_RATE, TTS_VOLUME, TTS_COOLDOWN_SECONDS

logging.basicConfig(level=logging.INFO)

class SpeechEngine:
    def __init__(self):
        """Initialize worker thread, audio queue, and TTS cooldown tracking."""
        self.speech_queue = queue.Queue()
        self.history = {}  # Format: {phrase: last_spoken_timestamp}
        self.running = True

        # Start dedicated speech thread
        self.thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.thread.start()

    def _speech_worker(self):
        """Worker thread loop. Keeps pyttsx3 isolated from OpenCV main thread."""
        try:
            # Initialize pyttsx3 inside thread to prevent COM concurrency crashes on Windows
            engine = pyttsx3.init()
            engine.setProperty('rate', TTS_RATE)
            engine.setProperty('volume', TTS_VOLUME)
        except Exception as e:
            logging.error(f"Failed to initialize pyttsx3 engine: {e}")
            return

        while self.running:
            try:
                # Wait for text from queue (timeout allows checking self.running flag)
                text = self.speech_queue.get(timeout=0.2)
                if text:
                    engine.say(text)
                    engine.runAndWait()
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in TTS execution worker: {e}")

    def speak(self, text: str, force: bool = False):
        """
        Public method to queue speech output with cooldown check.
        
        Args:
            text (str): Phrasing to be spoken.
            force (bool): If True, bypasses cooldown checks.
        """
        if not text or not text.strip():
            return

        text = text.strip()
        current_time = time.time()
        last_time = self.history.get(text, 0)

        # Check cooldown constraint
        if force or (current_time - last_time >= TTS_COOLDOWN_SECONDS):
            self.history[text] = current_time
            # Keep queue short (clear stale backlog)
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    break
            self.speech_queue.put(text)

    def stop(self):
        """Gracefully terminate worker thread."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)