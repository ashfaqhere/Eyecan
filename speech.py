"""
EyeCan Native Windows Speech Engine - Smart Duplicate Suppression
"""

import threading
import queue
import time
import logging
import win32com.client
import pythoncom

from config import TTS_COOLDOWN_SECONDS

logging.basicConfig(level=logging.INFO)

class SpeechEngine:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.last_spoken_text = None
        self.last_spoken_time = 0
        self.running = True

        # Start isolated worker thread
        self.thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.thread.start()

    def _speech_worker(self):
        """Worker thread using Windows Native SAPI with COM initialization."""
        pythoncom.CoInitialize()

        try:
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
        except Exception as e:
            logging.error(f"Failed to initialize Windows SAPI Voice: {e}")
            speaker = None

        while self.running:
            try:
                text = self.speech_queue.get(timeout=0.1)
                if text and speaker:
                    print(f"=== [SPEAKING NOW]: {text} ===")
                    speaker.Speak(text, 0)

                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in TTS worker: {e}")
        
        pythoncom.CoUninitialize()

    def speak(self, text: str, force: bool = False):
        """
        Only speaks if:
        1. It's a NEW phrase (different from the last spoken phrase), OR
        2. 'force=True' is passed, OR
        3. A extended re-announce interval has elapsed (e.g., 10 seconds).
        """
        if not text or not text.strip():
            return

        text = text.strip()
        current_time = time.time()

        # Re-announce the same static object only after 10 seconds, but announce new objects instantly
        repeat_interval = 10.0 if not force else 0.0

        if (text != self.last_spoken_text) or (current_time - self.last_spoken_time >= repeat_interval):
            self.last_spoken_text = text
            self.last_spoken_time = current_time

            # Clear backlog queue
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    break

            self.speech_queue.put(text)

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)