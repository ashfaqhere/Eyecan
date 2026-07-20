"""
EyeCan - Main Application Engine
Pipeline Integration, Audio Priority Cascade, and Real-time UI Display.
"""

import cv2
import time
import logging

from config import (
    CAMERA_SOURCE, FRAME_WIDTH, FRAME_HEIGHT, OCR_PROCESS_INTERVAL_FRAMES
)
from speech import SpeechEngine
from yolo_detector import YoloDetector
from ocr_reader import OCRReader
from traffic_light import TrafficLightDetector
from zebra_detector import ZebraCrossingDetector

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Starting EyeCan Vision Assistance System...")

    # Initialize Hardware / Speech Engine
    tts = SpeechEngine()
    tts.speak("EyeCan System Initialized. Starting camera feed.", force=True)

    # Initialize Perception Modules
    try:
        yolo = YoloDetector()
        ocr = OCRReader()
        traffic = TrafficLightDetector()
        zebra = ZebraCrossingDetector()
    except Exception as e:
        logging.error(f"Critical error initializing vision modules: {e}")
        return

    # Initialize Video Capture Pipeline
    logging.info(f"Opening Video Capture Source: {CAMERA_SOURCE}")
    cap = cv2.VideoCapture(CAMERA_SOURCE)

    if not cap.isOpened():
        logging.error(f"Unable to access camera source: {CAMERA_SOURCE}")
        tts.speak("Error. Camera feed not accessible.")
        return

    frame_count = 0
    cached_ocr_text = None
    cached_ocr_frame = None

    cv2.namedWindow("EyeCan Assistance System", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab frame from stream.")
                time.sleep(0.1)
                continue

            frame_count += 1

            # Standardize frame resolution
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            display_frame = frame.copy()

            # ----------------------------------------------------
            # PERCEPTION INFERENCE STAGE
            # ----------------------------------------------------

            # 1. Traffic Light Detection
            display_frame, light_status = traffic.detect(display_frame)

            # 2. Zebra Crossing Detection
            display_frame, is_zebra_detected = zebra.detect(display_frame)

            # 3. OCR Reading (Throttled for performance execution)
            if frame_count % OCR_PROCESS_INTERVAL_FRAMES == 0:
                cached_ocr_frame, cached_ocr_text = ocr.read_text(frame)
            
            # Blend visual bounding polygons for OCR text overlay
            if cached_ocr_text and cached_ocr_frame is not None:
                display_frame = cv2.addWeighted(display_frame, 0.7, cached_ocr_frame, 0.3, 0)

            # 4. YOLO Object Detection
            display_frame, detected_objects = yolo.detect(display_frame)

            # ----------------------------------------------------
            # STRICT PRIORITY SPEECH CASCADE
            # ----------------------------------------------------
            # Priority 1: Traffic Light
            if light_status == "RED":
                tts.speak("Red light. Please wait.")
            elif light_status == "GREEN":
                tts.speak("Green light. Safe to cross.")

            # Priority 2: Zebra Crossing
            elif is_zebra_detected:
                tts.speak("Zebra crossing detected ahead.")

            # Priority 3: OCR Text Reading
            elif cached_ocr_text:
                if "EXIT" in cached_ocr_text.upper():
                    tts.speak("Exit sign detected.")
                else:
                    tts.speak(f"Text detected: {cached_ocr_text}")

            # Priority 4: Object Detection
            elif len(detected_objects) > 0:
                # Speak first 2 detected priority objects to avoid voice clutter
                for obj in detected_objects[:2]:
                    tts.speak(f"{obj.capitalize()} ahead.")

            # ----------------------------------------------------
            # RENDER OVERLAYS AND DASHBOARD
            # ----------------------------------------------------
            # Render Top Status Banner
            status_banner = f"TL: {light_status if light_status else 'None'} | Zebra: {'YES' if is_zebra_detected else 'NO'}"
            cv2.putText(
                display_frame, status_banner, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

            # Render frame window
            cv2.imshow("EyeCan Assistance System", display_frame)

            # Exit Keyboard Shortcut: 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Quit key received. Stopping system...")
                break

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        # Cleanup Resources
        cap.release()
        cv2.destroyAllWindows()
        tts.stop()
        logging.info("EyeCan System shut down successfully.")

if __name__ == "__main__":
    main()