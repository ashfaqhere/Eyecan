"""
EyeCan - Main Engine with High-Priority Text Reading
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

    tts = SpeechEngine()
    tts.speak("EyeCan System Initialized. Starting camera feed.", force=True)

    try:
        yolo = YoloDetector()
        ocr = OCRReader()
        traffic = TrafficLightDetector()
        zebra = ZebraCrossingDetector()
    except Exception as e:
        logging.error(f"Critical error initializing vision modules: {e}")
        return

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        logging.error(f"Unable to access camera source: {CAMERA_SOURCE}")
        tts.speak("Error. Camera feed not accessible.", force=True)
        return

    frame_count = 0
    cached_ocr_text = None
    cached_ocr_frame = None
    detected_objects = []

    cv2.namedWindow("EyeCan Assistance System", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1
            clean_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            
            # 1. RUN DETECTORS
            light_anno, light_status = traffic.detect(clean_frame)
            zebra_anno, is_zebra_detected = zebra.detect(clean_frame)

            if frame_count % 2 == 0:
                yolo_anno, detected_objects = yolo.detect(clean_frame)
            else:
                yolo_anno = clean_frame.copy()

            # Run OCR scanning every few frames
            if frame_count % OCR_PROCESS_INTERVAL_FRAMES == 0:
                cached_ocr_frame, cached_ocr_text = ocr.read_text(clean_frame)

            # 2. COMBINE OVERLAYS
            display_frame = yolo_anno.copy()

            if light_status:
                display_frame = cv2.addWeighted(display_frame, 0.7, light_anno, 0.3, 0)

            if is_zebra_detected:
                display_frame = cv2.addWeighted(display_frame, 0.7, zebra_anno, 0.3, 0)

            if cached_ocr_text and cached_ocr_frame is not None:
                display_frame = cv2.addWeighted(display_frame, 0.7, cached_ocr_frame, 0.3, 0)

            # ----------------------------------------------------
            # 3. SPEECH PRIORITY (Safety > Text > Objects)
            # ----------------------------------------------------
            if light_status == "RED":
                tts.speak("Red light. Please wait.")
            elif light_status == "GREEN":
                tts.speak("Green light. Safe to cross.")
            elif is_zebra_detected:
                tts.speak("Zebra crossing detected ahead.")
            # HIGH PRIORITY: Read signboards/text out loud!
            elif cached_ocr_text:
                tts.speak(f"Text detected: {cached_ocr_text}")
            elif len(detected_objects) > 0:
                first_obj = detected_objects[0]
                tts.speak(f"{first_obj.capitalize()} ahead.")

            # Display Status Banner
            status_banner = f"TL: {light_status if light_status else 'None'} | Zebra: {'YES' if is_zebra_detected else 'NO'}"
            cv2.putText(
                display_frame, status_banner, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

            cv2.imshow("EyeCan Assistance System", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts.stop()

if __name__ == "__main__":
    main()