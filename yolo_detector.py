"""
YOLOv8 Object Detection Module using Ultralytics.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging

from config import YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD, TARGET_CLASSES

logging.basicConfig(level=logging.INFO)

class YoloDetector:
    def __init__(self):
        """Load pretrained YOLOv8 object detection model."""
        try:
            logging.info(f"Loading YOLO model: {YOLO_MODEL_PATH}...")
            self.model = YOLO(YOLO_MODEL_PATH)
            self.target_ids = list(TARGET_CLASSES.keys())
        except Exception as e:
            logging.error(f"Error initializing YOLO model: {e}")
            raise e

    def detect(self, frame: np.ndarray):
        """
        Perform detection on input frame.
        
        Args:
            frame (np.ndarray): OpenCV BGR frame.
            
        Returns:
            tuple: (annotated_frame, list of detected object strings)
        """
        annotated_frame = frame.copy()
        detected_names = []

        try:
            # Perform prediction filtering only target class IDs
            results = self.model.predict(
                source=frame,
                conf=YOLO_CONF_THRESHOLD,
                classes=self.target_ids,
                verbose=False
            )

            if len(results) > 0:
                result = results[0]
                boxes = result.boxes

                for box in boxes:
                    # Get box coordinates, confidence, and class ID
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    label_name = TARGET_CLASSES.get(cls_id, "object")
                    detected_names.append(label_name)

                    # Draw bounding box and label text
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    display_text = f"{label_name.capitalize()} {conf:.2f}"
                    cv2.putText(
                        annotated_frame, display_text, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )

        except Exception as e:
            logging.error(f"Error during YOLO inference: {e}")

        # Unique list preserving order
        unique_detections = list(dict.fromkeys(detected_names))
        return annotated_frame, unique_detections