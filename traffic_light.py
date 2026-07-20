"""
OpenCV HSV Color Mask Segmentation for Traffic Light State Detection.
Detects Red or Green light states.
"""

import cv2
import numpy as np
import logging

from config import (
    RED_LOWER1, RED_UPPER1, RED_LOWER2, RED_UPPER2,
    GREEN_LOWER, GREEN_UPPER, TRAFFIC_LIGHT_MIN_AREA
)

logging.basicConfig(level=logging.INFO)

class TrafficLightDetector:
    def __init__(self):
        pass

    def detect(self, frame: np.ndarray):
        """
        Process frame and output detected traffic light color status.
        
        Args:
            frame (np.ndarray): Input image array (BGR).
            
        Returns:
            tuple: (annotated_frame, status string or None)
        """
        annotated_frame = frame.copy()
        status = None

        try:
            # Convert to HSV color space and apply bilateral filter to reduce noise
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_filtered = cv2.bilateralFilter(hsv, d=9, sigmaColor=75, sigmaSpace=75)

            # Mask generation for RED
            mask_red1 = cv2.inRange(hsv_filtered, np.array(RED_LOWER1), np.array(RED_UPPER1))
            mask_red2 = cv2.inRange(hsv_filtered, np.array(RED_LOWER2), np.array(RED_UPPER2))
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            # Mask generation for GREEN
            mask_green = cv2.inRange(hsv_filtered, np.array(GREEN_LOWER), np.array(GREEN_UPPER))

            # Find contours
            contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            max_red_area = 0
            best_red_box = None
            for cnt in contours_red:
                area = cv2.contourArea(cnt)
                if area > max_red_area and area >= TRAFFIC_LIGHT_MIN_AREA:
                    max_red_area = area
                    best_red_box = cv2.boundingRect(cnt)

            max_green_area = 0
            best_green_box = None
            for cnt in contours_green:
                area = cv2.contourArea(cnt)
                if area > max_green_area and area >= TRAFFIC_LIGHT_MIN_AREA:
                    max_green_area = area
                    best_green_box = cv2.boundingRect(cnt)

            # Compare areas to avoid split triggers
            if max_red_area > max_green_area and best_red_box is not None:
                status = "RED"
                x, y, w, h = best_red_box
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(
                    annotated_frame, "TRAFFIC LIGHT: RED", (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
            elif max_green_area > max_red_area and best_green_box is not None:
                status = "GREEN"
                x, y, w, h = best_green_box
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(
                    annotated_frame, "TRAFFIC LIGHT: GREEN", (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

        except Exception as e:
            logging.error(f"Error in Traffic Light detection: {e}")

        return annotated_frame, status