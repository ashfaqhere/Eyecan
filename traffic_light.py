"""
OpenCV HSV Color & Shape Filter for Traffic Light State Detection.
Strictly filters out red clothes, walls, and non-glowing objects.
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
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        status = None

        try:
            # 1. UPPER ROI ONLY: Traffic lights live overhead in the top 50% of the frame
            roi_h = int(h * 0.50)
            roi = frame[0:roi_h, 0:w]

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hsv_filtered = cv2.bilateralFilter(hsv, d=9, sigmaColor=75, sigmaSpace=75)

            # 2. BRIGHT GLOWING RED MASK: Require high brightness (V >= 180) to ignore dull red shirts/objects
            red_lower1_bright = np.array([0, 100, 180])
            red_upper1_bright = np.array([10, 255, 255])
            red_lower2_bright = np.array([170, 100, 180])
            red_upper2_bright = np.array([180, 255, 255])

            mask_red1 = cv2.inRange(hsv_filtered, red_lower1_bright, red_upper1_bright)
            mask_red2 = cv2.inRange(hsv_filtered, red_lower2_bright, red_upper2_bright)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            # Bright Green Mask
            green_lower_bright = np.array([40, 100, 180])
            green_upper_bright = np.array([90, 255, 255])
            mask_green = cv2.inRange(hsv_filtered, green_lower_bright, green_upper_bright)

            # Clean up noise with morphological open/close
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
            mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

            contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            max_red_area = 0
            best_red_box = None
            for cnt in contours_red:
                area = cv2.contourArea(cnt)
                # Cap max area so large red objects (like red shirts) are ignored
                if 200 <= area <= 15000:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect_ratio = float(bw) / bh
                    
                    # 3. CIRCULARITY / COMPACTNESS CHECK: Traffic light lamps are nearly 1:1 ratio
                    if 0.7 <= aspect_ratio <= 1.3:
                        if area > max_red_area:
                            max_red_area = area
                            best_red_box = (x, y, bw, bh)

            max_green_area = 0
            best_green_box = None
            for cnt in contours_green:
                area = cv2.contourArea(cnt)
                if 200 <= area <= 15000:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect_ratio = float(bw) / bh
                    if 0.7 <= aspect_ratio <= 1.3:
                        if area > max_green_area:
                            max_green_area = area
                            best_green_box = (x, y, bw, bh)

            # Evaluate strongest match
            if max_red_area > max_green_area and best_red_box is not None:
                status = "RED"
                x, y, bw, bh = best_red_box
                cv2.rectangle(annotated_frame, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
                cv2.putText(
                    annotated_frame, "TRAFFIC LIGHT: RED", (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
            elif max_green_area > max_red_area and best_green_box is not None:
                status = "GREEN"
                x, y, bw, bh = best_green_box
                cv2.rectangle(annotated_frame, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
                cv2.putText(
                    annotated_frame, "TRAFFIC LIGHT: GREEN", (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

        except Exception as e:
            logging.error(f"Error in Traffic Light detection: {e}")

        return annotated_frame, status