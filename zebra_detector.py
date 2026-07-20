"""
Zebra Crossing Detector utilizing Edge Detection, Region-of-Interest (ROI) masking,
and Probabilistic Hough Line Transformation.
"""

import cv2
import numpy as np
import math
import logging

from config import (
    ZEBRA_CANNY_LOW, ZEBRA_CANNY_HIGH, ZEBRA_RHO, ZEBRA_THETA,
    ZEBRA_HOUGH_THRESHOLD, ZEBRA_MIN_LINE_LENGTH, ZEBRA_MAX_LINE_GAP,
    ZEBRA_MIN_PARALLEL_STRIPES
)

logging.basicConfig(level=logging.INFO)

class ZebraCrossingDetector:
    def __init__(self):
        pass

    def detect(self, frame: np.ndarray):
        """
        Detect parallel zebra crossing lines in the lower half of the frame.
        
        Args:
            frame (np.ndarray): OpenCV BGR Frame.
            
        Returns:
            tuple: (annotated_frame, bool: is_detected)
        """
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        is_detected = False

        try:
            # Region of Interest: Lower 60% of frame (street level)
            roi_y_start = int(h * 0.4)
            roi = frame[roi_y_start:h, 0:w]

            # Preprocessing: Grayscale -> Gaussian Blur
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Canny edge detector
            edges = cv2.Canny(blur, ZEBRA_CANNY_LOW, ZEBRA_CANNY_HIGH)

            # Detect lines using HoughLinesP
            lines = cv2.HoughLinesP(
                edges,
                rho=ZEBRA_RHO,
                theta=ZEBRA_THETA,
                threshold=ZEBRA_HOUGH_THRESHOLD,
                minLineLength=ZEBRA_MIN_LINE_LENGTH,
                maxLineGap=ZEBRA_MAX_LINE_GAP
            )

            near_horizontal_lines = []

            if lines is not None:
                for line in lines:
                    # Robust array flattening to handle varying OpenCV array dimensions
                    coords = np.array(line).flatten()
                    if len(coords) == 4:
                        x1, y1, x2, y2 = coords
                        # Calculate angle with horizontal axis
                        angle = abs(math.degrees(math.atan2((y2 - y1), (x2 - x1))))
                        
                        # Filter near-horizontal lines (0° to 35° or 145° to 180°)
                        if angle < 35 or angle > 145:
                            near_horizontal_lines.append((int(x1), int(y1) + roi_y_start, int(x2), int(y2) + roi_y_start))

            # Evaluate line grouping for zebra pattern validation
            if len(near_horizontal_lines) >= ZEBRA_MIN_PARALLEL_STRIPES:
                is_detected = True
                for x1, y1, x2, y2 in near_horizontal_lines:
                    cv2.line(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                cv2.putText(
                    annotated_frame, "ZEBRA CROSSING DETECTED", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
                )

        except Exception as e:
            logging.error(f"Error in Zebra Crossing detection: {e}")

        return annotated_frame, is_detected