"""
Zebra Crossing Detector - False-Positive Proof Implementation
Filters out chair legs, furniture, doors, and blinds using Color + Geometry checks.
"""

import cv2
import numpy as np
import math
import logging

from config import (
    ZEBRA_CANNY_LOW, ZEBRA_CANNY_HIGH, ZEBRA_RHO, ZEBRA_THETA,
    ZEBRA_HOUGH_THRESHOLD, ZEBRA_MIN_LINE_LENGTH, ZEBRA_MAX_LINE_GAP
)

logging.basicConfig(level=logging.INFO)

class ZebraCrossingDetector:
    def __init__(self):
        pass

    def detect(self, frame: np.ndarray):
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        is_detected = False

        try:
            # 1. REGION OF INTEREST (ROI): Only inspect lower 45% of the camera screen (the road level)
            roi_y_start = int(h * 0.55)
            roi = frame[roi_y_start:h, 0:w]

            # 2. COLOR MASK: Isolate bright white pavement paint (Filters out gray/metallic shiny steel)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # High brightness (V >= 180), low saturation (S <= 50) = White paint only
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 50, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)

            # 3. EDGE DETECTION: Run edge filter strictly on white masked areas
            blur = cv2.GaussianBlur(white_mask, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)

            # 4. HOUGH LINE TRANSFORM: Detect long continuous straight edges
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=70,             # Requires stronger edge alignment
                minLineLength=100,        # Filters out short chair legs/bars
                maxLineGap=20
            )

            valid_horizontal_lines = []

            if lines is not None:
                for line in lines:
                    coords = np.array(line).flatten()
                    if len(coords) == 4:
                        x1, y1, x2, y2 = coords
                        
                        # Calculate line length and angle
                        length = math.hypot(x2 - x1, y2 - y1)
                        angle = abs(math.degrees(math.atan2((y2 - y1), (x2 - x1))))

                        # GEOMETRY FILTER:
                        # Real zebra stripes cross horizontally in front of the walker (angle between -20° and +20°)
                        # Vertical lines like chair legs (60° to 120°) are rejected.
                        if (angle < 20 or angle > 160) and length > 100:
                            real_y1 = int(y1) + roi_y_start
                            real_y2 = int(y2) + roi_y_start
                            valid_horizontal_lines.append((int(x1), real_y1, int(x2), real_y2))

            # 5. SPATIAL DENSITY CHECK: Must detect at least 4 horizontal parallel white stripes, vertically stacked
            if len(valid_horizontal_lines) >= 4:
                # Check line spacing along the vertical axis
                y_centers = sorted([ (line[1] + line[3]) // 2 for line in valid_horizontal_lines ])
                y_diffs = np.diff(y_centers)
                
                # Check for consistent spacing between stripes (10px to 80px apart)
                valid_gaps = [d for d in y_diffs if 10 <= d <= 80]

                if len(valid_gaps) >= 3:
                    is_detected = True
                    for x1, y1, x2, y2 in valid_horizontal_lines:
                        cv2.line(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                    cv2.putText(
                        annotated_frame, "ZEBRA CROSSING DETECTED", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
                    )

        except Exception as e:
            logging.error(f"Error in Zebra Crossing detection: {e}")

        return annotated_frame, is_detected