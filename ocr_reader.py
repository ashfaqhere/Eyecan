"""
Optical Character Recognition (OCR) Engine powered by EasyOCR.
Extracts readable text from signboards, store names, exit signs, etc.
"""

import cv2
import numpy as np
import easyocr
import logging

from config import OCR_LANGUAGES, OCR_CONF_THRESHOLD

logging.basicConfig(level=logging.INFO)

class OCRReader:
    def __init__(self):
        """Initialize EasyOCR Reader instance."""
        try:
            logging.info("Initializing EasyOCR Model (CPU)...")
            self.reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
        except Exception as e:
            logging.error(f"Failed to initialize EasyOCR: {e}")
            raise e

    def read_text(self, frame: np.ndarray):
        """
        Process frame and detect printed text.
        
        Args:
            frame (np.ndarray): OpenCV input frame.
            
        Returns:
            tuple: (annotated_frame, extracted_string or None)
        """
        annotated_frame = frame.copy()
        extracted_text_list = []

        try:
            # Convert frame to RGB format required by EasyOCR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.reader.readtext(rgb_frame)

            for (bbox, text, prob) in results:
                if prob >= OCR_CONF_THRESHOLD and len(text.strip()) > 2:
                    clean_text = text.strip()
                    extracted_text_list.append(clean_text)

                    # Polygon bounds conversion
                    pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                    
                    # Compute top-left corner for text rendering
                    top_left = tuple(map(int, bbox[0]))
                    cv2.putText(
                        annotated_frame, clean_text, (top_left[0], max(top_left[1] - 5, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                    )

        except Exception as e:
            logging.error(f"Error executing OCR reader: {e}")

        final_text = " ".join(extracted_text_list) if extracted_text_list else None
        return annotated_frame, final_text