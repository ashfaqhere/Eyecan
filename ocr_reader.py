"""
EyeCan OCR Reader Module - Optimized for Signboard & Text Reading
"""

import cv2
import easyocr
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class OCRReader:
    def __init__(self, gpu=False):
        # Initialize EasyOCR reader for English
        logging.info("Initializing EasyOCR Engine...")
        self.reader = easyocr.Reader(['en'], gpu=gpu)

    def read_text(self, frame: np.ndarray):
        annotated_frame = frame.copy()
        clean_extracted_words = []

        try:
            # 1. Preprocess: Convert to Grayscale & Contrast Boost for outdoor signs
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 2. Run EasyOCR Detection
            results = self.reader.readtext(gray)

            for (bbox, text, prob) in results:
                # Only process confident text readings (> 35% confidence)
                if prob > 0.35:
                    cleaned_word = text.strip()
                    # Ignore random tiny noise (require at least 2 characters)
                    if len(cleaned_word) >= 2:
                        clean_extracted_words.append(cleaned_word)

                        # Draw bounding box on frame
                        (top_left, top_right, bottom_right, bottom_left) = bbox
                        top_left = (int(top_left[0]), int(top_left[1]))
                        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

                        cv2.rectangle(annotated_frame, top_left, bottom_right, (0, 255, 0), 2)
                        cv2.putText(
                            annotated_frame, cleaned_word, (top_left[0], max(top_left[1] - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                        )

        except Exception as e:
            logging.error(f"Error executing OCR reader: {e}")

        final_text = " ".join(clean_extracted_words) if clean_extracted_words else None
        return annotated_frame, final_text