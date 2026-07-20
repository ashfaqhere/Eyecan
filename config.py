"""
Configuration Settings for EyeCan
Contains hardware inputs, class mappings, threshold settings, and HSV bounds.
"""

import os

# ==========================================
# CAMERA / INPUT CONFIGURATION
# ==========================================
# Set CAMERA_SOURCE to:
# 0 -> Laptop Built-in Webcam
# "http://192.168.1.X:4747/video" -> DroidCam (IP / Port)
# "http://192.168.1.X:8080/video" -> IP Webcam App
CAMERA_SOURCE = 0

# Processing frame size (resizing speeds up inference & OCR)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ==========================================
# YOLO DETECTOR CONFIGURATION
# ==========================================
YOLO_MODEL_PATH = "yolov8n.pt"  # Nano model for real-time speed
YOLO_CONF_THRESHOLD = 0.50

# Target classes mapping COCO ID -> Friendly Name
# COCO IDs: 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 9: traffic light, 56: chair
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    9: "traffic light",
    56: "chair"
}

# ==========================================
# TRAFFIC LIGHT CONFIGURATION (HSV)
# ==========================================
# HSV Ranges for Color Segmentation
# Red spans across two bands in OpenCV (0-10 and 160-180)
RED_LOWER1 = (0, 120, 120)
RED_UPPER1 = (10, 255, 255)
RED_LOWER2 = (160, 120, 120)
RED_UPPER2 = (180, 255, 255)

GREEN_LOWER = (40, 100, 100)
GREEN_UPPER = (90, 255, 255)

TRAFFIC_LIGHT_MIN_AREA = 400  # Minimum pixel area to filter noise

# ==========================================
# ZEBRA CROSSING CONFIGURATION
# ==========================================
ZEBRA_CANNY_LOW = 50
ZEBRA_CANNY_HIGH = 150
ZEBRA_RHO = 1
ZEBRA_THETA = 3.14159 / 180  # 1 degree in radians
ZEBRA_HOUGH_THRESHOLD = 50
ZEBRA_MIN_LINE_LENGTH = 80
ZEBRA_MAX_LINE_GAP = 15
ZEBRA_MIN_PARALLEL_STRIPES = 3  # Minimum parallel lines to validate crossing

# ==========================================
# OCR CONFIGURATION
# ==========================================
OCR_LANGUAGES = ['en']
OCR_CONF_THRESHOLD = 0.40
OCR_PROCESS_INTERVAL_FRAMES = 15  # Process OCR every 15 frames for performance

# ==========================================
# TEXT-TO-SPEECH CONFIGURATION
# ==========================================
TTS_RATE = 160           # Words per minute
TTS_VOLUME = 1.0         # 0.0 to 1.0
TTS_COOLDOWN_SECONDS = 4 # Time before repeating the exact same audio output