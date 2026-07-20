# EyeCan – AI-Powered Vision Assistance System

EyeCan is a real-time computer vision system built to assist visually impaired individuals in navigating environments safely.

## Features
1. **YOLOv8 Object Detection**: Identifies nearby people, vehicles, obstacles, and furniture.
2. **Traffic Light Color Detection**: Recognizes Red and Green traffic signals via HSV space segmentation.
3. **Zebra Crossing Detection**: Uses Canny Edge Detection and Hough Transforms to identify pedestrian crossings.
4. **Optical Character Recognition (OCR)**: Reads signboards, shop names, and exit indicators.
5. **Non-blocking TTS Engine**: Delivers priority audio cues using `pyttsx3` without dropping camera video frames.

---

## Setup & Execution Instructions

### 1. Installation

1. Open PowerShell or Terminal in the project directory.
2. Create and activate a virtual environment (optional, but recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate