"""
EyeCan Interactive Hackathon Pitch Dashboard
Integrates Live WebCam, YOLO, Traffic Light, Zebra Crossing, and TTS.
Run with: python -m streamlit run app.py
"""

import streamlit as st
import cv2
import time
import logging
from datetime import datetime

from config import (
    CAMERA_SOURCE, FRAME_WIDTH, FRAME_HEIGHT, OCR_PROCESS_INTERVAL_FRAMES
)
from speech import SpeechEngine
from yolo_detector import YoloDetector
from ocr_reader import OCRReader
from traffic_light import TrafficLightDetector
from zebra_detector import ZebraCrossingDetector

# Streamlit Page Setup
st.set_page_config(
    page_title="EyeCan | Vision Assistance System",
    page_icon="👁️",
    layout="wide"
)

# Custom High-Contrast Assistive UI CSS
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stApp { color: #FFFFFF; }
    .speech-box {
        background-color: #1E222D;
        padding: 18px;
        border-radius: 10px;
        font-size: 22px;
        font-weight: bold;
        color: #00E5FF;
        border-left: 6px solid #00E5FF;
        text-align: center;
        margin-bottom: 20px;
    }
    .log-box {
        background-color: #161922;
        padding: 12px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 14px;
        color: #A0AAB0;
        max-height: 300px;
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)

st.title("👁️ EyeCan — AI Vision Assistant for the Visually Impaired")
st.caption("Real-Time Spatial Perception • Signboard Reader • Assistive Audio Pipeline")

# Layout Columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("🔊 Audio Feedback Status")
    speech_placeholder = st.empty()
    speech_placeholder.markdown("<div class='speech-box'>🔊 System Ready...</div>", unsafe_allow_html=True)

    st.subheader("📊 Live Detection Logs")
    log_placeholder = st.empty()

# Sidebar Controls
st.sidebar.header("⚙️ Pitch Presentation Controls")
run_system = st.sidebar.toggle("Start Camera Feed", value=True)
mute_audio = st.sidebar.checkbox("Mute Audio", value=False)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Demo Strategy:** Point camera at signboards, red/green items, or step into frame to trigger real-time audio announcements.")

# Pipeline Initialization
@st.cache_resource
def load_modules():
    tts = SpeechEngine()
    yolo = YoloDetector()
    ocr = OCRReader()
    traffic = TrafficLightDetector()
    zebra = ZebraCrossingDetector()
    return tts, yolo, ocr, traffic, zebra

# Initialize session state for detection logs so they persist cleanly across re-renders
if "detection_logs" not in st.session_state:
    st.session_state.detection_logs = []

if run_system:
    tts, yolo, ocr, traffic, zebra = load_modules()
    
    # Speak ready prompt once on start
    if "spoken_welcome" not in st.session_state:
        if not mute_audio:
            tts.speak("EyeCan Dashboard Active", force=True)
        st.session_state.spoken_welcome = True

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    # Enable DirectShow or MSMF buffer optimization on Windows if using USB/DroidCam
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    cached_ocr_text = None
    cached_ocr_frame = None
    detected_objects = []
    
    # Keep track of last spoken message to avoid repeating TTS on every frame
    last_spoken_message = None

    try:
        while run_system:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from video source. Please check your camera index/DroidCam connection.")
                break

            frame_count += 1
            clean_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # 1. Run Detection Pipeline
            light_anno, light_status = traffic.detect(clean_frame)
            zebra_anno, is_zebra_detected = zebra.detect(clean_frame)

            if frame_count % 2 == 0:
                yolo_anno, detected_objects = yolo.detect(clean_frame)
            else:
                yolo_anno = clean_frame.copy()

            if frame_count % OCR_PROCESS_INTERVAL_FRAMES == 0:
                cached_ocr_frame, cached_ocr_text = ocr.read_text(clean_frame)

            # 2. Layer Overlays for Web Stream
            display_frame = yolo_anno.copy()

            if light_status:
                display_frame = cv2.addWeighted(display_frame, 0.7, light_anno, 0.3, 0)
            if is_zebra_detected:
                display_frame = cv2.addWeighted(display_frame, 0.7, zebra_anno, 0.3, 0)
            if cached_ocr_text and cached_ocr_frame is not None:
                display_frame = cv2.addWeighted(display_frame, 0.7, cached_ocr_frame, 0.3, 0)

            # 3. Speech Cascade & UI Status Update
            current_speech = None

            if light_status == "RED":
                current_speech = "Red light. Please wait."
            elif light_status == "GREEN":
                current_speech = "Green light. Safe to cross."
            elif is_zebra_detected:
                current_speech = "Zebra crossing detected ahead."
            elif cached_ocr_text:
                current_speech = f"Text detected: {cached_ocr_text}"
            elif len(detected_objects) > 0:
                first_obj = detected_objects[0]
                current_speech = f"{first_obj.capitalize()} ahead."

            if current_speech:
                # Only trigger TTS and logging if message changed or hasn't been spoken recently
                if current_speech != last_spoken_message:
                    if not mute_audio:
                        tts.speak(current_speech)
                    last_spoken_message = current_speech

                    # Update Log Feed
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_entry = f"[{timestamp}] {current_speech}"
                    
                    st.session_state.detection_logs.insert(0, log_entry)
                    if len(st.session_state.detection_logs) > 8:
                        st.session_state.detection_logs.pop()

                # Always update the UI card displaying current feedback status
                speech_placeholder.markdown(
                    f"<div class='speech-box'>🔊 {current_speech}</div>", 
                    unsafe_allow_html=True
                )

            # Render Log Box
            log_html = "<br>".join(st.session_state.detection_logs) if st.session_state.detection_logs else "Awaiting detections..."
            log_placeholder.markdown(f"<div class='log-box'>{log_html}</div>", unsafe_allow_html=True)

            # 4. Stream BGR Frame to Streamlit (Convert BGR to RGB)
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            time.sleep(0.01)

    finally:
        cap.release()
else:
    # Reset welcome voice flag when camera toggle is turned off
    if "spoken_welcome" in st.session_state:
        del st.session_state["spoken_welcome"]