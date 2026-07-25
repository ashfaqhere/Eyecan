from ultralytics import YOLO

# 1. Load your YOLOv8 PyTorch weights file
model = YOLO('yolov8n.pt')

# 2. Export the model to TFLite format
# (Note: format="litert" or "tflite" works)
model.export(format='litert')
print("Model export completed successfully!")