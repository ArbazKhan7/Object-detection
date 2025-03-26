import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model
model = YOLO("yolov8s.pt")  # Ensure this model is in the working directory

st.title("🔍 YOLOv8 Object Detection")
st.write("Upload an image to detect objects using YOLOv8.")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg","jfif"])

if uploaded_file is not None:
    # Convert uploaded file to an OpenCV image
    image = Image.open(uploaded_file)
    image_cv2 = np.array(image)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Run YOLO object detection
    results = model(image_cv2)

    # Process results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        classes = result.boxes.cls.cpu().numpy()  # Class labels
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores

        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            accuracy = f"{conf * 100:.2f}%"  # Confidence percentage

            # Draw bounding box
            cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display label + accuracy
            text = f"{label}: {accuracy}"
            cv2.putText(image_cv2, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert OpenCV image back to PIL format for Streamlit
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Detected Objects", use_column_width=True)

st.write("🔹 Model: YOLOv8 (Small Version) | 🔹 Upload an image to test.")
