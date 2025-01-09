import torch
import cv2
import numpy as np
import time
import os
from playsound import playsound
from gtts import gTTS  # Google Text-to-Speech is used here

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use 'yolov5s'
cap = cv2.VideoCapture(0)  # Use 0 for default camera
SIZE_THRESHOLD = 400  # Minimum bounding box width or height to trigger alert message for person too close
CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence level to trigger alert(shown in the box itself)

alert_triggered = False  # To track if an alert was already triggered
last_alert_time = 0  # Timer to space alerts
ALERT_COOLDOWN = 5  # Minimum seconds between alerts

# Variables for FPS calculation
start_time = time.time()
fps_smooth = 0
alpha = 0.1  # Weight for FPS smoothing

# Object tracker
tracker = cv2.legacy.TrackerCSRT_create()
tracking_initialized = False  # To ensure tracker is initialized only once

# Function to generate and save alert audio
def generate_alert_audio(file_name, text):
    # Check if the file already exists with the same name
    if not os.path.exists(file_name):
        # Create the audio file only if it doesn't exist
        tts = gTTS(text=text, lang='en')
        tts.save(file_name)
        print(f"Audio file '{file_name}' generated.")
    else:
        print(f"Audio file '{file_name}' already exists, using the existing one.")

# Generate alert audio if not already present
alert_text = "Alert! Person too close to camera"
alert_audio_file = "alert.mp3"
generate_alert_audio(alert_audio_file, alert_text)

cv2.namedWindow("YOLOv5 Object Detection and Tracking", cv2.WINDOW_NORMAL) 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for better performance
    resized_frame = cv2.resize(frame, (640, 480))

    # Perform object detection with YOLOv5
    results = model(resized_frame)
    detections = results.xywh[0]  # Get detections from YOLOv5 (if any)

    # Initialize the tracker only once with the first detection
    if not tracking_initialized and len(detections) > 0:
        for detection in detections:
            x_center, y_center, width, height, conf, cls = detection.tolist()
            class_name = model.names[int(cls)]  # Get class name

            # Check if the detection is for a "person" with confidence and size conditions
            if class_name == "person" and conf >= CONFIDENCE_THRESHOLD and max(width, height) >= SIZE_THRESHOLD:
                x, y, w, h = int(x_center - width/2), int(y_center - height/2), int(width), int(height)
                tracker.init(resized_frame, (x, y, w, h))  # Initialize the tracker with bounding box
                tracking_initialized = True
                break  # Only initialize tracker once

    # Update the tracker and check if tracking is successful
    if tracking_initialized:
        success, bbox = tracker.update(resized_frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
        else:
            # Reset tracker if tracking fails
            tracking_initialized = False

    # Handle alert if person is detected close to the camera
    for detection in detections:
        x_center, y_center, width, height, conf, cls = detection.tolist()
        class_name = model.names[int(cls)]  # Get class name

        # Check if the detection is for a "person"
        if class_name == "person" and conf >= CONFIDENCE_THRESHOLD and max(width, height) >= SIZE_THRESHOLD:
            # Trigger alert only if cooldown has passed
            current_time = time.time()
            if not alert_triggered or (current_time - last_alert_time) >= ALERT_COOLDOWN:
                print("ALERT: Person detected close to the camera!")
                playsound(alert_audio_file)  # Play an audio alert (replace with your alert sound file)
                alert_triggered = True
                last_alert_time = current_time

    # Render YOLOv5 results (bounding boxes and labels)
    result_frame = np.squeeze(results.render())

    # Calculate and smooth FPS
    fps = 1 / (time.time() - start_time)
    fps_smooth = alpha * fps + (1 - alpha) * fps_smooth
    start_time = time.time()

    # Display FPS on the frame
    cv2.putText(result_frame, f"FPS: {fps_smooth:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.resizeWindow("YOLOv5 Object Detection and Tracking", resized_frame.shape[1], resized_frame.shape[0])
    # Show the result frame
    cv2.imshow("YOLOv5 Object Detection and Tracking", result_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the last output frame as an image
cv2.imwrite("output_frame.jpg", result_frame)

# Release resources
cap.release()
cv2.destroyAllWindows()