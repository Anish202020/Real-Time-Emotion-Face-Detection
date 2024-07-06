# Real-Time Emotion Detection Website
## Overview
This documentation provides a comprehensive guide to building a real-time emotion detection website using Streamlit and various Python libraries. The website analyzes facial expressions via a camera to detect emotions in real-time.

<img src="https://i.ibb.co/x5bmYWp/Screenshot-119.png" alt="Screenshot-119" border="0">
## Purpose
The primary purpose of this website is to detect emotions in real-time using facial expressions captured via a camera.

## Prerequisites
Before setting up the development environment, ensure you have the following:

- Python installed
- Required libraries installed:
    - `opencv-python` 
    - `tensorflow` 
    - `keras` 
    - `fer` 
- Streamlit installed
- A webcam or camera
## Setup Instructions
### Step 1: Install Python and Required Libraries
Ensure Python is installed on your system. Install the required libraries using pip:

```sh
pip install opencv-python tensorflow keras fer
```
### Step 2: Set Up Streamlit
Streamlit is used to create the web interface for real-time emotion detection. Ensure Streamlit is installed:

```sh
pip install streamlit
```
### Step 3: Configure the Webcam
Ensure your webcam or camera is properly configured and accessible by your system.

### Step 4: Run the Streamlit Application
Create a Python script (e.g., `real_time_emotion_facedetect.py`) with the necessary code to set up the Streamlit application and integrate the emotion detection functionality. Run the application using:

```sh
streamlit run real_time_emotion_facedetect.py
```
## Code Structure
Provide an overview of the code structure and key components of the application.

### Example Code Snippet
```python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st
import cv2
from fer import FER
import numpy as np
import uuid  # Import UUID library to generate unique keys

# Initialize the FER emotion detector with MTCNN for face detection
try:
    detector = FER(mtcnn=True)
except Exception as e:
    st.error("Error initializing FER with MTCNN: {}".format(e))
    st.stop()

# Streamlit app setup
st.title('Real-time Emotion Detection')
st.write('This application detects faces and their emotions in real-time using your webcam.')
st.title('Team Members')
st.write('ANISH KUMAR 1AY21CS028')
st.write('ADITYA KHATRIYA 1AY21CS018')
st.write('ADITYA JYOTI SAHU 1AY21CS017')
st.write('ADITYA ARUN KUMAR 1AY21CS016')


# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error: Could not open webcam.")
    st.stop()

# Function to draw text with background
def draw_text(frame, text, x, y):
    font_scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = x
    text_y = y - text_size[1]
    cv2.rectangle(frame, (text_x, text_y), (text_x + text_size[0], text_y + text_size[1]), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (text_x, y), font, font_scale, (255, 255, 255), font_thickness)

# Streamlit video capture and processing
frame_window = st.image([])


while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Could not read frame from webcam.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = detector.detect_emotions(rgb_frame)

    for face in result:
        (x, y, w, h) = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        dominant_emotion = max(face['emotions'], key=face['emotions'].get)
        score = face['emotions'][dominant_emotion]

        draw_text(frame, f'{dominant_emotion} ({score:.2f})', x, y - 10)

    frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Generate a unique key for each button instance using UUID
   

cap.release()
```
## Conclusion
This documentation provides the necessary steps to set up and run a real-time emotion detection website using Streamlit and various Python libraries. Follow the setup instructions and use the provided code structure to build and customize your application.



### Key Points:
- **FER Initialization**: The FER library is initialized with the `mtcnn=True`  parameter to use the MTCNN face detector, which is more robust.
- **Webcam Initialization**: The webcam is initialized using OpenCV's `VideoCapture` .
- **Text Drawing Function**: The `draw_text`  function is defined to draw text with a background rectangle on the video frames.
- **Main Loop**:
    - The webcam captures each frame.
    - The frame is converted from BGR to RGB since the FER library requires RGB input.
    - Emotions are detected in the frame using the FER library.
    - For each detected face, a rectangle is drawn around the face, and the dominant emotion is displayed on the frame.
- **Exit Condition**: Pressing the 'q' key will exit the loop and release the webcam resources.








