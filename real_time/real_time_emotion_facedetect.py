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

