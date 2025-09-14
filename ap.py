import cv2
import numpy as np
import time
import threading
import platform
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Function to play system beep sound
def play_system_beep():
    try:
        if platform.system() == 'Windows':
            import winsound
            winsound.Beep(800, 500)
        else:
            os.system('echo -e "\\a"')
    except:
        print("\a🚨 DROWSINESS ALERT! 🚨")

# Video Transformer Class for Drowsiness Detection
class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.CLOSED_EYES_FRAME_THRESHOLD = 15
        self.eyes_closed_frames = 0
        self.alarm_on = False
        self.last_alarm_time = 0
        self.eye_state_history = []
        self.history_length = 10
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def detect_eyes_closed(self, face_roi):
        eyes = self.eye_cascade.detectMultiScale(
            face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(10,10), maxSize=(50,50))
        return len(eyes), eyes

    def trigger_alarm(self):
        current_time = time.time()
        if not self.alarm_on and (current_time - self.last_alarm_time) > 3:
            self.alarm_on = True
            self.last_alarm_time = current_time
            threading.Thread(target=play_system_beep, daemon=True).start()

    def reset_alarm(self):
        current_time = time.time()
        if self.alarm_on and (current_time - self.last_alarm_time) > 2:
            self.alarm_on = False
            self.eyes_closed_frames = 0

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

        drowsiness_detected = False

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_roi = gray[y:y + h, x:x + w]

            eye_count, eyes = self.detect_eyes_closed(face_roi)

            self.eye_state_history.append(eye_count)
            if len(self.eye_state_history) > self.history_length:
                self.eye_state_history.pop(0)

            if len(self.eye_state_history) >= 5:
                recent_avg = sum(self.eye_state_history[-5:]) / 5
                if recent_avg < 1.0:
                    self.eyes_closed_frames += 1
                else:
                    self.eyes_closed_frames = max(0, self.eyes_closed_frames - 2)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

            if self.eyes_closed_frames > self.CLOSED_EYES_FRAME_THRESHOLD:
                drowsiness_detected = True
                self.trigger_alarm()

        if not drowsiness_detected:
            self.reset_alarm()

        alert_text = "🚨 DROWSINESS ALERT!" if self.alarm_on else "😊 Monitoring"
        alert_color = (0, 0, 255) if self.alarm_on else (0, 255, 0)
        cv2.putText(image, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 3)

        return image

# Streamlit App UI
st.title("🚗 Driver Drowsiness Detection System")

webrtc_streamer(
    key="drowsiness-detector",
    video_transformer_factory=DrowsinessTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
