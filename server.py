from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import base64

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)
hearts = []

@app.route('/')
def index():
    return "Cupid's Arrow Backend Running"

def generate_frame():
    global hearts
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[8]
                x, y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
                hearts.append((x, y))

        for x, y in hearts:
            cv2.circle(frame, (x, y), 20, (0, 0, 255), -1)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('stream_frame', {'image': frame_base64})

@socketio.on('click')
def handle_click(data):
    x, y = data['x'], data['y']
    hearts.append((x, y))

if __name__ == '__main__':
    socketio.start_background_task(target=generate_frame)
    socketio.run(app, host='0.0.0.0', port=5000)
