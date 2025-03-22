





import cv2
import mediapipe as mp
import numpy as np
import math
from datetime import datetime

class CupidsArrowGame:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils
        self.score = 0
        self.hearts = []
        self.cooldown = 1.0  
        self.last_spawn_time = datetime.now()
    
    def spawn_heart(self, x, y):
        self.hearts.append({'x': x, 'y': y})
    
    def detect_finger_tap(self, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                return int(index_finger_tip.x * 640), int(index_finger_tip.y * 480)
        return None
    
    def run_game(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(rgb_frame)
            hand_results = self.hands.process(rgb_frame)
            overlay = frame.copy()
            
            tap_position = self.detect_finger_tap(hand_results)
            if tap_position and (datetime.now() - self.last_spawn_time).total_seconds() > self.cooldown:
                self.spawn_heart(*tap_position)
                self.last_spawn_time = datetime.now()
            
            for heart in self.hearts:
                x, y = heart['x'], heart['y']
                size = 40
                cv2.circle(overlay, (x-size//4, y), size//2, (0, 0, 255), -1)
                cv2.circle(overlay, (x+size//4, y), size//2, (0, 0, 255), -1)
                pts = np.array([[x-size//2, y+size//4], 
                                [x, y+size], 
                                [x+size//2, y+size//4]], np.int32)
                cv2.fillPoly(overlay, [pts], (0, 0, 255))
                
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.imshow("Cupid's Arrow AR Game", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = CupidsArrowGame()
    game.run_game()




