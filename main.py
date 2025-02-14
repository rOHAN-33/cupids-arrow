import cv2
import mediapipe as mp
import numpy as np
import math
from datetime import datetime

class CupidsArrowGame:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils
        self.score = 0
        self.hearts = []
        self.arrows = []
        self.is_drawing_bow = False
        self.last_shot_time = datetime.now()
        self.cooldown = 1.0  
        self.heart_img = cv2.imread('heart.png', cv2.IMREAD_UNCHANGED)
        self.heart_img = cv2.resize(self.heart_img, (50, 50))
        
    def spawn_heart(self):
        if len(self.hearts) < 5:  
            heart = {
                'x': np.random.randint(100, 540),
                'y': np.random.randint(100, 380),
                'speed': np.random.randint(2, 5)
            }
            self.hearts.append(heart)
    
    def draw_bow(self, frame, right_elbow, right_wrist):
        if right_elbow and right_wrist:
            bow_center = (int(right_elbow[0]), int(right_elbow[1]))
            bow_radius = 150  
            cv2.ellipse(frame, bow_center, (bow_radius, bow_radius), 0, -45, 45, (255, 0, 255), 12)  
            cv2.ellipse(frame, bow_center, (bow_radius+5, bow_radius+5), 0, -45, 45, (200, 0, 200), 8)  
            cv2.line(frame, (int(right_elbow[0]), int(right_elbow[1])), (int(right_wrist[0]), int(right_wrist[1])), (200, 200, 255), 12)  
            cv2.line(frame, (int(right_elbow[0]), int(right_elbow[1])), (int(right_wrist[0]), int(right_wrist[1])), (255, 255, 255), 6)  
            distance = math.dist(right_elbow, right_wrist)
            draw_percentage = min(distance / 100.0, 1.0)
            return draw_percentage
        return 0
    
    def shoot_arrow(self, start_pos, direction):
        arrow = {
            'x': start_pos[0],
            'y': start_pos[1],
            'dx': direction[0] * 10,
            'dy': direction[1] * 10
        }
        self.arrows.append(arrow)
    
    def update_game_objects(self):
        for heart in self.hearts[:]:
            heart['x'] += math.sin(datetime.now().timestamp()) * heart['speed']
            if heart['x'] < 0 or heart['x'] > 640:
                self.hearts.remove(heart)
        
        for arrow in self.arrows[:]:
            arrow['x'] += arrow['dx']
            arrow['y'] += arrow['dy']
            
            for heart in self.hearts[:]:
                if math.dist([arrow['x'], arrow['y']], [heart['x'], heart['y']]) < 25:
                    self.hearts.remove(heart)
                    self.arrows.remove(arrow)
                    self.score += 10
                    break
            
            if (arrow['x'] < 0 or arrow['x'] > 640 or arrow['y'] < 0 or arrow['y'] > 480):
                self.arrows.remove(arrow)
    
    def show_welcome_screen(self):
        welcome_screen = np.zeros((480, 640, 3), dtype=np.uint8)
        animation_counter = 0
        while True:  
            screen = welcome_screen.copy()
            animation_counter = (animation_counter + 5) % 255
            
            title = "CUPID'S ARROW"
            cv2.putText(screen, title, (122, 152), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, animation_counter//2), 3)
            cv2.putText(screen, title, (120, 150), cv2.FONT_HERSHEY_TRIPLEX, 2, (animation_counter, 0, 255), 3)
            subtitle = "THE LOVE GAME"
            cv2.putText(screen, subtitle, (170, 220), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, animation_counter, animation_counter), 2)
            
            instructions = [
                "HOW TO PLAY:",
                "1. RAISE YOUR RIGHT ARM",
                "2. DRAW BACK TO SHOOT",
                "3. HIT THE HEARTS",
                "",
                "PRESS ENTER TO START" 
            ]
            
            for idx, text in enumerate(instructions):
                alpha = abs(math.sin(animation_counter/50 + idx/2)) * 255
                cv2.putText(screen, text, (180, 280 + idx*40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (alpha, alpha, alpha), 2)
            
            for y in range(0, 480, 2):
                screen[y:y+1, :] = screen[y:y+1, :] * 0.5
            
            cv2.imshow('Cupid\'s Arrow AR Game', screen)
            key = cv2.waitKey(50)  

            if key == 13:
                break
            
        for i in range(255, 0, -5):
            screen = welcome_screen.copy()
            screen = cv2.multiply(screen, np.array([i/255]))
            cv2.imshow('Cupid\'s Arrow AR Game', screen)
            cv2.waitKey(1)

    def run_game(self):
        self.show_welcome_screen()
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            overlay = frame.copy()
            
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    h, w, c = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(overlay, (cx, cy), 5, (0, 255, 0), -1)
                
                right_shoulder = (
                    results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * 640,
                    results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * 480
                )
                right_elbow = (
                    results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x * 640,
                    results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y * 480
                )
                right_wrist = (
                    results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x * 640,
                    results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y * 480
                )
                
                draw_percentage = self.draw_bow(frame, right_elbow, right_wrist)
                
                if (draw_percentage > 0.8 and not self.is_drawing_bow and 
                    (datetime.now() - self.last_shot_time).total_seconds() > self.cooldown):
                    direction = np.array([right_wrist[0] - right_elbow[0],
                                       right_wrist[1] - right_elbow[1]])
                    direction = direction / np.linalg.norm(direction)
                    self.shoot_arrow(right_shoulder, direction)
                    self.last_shot_time = datetime.now()
                
                self.is_drawing_bow = draw_percentage > 0.8
            
            if np.random.random() < 0.02:
                self.spawn_heart()
            
            self.update_game_objects()
            
            for heart in self.hearts:
                x, y = int(heart['x']), int(heart['y'])
                size = 40
                cv2.circle(overlay, (x-size//4, y), size//2, (0, 0, 255), -1)
                cv2.circle(overlay, (x+size//4, y), size//2, (0, 0, 255), -1)
                pts = np.array([[x-size//2, y+size//4], 
                              [x, y+size], 
                              [x+size//2, y+size//4]], np.int32)
                cv2.fillPoly(overlay, [pts], (0, 0, 255))
                cv2.circle(overlay, (x, y), size+10, (0, 0, 255), 4)
            
            for arrow in self.arrows:
                x, y = int(arrow['x']), int(arrow['y'])
                cv2.line(overlay,
                        (int(x - arrow['dx']*5), int(y - arrow['dy']*5)),
                        (x, y),
                        (0, 255, 255), 8)
                cv2.circle(overlay, (x, y), 10, (0, 255, 0), -1)
            
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            score_text = f'Score: {self.score}'
            cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
            cv2.putText(frame, score_text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
            
            cv2.imshow('Cupid\'s Arrow AR Game', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = CupidsArrowGame()
    game.run_game()
