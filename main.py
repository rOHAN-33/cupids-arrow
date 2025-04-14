






import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime

class CupidsArrowGame:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.score = 0
        self.figures = []  # Changed from hearts to figures
        self.cooldown = 1.0
        self.last_spawn_time = datetime.now()
        self.game_active = False
        self.loading_complete = False
        self.button_hover = False
        self.mouse_x, self.mouse_y = 0, 0
        
        # Load custom figure image
        self.figure_img = cv2.imread('chill gut.png', cv2.IMREAD_UNCHANGED)
        if self.figure_img is None:
            raise FileNotFoundError("Could not load 'chill_gut.png'. Please ensure it's in the same directory.")
        self.figure_size = 100  # Adjust size as needed

    def on_mouse(self, event, x, y, flags, param):
        """Mouse callback function to track mouse position"""
        self.mouse_x, self.mouse_y = x, y

    def spawn_figure(self, x, y):
        """Add a new figure at the specified position"""
        self.figures.append({'x': x, 'y': y})
        self.score += 1

    def detect_finger_tap(self, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                return int(index_finger_tip.x * 640), int(index_finger_tip.y * 480)
        return None

    def draw_gradient_background(self, frame, color1=(50, 50, 150), color2=(150, 50, 50)):
        """Draw a vertical gradient background"""
        height, width = frame.shape[:2]
        for y in range(height):
            ratio = y / height
            b = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            r = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            cv2.line(frame, (0, y), (width, y), (b, g, r), 1)

    def show_start_screen(self, frame):
        # Draw gradient background
        self.draw_gradient_background(frame, (30, 30, 60), (60, 30, 30))
        
        # Draw game title with shadow
        text = "Make The Figure"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.5
        thickness = 3
        
        # Shadow
        cv2.putText(frame, text, (123, 153), font, scale, (0, 0, 0), thickness + 2)
        # Main text
        cv2.putText(frame, text, (120, 150), font, scale, (0, 200, 255), thickness)
        
        # Draw animated button
        button_y = 250
        button_height = 70
        button_width = 240
        button_x = 320 - button_width//2
        
        # Check if mouse is over button
        self.button_hover = (button_x <= self.mouse_x <= button_x + button_width and 
                           button_y <= self.mouse_y <= button_y + button_height)
        
        # Button glow effect when hovered
        if self.button_hover:
            glow = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            cv2.circle(glow, (320, button_y + button_height//2), 
                       button_width//2 + 10, (0, 255, 255), -1)
            frame = cv2.addWeighted(frame, 1.0, glow, 0.3, 0)
        
        # Button
        cv2.rectangle(frame, (button_x, button_y),
                     (button_x + button_width, button_y + button_height),
                     (0, 180 + 50 * self.button_hover, 0), -1)
        
        # Button border
        cv2.rectangle(frame, (button_x, button_y),
                     (button_x + button_width, button_y + button_height),
                     (255, 255, 255), 2)
        
        # Button text
        cv2.putText(frame, "PRESS ENTER", (320 - 90, button_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return (button_x, button_y, button_x + button_width, button_y + button_height)

    def show_loading_screen(self, frame, progress):
        # Dark background
        self.draw_gradient_background(frame, (10, 10, 30), (30, 10, 10))
        
        # Loading text
        cv2.putText(frame, "LOADING...", (220, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255), 2)
        
        # Progress bar container
        bar_width = 400
        bar_height = 30
        bar_x = 320 - bar_width//2
        bar_y = 250
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        
        # Progress bar fill
        fill_width = int(bar_width * progress)
        gradient = np.linspace(0, 180, fill_width)
        for i in range(fill_width):
            color = (0, int(gradient[i]), 255 - int(gradient[i]))
            cv2.line(frame, (bar_x + i, bar_y), (bar_x + i, bar_y + bar_height), color, 1)
        
        # Percentage text
        cv2.putText(frame, f"{int(progress * 100)}%", (300, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Loading animation (rotating circle)
        if progress < 1.0:
            radius = 20
            center = (320, 370)
            angle = int(time.time() * 10) % 360
            cv2.ellipse(frame, center, (radius, radius), 0, angle - 30, angle + 210,
                        (0, 255, 255), 4)

    def draw_figure(self, frame, x, y):
        """Draw the custom figure image at specified position"""
        if self.figure_img is None:
            return
            
        # Resize image if needed
        img = cv2.resize(self.figure_img, (self.figure_size, self.figure_size))
        
        # Calculate position
        y_start = max(0, y - self.figure_size//2)
        y_end = min(frame.shape[0], y + self.figure_size//2)
        x_start = max(0, x - self.figure_size//2)
        x_end = min(frame.shape[1], x + self.figure_size//2)
        
        # Handle alpha channel if exists
        if img.shape[2] == 4:
            alpha = img[:, :, 3] / 255.0
            for c in range(0, 3):
                frame[y_start:y_end, x_start:x_end, c] = (
                    alpha * img[:, :, c] + 
                    (1 - alpha) * frame[y_start:y_end, x_start:x_end, c]
                )
        else:
            frame[y_start:y_end, x_start:x_end] = img

    def run_game(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Make The Figure")
        cv2.setMouseCallback("Make The Figure", self.on_mouse)
        
        # Start screen
        while not self.game_active:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            button_pos = self.show_start_screen(frame)
            
            cv2.imshow("Make The Figure", frame)
            
            key = cv2.waitKey(1)
            if key == 13:  # Enter key
                self.game_active = True
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        # Loading screen
        start_time = time.time()
        loading_duration = 3  # 3 seconds
        
        while not self.loading_complete:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            elapsed = time.time() - start_time
            progress = min(elapsed / loading_duration, 1.0)
            
            self.show_loading_screen(frame, progress)
            cv2.imshow("Make The Figure", frame)
            
            if progress >= 1.0:
                self.loading_complete = True
                time.sleep(0.5)  # Small delay before game starts
                cv2.destroyWindow("Make The Figure")
                break
            
            if cv2.waitKey(1) == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        # Main game loop
        cv2.namedWindow("Make The Figure - Game")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(rgb_frame)
            hand_results = self.hands.process(rgb_frame)
            
            tap_position = self.detect_finger_tap(hand_results)
            if tap_position and (datetime.now() - self.last_spawn_time).total_seconds() > self.cooldown:
                self.spawn_figure(*tap_position)
                self.last_spawn_time = datetime.now()
            
            # Draw all figures
            for figure in self.figures:
                self.draw_figure(frame, figure['x'], figure['y'])
            
            # Display score with fancy background
            cv2.rectangle(frame, (5, 5), (300, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"Figures: {self.score}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Make The Figure - Game", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = CupidsArrowGame()
    game.run_game()