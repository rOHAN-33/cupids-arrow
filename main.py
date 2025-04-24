














import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

class InteractiveFigureGame:
    def __init__(self):
        # Initialize mediapipe components
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        
        # Game state variables
        self.score = 0
        self.figures = []
        self.cooldown = 1.0
        self.last_spawn_time = datetime.now()
        self.game_active = False
        self.loading_complete = False
        self.button_hover = False
        self.mouse_x, self.mouse_y = 0, 0
        self.custom_img = None
        self.figure_size = 100
        
        # UI elements
        self.select_img_button = (200, 350, 440, 400)
        self.start_button = (200, 420, 440, 470)
        self.default_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(self.default_img, "No Image", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def load_image(self, path):
        """Load image with transparency support"""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            # Convert BGRA to RGBA if needed
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
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
        """Start screen with image selection"""
        self.draw_gradient_background(frame)
        
        # Title
        cv2.putText(frame, "MAKE THE FIGURE", (120, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)
        
        # Selected image preview area
        preview_size = 150
        start_x = frame.shape[1]//2 - preview_size//2
        start_y = 150
        
        # Display current image or placeholder
        if self.custom_img is not None:
            preview = cv2.resize(self.custom_img, (preview_size, preview_size))
            if preview.shape[2] == 4:  # With alpha channel
                alpha = preview[:, :, 3] / 255.0
                for c in range(3):
                    frame[start_y:start_y+preview_size, start_x:start_x+preview_size, c] = \
                        alpha * preview[:, :, c] + \
                        (1 - alpha) * frame[start_y:start_y+preview_size, start_x:start_x+preview_size, c]
            else:
                frame[start_y:start_y+preview_size, start_x:start_x+preview_size] = preview
        else:
            placeholder = cv2.resize(self.default_img, (preview_size, preview_size))
            frame[start_y:start_y+preview_size, start_x:start_x+preview_size] = placeholder
        
        # "Select Image" button
        btn_x1, btn_y1, btn_x2, btn_y2 = self.select_img_button
        btn_color = (0, 120, 255) if not self.button_hover else (0, 150, 255)
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), btn_color, -1)
        cv2.putText(frame, "SELECT IMAGE", (btn_x1 + 20, btn_y1 + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # "Start Game" button
        btn_x1, btn_y1, btn_x2, btn_y2 = self.start_button
        btn_color = (0, 180, 0) if self.custom_img is not None else (100, 100, 100)
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), btn_color, -1)
        cv2.putText(frame, "START GAME", (btn_x1 + 30, btn_y1 + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def check_button_hover(self):
        """Check if mouse is hovering over buttons"""
        x, y = self.mouse_x, self.mouse_y
        btn_x1, btn_y1, btn_x2, btn_y2 = self.select_img_button
        self.button_hover = (btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2)

    def handle_click(self, x, y):
        """Handle mouse clicks in start screen"""
        # Check "Select Image" button
        btn_x1, btn_y1, btn_x2, btn_y2 = self.select_img_button
        if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
            path = self.open_file_dialog()
            if path:
                self.custom_img = self.load_image(path)
        
        # Check "Start Game" button
        btn_x1, btn_y1, btn_x2, btn_y2 = self.start_button
        if self.custom_img is not None and (btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2):
            self.game_active = True

    def open_file_dialog(self):
        """Create a simple file dialog"""
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        return file_path

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

    def show_loading_screen(self, frame, progress):
        """Loading screen with progress bar"""
        self.draw_gradient_background(frame, (10, 10, 30), (30, 10, 10))
        
        # Loading text
        cv2.putText(frame, "LOADING...", (220, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255), 2)
        
        # Progress bar
        bar_width = 400
        bar_height = 30
        bar_x = frame.shape[1]//2 - bar_width//2
        bar_y = 250
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        
        fill_width = int(bar_width * progress)
        gradient = np.linspace(0, 180, fill_width)
        for i in range(fill_width):
            color = (0, int(gradient[i]), 255 - int(gradient[i]))
            cv2.line(frame, (bar_x + i, bar_y), (bar_x + i, bar_y + bar_height), color, 1)
        
        # Percentage
        cv2.putText(frame, f"{int(progress * 100)}%", (frame.shape[1]//2 - 30, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Spinner
        if progress < 1.0:
            radius = 20
            center = (frame.shape[1]//2, 370)
            angle = int(time.time() * 10) % 360
            cv2.ellipse(frame, center, (radius, radius), 0, angle - 30, angle + 210,
                        (0, 255, 255), 4)

    def draw_figure(self, frame, x, y):
        """Draw the custom image with transparency support"""
        if self.custom_img is None:
            return
            
        img = cv2.resize(self.custom_img, (self.figure_size, self.figure_size))
        h, w = img.shape[:2]
        
        y1, y2 = max(0, y - h//2), min(frame.shape[0], y + h//2)
        x1, x2 = max(0, x - w//2), min(frame.shape[1], x + w//2)
        
        if img.shape[2] == 4:
            alpha = img[:, :, 3] / 255.0
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    alpha * img[:, :, c] + 
                    (1 - alpha) * frame[y1:y2, x1:x2, c]
                )
        else:
            frame[y1:y2, x1:x2] = img

    def on_mouse(self, event, x, y, flags, param):
        """Handle mouse events"""
        self.mouse_x, self.mouse_y = x, y
        self.check_button_hover()
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.handle_click(x, y)

    def run_game(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Make The Figure")
        cv2.setMouseCallback("Make The Figure", self.on_mouse)
        
        # Start screen loop
        while not self.game_active:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            self.show_start_screen(frame)
            
            cv2.imshow("Make The Figure", frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        # Loading screen
        start_time = time.time()
        loading_duration = 3
        
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
                time.sleep(0.5)
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
            hand_results = self.hands.process(rgb_frame)
            
            tap_position = self.detect_finger_tap(hand_results)
            if tap_position and (datetime.now() - self.last_spawn_time).total_seconds() > self.cooldown:
                self.spawn_figure(*tap_position)
                self.last_spawn_time = datetime.now()
            
            # Draw all figures
            for figure in self.figures:
                self.draw_figure(frame, figure['x'], figure['y'])
            
            # Display score
            cv2.rectangle(frame, (5, 5), (250, 50), (0, 0, 0, 0.7), -1)
            cv2.putText(frame, f"Figures: {self.score}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Make The Figure - Game", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = InteractiveFigureGame()
    game.run_game()