# Cupid's Arrow

Cupid's Arrow is an interactive augmented reality (AR) game that uses OpenCV and MediaPipe to track hand movements and generate heart animations when a user taps their finger.

## Features
- Hand tracking using MediaPipe
- Finger tap detection to spawn heart animations
- Real-time camera processing with OpenCV
- Server integration with Flask & Socket.IO for streaming frames

## Technologies Used
- Python
- OpenCV
- MediaPipe
- Flask
- Flask-SocketIO
- NumPy

## Installation
1. **Clone the Repository**
   ```sh
   git clone https://github.com/your-repo/cupids-arrow.git
   cd cupids-arrow
   ```

2. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Game**
   ```sh
   python main.py
   ```

4. **Run the Server (Optional for Streaming)**
   ```sh
   python server.py
   ```

## Usage
- Launch the game (`main.py`), and the webcam will activate.
- Use your index finger to tap in the air to spawn hearts.
- Press `q` to exit.

## Project Structure
```
📂 cupids-arrow
├── main.py          # Runs the AR game
├── server.py        # Flask server for streaming
├── heart.png        # Heart image for rendering
├── README.md        # Documentation
└── requirements.txt # Dependencies
```

## Dependencies
Make sure you have Python 3 installed. The dependencies include:
- OpenCV (`cv2`)
- MediaPipe
- Flask
- Flask-SocketIO
- NumPy

Install them using:
```sh
pip install opencv-python mediapipe flask flask-socketio numpy
```

## How It Works
- The camera captures frames using OpenCV.
- MediaPipe detects hand landmarks and tracks the index finger.
- A heart is spawned at the fingertip location when a tap is detected.
- The game overlays animated hearts on the video feed.
- The Flask server allows real-time streaming of frames.


"# cupid-arrow" 
