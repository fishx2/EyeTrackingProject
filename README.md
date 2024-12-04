# Eye Tracking System
# author: Tuan Tran (Fishx2)
A Python-based eye tracking system that can detect gaze direction, monitor blink rate, and identify potential fatigue.

## Features

- Real-time eye tracking using webcam
- Gaze direction detection (Left, Right, Up, Down)
- Blink rate monitoring
- Fatigue detection based on blink frequency
- Modern GUI interface using CustomTkinter

## Requirements

- Python 3.7+
- Webcam

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python eye_tracker.py
```

The application will open your webcam and start tracking your eyes. The GUI displays:
- Live video feed with eye tracking visualization
- Current gaze direction
- Blink rate (blinks per minute)
- Fatigue status warning

## How it Works

- Uses MediaPipe Face Mesh for facial landmark detection
- Calculates Eye Aspect Ratio (EAR) for blink detection
- Determines gaze direction based on iris position relative to eye center
- Monitors blink rate over time to detect potential fatigue

## Notes

- Ensure good lighting conditions for optimal tracking
- Position yourself at a comfortable distance from the webcam
- The fatigue warning triggers if blink rate exceeds 30 blinks per minute
