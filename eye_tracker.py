import cv2
import mediapipe as mp
import numpy as np
import time
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import messagebox

class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize variables for blink detection
        self.blink_counter = 0
        self.blink_start_time = time.time()
        self.BLINK_THRESHOLD = 0.3  # Threshold for eye aspect ratio to consider as blink
        self.BLINK_RATE_THRESHOLD = 30  # Blinks per minute threshold for fatigue
        
        # Eye landmarks indices for MediaPipe Face Mesh
        # Left eye landmarks
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.LEFT_IRIS = [474, 475, 476, 477]
        # Right eye landmarks
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Initialize the camera
        self.cap = cv2.VideoCapture(0)
        
        # Setup GUI
        self.setup_gui()
        
        # Show welcome message
        messagebox.showinfo("Eye Tracking System", "Welcome to the Eye Tracking System!\n\nThis application will track your eye movements and monitor your fatigue level.\n\nPlease ensure good lighting and position yourself comfortably in front of the camera.")
        
    def setup_gui(self):
        self.root = ctk.CTk()
        self.root.title("Eye Tracking System")
        self.root.geometry("1200x800")
        
        # Create frames
        self.video_frame = ctk.CTkFrame(self.root)
        self.video_frame.pack(side="left", padx=10, pady=10)
        
        self.info_frame = ctk.CTkFrame(self.root)
        self.info_frame.pack(side="right", padx=10, pady=10, fill="y")
        
        # Create video label
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack()
        
        # Create info labels
        self.gaze_label = ctk.CTkLabel(self.info_frame, text="Gaze Direction: ")
        self.gaze_label.pack(pady=10)
        
        self.blink_rate_label = ctk.CTkLabel(self.info_frame, text="Blink Rate: 0 bpm")
        self.blink_rate_label.pack(pady=10)
        
        self.fatigue_label = ctk.CTkLabel(self.info_frame, text="Fatigue Status: Normal")
        self.fatigue_label.pack(pady=10)
        
    def calculate_ear(self, eye_points):
        """Calculate eye aspect ratio"""
        height1 = np.linalg.norm(eye_points[1] - eye_points[5])
        height2 = np.linalg.norm(eye_points[2] - eye_points[4])
        width = np.linalg.norm(eye_points[0] - eye_points[3])
        ear = (height1 + height2) / (2.0 * width)
        return ear
    
    def detect_gaze_direction(self, landmarks, frame):
        """Detect gaze direction based on iris position relative to eye corners"""
        frame_height, frame_width = frame.shape[:2]
        
        def get_normalized_coords(landmark_index):
            return np.array([
                landmarks.landmark[landmark_index].x * frame_width,
                landmarks.landmark[landmark_index].y * frame_height
            ])
        
        # Get left eye landmarks
        left_iris_center = get_normalized_coords(self.LEFT_IRIS[0])
        left_eye_left = get_normalized_coords(self.LEFT_EYE[0])
        left_eye_right = get_normalized_coords(self.LEFT_EYE[3])
        
        # Get right eye landmarks
        right_iris_center = get_normalized_coords(self.RIGHT_IRIS[0])
        right_eye_left = get_normalized_coords(self.RIGHT_EYE[0])
        right_eye_right = get_normalized_coords(self.RIGHT_EYE[3])
        
        # Calculate relative position for both eyes
        def get_gaze_ratio(iris_center, eye_left, eye_right):
            eye_width = np.linalg.norm(eye_right - eye_left)
            if eye_width == 0:
                return 0.5
            iris_pos = (iris_center[0] - eye_left[0]) / eye_width
            return iris_pos
        
        left_ratio = get_gaze_ratio(left_iris_center, left_eye_left, left_eye_right)
        right_ratio = get_gaze_ratio(right_iris_center, right_eye_left, right_eye_right)
        
        # Average the ratios from both eyes
        avg_ratio = (left_ratio + right_ratio) / 2
        
        # Determine vertical gaze
        left_eye_top = get_normalized_coords(self.LEFT_EYE[1])
        left_eye_bottom = get_normalized_coords(self.LEFT_EYE[5])
        vertical_ratio = (left_iris_center[1] - left_eye_top[1]) / (left_eye_bottom[1] - left_eye_top[1])
        
        # Thresholds for gaze direction
        if avg_ratio < 0.35:
            return "Left"
        elif avg_ratio > 0.65:
            return "Right"
        elif vertical_ratio < 0.35:
            return "Up"
        elif vertical_ratio > 0.65:
            return "Down"
        else:
            return "Center"
    
    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Get eye landmarks
            left_eye_points = np.array([[landmarks.landmark[i].x * frame.shape[1],
                                       landmarks.landmark[i].y * frame.shape[0]]
                                      for i in self.LEFT_EYE])
            right_eye_points = np.array([[landmarks.landmark[i].x * frame.shape[1],
                                        landmarks.landmark[i].y * frame.shape[0]]
                                       for i in self.RIGHT_EYE])
            
            # Calculate EAR
            left_ear = self.calculate_ear(left_eye_points)
            right_ear = self.calculate_ear(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Detect blink
            if avg_ear < self.BLINK_THRESHOLD:
                self.blink_counter += 1
            
            # Calculate blink rate (blinks per minute)
            elapsed_time = time.time() - self.blink_start_time
            if elapsed_time >= 60:  # Update every minute
                blink_rate = (self.blink_counter / elapsed_time) * 60
                self.blink_rate_label.configure(text=f"Blink Rate: {blink_rate:.1f} bpm")
                
                # Check for fatigue
                if blink_rate > self.BLINK_RATE_THRESHOLD:
                    self.fatigue_label.configure(text="Fatigue Status: FATIGUE DETECTED!", 
                                              text_color="red")
                    messagebox.showwarning("Fatigue Warning", "High blink rate detected! Please take a break.")
                else:
                    self.fatigue_label.configure(text="Fatigue Status: Normal",
                                              text_color="white")
                
                # Reset counters
                self.blink_counter = 0
                self.blink_start_time = time.time()
            
            # Detect gaze direction
            gaze_direction = self.detect_gaze_direction(landmarks, frame)
            self.gaze_label.configure(text=f"Gaze Direction: {gaze_direction}")
            
            # Draw eye landmarks and iris
            for eye_points in [left_eye_points, right_eye_points]:
                for point in eye_points:
                    cv2.circle(frame, tuple(point.astype(int)), 2, (0, 255, 0), -1)
            
            # Draw iris landmarks
            for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
                x = int(landmarks.landmark[idx].x * frame.shape[1])
                y = int(landmarks.landmark[idx].y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        
        # Convert frame to PhotoImage
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo
        
        # Schedule the next update
        self.root.after(10, self.update)
    
    def run(self):
        self.update()
        self.root.mainloop()
    
    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    tracker = EyeTracker()
    tracker.run()
