import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
from datetime import datetime

class GestureDataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        
        # Actions to collect data for
        self.actions = {
            "Translation Mode": {
                8: "Forward movement",
                2: "Backward movement",
                4: "Left movement",
                6: "Right movement",
                7: "Up movement",
                1: "Down movement"
            },
            "Orientation Mode": {
                8: "Rotate X+",
                2: "Rotate X-",
                4: "Rotate Z+",
                6: "Rotate Z-",
                7: "Rotate Y+",
                1: "Rotate Y-"
            },
            "Gripper Mode": {
                8: "Open gripper",
                2: "Close gripper"
            }
        }
        
        # Create data directory
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "gesture_data")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory at: {self.data_dir}")
            
        # Load or create dataset info
        self.info_file = os.path.join(self.data_dir, "dataset_info.json")
        if os.path.exists(self.info_file):
            with open(self.info_file, 'r') as f:
                self.dataset_info = json.load(f)
        else:
            self.dataset_info = {"samples": {}}

    def collect_data(self):
        """Collect gesture data with variations"""
        print("\nWelcome to Gesture Data Collection!")
        print(f"Data will be saved to: {self.data_dir}")
        print("\nFor each gesture, we'll collect multiple variations.")
        print("Instructions:")
        print("- Press SPACE to start recording a variation")
        print("- Move your hand slightly during recording to create variations")
        print("- Press ESC to skip a gesture")
        print("- Press Q to quit\n")
        
        for mode, actions in self.actions.items():
            print(f"\n=== {mode} ===")
            for action_id, action_name in actions.items():
                if not self._collect_gesture_variations(action_id, action_name):
                    break
        
        # Save dataset info
        with open(self.info_file, 'w') as f:
            json.dump(self.dataset_info, f, indent=2)
            print(f"\nSaved dataset info to: {self.info_file}")
        
        self.cleanup()
        
    def _collect_gesture_variations(self, action_id, action_name, num_variations=30, frames_per_variation=10):
        """Collect multiple variations of a gesture"""
        print(f"\nCollecting data for: {action_name}")
        print(f"Please show {num_variations} variations of the gesture")
        
        variation_count = 0
        frame_count = 0
        recording = False
        current_frames = []
        
        action_dir = os.path.join(self.data_dir, str(action_id))
        if not os.path.exists(action_dir):
            os.makedirs(action_dir)
            print(f"Created directory for action {action_id}: {action_dir}")
        
        while variation_count < num_variations:
            success, image = self.cap.read()
            if not success:
                continue
                
            # Flip the image horizontally
            image = cv2.flip(image, 1)
            
            # Process the image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            # Create overlay
            overlay = image.copy()
            
            # Draw hand landmarks if detected
            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        overlay, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    if recording:
                        # Extract features
                        features = self._extract_features(hand_landmarks)
                        current_frames.append(features)
                        frame_count += 1
                        
                        if frame_count >= frames_per_variation:
                            # Save the variation
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"var_{variation_count}_{timestamp}.npy"
                            filepath = os.path.join(action_dir, filename)
                            np.save(filepath, np.array(current_frames))
                            
                            # Store relative path in dataset info
                            rel_path = os.path.join(str(action_id), filename)
                            self.dataset_info["samples"][rel_path] = {
                                "action_id": action_id,
                                "action_name": action_name,
                                "variation": variation_count
                            }
                            
                            # Reset for next variation
                            current_frames = []
                            frame_count = 0
                            recording = False
                            variation_count += 1
                            print(f"Variation {variation_count}/{num_variations} captured")
            
            # Display status
            cv2.putText(overlay, f"Action: {action_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(overlay, f"Variation: {variation_count}/{num_variations}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if recording:
                cv2.putText(overlay, f"Recording frames: {frame_count}/{frames_per_variation}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(overlay, "Press SPACE to record variation", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Show hand detection status
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.putText(overlay, "Hand Detected" if hand_detected else "No Hand Detected",
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Show the image
            cv2.imshow('Gesture Data Collection', overlay)
            
            # Handle key presses
            key = cv2.waitKey(1)
            if key == ord(' '):  # Space key
                if not recording and hand_detected:
                    recording = True
                    current_frames = []
                    frame_count = 0
            elif key == 27:  # ESC key
                print(f"Skipped: {action_name}")
                return True
            elif key == ord('q'):  # Q key
                print("\nData collection process cancelled.")
                return False
        
        return True
        
    def _extract_features(self, hand_landmarks):
        """Extract features from hand landmarks"""
        features = []
        
        # Extract normalized 3D coordinates
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(features)
        
    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nData collection completed!")
        print(f"Total samples collected: {len(self.dataset_info['samples'])}")

if __name__ == "__main__":
    collector = GestureDataCollector()
    collector.collect_data() 