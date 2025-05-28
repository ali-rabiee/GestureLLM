import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
from datetime import datetime
from config import USER_ID, ACTIONS

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
        
        # Use centralized ACTIONS list
        self.actions = ACTIONS
        
        # Create data directory
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.user_data_dir = os.path.join("gesture_data", USER_ID)
        os.makedirs(self.user_data_dir, exist_ok=True)
        print(f"Created data directory at: {self.user_data_dir}")
            
        # Load or create dataset info
        self.info_file = os.path.join(self.user_data_dir, "dataset_info.json")
        if os.path.exists(self.info_file):
            with open(self.info_file, 'r') as f:
                self.dataset_info = json.load(f)
        else:
            self.dataset_info = {"samples": {}, "action_map": {}}

    def collect_data(self):
        print("\nWelcome to Gesture Data Collection!")
        print(f"Data will be saved to: {self.user_data_dir}")
        print("\nAll prompts and options will appear in the webcam window.")
        print("Instructions:")
        print("- Press 'r' to record a new gesture for the current action")
        print("- Press 'u' to reuse a gesture from another action")
        print("- Press 's' to skip the current action")
        print("- Press 'q' to quit\n")
        
        recorded_actions = {}
        action_idx = 0
        total_actions = len(self.actions)
        while action_idx < total_actions:
            action_id, action_name = self.actions[action_idx]
            state = 'await_choice'  # 'await_choice', 'record', 'reuse_select', 'done', 'quit'
            reuse_selected = None
            message = ""
            while True:
                success, image = self.cap.read()
                if not success:
                    continue
                image = cv2.flip(image, 1)
                overlay = image.copy()
                y0 = 30
                dy = 40
                # Draw current action info
                cv2.putText(overlay, f"Action {action_id+1}/{total_actions}: {action_name}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                y = y0 + dy
                if state == 'await_choice':
                    cv2.putText(overlay, "Press 'r' to record, 'u' to reuse, 's' to skip, 'q' to quit", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                    y += dy
                    if message:
                        cv2.putText(overlay, message, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                elif state == 'reuse_select':
                    cv2.putText(overlay, "Select action to reuse (press number key):", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                    y += dy
                    for idx, aid in enumerate(recorded_actions):
                        aname = dict(self.actions)[aid]
                        cv2.putText(overlay, f"{idx+1}: {aname}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)
                        y += dy
                    if message:
                        cv2.putText(overlay, message, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                elif state == 'record':
                    cv2.putText(overlay, "Recording gesture...", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    y += dy
                    cv2.putText(overlay, "Press SPACE to start/stop each variation", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                elif state == 'done':
                    cv2.putText(overlay, "Done! Moving to next action...", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                elif state == 'quit':
                    cv2.putText(overlay, "Data collection cancelled.", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.imshow('Gesture Data Collection', overlay)
                key = cv2.waitKey(1) & 0xFF
                if state == 'await_choice':
                    if key == ord('r'):
                        state = 'record'
                        message = ""
                    elif key == ord('u'):
                        if not recorded_actions:
                            message = "No gestures recorded yet. Record at least one first."
                        else:
                            state = 'reuse_select'
                            message = ""
                    elif key == ord('s'):
                        state = 'done'
                        message = f"Skipped: {action_name}"
                        self.dataset_info["action_map"][str(action_id)] = None
                    elif key == ord('q'):
                        state = 'quit'
                        break
                elif state == 'reuse_select':
                    if key in [ord(str(i+1)) for i in range(len(recorded_actions))]:
                        idx = int(chr(key)) - 1
                        reuse_id = list(recorded_actions.keys())[idx]
                        self.dataset_info["action_map"][str(action_id)] = reuse_id
                        message = f"Action {action_id} will reuse gesture from action {reuse_id}."
                        state = 'done'
                    elif key == ord('b'):
                        state = 'await_choice'
                        message = ""
                elif state == 'record':
                    ok = self._collect_gesture_variations(action_id, action_name)
                    if ok:
                        recorded_actions[action_id] = action_id
                        self.dataset_info["action_map"][str(action_id)] = action_id
                        message = f"Gesture recorded for action {action_id}."
                    else:
                        message = f"Recording cancelled for action {action_id}."
                    state = 'done'
                elif state == 'done':
                    cv2.waitKey(500)
                    break
                elif state == 'quit':
                    break
            if state == 'quit':
                break
            action_idx += 1
        
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
        
        action_dir = os.path.join(self.user_data_dir, str(action_id))
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
                return False
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

    def save_gesture_data(self, gesture_name, data):
        # Save gesture data to user-specific directory
        file_path = os.path.join(self.user_data_dir, f"{gesture_name}.json")
        with open(file_path, 'w') as f:
            json.dump(data, f)
        print(f"Saved gesture data for '{gesture_name}' to {file_path}")

if __name__ == "__main__":
    collector = GestureDataCollector()
    collector.collect_data() 