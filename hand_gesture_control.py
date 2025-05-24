import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
import pickle
from gesture_model import GestureModelTrainer

class HandGestureController:
    def __init__(self, mode="default"):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        
        # Control states
        self.mode = 0  # 0: Translation, 1: Orientation, 2: Gripper
        self.state = -1
        self.prev_state = -1
        self.modes = ['Translation', 'Orientation', 'Gripper']
        
        # Gesture mode and data
        self.gesture_mode = mode
        self.custom_gestures = {}
        self.gesture_images = {}
        self.available_gestures = set()  # Track which gestures are available
        self.debug_mode = True  # Enable debug output
        self.load_custom_gestures()
        
        # Initialize model if using AI mode
        if self.gesture_mode == "ai":
            try:
                print("\nInitializing AI gesture recognition model...")
                self.model_trainer = GestureModelTrainer()
                self.model_trainer.load_trained_model()
                print("AI model initialized successfully!")
                
                # Load label mapping if available
                self.label_mapping_file = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "gesture_data",
                    "label_mapping.json"
                )
                if os.path.exists(self.label_mapping_file):
                    with open(self.label_mapping_file, 'r') as f:
                        self.label_mapping = json.load(f)
                    print("\nGesture mappings loaded:")
                    for original, encoded in sorted(self.label_mapping.items()):
                        print(f"Action {original}: {self.get_action_description(int(original))}")
            except Exception as e:
                print(f"\nError initializing AI mode: {e}")
                print("Please make sure you have collected data and trained the model first:")
                print("1. python main.py --collect-data")
                print("2. python main.py --train-model")
                raise
        
        # Gesture detection parameters
        self.prev_gesture_time = time.time()
        self.gesture_cooldown = 0.1  # Reduced from 0.2
        self.gesture_hold_time = 0.3  # Reduced from 0.5
        self.mode_switch_hold_time = 1.0
        self.current_gesture_start = 0
        self.current_gesture = None
        self.mode_switch_start = 0
        self.switching_mode = False
        
        # Frame buffer for gesture recognition
        self.frame_buffer = []
        self.buffer_size = 10
        self.sliding_window_step = 3  # Reduced from 5
        self.prediction_cooldown = 0.05  # Reduced from 0.1
        self.last_prediction_time = 0
        self.gesture_confidence_threshold = 2  # Reduced from 3
        self.recent_predictions = []
        self.max_recent_predictions = 5
        
        # Add gesture state tracking
        self.active_gesture = None
        self.active_gesture_start = 0
        self.gesture_timeout = 0.5  # How long to maintain gesture without new detection
        self.last_active_time = 0
        
        # Gesture comparison parameters
        self.comparison_threshold = 0.2  # Increased threshold for more lenient matching

    def load_custom_gestures(self):
        """Load custom gestures if they exist"""
        if os.path.exists('custom_gestures.pkl'):
            try:
                with open('custom_gestures.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.custom_gestures = data.get('gestures', {})
                    self.gesture_images = data.get('images', {})
                    # Update available gestures
                    self.available_gestures = set(int(k) for k in self.custom_gestures.keys())
                    print("\nLoaded custom gestures:")
                    print("Available actions:", sorted(list(self.available_gestures)))
                    print(f"Number of gestures loaded: {len(self.custom_gestures)}")
                    if self.debug_mode:
                        for action in sorted(self.available_gestures):
                            print(f"Action {action}: Feature shape = {self.custom_gestures[str(action)].shape}")
            except Exception as e:
                print(f"Error loading custom gestures: {e}")
                self.custom_gestures = {}
                self.gesture_images = {}
                self.available_gestures = set()

    def save_custom_gestures(self):
        """Save custom gestures to file"""
        with open('custom_gestures.pkl', 'wb') as f:
            pickle.dump({
                'gestures': self.custom_gestures,
                'images': self.gesture_images
            }, f)

    def get_hand_landmarks_features(self, hand_landmarks):
        """Extract features from hand landmarks for gesture recognition"""
        features = []
        
        # Extract normalized 3D coordinates in a consistent order
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(features)

    def compare_gestures(self, gesture1, gesture2, threshold=None):
        """Compare two gestures using weighted feature comparison"""
        if threshold is None:
            threshold = self.comparison_threshold
            
        try:
            # Ensure gestures have the same shape
            if gesture1.shape != gesture2.shape:
                return False
                
            # Split features into angles, positions, and distances
            n_angles = 15  # 3 angles per finger * 5 fingers
            n_positions = 15  # 3 coordinates per fingertip * 5 fingers
            
            angles1 = gesture1[:n_angles]
            angles2 = gesture2[:n_angles]
            
            positions1 = gesture1[n_angles:n_angles + n_positions]
            positions2 = gesture2[n_angles:n_angles + n_positions]
            
            distances1 = gesture1[n_angles + n_positions:]
            distances2 = gesture2[n_angles + n_positions:]
            
            # Calculate weighted differences
            angle_diff = np.mean(np.abs(angles1 - angles2))
            position_diff = np.mean(np.abs(positions1 - positions2))
            distance_diff = np.mean(np.abs(distances1 - distances2))
            
            # Weighted sum (give more weight to angles and distances)
            total_diff = (0.4 * angle_diff + 
                         0.2 * position_diff + 
                         0.4 * distance_diff)
            
            if self.debug_mode:
                print(f"Angle diff: {angle_diff:.3f}, "
                      f"Position diff: {position_diff:.3f}, "
                      f"Distance diff: {distance_diff:.3f}, "
                      f"Total diff: {total_diff:.3f}")
            
            return total_diff < threshold
            
        except Exception as e:
            print(f"Error comparing gestures: {e}")
            return False

    def get_action_description(self, action_id):
        """Get description for an action ID"""
        descriptions = {
            8: "Forward/Open Gripper",
            2: "Backward/Close Gripper",
            4: "Left/Rotate Z+",
            6: "Right/Rotate Z-",
            7: "Up/Rotate Y+",
            1: "Down/Rotate Y-"
        }
        return descriptions.get(action_id, f"Action {action_id}")

    def detect_gesture(self, hand_landmarks):
        """Detect gesture based on current mode"""
        if self.gesture_mode == "ai":
            # Extract features from current frame
            features = self.get_hand_landmarks_features(hand_landmarks)
            
            # Update frame buffer
            self.frame_buffer.append(features)
            
            # Keep only the most recent frames
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer = self.frame_buffer[-self.buffer_size:]
            
            # Only predict if we have enough frames and enough time has passed
            current_time = time.time()
            if (len(self.frame_buffer) >= self.buffer_size and 
                current_time - self.last_prediction_time >= self.prediction_cooldown):
                
                try:
                    # Convert buffer to numpy array with correct shape
                    frames = np.array(self.frame_buffer)
                    frames = frames.reshape(1, len(self.frame_buffer), -1)
                    
                    if self.debug_mode:
                        print(f"\nFrame buffer shape: {frames.shape}")
                    
                    # Make prediction
                    predicted_action = self.model_trainer.predict(frames)
                    self.last_prediction_time = current_time
                    
                    # Update recent predictions
                    self.recent_predictions.append(predicted_action)
                    if len(self.recent_predictions) > self.max_recent_predictions:
                        self.recent_predictions.pop(0)
                    
                    # Check if we have consistent predictions
                    if len(self.recent_predictions) >= self.gesture_confidence_threshold:
                        most_common = max(set(self.recent_predictions), 
                                        key=self.recent_predictions.count)
                        count = self.recent_predictions.count(most_common)
                        
                        if count >= self.gesture_confidence_threshold:
                            if self.debug_mode:
                                print(f"Confident prediction: {most_common} "
                                      f"({self.get_action_description(most_common)})")
                            return most_common
                    
                    if self.debug_mode:
                        print(f"Recent predictions: {self.recent_predictions}")
                    
                except Exception as e:
                    if self.debug_mode:
                        print(f"Error in AI prediction: {e}")
                    # Don't clear buffer on error, just skip this prediction
                    pass
                
            return None
            
        elif self.gesture_mode == "custom" and self.custom_gestures:
            return self._detect_custom_gesture(hand_landmarks)
        else:
            return self._detect_default_gesture(hand_landmarks)

    def _detect_default_gesture(self, hand_landmarks):
        """Default gesture detection logic"""
        fingertips = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]
        mcps = [hand_landmarks.landmark[i] for i in [5, 9, 13, 17]]
        fingers_up = [tip.y < mcp.y for tip, mcp in zip(fingertips, mcps)]
        
        if self.mode == 0:  # Translation mode
            if fingers_up == [True, False, False, False]: return 8  # Forward
            elif fingers_up == [False, True, True, True]: return 2  # Backward
            elif fingers_up == [True, True, False, False]: return 4  # Left
            elif fingers_up == [True, True, True, False]: return 6  # Right
            elif fingers_up == [False, False, False, True]: return 7  # Up
            elif fingers_up == [False, False, True, True]: return 1  # Down
        elif self.mode == 1:  # Orientation mode
            if fingers_up == [True, False, False, False]: return 8  # Rotate X+
            elif fingers_up == [False, True, False, False]: return 2  # Rotate X-
            elif fingers_up == [True, True, False, False]: return 4  # Rotate Z+
            elif fingers_up == [True, True, True, False]: return 6  # Rotate Z-
            elif fingers_up == [False, False, False, True]: return 7  # Rotate Y+
            elif fingers_up == [False, False, True, False]: return 1  # Rotate Y-
        elif self.mode == 2:  # Gripper mode
            if fingers_up == [True, True, False, False]: return 8  # Open gripper
            elif fingers_up == [True, False, False, False]: return 2  # Close gripper
        return None

    def show_gesture_guide(self):
        """Show the gesture guide window with current gestures"""
        if self.gesture_mode == "custom" and self.gesture_images:
            guide_window = np.zeros((600, 800, 3), dtype=np.uint8)
            row, col = 0, 0
            for action, img in self.gesture_images.items():
                if img is not None:
                    # Resize image to fit in guide
                    display_img = cv2.resize(img, (200, 150))
                    x = col * 200
                    y = row * 150
                    
                    # Add image and label
                    guide_window[y:y+150, x:x+200] = display_img
                    label = self.get_action_label(int(action))
                    cv2.putText(guide_window, label, (x+5, y+145), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    col += 1
                    if col >= 4:  # 4 columns max
                        col = 0
                        row += 1
            
            cv2.imshow('Gesture Guide', guide_window)
            cv2.waitKey(1)

    def get_action_label(self, action):
        """Get label for an action number"""
        labels = {
            8: "Forward/Open", 2: "Backward/Close", 4: "Left/RotZ+",
            6: "Right/RotZ-", 7: "Up/RotY+", 1: "Down/RotY-"
        }
        return labels.get(action, str(action))

    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()

    def get_hand_state(self):
        """Get the current hand state and control mode"""
        success, image = self.cap.read()
        if not success:
            return -1, self.mode
            
        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = self.hands.process(image_rgb)
        
        current_time = time.time()
        state = -1  # Default to neutral state
        
        # Draw hand landmarks and process gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Check for mode switch first
                if self.is_mode_switch_gesture(hand_landmarks):
                    if not self.switching_mode:
                        self.switching_mode = True
                        self.mode_switch_start = current_time
                    elif current_time - self.mode_switch_start > self.mode_switch_hold_time:
                        self.mode = (self.mode + 1) % 3
                        self.switching_mode = False
                        self.active_gesture = None
                        self.current_gesture = None
                        self.current_gesture_start = 0
                        self.prev_gesture_time = current_time
                else:
                    self.switching_mode = False
                    
                    if self.is_neutral_gesture(hand_landmarks):
                        # Reset gesture tracking in neutral position
                        self.active_gesture = None
                        self.current_gesture = None
                        self.current_gesture_start = 0
                        state = -1
                    else:
                        # Get current gesture
                        new_gesture = self.detect_gesture(hand_landmarks)
                        
                        if new_gesture is not None:
                            # Update active gesture
                            if new_gesture != self.active_gesture:
                                self.active_gesture = new_gesture
                                self.active_gesture_start = current_time
                            self.last_active_time = current_time
                            
                        # Check if we should maintain the active gesture
                        if (self.active_gesture is not None and 
                            current_time - self.last_active_time < self.gesture_timeout):
                            state = self.active_gesture
        else:
            # No hand detected, reset all states
            self.active_gesture = None
            self.current_gesture = None
            self.current_gesture_start = 0
            self.switching_mode = False
            
        # Display mode and state
        cv2.putText(image, f'Mode: {self.modes[self.mode]}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display current gesture mode
        cv2.putText(image, f'Gesture Mode: {self.gesture_mode}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display active gestures for current mode
        if self.gesture_mode == "custom":
            valid_actions = set([1, 2, 4, 6, 7, 8]) & self.available_gestures
            cv2.putText(image, f'Available Actions: {sorted(list(valid_actions))}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display mode switching progress
        if self.switching_mode:
            progress = min(1.0, (current_time - self.mode_switch_start) / self.mode_switch_hold_time)
            bar_width = int(200 * progress)
            cv2.rectangle(image, (10, 80), (210, 100), (0, 0, 0), 2)
            cv2.rectangle(image, (10, 80), (10 + bar_width, 100), (0, 255, 255), -1)
            cv2.putText(image, 'Switching Mode...', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # Display gesture state
        elif self.active_gesture is None:
            cv2.putText(image, 'Neutral', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, f'Active: {self.get_action_description(self.active_gesture)}', 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show the image
        cv2.imshow('Hand Gesture Control', image)
        cv2.waitKey(1)
        
        return state, self.mode

    def is_neutral_gesture(self, hand_landmarks):
        """Neutral gesture: Hand in a fist position"""
        fingertips = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]  # Index, Middle, Ring, Pinky tips
        pips = [hand_landmarks.landmark[i] for i in [6, 10, 14, 18]]  # PIP joints
        
        # Check if all fingers are curled (fingertips below PIP joints)
        fingers_curled = all(tip.y > pip.y for tip, pip in zip(fingertips, pips))
        
        # Check thumb is curled too
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_curled = thumb_tip.x > thumb_ip.x if thumb_tip.x > 0 else thumb_tip.x < thumb_ip.x
        
        return fingers_curled and thumb_curled

    def is_mode_switch_gesture(self, hand_landmarks):
        """Mode switch gesture: Thumb up only"""
        # Check thumb is up
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_up = thumb_tip.y < thumb_ip.y
        
        # Check all other fingers are down
        other_fingertips = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]  # Index, Middle, Ring, Pinky
        other_pips = [hand_landmarks.landmark[i] for i in [6, 10, 14, 18]]
        others_down = all(tip.y > pip.y for tip, pip in zip(other_fingertips, other_pips))
        
        return thumb_up and others_down 