import cv2
import mediapipe as mp
import numpy as np
import time

class HandGestureController:
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
        
        # Control states
        self.mode = 0  # 0: Translation, 1: Orientation, 2: Gripper
        self.state = -1
        self.prev_state = -1
        self.modes = ['Translation', 'Orientation', 'Gripper']
        
        # Gesture detection parameters
        self.prev_gesture_time = time.time()
        self.gesture_cooldown = 0.2  # seconds
        self.gesture_hold_time = 0.5  # seconds to hold gesture before action
        self.mode_switch_hold_time = 1.0  # seconds to hold for mode switch
        self.current_gesture_start = 0
        self.current_gesture = None
        self.mode_switch_start = 0
        self.switching_mode = False
        
    def get_hand_state(self):
        success, image = self.cap.read()
        if not success:
            return -1, self.mode
            
        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = self.hands.process(image_rgb)
        
        state = -1  # Default to neutral state
        current_time = time.time()
        
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
                        self.current_gesture = None
                        self.current_gesture_start = 0
                        self.prev_gesture_time = current_time
                else:
                    self.switching_mode = False
                    
                    if self.is_neutral_gesture(hand_landmarks):
                        # Reset gesture tracking in neutral position
                        self.current_gesture = None
                        self.current_gesture_start = 0
                        state = -1
                    else:
                        # Get current gesture
                        new_gesture = self.detect_gesture(hand_landmarks)
                        
                        # If gesture changed, reset timer
                        if new_gesture != self.current_gesture:
                            self.current_gesture = new_gesture
                            self.current_gesture_start = current_time
                        # If gesture held long enough, activate it
                        elif (self.current_gesture is not None and 
                              current_time - self.current_gesture_start > self.gesture_hold_time):
                            state = self.current_gesture
        else:
            # No hand detected, reset all states
            self.current_gesture = None
            self.current_gesture_start = 0
            self.switching_mode = False
            
        # Display mode and state
        cv2.putText(image, f'Mode: {self.modes[self.mode]}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display mode switching progress
        if self.switching_mode:
            progress = min(1.0, (current_time - self.mode_switch_start) / self.mode_switch_hold_time)
            bar_width = int(200 * progress)
            cv2.rectangle(image, (10, 80), (210, 100), (0, 0, 0), 2)
            cv2.rectangle(image, (10, 80), (10 + bar_width, 100), (0, 255, 255), -1)
            cv2.putText(image, 'Switching Mode...', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # Display gesture state
        elif self.current_gesture is None:
            cv2.putText(image, 'Neutral', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif state == -1:
            progress = min(1.0, (current_time - self.current_gesture_start) / self.gesture_hold_time)
            bar_width = int(200 * progress)
            cv2.rectangle(image, (10, 80), (210, 100), (0, 0, 0), 2)
            cv2.rectangle(image, (10, 80), (10 + bar_width, 100), (0, 255, 255), -1)
            cv2.putText(image, 'Hold gesture...', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(image, 'Active', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
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
        
    def detect_gesture(self, hand_landmarks):
        # Get y coordinates of fingertips and their corresponding MCP joints
        fingertips = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]  # Index, Middle, Ring, Pinky
        mcps = [hand_landmarks.landmark[i] for i in [5, 9, 13, 17]]  # Corresponding MCP joints
        
        # Check if fingers are extended (up) or flexed (down)
        fingers_up = [tip.y < mcp.y for tip, mcp in zip(fingertips, mcps)]
        
        # Define gestures based on finger positions
        if self.mode == 0:  # Translation mode
            if fingers_up == [True, False, False, False]:  # Index up only
                return 8  # Move forward
            elif fingers_up == [False, True, True, True]:  # Middle up only
                return 2  # Move backward
            elif fingers_up == [True, True, False, False]:  # Index and Middle up
                return 4  # Move left
            elif fingers_up == [True, True, True, False]:  # Index, Middle, Ring up
                return 6  # Move right
            elif fingers_up == [False, False, False, True]:  # Only pinky up
                return 7  # Move up
            elif fingers_up == [False, False, True, True]:  # Only ring up
                return 1  # Move down
                
        elif self.mode == 1:  # Orientation mode
            if fingers_up == [True, False, False, False]:  # Index up only
                return 8  # Rotate X+
            elif fingers_up == [False, True, False, False]:  # Middle up only
                return 2  # Rotate X-
            elif fingers_up == [True, True, False, False]:  # Index and Middle up
                return 4  # Rotate Z+
            elif fingers_up == [True, True, True, False]:  # Index, Middle, Ring up
                return 6  # Rotate Z-
            elif fingers_up == [False, False, False, True]:  # Only pinky up
                return 7  # Rotate Y+
            elif fingers_up == [False, False, True, False]:  # Only ring up
                return 1  # Rotate Y-
                
        elif self.mode == 2:  # Gripper mode
            if fingers_up == [True, True, False, False]:  # Index up only
                return 8  # Open gripper
            elif fingers_up == [True, False, False, False]:  # Middle up only
                return 2  # Close gripper
                
        return None
        
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
        
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows() 