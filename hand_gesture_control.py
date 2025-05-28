import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
import pickle
from gesture_model import GestureModelTrainer
from config import DEBUG_MODE, USER_ID, ACTIONS

class HandGestureController:
    def __init__(self, mode=None):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.mode = 0  # 0: Translation, 1: Orientation, 2: Gripper
        self.state = -1
        self.prev_state = -1
        self.modes = ['Translation', 'Orientation', 'Gripper']
        self.debug_mode = DEBUG_MODE
        self.actions = ACTIONS
        self.gesture_class_map = None
        self.action_to_gesture_class = None
        # Always use AI mode
        try:
            print(f"\nInitializing AI gesture recognition model for user: {USER_ID} ...")
            self.model_trainer = GestureModelTrainer()
            self.model_trainer.load_trained_model()
            print("AI model initialized successfully!")
            # Load gesture class map
            gesture_class_map_file = os.path.join(
                "gesture_data", USER_ID, "gesture_class_map.json")
            if os.path.exists(gesture_class_map_file):
                with open(gesture_class_map_file, 'r') as f:
                    self.gesture_class_map = {int(k): v for k, v in json.load(f).items()}
                # Build action to gesture class reverse map
                self.action_to_gesture_class = {}
                for gidx, acts in self.gesture_class_map.items():
                    for aid in acts:
                        self.action_to_gesture_class[aid] = gidx
            else:
                print("Warning: gesture_class_map.json not found!")
        except Exception as e:
            print(f"\nError initializing AI mode: {e}")
            raise
        self.prev_gesture_time = time.time()
        self.gesture_cooldown = 0.1
        self.gesture_hold_time = 0.3
        self.mode_switch_hold_time = 1.0
        self.current_gesture_start = 0
        self.current_gesture = None
        self.mode_switch_start = 0
        self.switching_mode = False
        self.frame_buffer = []
        self.buffer_size = 10
        self.sliding_window_step = 3
        self.prediction_cooldown = 0.05
        self.last_prediction_time = 0
        self.gesture_confidence_threshold = 2
        self.recent_predictions = []
        self.max_recent_predictions = 5
        self.active_gesture = None
        self.active_gesture_start = 0
        self.gesture_timeout = 0.5
        self.last_active_time = 0
        self.last_caption = ""

    def get_hand_landmarks_features(self, hand_landmarks):
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features)

    def get_action_description(self, action_id):
        action_dict = dict(self.actions)
        return action_dict.get(action_id, f"Action {action_id}")

    def get_caption_for_gesture_class(self, gesture_class):
        if self.gesture_class_map is None:
            return ""
        action_names = [self.get_action_description(aid) for aid in self.gesture_class_map.get(gesture_class, [])]
        return " / ".join(action_names)

    def detect_gesture(self, hand_landmarks):
        features = self.get_hand_landmarks_features(hand_landmarks)
        self.frame_buffer.append(features)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer = self.frame_buffer[-self.buffer_size:]
        current_time = time.time()
        if (len(self.frame_buffer) >= self.buffer_size and 
            current_time - self.last_prediction_time >= self.prediction_cooldown):
            try:
                frames = np.array(self.frame_buffer)
                frames = frames.reshape(1, len(self.frame_buffer), -1)
                if self.debug_mode:
                    print(f"\nFrame buffer shape: {frames.shape}")
                predicted_gesture_class = self.model_trainer.predict(frames)
                self.last_prediction_time = current_time
                self.recent_predictions.append(predicted_gesture_class)
                if len(self.recent_predictions) > self.max_recent_predictions:
                    self.recent_predictions.pop(0)
                if len(self.recent_predictions) >= self.gesture_confidence_threshold:
                    most_common = max(set(self.recent_predictions), 
                                    key=self.recent_predictions.count)
                    count = self.recent_predictions.count(most_common)
                    if count >= self.gesture_confidence_threshold:
                        if self.debug_mode:
                            print(f"Confident prediction: gesture class {most_common} (caption: {self.get_caption_for_gesture_class(most_common)})")
                        return most_common
                if self.debug_mode:
                    print(f"Recent predictions: {self.recent_predictions}")
            except Exception as e:
                if self.debug_mode:
                    print(f"Error in AI prediction: {e}")
                pass
        return None

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_hand_state(self):
        success, image = self.cap.read()
        if not success:
            return -1, self.mode
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        current_time = time.time()
        state = -1
        caption = ""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
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
                        self.active_gesture = None
                        self.current_gesture = None
                        self.current_gesture_start = 0
                        state = -1
                        caption = ""
                    else:
                        gesture_class = self.detect_gesture(hand_landmarks)
                        if gesture_class is not None:
                            actions = self.gesture_class_map.get(gesture_class, []) if self.gesture_class_map else []
                            action_for_mode = None
                            for aid in actions:
                                if (self.mode == 0 and 0 <= aid <= 5) or (self.mode == 1 and 6 <= aid <= 11) or (self.mode == 2 and 12 <= aid <= 13):
                                    action_for_mode = aid
                                    break
                            if action_for_mode is not None:
                                state = action_for_mode
                                self.active_gesture = gesture_class
                                self.last_active_time = current_time
                                caption = self.get_caption_for_gesture_class(gesture_class)
                                self.last_caption = caption
                            else:
                                state = -1
                                caption = self.get_caption_for_gesture_class(gesture_class)
                                self.last_caption = caption
                        elif (self.active_gesture is not None and current_time - self.last_active_time < self.gesture_timeout):
                            state = -1
                            caption = self.last_caption
        else:
            self.active_gesture = None
            self.current_gesture = None
            self.current_gesture_start = 0
            self.switching_mode = False
            caption = ""
        cv2.putText(image, f'Mode: {self.modes[self.mode]}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f'Gesture Mode: AI', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if self.switching_mode:
            progress = min(1.0, (current_time - self.mode_switch_start) / self.mode_switch_hold_time)
            bar_width = int(200 * progress)
            cv2.rectangle(image, (10, 80), (210, 100), (0, 0, 0), 2)
            cv2.rectangle(image, (10, 80), (10 + bar_width, 100), (0, 255, 255), -1)
            cv2.putText(image, 'Switching Mode...', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif self.active_gesture is None:
            cv2.putText(image, 'Neutral', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, f'Active: {caption}', 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Hand Gesture Control', image)
        cv2.waitKey(1)
        return state, self.mode

    def is_neutral_gesture(self, hand_landmarks):
        fingertips = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]
        pips = [hand_landmarks.landmark[i] for i in [6, 10, 14, 18]]
        fingers_curled = all(tip.y > pip.y for tip, pip in zip(fingertips, pips))
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_curled = thumb_tip.x > thumb_ip.x if thumb_tip.x > 0 else thumb_tip.x < thumb_ip.x
        return fingers_curled and thumb_curled

    def is_mode_switch_gesture(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_up = thumb_tip.y < thumb_ip.y
        other_fingertips = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]
        other_pips = [hand_landmarks.landmark[i] for i in [6, 10, 14, 18]]
        others_down = all(tip.y > pip.y for tip, pip in zip(other_fingertips, other_pips))
        return thumb_up and others_down