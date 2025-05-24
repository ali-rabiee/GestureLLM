import cv2
import mediapipe as mp
import numpy as np
import time
import os
from hand_gesture_control import HandGestureController

class GestureCustomizer:
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
        
        # Actions to customize
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
        
    def customize_gestures(self):
        """Run the gesture customization process"""
        custom_gestures = {}
        gesture_images = {}
        
        print("\nWelcome to Gesture Customization!")
        print("Press 'SPACE' to capture a gesture when ready.")
        print("Press 'ESC' to skip a gesture.")
        print("Press 'Q' to quit the customization process.")
        print("Press 'R' to retry the last gesture.\n")
        
        for mode, actions in self.actions.items():
            print(f"\n=== {mode} ===")
            for action_id, action_name in actions.items():
                while True:  # Allow retries for each gesture
                    result = self._capture_gesture(action_name, action_id, custom_gestures, gesture_images)
                    if result == "quit":
                        return
                    elif result == "skip":
                        break
                    elif result == "retry":
                        print(f"Retrying gesture for: {action_name}")
                        continue
                    else:  # Successful capture
                        break
        
        if custom_gestures:
            # Save the custom gestures
            controller = HandGestureController()
            controller.custom_gestures = custom_gestures
            controller.gesture_images = gesture_images
            controller.save_custom_gestures()
            print("\nCustom gestures saved successfully!")
            print(f"Number of gestures saved: {len(custom_gestures)}")
            print("Saved gestures for actions:", sorted([int(k) for k in custom_gestures.keys()]))
        
        self.cleanup()
        
    def _capture_gesture(self, action_name, action_id, custom_gestures, gesture_images):
        """Capture a single gesture"""
        print(f"\nShow gesture for: {action_name}")
        countdown_start = None
        capturing = False
        preview_time = None
        
        while True:
            success, image = self.cap.read()
            if not success:
                continue
                
            # Flip the image horizontally for a later selfie-view display
            image = cv2.flip(image, 1)
            
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect hands
            results = self.hands.process(image_rgb)
            
            # Clear the image
            overlay = image.copy()
            
            # Draw hand landmarks if detected
            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        overlay, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Add instructions
            cv2.putText(overlay, f"Action: {action_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(overlay, "SPACE: Capture  ESC: Skip  Q: Quit  R: Retry", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show hand detection status
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.putText(overlay, "Hand Detected" if hand_detected else "No Hand Detected",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Handle countdown
            if countdown_start is not None:
                elapsed = time.time() - countdown_start
                if elapsed < 3:
                    # Show countdown
                    count = 3 - int(elapsed)
                    cv2.putText(overlay, str(count), (overlay.shape[1]//2 - 20, overlay.shape[0]//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)
                else:
                    # Capture the gesture
                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        features = []
                        for landmark in hand_landmarks.landmark:
                            features.extend([landmark.x, landmark.y, landmark.z])
                        custom_gestures[str(action_id)] = np.array(features)
                        gesture_images[str(action_id)] = image.copy()
                        print(f"Gesture captured for: {action_name}")
                        preview_time = time.time()
                        countdown_start = None
                    else:
                        print("No hand detected during capture!")
                        return "retry"
            
            # Show preview of captured gesture
            if preview_time is not None:
                if time.time() - preview_time < 2:  # Show preview for 2 seconds
                    cv2.putText(overlay, "Gesture Captured!", (overlay.shape[1]//2 - 100, overlay.shape[0]//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    return True
            
            # Show the image
            cv2.imshow('Gesture Customization', overlay)
            
            # Handle key presses
            key = cv2.waitKey(1)
            if key == ord(' '):  # Space key
                if countdown_start is None and hand_detected:
                    countdown_start = time.time()
                elif not hand_detected:
                    print("Please show your hand before capturing!")
            elif key == 27:  # ESC key
                print(f"Skipped: {action_name}")
                return "skip"
            elif key == ord('q'):  # Q key
                print("\nCustomization process cancelled.")
                return "quit"
            elif key == ord('r'):  # R key
                return "retry"
                
        return True
        
    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    customizer = GestureCustomizer()
    customizer.customize_gestures() 