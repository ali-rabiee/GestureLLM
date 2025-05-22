# GestureLLM - Hand Gesture Controlled Robot Arm

A Python-based system for controlling a Jaco robotic arm using hand gestures through webcam input.

## Project Structure

- `main.py`: Entry point of the application
- `simulation_manager.py`: Handles the PyBullet simulation and robot control
- `hand_gesture_control.py`: Manages hand gesture recognition using MediaPipe
- `requirements.txt`: Lists all required Python packages

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Control modes:
   - Translation Mode (0): Control robot position
   - Orientation Mode (1): Control end-effector orientation
   - Gripper Mode (2): Control gripper opening/closing

3. Gestures:
   - Neutral Position: Make a fist
   - Mode Switch: Thumb up only (hold for 1 second)
   
   Translation Mode:
   - Index up: Forward
   - Middle up: Backward
   - Index + Middle up: Left
   - Index + Middle + Ring up: Right
   - Pinky up: Up
   - Ring up: Down

   Orientation Mode:
   - Similar finger patterns for rotation around X, Y, Z axes

   Gripper Mode:
   - Index up: Open gripper
   - Middle up: Close gripper

## Adding New Features

The code is structured to make it easy to add new features:

1. For new robot capabilities:
   - Add methods to `SimulationManager` class in `simulation_manager.py`

2. For new gesture controls:
   - Add gesture detection in `HandGestureController` class in `hand_gesture_control.py`

3. For new modes or features:
   - Add new mode constants and handling in both classes
   - Update the main loop in `main.py` if needed

## License

[Your chosen license]