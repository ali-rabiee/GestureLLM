# GestureLLM

A Python framework for controlling a robot arm using hand gestures via webcam, MediaPipe, and PyBullet. Supports two gesture control modes:

- **Default mode:** Hardcoded gestures for basic robot control.
- **AI mode:** Deep learning model trained on user-collected gesture data.

## Features
- Real-time hand gesture recognition using MediaPipe
- Robot arm simulation with PyBullet (Jaco arm)
- Two gesture control modes: default (hardcoded) and AI (deep learning)
- Data collection and model training workflow for AI mode
- Robust gripper and object handling in simulation

## Modes

### 1. Default Mode
- **Use case:** Quick start, no customization or training required.
- **Gestures:** Predefined hand poses mapped to robot actions.
- **How to run:**
  ```bash
  python main.py --mode default
  ```

### 2. AI Mode
- **Use case:** Robust, user-adaptable gesture recognition via deep learning.
- **Workflow:**
  1. Collect gesture data:
     ```bash
     python main.py --collect-data
     ```
  2. Train the model:
     ```bash
     python main.py --train-model
     ```
  3. Run with AI gesture recognition:
     ```bash
     python main.py --mode ai
     ```

## Command-Line Arguments
- `--mode [default|ai]` : Select gesture mode (default: `default`)
- `--collect-data` : Run gesture data collection for AI mode
- `--train-model` : Train the AI model on collected data

## Example Workflows

### Default Mode (no training required)
```bash
python main.py --mode default
```

### AI Mode (from scratch)
```bash
python main.py --collect-data
python main.py --train-model
python main.py --mode ai
```

## Gesture Mapping (Default)
| Action   | Description                |
|----------|----------------------------|
| 8        | Forward / Open Gripper     |
| 2        | Backward / Close Gripper   |
| 4        | Left / Rotate Z+           |
| 6        | Right / Rotate Z-          |
| 7        | Up / Rotate Y+             |
| 1        | Down / Rotate Y-           |

## Requirements
- Python 3.8+
- PyBullet
- OpenCV
- MediaPipe
- PyTorch (for AI mode)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Notes
- For AI mode, ensure you have a CUDA-enabled GPU and the correct PyTorch version for GPU acceleration.
- The simulation window (PyBullet) provides camera controls (see terminal output for keys).
- If you want to retrain from scratch, delete the `gesture_data` directory contents before collecting new data.

## Troubleshooting
- **Webcam not detected:** Check your camera index in `hand_gesture_control.py`.
- **PyBullet errors:** Ensure you have the correct URDFs and PyBullet installed.
- **AI mode errors:** Make sure you have collected data and trained the model before running in AI mode.

## License
MIT