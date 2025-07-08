# GestureLLM

A Python framework for controlling a robot arm using either hand gestures or a graphical user interface, with PyBullet simulation. The system supports multiple control methods:

- **Gesture Control:** Real-time hand gesture recognition using AI-trained models
- **GUI Control:** Mouse and button-based control interface with pygame

## Features
- **Dual Control Methods:** Choose between gesture control or GUI control at startup
- Real-time hand gesture recognition using MediaPipe and custom AI models
- Intuitive GUI control with pygame for precise robot manipulation
- Robot arm simulation with PyBullet (Jaco arm)
- **Realistic Daily Objects:** Random placement of everyday items (cups, bottles, books, phones, etc.)
- User-specific AI model training for personalized gesture recognition
- Robust gripper and object handling in simulation
- Easy-to-use launcher scripts for Windows

## Control Methods

### 1. Gesture Control (AI Mode)
- **Use case:** Hands-free control using webcam and hand gestures
- **Features:** 
  - AI model trained on user-collected gesture data
  - Three control modes: Translation, Orientation, and Gripper
  - Real-time gesture recognition with confidence thresholds
- **Setup:** First-time users need to collect gesture data and train the model

### 2. GUI Control
- **Use case:** Precise control using mouse and buttons
- **Features:**
  - Intuitive button layout for all robot movements
  - Mode switching between Translation, Orientation, and Gripper
  - Press-and-hold buttons for continuous movement
  - Visual feedback and status indicators
- **Setup:** No training required, ready to use immediately

## Quick Start

### Easy Launch (Windows)
For Windows users, use the provided launcher scripts:

**Option 1: Batch File**
```cmd
run_controller.bat
```

**Option 2: PowerShell**
```powershell
.\run_controller.ps1
```

These scripts will automatically:
1. Activate the virtual environment
2. Launch the control selection menu
3. Handle cleanup when finished

### Manual Launch
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run the controller
python main.py
```

## Control Interface

When you run the program, you'll see a menu to choose your control method:

```
🤖 ROBOTIC ARM CONTROL SYSTEM
==================================================

Choose your control method:
1. Gesture Control (Hand gestures via camera)
2. GUI Control (Mouse and buttons)
3. Exit
```

### Gesture Control Interface
- **Camera window:** Shows real-time hand detection and current mode
- **Mode switching:** Make a fist and hold for 1 second to switch modes
- **Actions:** Use trained gestures to control robot movements
- **Modes:**
  - Translation: Forward, Backward, Left, Right, Up, Down
  - Orientation: X+/X-, Y+/Y-, Z+/Z- rotations
  - Gripper: Open, Close

### GUI Control Interface
- **Mode buttons:** Click to switch between Translation, Orientation, and Gripper modes
- **Action buttons:** Hold down to move the robot continuously
- **Visual feedback:** Current mode and action are clearly displayed
- **Controls:**
  - Translation mode: Forward, Backward, Left, Right, Up, Down buttons
  - Orientation mode: X+/X-, Y+/Y-, Z+/Z- rotation buttons
  - Gripper mode: Open, Close buttons

## First-Time Setup for Gesture Control

If you choose gesture control and no trained model exists:

1. **Data Collection:** Follow on-screen prompts to record gesture samples
2. **Model Training:** The system will automatically train an AI model
3. **Ready to Use:** Start controlling the robot with your trained gestures

## Robot Control Modes

Both control methods support three operational modes:

| Mode | Actions | Description |
|------|---------|-------------|
| **Translation** | Forward, Backward, Left, Right, Up, Down | Move the robot arm in 3D space |
| **Orientation** | X+/X-, Y+/Y-, Z+/Z- | Rotate the end-effector |
| **Gripper** | Open, Close | Control the gripper to grasp objects |

## Simulation Environment

The system creates a realistic tabletop environment with:

- **Table Setup:** Wooden table surface for object placement
- **Realistic URDF Objects:** 6-8 randomly selected items using PyBullet's built-in 3D models:
  - ☕ **Coffee Mugs** - Detailed ceramic mugs with handles
  - 📦 **Containers** - Various shaped boxes and containers
  - 📚 **Flat Objects** - Book-like and tablet-shaped items
  - 🔧 **Tools** - Small household and office items
  - 🏺 **Kitchen Items** - Realistic kitchen utensils and containers
  - 🖥️ **Office Items** - Desk accessories and small electronics
  - 📱 **Compact Objects** - Phone-sized and small gadgets
  - 🎯 **Varied Shapes** - Cylindrical, rectangular, and complex 3D models

- **Realistic 3D Models:** High-quality URDF files with detailed meshes and textures
- **Physics Properties:** Accurate mass, friction, and collision detection
- **Smart Scaling:** Objects automatically scaled for appropriate tabletop sizes
- **Random Placement:** Objects spawn at random positions and orientations
- **Collision Prevention:** Smart placement algorithm prevents object overlap

## Installation

### Prerequisites
- Python 3.8+
- Windows 10/11 (for provided launcher scripts)
- Webcam (for gesture control)

### Setup
1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   ```
3. Install dependencies:
   ```bash
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Requirements
```
numpy>=1.19.0
opencv-python>=4.5.0
mediapipe>=0.8.9
pygame>=2.0.0
pybullet>=3.2.0 
torch>=1.9.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
```

## Configuration

Edit `config.py` to customize:
- `USER_ID`: Unique identifier for gesture model storage
- `DEBUG_MODE`: Enable/disable debug output
- `ACTIONS`: Customize action mappings

## Troubleshooting

### Common Issues
- **Virtual environment not found:** Ensure `venv` folder exists and is properly created
- **Webcam not detected:** Check camera permissions and try different camera indices
- **PyBullet errors:** Ensure all URDF files are present in the correct directories
- **Gesture model errors:** Delete gesture data and retrain if experiencing issues
- **GUI not responding:** Check pygame installation and display settings

### Gesture Control Specific
- **Poor recognition:** Retrain the model with more diverse gesture samples
- **Mode switching issues:** Ensure clear fist gesture for mode switching
- **Camera lag:** Reduce camera resolution or improve lighting

### GUI Control Specific
- **Buttons not responsive:** Check mouse sensitivity and pygame version
- **Display issues:** Verify monitor resolution and scaling settings

## File Structure
```
GestureLLM/
├── main.py                 # Main application entry point
├── gui_controller.py       # Pygame GUI controller
├── hand_gesture_control.py # Gesture recognition system
├── simulation_manager.py   # PyBullet robot simulation
├── gesture_collector.py    # Data collection for AI training
├── gesture_model.py        # AI model training and inference
├── config.py              # Configuration settings
├── run_controller.bat     # Windows batch launcher
├── run_controller.ps1     # PowerShell launcher
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## License
MIT