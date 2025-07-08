# from isaac_sim_manager import SimulationManager
from simulation_manager import SimulationManager
from hand_gesture_control import HandGestureController
from gui_controller import GUIController
from gesture_collector import GestureDataCollector
from gesture_model import GestureModelTrainer
from config import USER_ID
import time
import os

def choose_control_mode():
    """Allow user to choose between gesture and GUI control"""
    print("\n" + "="*50)
    print("🤖 ROBOTIC ARM CONTROL SYSTEM")
    print("="*50)
    print("\nChoose your control method:")
    print("1. Gesture Control (Hand gestures via camera)")
    print("2. GUI Control (Mouse and buttons)")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1, 2, or 3): ").strip()
            if choice == '1':
                return 'gesture'
            elif choice == '2':
                return 'gui'
            elif choice == '3':
                print("Exiting...")
                return None
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None



def main():
    # Choose control mode
    control_mode = choose_control_mode()
    if control_mode is None:
        return
    
    print(f"✅ Selected: YCB Objects")
    
    # User-specific model and data paths (only needed for gesture control)
    user_data_dir = os.path.join("gesture_data", USER_ID)
    model_path = os.path.join(user_data_dir, "gesture_model.pt")
    dataset_info_path = os.path.join(user_data_dir, "dataset_info.json")

    # If gesture mode is selected and model does not exist, collect data and train
    if control_mode == 'gesture' and not os.path.exists(model_path):
        print(f"No trained model found for user '{USER_ID}'. Starting data collection and training...")
        collector = GestureDataCollector()
        collector.collect_data()
        trainer = GestureModelTrainer()
        trainer.train_model()
        print("\nModel training complete.")
    elif control_mode == 'gesture':
        print(f"Found trained model for user '{USER_ID}'. Skipping data collection and training.")

    # Initialize simulation manager
    print(f"\n🚀 Starting {control_mode.upper()} control mode with YCB Objects...")
    sim = SimulationManager(enable_logging=False)
    
    # Initialize appropriate controller
    controller = None
    try:
        if control_mode == 'gesture':
            controller = HandGestureController(mode="ai")
            print("✅ Gesture controller initialized successfully!")
            print("📷 Camera window should be open. Use hand gestures to control the robot.")
            print("💡 Make a fist and hold for 1 second to switch between modes.")
            
        elif control_mode == 'gui':
            controller = GUIController()
            print("✅ GUI controller initialized successfully!")
            print("🖱️  GUI window should be open. Use mouse and buttons to control the robot.")
        
        print(f"\n🎮 Robot arm is ready for control with YCB Objects!")
        print("⏹️  Press Ctrl+C to stop")
        print("🏷️  You're using realistic YCB objects - perfect for manipulation tasks!")
        
        # Main control loop
        while True:
            if control_mode == 'gesture':
                # Get gesture input
                state, mode = controller.get_hand_state()
                # Process commands
                sim.process_command(state, mode)
                # Small sleep to prevent CPU overload
                time.sleep(0.01)
                
            elif control_mode == 'gui':
                # Update GUI and get input
                if not controller.update():
                    break  # GUI was closed
                # Get GUI input
                state, mode = controller.get_robot_state()
                # Process commands
                sim.process_command(state, mode)
                # Small sleep to prevent CPU overload
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
    finally:
        # Cleanup
        print("🧹 Cleaning up resources...")
        if controller:
            controller.cleanup()
        sim.cleanup()
        print("✅ Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main() 