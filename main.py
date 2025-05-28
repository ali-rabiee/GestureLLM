from simulation_manager import SimulationManager
from hand_gesture_control import HandGestureController
from gesture_collector import GestureDataCollector
from gesture_model import GestureModelTrainer
from config import USER_ID
import time
import os

def main():
    # User-specific model and data paths
    user_data_dir = os.path.join("gesture_data", USER_ID)
    model_path = os.path.join(user_data_dir, "gesture_model.pt")
    dataset_info_path = os.path.join(user_data_dir, "dataset_info.json")

    # If model does not exist, collect data and train
    if not os.path.exists(model_path):
        print(f"No trained model found for user '{USER_ID}'. Starting data collection and training...")
        collector = GestureDataCollector()
        collector.collect_data()
        trainer = GestureModelTrainer()
        trainer.train_model()
        print("\nModel training complete.")
    else:
        print(f"Found trained model for user '{USER_ID}'. Skipping data collection and training.")

    # Initialize controllers (always AI mode)
    sim = SimulationManager(enable_logging=False)
    gesture_controller = HandGestureController(mode="ai")
    
    try:
        while True:
            # Get gesture input
            state, mode = gesture_controller.get_hand_state()
            # Process commands
            sim.process_command(state, mode)
            # Small sleep to prevent CPU overload
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        # Cleanup
        gesture_controller.cleanup()
        sim.cleanup()

if __name__ == "__main__":
    main() 