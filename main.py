from simulation_manager import SimulationManager
from hand_gesture_control import HandGestureController
import time

def main():
    # Initialize controllers
    sim = SimulationManager(enable_logging=False)
    gesture_controller = HandGestureController()
    
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