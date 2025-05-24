from simulation_manager import SimulationManager
from hand_gesture_control import HandGestureController
from gesture_customizer import GestureCustomizer
from gesture_collector import GestureDataCollector
from gesture_model import GestureModelTrainer
import time
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Robot Control with Hand Gestures')
    parser.add_argument('--mode', choices=['default', 'custom', 'ai'], default='default',
                       help='Gesture mode: default, custom, or AI-based')
    parser.add_argument('--customize', action='store_true',
                       help='Run gesture customization before starting')
    parser.add_argument('--collect-data', action='store_true',
                       help='Run data collection for AI training')
    parser.add_argument('--train-model', action='store_true',
                       help='Train the AI model on collected data')
    args = parser.parse_args()
    
    # Collect data if requested
    if args.collect_data:
        print("\nStarting gesture data collection...")
        collector = GestureDataCollector()
        collector.collect_data()
        print("\nData collection complete.")
        return
        
    # Train model if requested
    if args.train_model:
        print("\nStarting model training...")
        trainer = GestureModelTrainer()
        trainer.train_model()
        print("\nModel training complete.")
        return
    
    # Run customization if requested
    if args.customize:
        print("\nStarting gesture customization...")
        customizer = GestureCustomizer()
        customizer.customize_gestures()
        print("\nCustomization complete. Starting robot control...")
    
    # Initialize controllers
    sim = SimulationManager(enable_logging=False)
    gesture_controller = HandGestureController(mode=args.mode)
    
    # Show gesture guide if using custom mode
    if args.mode == "custom":
        gesture_controller.show_gesture_guide()
    
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