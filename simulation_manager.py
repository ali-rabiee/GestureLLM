import pybullet as p
import pybullet_data
import numpy as np
from datetime import datetime
import time
import os
import csv
import random
from scipy.spatial.transform import Rotation as R
from config import RANDOM_SEED
from collections import deque

try:
    from pybullet_object_models import ycb_objects
    YCB_AVAILABLE = True
except ImportError:
    print("WARNING: pybullet-object-models not installed. Install with: pip install git+https://github.com/eleramp/pybullet-object-models.git")
    YCB_AVAILABLE = False

class SimulationManager:
    def __init__(self, enable_logging=False):
        self.enable_logging = enable_logging
        # Set random seed for reproducible object placement
        if RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
        self.setup_simulation()
        self.setup_robot()
        self.setup_workspace()
        self.setup_control_params()
        self.setup_logging()
        # Keep short action history for shared autonomy heuristics
        self.recent_actions = deque(maxlen=240)  # ~4s at 60 Hz
        
    def setup_simulation(self):
        """Initialize PyBullet simulation"""
        # Direct connection to GUI without shared memory
        p.connect(p.GUI)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        
        # Configure visualizer for better interaction
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
        
        # Disable unnecessary rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        
        # Set camera parameters for better initial view
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=-40,
            cameraPitch=-30,
            cameraTargetPosition=[-0.2, 0.0, 0.0]
        )
        
        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
        # Improved physics parameters for better stability and collision handling
        p.setPhysicsEngineParameter(numSolverIterations=100)  # Increased from 50
        p.setPhysicsEngineParameter(enableConeFriction=1)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.0001)  # Decreased for more precise contact
        p.setPhysicsEngineParameter(allowedCcdPenetration=0.0001)  # Add continuous collision detection
        
        # Increased gravity for better stability
        p.setGravity(0, 0, -20)  # Increased from -15
        p.setRealTimeSimulation(True)
        
        print("\nViewer Controls:")
        print("- ALT + Left Mouse Button: Rotate view")
        print("- ALT + Middle Mouse Button: Pan view")
        print("- ALT + Right Mouse Button (up/down): Zoom in/out")
        print("- R key: Reset camera view\n")
        
    def setup_robot(self):
        """Setup Jaco robot and its parameters"""
        # Load robot
        self.jacoId = p.loadURDF("jaco/j2n6s300.urdf", [0,0,0], useFixedBase=True)
        self.jacoEndEffectorIndex = 8
        self.jacoArmJoints = [2, 3, 4, 5, 6, 7]
        self.jacoFingerJoints = [9, 11, 13]
        
        # Improved joint parameters for better control
        self.jd = [2.0] * 10  # Increased joint damping for more stability
        self.rp = [0, np.pi/4, np.pi, 1.0*np.pi, 1.8*np.pi, 0*np.pi, 1.75*np.pi, 0.5*np.pi]
        
        # Initialize robot pose
        self.reset_robot_pose()
        
        # Get initial state
        self.ls = p.getLinkState(self.jacoId, self.jacoEndEffectorIndex)
        self.pos = list(self.ls[4])
        self.orn = list(self.ls[5])
        
        # Make gripper fingers extremely rigid and powerful
        for joint in self.jacoFingerJoints:
            p.changeDynamics(self.jacoId, joint,
                           lateralFriction=50.0,     # Extremely high friction for perfect grip
                           spinningFriction=50.0,    # Prevent any spinning in grip
                           rollingFriction=40.0,     # Prevent any rolling in grip
                           contactStiffness=1000000, # Extremely stiff contact - no deformation
                           contactDamping=200000,    # Very high damping for stability
                           frictionAnchor=1,
                           mass=0.1,                 # Slightly heavier fingers for more power
                           localInertiaDiagonal=[0.001, 0.001, 0.001])  # High inertia for stability
        
    def setup_workspace(self):
        """Setup workspace and objects"""
        # Load ground plane with high friction
        plane_id = p.loadURDF("plane.urdf", [0,0,-.65])
        p.changeDynamics(plane_id, -1,
                        lateralFriction=3.0,
                        spinningFriction=3.0,
                        rollingFriction=3.0)
                        
        # Load table with high friction
        self.tableId = p.loadURDF("table/table.urdf", basePosition=[-0.4,0.0,-0.65])
        p.changeDynamics(self.tableId, -1,
                        lateralFriction=3.0,
                        spinningFriction=3.0,
                        rollingFriction=3.0,
                        contactStiffness=50000,
                        contactDamping=10000)
        
        # Create YCB objects
        self.objects = []
        self._setup_ycb_objects()
    
    def _setup_ycb_objects(self):
        """Setup YCB objects on the table."""
        if not YCB_AVAILABLE:
            print("❌ YCB objects not available. Please install: pip install git+https://github.com/eleramp/pybullet-object-models.git")
            return
            
        # Available YCB objects with their properties
        ycb_objects_list = [
            # {'name': 'YcbBanana', 'folder': 'YcbBanana', 'mass': 0.12, 'description': 'Yellow banana fruit'},
            {'name': 'YcbMustardBottle', 'folder': 'YcbMustardBottle', 'mass': 0.6, 'description': 'Mustard bottle container'},
            {'name': 'YcbTomatoSoupCan', 'folder': 'YcbTomatoSoupCan', 'mass': 0.35, 'description': 'Tomato soup can'},
            {'name': 'YcbCrackerBox', 'folder': 'YcbCrackerBox', 'mass': 0.4, 'description': 'Cheez-It cracker box'},
            {'name': 'YcbSugar', 'folder': 'YcbSugar', 'mass': 0.5, 'description': 'Sugar box container'},
            {'name': 'YcbChipsCan', 'folder': 'YcbChipsCan', 'mass': 0.2, 'description': 'Pringles chips can'},
            # {'name': 'YcbHammer', 'folder': 'YcbHammer', 'mass': 0.8, 'description': 'Claw hammer tool'},
            # {'name': 'YcbStrawberry', 'folder': 'YcbStrawberry', 'mass': 0.02, 'description': 'Fresh strawberry fruit'},
            # {'name': 'YcbApple', 'folder': 'YcbApple', 'mass': 0.15, 'description': 'Red apple fruit'},
            # {'name': 'YcbPear', 'folder': 'YcbPear', 'mass': 0.18, 'description': 'Green pear fruit'},
            {'name': 'YcbPowerDrill', 'folder': 'YcbPowerDrill', 'mass': 1.5, 'description': 'Power drill tool'},
            # {'name': 'YcbScissors', 'folder': 'YcbScissors', 'mass': 0.1, 'description': 'Cutting scissors'},
            {'name': 'YcbMasterChefCan', 'folder': 'YcbMasterChefCan', 'mass': 0.4, 'description': 'Master Chef can'},
            {'name': 'YcbGelatinBox', 'folder': 'YcbGelatinBox', 'mass': 0.1, 'description': 'Gelatin dessert box'},
            # {'name': 'YcbMediumClamp', 'folder': 'YcbMediumClamp', 'mass': 0.2, 'description': 'Medium-sized clamp tool'}
        ]
        
        print(f"🎲 Spawning random YCB objects (seed: {RANDOM_SEED})...")
        
        # Define table boundaries for random placement
        min_x, max_x = -0.6, -0.2
        min_y, max_y = -0.3, 0.3
        min_distance = 0.1
        
        # Randomly select 5-7 objects to spawn
        num_objects = np.random.randint(5, 8)
        selected_objects = random.sample(ycb_objects_list, min(num_objects, len(ycb_objects_list)))
        
        positions = []
        spawned_count = 0
        
        for obj_info in selected_objects:
            # Try to find a valid position
            position_found = False
            for _ in range(50):  # Max attempts
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                
                # Check distance from existing objects
                valid_position = True
                for pos in positions:
                    if np.sqrt((x - pos[0])**2 + (y - pos[1])**2) < min_distance:
                        valid_position = False
                        break
                
                if valid_position:
                    positions.append([x, y])
                    position_found = True
                    break
            
            if not position_found:
                # Fallback position
                x = min_x + (spawned_count * (max_x - min_x) / num_objects)
                y = 0
            
            # Load YCB object
            try:
                urdf_path = os.path.join(ycb_objects.getDataPath(), obj_info['folder'], "model.urdf")
                rotation = np.random.uniform(0, 2 * np.pi)
                
                obj_id = p.loadURDF(
                    urdf_path,
                    basePosition=[x, y, 0.05],
                    baseOrientation=p.getQuaternionFromEuler([0, 0, rotation]),
                    useFixedBase=False,
                    flags=p.URDF_USE_INERTIA_FROM_FILE
                )
                
                # Set physics properties with very high friction to prevent slipping
                p.changeDynamics(obj_id, -1,
                               lateralFriction=30.0,    # Much higher friction for no slipping
                               spinningFriction=30.0,   # Prevent spinning when gripped
                               rollingFriction=15.0,    # Prevent rolling when gripped
                               restitution=0.05,        # Less bouncy
                               mass=obj_info['mass'],
                               linearDamping=0.5,       # More damping for stability
                               angularDamping=0.5,      # More angular damping
                               contactStiffness=100000, # High contact stiffness
                               contactDamping=20000)    # High contact damping
                
                self.objects.append(obj_id)
                spawned_count += 1
                print(f"✅ Spawned {obj_info['name']} - {obj_info['description']} at ({x:.2f}, {y:.2f})")
                
            except Exception as e:
                print(f"❌ Failed to load {obj_info['name']}: {e}")
                continue
        
        print(f"✅ Successfully spawned {spawned_count} YCB objects")
        
        # Set camera view
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=-40,
            cameraPitch=-30,
            cameraTargetPosition=[-0.2, 0.0, 0.0]
        )
        
    def setup_control_params(self):
        """Setup control parameters"""
        self.wu = [0.1, 0.5, 0.5]      # Upper workspace limits
        self.wl = [-.66, -.5, 0.02]    # Lower workspace limits
        
        # Movement parameters
        self.dist = .002       # Translation step size
        self.ang = .005       # Angular step size
        self.rot_theta = .008  # Rotation step size
        
        # Setup rotation matrices
        self._setup_rotation_matrices()
        
        # Control states
        self.JP = list(self.rp[2:9])
        self.gripper_state = "open"  # Can be "open" or "closed"
        self.gripper_open_pos = 0.0
        self.gripper_closed_pos = 1.2  # Reduced to prevent finger overlap
        self.gripper_target_pos = self.gripper_open_pos  # Smooth target tracking
        self.fing = self.gripper_open_pos
        self.newPosInput = 1
        
        # Smart gripper collision detection
        self.gripper_closing_step = 0.05  # Larger steps for more forceful closing
        self.max_gripper_force_threshold = 200.0  # Much higher threshold for powerful grip
        self.min_finger_contacts = 2  # Require at least 2 fingers to make contact
        self.gripper_hold_force = 5000.0  # Extremely high force when holding objects
        self.gripper_push_force = 8000.0  # Very high force to push objects when closing
        
        # Object height for collision prevention (using max possible object height)
        self.max_object_height = 0.20  # Maximum height of objects (water bottle)
        self.grasp_height_offset = 0.02  # Small offset above objects for grasping
        self.min_z_height = self.wl[2] + 0.02  # Minimum Z height to prevent collision with table
        
        # Improved gripper parameters - extremely powerful force control
        self.grip_force = 2000.0  # High normal closing force
        self.grip_speed = 6.0     # Faster speed for more forceful movement
        
    def _setup_rotation_matrices(self):
        """Setup rotation matrices for orientation control"""
        theta = self.rot_theta
        # Positive rotations
        self.Rx = np.array([[1., 0., 0.],
                           [0., np.cos(theta), -np.sin(theta)],
                           [0., np.sin(theta), np.cos(theta)]])
        self.Ry = np.array([[np.cos(theta), 0., np.sin(theta)],
                           [0., 1., 0.],
                           [-np.sin(theta), 0., np.cos(theta)]])
        self.Rz = np.array([[np.cos(theta), -np.sin(theta), 0.],
                           [np.sin(theta), np.cos(theta), 0.],
                           [0., 0., 1.]])
        # Negative rotations
        self.Rxm = np.array([[1., 0., 0.],
                            [0., np.cos(-theta), -np.sin(-theta)],
                            [0., np.sin(-theta), np.cos(-theta)]])
        self.Rym = np.array([[np.cos(-theta), 0., np.sin(-theta)],
                            [0., 1., 0.],
                            [-np.sin(-theta), 0., np.cos(-theta)]])
        self.Rzm = np.array([[np.cos(-theta), -np.sin(-theta), 0.],
                            [np.sin(-theta), np.cos(-theta), 0.],
                            [0., 0., 1.]])
                            
    def setup_logging(self):
        """Setup data logging if enabled"""
        self.log_file = None
        self.file_obj = None
        
        if self.enable_logging:
            data_dir = time.strftime("%Y%m%d")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            trial_ind = len(os.listdir(data_dir))
            fn = data_dir + "/simdata00" + str(trial_ind) + ".csv"
            self.log_file = open(fn, 'w', newline='')
            self.file_obj = csv.writer(self.log_file)
            
    def reset_robot_pose(self):
        """Reset robot to initial pose"""
        for i in range(8):
            p.resetJointState(self.jacoId, i, self.rp[i])
            
    def process_command(self, state, mode):
        """Process control commands based on gesture input and current mode"""
        if state == -1:
            # Even if idle, append idle to history for timing-based heuristics
            self._append_action_history(state, mode)
            return
        self._append_action_history(state, mode)
        baseTheta = self.JP[0]
        s = np.sin(baseTheta)
        c = np.cos(baseTheta)
        n = np.sqrt(self.pos[0]**2 + self.pos[1]**2)
        dx = -self.pos[1]/n if n > 0 else 0
        dy = self.pos[0]/n if n > 0 else 0
        Rrm = R.from_quat(self.orn)
        Rnew = Rrm.as_matrix()
        # Only allow actions for the current mode
        if mode == 0 and state in [0, 1, 2, 3, 4, 5]:
            self._process_translation_flat(state, dx, dy, s, c)
        elif mode == 1 and state in [6, 7, 8, 9, 10, 11]:
            self._process_orientation_flat(state)
        elif mode == 2 and state in [12, 13]:
            self._process_gripper_flat(state)
        self._apply_workspace_limits()
        self._update_robot_state()
        self._log_data()

    def _process_translation_flat(self, state, dx, dy, s, c):
        if state == 0:  # Forward
            self.pos[0] += self.dist * c
            self.pos[1] -= self.dist * s
        elif state == 1:  # Backward
            self.pos[0] -= self.dist * c
            self.pos[1] += self.dist * s
        elif state == 2:  # Left
            self.pos[0] += self.dist * dx
            self.pos[1] += self.dist * dy
        elif state == 3:  # Right
            self.pos[0] -= self.dist * dx
            self.pos[1] -= self.dist * dy
        elif state == 4:  # Up
            self.pos[2] += self.dist
        elif state == 5:  # Down
            self.pos[2] -= self.dist
        self.newPosInput = 1

    def _process_orientation_flat(self, state):
        Rrm = R.from_quat(self.orn)
        if state == 6:  # X+
            Rnew = Rrm.as_matrix() @ self.Rx
        elif state == 7:  # X-
            Rnew = Rrm.as_matrix() @ self.Rxm
        elif state == 8:  # Y+
            Rnew = Rrm.as_matrix() @ self.Ry
        elif state == 9:  # Y-
            Rnew = Rrm.as_matrix() @ self.Rym
        elif state == 10:  # Z+
            Rnew = Rrm.as_matrix() @ self.Rz
        elif state == 11:  # Z-
            Rnew = Rrm.as_matrix() @ self.Rzm
        else:
            return
        Rn = R.from_matrix(Rnew)
        self.orn = Rn.as_quat()
        self.newPosInput = 1

    def _process_gripper_flat(self, state):
        if state == 12 and self.gripper_state != "open":
            self.gripper_state = "open"
            self.gripper_target_pos = self.gripper_open_pos
            self.fing = self.gripper_open_pos
        elif state == 13 and self.gripper_state != "closed":
            self.gripper_state = "closing"  # Start closing process
            self.gripper_target_pos = self.gripper_closed_pos
        
        # Intelligent gripper control with collision detection
        if self.gripper_state == "closing":
            self._smart_gripper_close()
        
        # Apply gripper control with adaptive force based on state
        if self.gripper_state == "closed":
            current_force = self.gripper_hold_force
        elif self.gripper_state == "closing":
            current_force = self.gripper_push_force  # Use very high force when closing to push objects
        else:
            current_force = self.grip_force
        
        for joint in self.jacoFingerJoints:
            p.setJointMotorControl2(
                self.jacoId,
                joint,
                p.POSITION_CONTROL,
                targetPosition=self.fing,
                force=current_force,  # Use higher force when holding objects
                maxVelocity=self.grip_speed,
                positionGain=1.0,
                velocityGain=1.0
            )
    
    def _smart_gripper_close(self):
        """Intelligently close gripper with powerful force to push objects"""
        # Count how many fingers are in contact with objects
        finger_contacts = [False, False, False]  # Track each finger's contact
        total_contacts = 0
        high_force_detected = False
        
        # Check contact forces on finger joints
        for i, joint in enumerate(self.jacoFingerJoints):
            joint_state = p.getJointState(self.jacoId, joint)
            applied_force = abs(joint_state[3])  # Joint reaction force
            
            if applied_force > self.max_gripper_force_threshold:
                high_force_detected = True
                finger_contacts[i] = True
                total_contacts += 1
        
        # Check for contacts between fingers and objects
        for obj_id in self.objects:
            for i, finger_joint in enumerate(self.jacoFingerJoints):
                contacts = p.getContactPoints(bodyA=self.jacoId, bodyB=obj_id, linkIndexA=finger_joint)
                if len(contacts) > 0 and not finger_contacts[i]:
                    finger_contacts[i] = True
                    total_contacts += 1
        
        # Smart closing logic - always close with powerful force
        if total_contacts >= self.min_finger_contacts or high_force_detected or self.fing >= self.gripper_target_pos:
            # We have sufficient contact or reached target - close and hold
            self.gripper_state = "closed"
            self.fing = self.gripper_closed_pos  # Ensure full closure
        else:
            # Continue closing with powerful force to push objects out of the way
            self.fing = min(self.fing + self.gripper_closing_step, self.gripper_target_pos)

    def _apply_workspace_limits(self):
        """Apply workspace limits to robot position with improved Z-axis control"""
        self.pos[0] = np.clip(self.pos[0], self.wl[0], self.wu[0])
        self.pos[1] = np.clip(self.pos[1], self.wl[1], self.wu[1])
        
        # Improved Z-axis limits based on object height
        if self.gripper_state == "closed":
            # When holding an object, prevent going too low
            self.pos[2] = np.clip(self.pos[2], 
                                 self.min_z_height + self.max_object_height/2 + self.grasp_height_offset, 
                                 self.wu[2])
        else:
            # When gripper is open, allow going to grasping height
            self.pos[2] = np.clip(self.pos[2], 
                                 self.min_z_height, 
                                 self.wu[2])
        
    def _update_robot_state(self):
        """Update robot joint positions"""
        if self.newPosInput:
            jointPoses = p.calculateInverseKinematics(
                self.jacoId,
                self.jacoEndEffectorIndex,
                self.pos,
                self.orn,
                jointDamping=self.jd,
                solver=0,
                maxNumIterations=100,
                residualThreshold=.01
            )
            self.JP = list(jointPoses)
            
            # Update joint positions
            for i, joint in enumerate(self.jacoArmJoints):
                p.setJointMotorControl2(
                    self.jacoId, joint,
                    p.POSITION_CONTROL,
                    self.JP[i]
                )
            
            # Update gripper with adaptive force for YCB objects
            if self.gripper_state == "closed":
                current_force = self.gripper_hold_force
            elif self.gripper_state == "closing":
                current_force = self.gripper_push_force  # Use very high force when closing to push objects
            else:
                current_force = self.grip_force
            
            for joint in self.jacoFingerJoints:
                p.setJointMotorControl2(
                    self.jacoId, joint,
                    p.POSITION_CONTROL,
                    targetPosition=self.fing,
                    force=current_force,  # Use higher force when holding objects
                    maxVelocity=self.grip_speed,
                    positionGain=1.0,
                    velocityGain=1.0
                )
                
            self.ls = p.getLinkState(self.jacoId, self.jacoEndEffectorIndex)
            self.newPosInput = 0
            
    def _log_data(self):
        """Log simulation data if enabled"""
        if self.enable_logging:
            lsr = p.getLinkState(self.jacoId, self.jacoEndEffectorIndex)
            lsc = p.getBasePositionAndOrientation(self.tableId)
            ln = [time.time(),
                  lsr[4][0], lsr[4][1], lsr[4][2],
                  lsr[5][0], lsr[5][1], lsr[5][2], lsr[5][3],
                  self.fing,
                  lsc[0][0], lsc[0][1], lsc[0][2],
                  lsc[1][0], lsc[1][1], lsc[1][2], lsc[1][3]]
            ln_rnd = [round(num, 4) for num in ln]
            self.file_obj.writerow(ln_rnd)
            
    def cleanup(self):
        """Cleanup simulation resources"""
        if self.enable_logging and self.log_file:
            self.log_file.close()
        p.disconnect() 

    # ------------------ Shared autonomy helpers ------------------
    def _append_action_history(self, state: int, mode: int) -> None:
        """Append an action entry to the recent history with timestamp."""
        try:
            self.recent_actions.append({
                "t": time.time(),
                "state": int(state) if state is not None else -1,
                "mode": int(mode) if mode is not None else -1,
            })
        except Exception:
            # Be resilient to any unexpected inputs
            pass

    def get_action_history(self, within_seconds=None):
        """Return a list of recent action entries, optionally filtered by window size."""
        if within_seconds is None:
            return list(self.recent_actions)
        now = time.time()
        return [e for e in self.recent_actions if (now - e.get("t", now)) <= within_seconds]

    def get_state(self):
        """Return a snapshot of robot and environment state for shared autonomy.

        Structure:
          {
            "robot_state": {"position": [x,y,z], "orientation": [x,y,z,w], "gripper": str},
            "environment": {"objects": [{"id": int, "position": [x,y,z], "orientation": [x,y,z,w]}]},
            "time": float,
          }
        """
        objects_state = []
        for obj_id in getattr(self, "objects", []) or []:
            try:
                pos, orn = p.getBasePositionAndOrientation(obj_id)
                objects_state.append({
                    "id": obj_id,
                    "position": list(pos),
                    "orientation": list(orn),
                })
            except Exception:
                continue

        return {
            "robot_state": {
                "position": list(self.pos),
                "orientation": list(self.orn),
                "gripper": self.gripper_state,
            },
            "environment": {
                "objects": objects_state,
            },
            "time": time.time(),
        }