import pybullet as p
import pybullet_data
import numpy as np
from datetime import datetime
import time
import os
import csv
from scipy.spatial.transform import Rotation as R

class SimulationManager:
    def __init__(self, enable_logging=False):
        self.enable_logging = enable_logging
        self.setup_simulation()
        self.setup_robot()
        self.setup_workspace()
        self.setup_control_params()
        self.setup_logging()
        
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
        
        # Improved physics parameters for better grasping
        p.setPhysicsEngineParameter(numSolverIterations=50)
        p.setPhysicsEngineParameter(enableConeFriction=1)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        
        # Increased gravity for better stability
        p.setGravity(0, 0, -15)
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
        self.jd = [1.0] * 10  # Increased joint damping for stability
        self.rp = [0, np.pi/4, np.pi, 1.0*np.pi, 1.8*np.pi, 0*np.pi, 1.75*np.pi, 0.5*np.pi]
        
        # Initialize robot pose
        self.reset_robot_pose()
        
        # Get initial state
        self.ls = p.getLinkState(self.jacoId, self.jacoEndEffectorIndex)
        self.pos = list(self.ls[4])
        self.orn = list(self.ls[5])
        
        # Improve gripper friction
        for joint in self.jacoFingerJoints:
            p.changeDynamics(self.jacoId, joint,
                           lateralFriction=1.0,
                           spinningFriction=1.0,
                           rollingFriction=0.1,
                           frictionAnchor=1)
        
    def setup_workspace(self):
        """Setup workspace and objects"""
        # Load ground plane and table
        p.loadURDF("plane.urdf", [0,0,-.65])
        self.tableId = p.loadURDF("table/table.urdf", basePosition=[-0.4,0.0,-0.65])
        
        # Define object scales and colors
        colors = [
            [1, 0, 0, 1],    # Red
            [0, 1, 0, 1],    # Green
            [0, 0, 1, 1],    # Blue
            [1, 1, 0, 1],    # Yellow
            [1, 0, 1, 1],    # Magenta
            [0, 1, 1, 1]     # Cyan
        ]
        
        # Create standing rectangular objects
        self.objects = []
        
        # Parameters for rectangular blocks - increased size
        width = 0.05   # Increased from 0.03
        depth = 0.05   # Increased from 0.03
        height = 0.15  # Increased from 0.09
        
        # Define table boundaries for random placement
        min_x = -0.6
        max_x = -0.2
        min_y = -0.3
        max_y = 0.3
        
        # Minimum distance between objects to prevent overlap
        min_distance = 0.12
        
        # List to store positions for collision checking
        positions = []
        for i, color in enumerate(colors):
            # Create visual and collision shapes for standing rectangular block
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[width/2, depth/2, height/2],
                rgbaColor=color)
            collision_shape_id = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[width/2, depth/2, height/2])
            
            
            # Try to find a valid position that doesn't overlap with other objects
            max_attempts = 100
            position_found = False
            
            for _ in range(max_attempts):
                # Generate random position
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                
                # Check distance from all existing objects
                valid_position = True
                for pos in positions:
                    distance = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                    if distance < min_distance:
                        valid_position = False
                        break
                
                if valid_position:
                    positions.append([x, y])
                    position_found = True
                    break
            
            if not position_found:
                # If no valid position found, use backup position
                x = min_x + (i * (max_x - min_x) / (len(colors) - 1))
                y = 0
                positions.append([x, y])
            
            # Random rotation around vertical axis
            rotation = np.random.uniform(0, 2 * np.pi)
            
            # Create the object slightly above the table to let it fall into place
            obj_id = p.createMultiBody(
                baseMass=0.2,  # Slightly heavier for stability
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=[x, y, height/2],
                baseOrientation=p.getQuaternionFromEuler([0, 0, rotation]))
            
            self.objects.append(obj_id)
            
            # Set dynamics properties for better stability
            p.changeDynamics(obj_id, -1,
                           lateralFriction=1.0,
                           spinningFriction=1.0,
                           rollingFriction=0.1,
                           restitution=0.1,
                           contactStiffness=10000,
                           contactDamping=100,
                           mass=0.2)
        
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
        self.wl = [-.66, -.5, 0.00]    # Lower workspace limits
        
        # Movement parameters
        self.dist = .002       # Translation step size
        self.ang = .005       # Angular step size
        self.rot_theta = .008  # Rotation step size
        
        # Setup rotation matrices
        self._setup_rotation_matrices()
        
        # Control states
        self.JP = list(self.rp[2:9])
        self.fing = 0
        self.newPosInput = 1
        
        # Improved gripper parameters
        self.grip_force = 100.0  # Increased gripping force
        self.grip_speed = 2.0    # Increased gripper speed (was 0.5)
        
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
        """Process control commands based on gesture input"""
        if state == -1:
            return
            
        baseTheta = self.JP[0]
        s = np.sin(baseTheta)
        c = np.cos(baseTheta)
        
        n = np.sqrt(self.pos[0]**2 + self.pos[1]**2)
        dx = -self.pos[1]/n if n > 0 else 0
        dy = self.pos[0]/n if n > 0 else 0
        
        Rrm = R.from_quat(self.orn)
        Rnew = Rrm.as_matrix()
        
        # Process commands based on mode
        if mode == 0:  # Translation mode
            self._process_translation(state, dx, dy, s, c)
        elif mode == 1:  # Orientation mode
            self._process_orientation(state, Rrm)
        elif mode == 2:  # Gripper mode
            self._process_gripper(state)
            
        # Apply workspace limits
        self._apply_workspace_limits()
        
        # Update robot state
        self._update_robot_state()
        
        # Log data if enabled
        self._log_data()
        
    def _process_translation(self, state, dx, dy, s, c):
        """Process translation commands"""
        if state == 4:
            self.pos[0] += self.dist * dx
            self.pos[1] += self.dist * dy
        elif state == 6:
            self.pos[0] -= self.dist * dx
            self.pos[1] -= self.dist * dy
        elif state == 8:
            self.pos[0] += self.dist * c
            self.pos[1] -= self.dist * s
        elif state == 2:
            self.pos[0] -= self.dist * c
            self.pos[1] += self.dist * s
        elif state == 7:
            self.pos[2] += self.dist
        elif state == 1:
            self.pos[2] -= self.dist
        self.newPosInput = 1
        
    def _process_orientation(self, state, Rrm):
        """Process orientation commands"""
        if state == 4:
            Rnew = Rrm.as_matrix() @ self.Rz
        elif state == 6:
            Rnew = Rrm.as_matrix() @ self.Rzm
        elif state == 8:
            Rnew = Rrm.as_matrix() @ self.Rx
        elif state == 2:
            Rnew = Rrm.as_matrix() @ self.Rxm
        elif state == 7:
            Rnew = Rrm.as_matrix() @ self.Ry
        elif state == 1:
            Rnew = Rrm.as_matrix() @ self.Rym
        else:
            return
            
        Rn = R.from_matrix(Rnew)
        self.orn = Rn.as_quat()
        self.newPosInput = 1
        
    def _process_gripper(self, state):
        """Process gripper commands with improved control"""
        if state == 8:
            self.fing = max(0, self.fing - self.dist * 10 * self.grip_speed)  # Increased speed multiplier
        elif state == 2:
            self.fing = min(1.35, self.fing + self.dist * 10 * self.grip_speed)  # Increased speed multiplier
            
        # Apply grip force
        for joint in self.jacoFingerJoints:
            p.setJointMotorControl2(
                self.jacoId,
                joint,
                p.POSITION_CONTROL,
                targetPosition=self.fing,
                force=self.grip_force
            )
            
    def _apply_workspace_limits(self):
        """Apply workspace limits to robot position"""
        self.pos[0] = np.clip(self.pos[0], self.wl[0], self.wu[0])
        self.pos[1] = np.clip(self.pos[1], self.wl[1], self.wu[1])
        self.pos[2] = np.clip(self.pos[2], self.wl[2], self.wu[2])
        self.fing = np.clip(self.fing, 0, 1.35)
        
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
            
            # Update gripper
            for joint in self.jacoFingerJoints:
                p.setJointMotorControl2(
                    self.jacoId, joint,
                    p.POSITION_CONTROL,
                    self.fing
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