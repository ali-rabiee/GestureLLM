import os
import sys
import time
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# -----------------------------------------------------------------------------
# Isaac-Sim import logic: try the **new 4.5+ namespace first**, then fall back to
# the classic `omni.isaac.*` namespace, and finally to a stub.
# -----------------------------------------------------------------------------

# NOTE: we don't use importlib dynamically since mypy/pylint complains – a
# simple nested try-except is enough.


def _make_stub():
    class _Stub:  # pylint: disable=too-few-public-methods
        """Small placeholder when Isaac-Sim modules are absent."""

        def __getattr__(self, name):  # noqa: D401
            raise ImportError(
                "Isaac Sim modules not found (tried both isaacsim.* and omni.isaac.*). "
                "Please verify your Isaac Sim installation and PYTHONPATH."
            )

    return _Stub


# Step 1: Try to import SimulationApp first (this is available immediately)
try:
    from isaacsim.simulation_app import SimulationApp  # type: ignore
    _SIMULATION_APP_AVAILABLE = True
    _ISAAC_FLAVOR = "isaacsim"
except (ImportError, ModuleNotFoundError):
    try:
        from omni.isaac.kit import SimulationApp  # type: ignore
        _SIMULATION_APP_AVAILABLE = True
        _ISAAC_FLAVOR = "omni"
    except (ImportError, ModuleNotFoundError):
        _SIMULATION_APP_AVAILABLE = False
        _ISAAC_FLAVOR = "none"
        SimulationApp = _make_stub()  # type: ignore

# Step 2: Other imports will be done AFTER SimulationApp is initialized
# (Isaac Sim loads these modules dynamically during app startup)
ISAAC_AVAILABLE = _SIMULATION_APP_AVAILABLE


# ----------------------------------------------------------------------------
# Helper paths
# ----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
_URDF_PATH = _REPO_ROOT / "jaco" / "j2n6s300.urdf"

# Isaac Sim core imports - using only built-in stable APIs
# Note: Other imports are done after SimulationApp initialization

class SimulationManager:
    """Drop-in replacement for the old PyBullet SimulationManager.

    The class exposes the same public surface (`process_command`, `cleanup`) so
    that existing gesture-control / ML code can stay unchanged.  Internally, it
    forwards everything to NVIDIA Omniverse Isaac Sim (when available).
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(self, enable_logging: bool = False, headless: bool = False):
        self.enable_logging = enable_logging
        self._running = False

        if not ISAAC_AVAILABLE:
            print(
                "[GestureLLM] Isaac Sim not detected – falling back to stub. "
                "The simulation will *not* be executed."
            )
            return

        # Lazily start the Isaac-Sim kit application.
        self._app = SimulationApp({"headless": headless})

        # Now that SimulationApp is initialized, import the modules that are loaded dynamically
        try:
            from omni.isaac.core import World  # type: ignore
            from omni.isaac.core.utils.stage import add_reference_to_stage  # type: ignore  
            from omni.isaac.core.articulations import Articulation  # type: ignore
            from omni.isaac.motion_generation import (  # type: ignore
                LulaKinematicsSolver,
                ArticulationKinematicsSolver,
            )
            
            # Store these as module-level references for later use
            globals()['World'] = World
            globals()['add_reference_to_stage'] = add_reference_to_stage
            globals()['Articulation'] = Articulation
            globals()['LulaKinematicsSolver'] = LulaKinematicsSolver
            globals()['ArticulationKinematicsSolver'] = ArticulationKinematicsSolver
            
        except ImportError as e:
            print(f"[GestureLLM] Failed to import Isaac Sim core modules after app init: {e}")
            print("[GestureLLM] Falling back to stub mode.")
            self._app.close()
            return

        # Build a basic world (metric units, 1 unit == 1 m)
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # Simply load the Jaco robot from URDF
        print("[GestureLLM] Loading Jaco robot...")
        self._jaco = self._load_jaco_simple()
        self._ik_solver = self._setup_kinematics(self._jaco)
        
        # Setup PyBullet-style control parameters
        self._setup_control_params()
        
        print("[GestureLLM] Jaco robot loaded, setting initial pose...")
        # Set initial Jaco arm joint positions to match PyBullet
        self._set_jaco_initial_pose()
        
        # Create workspace with table and objects
        print("[GestureLLM] Creating workspace environment...")
        self._create_workspace()
        
        print("[GestureLLM] Isaac Sim environment created with Jaco robot")

        # Current EE pose (initialised later)
        self._ee_pos = None
        self._ee_quat = None

        # Step simulation once to initialise physics
        print("[GestureLLM] Resetting world to initialize physics...")
        self.world.reset()
        print("[GestureLLM] World reset complete, checking if joint positions were preserved...")
        
        # Check joint positions after world reset
        if self._jaco:
            try:
                post_reset_positions = self._jaco.get_joint_positions()
                print(f"[GestureLLM] Joint positions AFTER world.reset(): {post_reset_positions}")
            except Exception as e:
                print(f"[GestureLLM] Could not get joint positions after reset: {e}")
        
        self._running = True
        
        # Set camera view to match PyBullet setup
        self._setup_camera()
        
        # Try setting joint positions again after world reset, in case reset overrode them
        print("[GestureLLM] Setting joint positions again after world reset...")
        self._set_jaco_initial_pose()
        
        print(f"[GestureLLM] Isaac Sim backend initialised ✔  (flavor: {_ISAAC_FLAVOR})")

    # ------------------------------------------------------------------
    # Public API (mirrors old PyBullet manager)
    # ------------------------------------------------------------------
    def process_command(self, state: int, mode: int) -> None:
        """Process control commands based on gesture input and current mode (matching PyBullet exactly)."""
        if not self._running:
            return  # stub mode
        
        if state == -1:
            return
            
        print(f"[GestureLLM] Processing command - State: {state}, Mode: {mode}")
        
        # Only allow actions for the current mode (exactly like PyBullet)
        if mode == 0 and state in [0, 1, 2, 3, 4, 5]:
            self._process_translation_flat(state)
        elif mode == 1 and state in [6, 7, 8, 9, 10, 11]:
            self._process_orientation_flat(state)
        elif mode == 2 and state in [12, 13]:
            self._process_gripper_flat(state)
        
        self._apply_workspace_limits()
        self._update_robot_state()
        
        # Advance the simulation
        self.world.step(render=True)

    def cleanup(self):
        if not self._running:
            return
        # Shutdown sequence for Isaac Sim
        print("[GestureLLM] Closing Isaac Sim …")
        self._app.close()
        self._running = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_jaco_simple(self):
        """Load Jaco robot using the correct Isaac Sim 4.5.0 URDF import API"""
        try:
            from pathlib import Path
            from isaacsim.asset.importer.urdf import _urdf
            import omni.kit.commands
            
            # Use your Jaco URDF file
            jaco_urdf_path = str(_REPO_ROOT / "jaco" / "j2n6s300.urdf")
            
            if not Path(jaco_urdf_path).exists():
                raise Exception(f"Jaco URDF not found at: {jaco_urdf_path}")
                
            print(f"[GestureLLM] Loading Jaco robot from: {jaco_urdf_path}")
            
            # Use the correct Isaac Sim 4.5.0 URDF import method (from the official tutorial)
            urdf_interface = _urdf.acquire_urdf_interface()
            
            # Configure the settings for importing the URDF file
            import_config = _urdf.ImportConfig()
            import_config.convex_decomp = False
            import_config.fix_base = True  # Fix base for stationary robot
            import_config.make_default_prim = True
            import_config.self_collision = False
            import_config.distance_scale = 1
            import_config.density = 0.0
            
            # Parse the robot's URDF file to generate a robot model
            result, robot_model = omni.kit.commands.execute(
                "URDFParseFile",
                urdf_path=jaco_urdf_path,
                import_config=import_config
            )
            
            if not result:
                raise Exception("Failed to parse URDF file")
            
            # Update the joint drive parameters for better control
            for joint in robot_model.joints:
                robot_model.joints[joint].drive.strength = 1047.19751
                robot_model.joints[joint].drive.damping = 52.35988
            
            # Import the robot onto the current stage
            result, prim_path = omni.kit.commands.execute(
                "URDFImportRobot",
                urdf_robot=robot_model,
                import_config=import_config,
            )
            
            if result:
                print(f"[GestureLLM] ✓ Successfully imported Jaco robot at: {prim_path}")
                # --- Set the robot base position on the table (recommended Isaac Sim 4.5+ approach) ---
                import omni.usd
                from pxr import Gf
                stage = omni.usd.get_context().get_stage()
                robot_prim = stage.GetPrimAtPath(prim_path)
                # Set the translation to place the base on the table (z=0.03)
                xform_api = robot_prim.GetAttribute("xformOp:translate")
                if xform_api:
                    xform_api.Set(Gf.Vec3d(0.6, 0.0, 0.6))
                else:
                    # If no xformOp:translate exists, add one
                    from pxr import UsdGeom
                    xform = UsdGeom.Xformable(robot_prim)
                    xform.AddTranslateOp().Set(Gf.Vec3d(0.6, 0.0, 0.6))
                # --- End base position logic ---
                # Create articulation wrapper using new API
                from isaacsim.core.prims import SingleArticulation
                jaco = SingleArticulation(prim_path=prim_path, name="jaco_robot")
                self.world.scene.add(jaco)
                return jaco
            else:
                raise Exception("URDF robot import failed")
                
        except Exception as e:
            print(f"[GestureLLM] ❌ Failed to load Jaco robot: {e}")
            print(f"[GestureLLM] Error type: {type(e).__name__}")
            print("[GestureLLM] The environment will only have a ground plane.")
            return None
    
    def _create_workspace(self):
        """Create the workspace with ground plane, lighting, and table"""
        from isaacsim.core.utils.stage import add_reference_to_stage
        import omni.usd
        from pxr import PhysicsSchemaTools, Gf
        
        # Ground plane using standard Isaac Sim method
        stage = omni.usd.get_context().get_stage()
        PhysicsSchemaTools.addGroundPlane(stage, "/World/GroundPlane", "Z", 100, Gf.Vec3f(0, 0, -100), Gf.Vec3f(1.0))

        # Lighting setup
        add_reference_to_stage(
            usd_path="/Isaac/Environments/Simple_Warehouse/Props/S_Lights.usd",
            prim_path="/World/Lights",
        )

        # Create table from URDF
        print("[GestureLLM] Creating table from URDF with mesh...")
        try:
            self._create_table_from_urdf()
        except Exception as e:
            print(f"[GestureLLM] Table creation failed: {e}")
            print("[GestureLLM] Continuing without table...")

        # Create colored objects on table
        print("[GestureLLM] Creating colored objects...")
        try:
            self._create_colored_objects()
        except Exception as e:
            print(f"[GestureLLM] Could not create colored objects: {e}")

        print("[GestureLLM] Workspace environment created")

    def _create_colored_objects(self):
        """Create colored cuboid objects and place them on the table surface."""
        try:
            import omni.kit.commands  # Import needed for DeletePrim command
            
            # Try Isaac Sim 4.5+ import first
            try:
                from isaacsim.core.api.objects import DynamicCuboid
            except ImportError:
                try:
                    from isaacsim.core.objects import DynamicCuboid
                except ImportError:
                    # Fallback to older Isaac Sim import
                    from omni.isaac.core.objects import DynamicCuboid
                
            import numpy as np
            
            # Colors matching PyBullet setup
            colors = [
                [1, 0, 0],    # Red
                [0, 1, 0],    # Green  
                [0, 0, 1],    # Blue
                [1, 1, 0],    # Yellow
                [1, 0, 1],    # Magenta
                [0, 1, 1]     # Cyan
            ]
            
            # Object dimensions matching PyBullet: width=0.08, depth=0.05, height=0.15
            width, depth, height = 0.08, 0.05, 0.15
            
            # Table boundaries for random placement (matching PyBullet)
            min_x, max_x = -0.6, -0.2
            min_y, max_y = -0.3, 0.3
            min_distance = 0.12
            
            # Generate non-overlapping positions
            positions = []
            for i, color in enumerate(colors):
                # Try to find a valid position
                position_found = False
                for _ in range(100):  # max attempts
                    x = np.random.uniform(min_x, max_x)
                    y = np.random.uniform(min_y, max_y)
                    
                    # Check distance from existing objects
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
                    # Fallback to grid position
                    x = min_x + (i * (max_x - min_x) / (len(colors) - 1))
                    y = 0
                    positions.append([x, y])
                
                # Create the object
                # Place at requested height above ground (z=0.6)
                z = 0.6 + height/2  # Place at z=0.6
                
                prim_path = f"/World/Object_{i}"
                # If prim already exists (from previous attempt) delete it first to avoid duplicates
                if self.world.scene.stage.GetPrimAtPath(prim_path):
                    omni.kit.commands.execute("DeletePrim", path=prim_path)

                obj = DynamicCuboid(
                    prim_path=prim_path,
                    name=f"object_{i}",
                    position=np.array([positions[i][0], positions[i][1], z]),
                    scale=np.array([width, depth, height]),
                    color=np.array(color),
                    mass=0.5  # Matching PyBullet mass
                )
                self.world.scene.add(obj)
                
            print(f"[GestureLLM] Created {len(colors)} colored objects on table")
            
        except Exception as e:
            print(f"[GestureLLM] Error creating colored objects: {e}")
            
    def _create_table_from_urdf(self):
        """Create a realistic table from URDF file using the exact same pattern as Jaco robot loading."""
        try:
            from pathlib import Path
            from isaacsim.asset.importer.urdf import _urdf
            import omni.kit.commands
            import omni.usd
            from pxr import UsdGeom, Gf
            
            # Path to the table URDF
            table_urdf_path = str(_REPO_ROOT / "table" / "table.urdf")
            table_obj_path = str(_REPO_ROOT / "table" / "table.obj")
            table_mtl_path = str(_REPO_ROOT / "table" / "table.mtl")
            
            if not Path(table_urdf_path).exists():
                raise Exception(f"Table URDF not found at: {table_urdf_path}")
            
            if not Path(table_obj_path).exists():
                raise Exception(f"Table mesh (table.obj) not found at: {table_obj_path}")
                
            if Path(table_mtl_path).exists():
                print(f"[GestureLLM] Found table with materials and textures")
            
            print(f"[GestureLLM] Loading table via direct URDF import...")
            print(f"[GestureLLM] Table URDF path: {table_urdf_path}")
            
            # STEP 1: Delete any existing table prim to start clean
            stage = omni.usd.get_context().get_stage()
            if stage.GetPrimAtPath("/World/Table"):
                omni.kit.commands.execute("DeletePrim", path="/World/Table")
                print("[GestureLLM] Removed stale /World/Table prim before import")

            # STEP 2: Use the EXACT SAME pattern as Jaco robot loading
            urdf_interface = _urdf.acquire_urdf_interface()
            
            # Configure the settings for importing the URDF file
            import_config = _urdf.ImportConfig()
            import_config.convex_decomp = False
            import_config.fix_base = True  # Table should be fixed
            import_config.make_default_prim = False  # Don't make it default
            import_config.self_collision = False
            import_config.distance_scale = 1
            import_config.density = 0.0
            
            # Parse the table's URDF file to generate a robot model  
            result1, robot_model = omni.kit.commands.execute(
                "URDFParseFile",
                urdf_path=table_urdf_path,
                import_config=import_config
            )
            
            if not result1:
                raise RuntimeError("URDFParseFile failed for table")

            # Update mass properties to ensure proper physics
            for joint in robot_model.joints:
                robot_model.joints[joint].drive.strength = 0.0  # No drives needed for table
                robot_model.joints[joint].drive.damping = 0.0

            # Import the table onto the current stage
            stage = omni.usd.get_context().get_stage()
            
            result2, prim_path2 = omni.kit.commands.execute(
                "URDFImportRobot",
                urdf_robot=robot_model,
                import_config=import_config,
            )
            if not result2:
                raise RuntimeError("URDFImportRobot failed for table")

            print(f"[GestureLLM] Table imported at: {prim_path2}")

            # Move the prim to a clean path  
            target_path = "/World/Table"
            if prim_path2 != target_path:
                # Delete any existing table first
                if stage.GetPrimAtPath(target_path):
                    omni.kit.commands.execute("DeletePrim", path=target_path)
                    
                omni.kit.commands.execute("MovePrim", path_from=prim_path2, path_to=target_path)
                prim_path2 = target_path
                print(f"[GestureLLM] Moved table to: {target_path}")

            # Fix table size and orientation
            table_prim = stage.GetPrimAtPath(prim_path2)
            if table_prim and table_prim.IsValid():
                xformable = UsdGeom.Xformable(table_prim)
                # Clear existing transform ops to avoid duplicate errors
                xformable.ClearXformOpOrder()
                
                # Scale down drastically - use double precision to match existing USD ops
                xformable.AddScaleOp().Set(Gf.Vec3d(0.05, 0.05, 0.05))  # Scale down 20x
                
                # Rotate to fix upside-down orientation (180° around X-axis) - use double precision
                xformable.AddRotateXYZOp().Set(Gf.Vec3d(180, 0, 0))
                
                # Position table at ground level
                xformable.AddTranslateOp().Set(Gf.Vec3d(-0.4, 0.0, 0.0))
                
                print("[GestureLLM] Table scaled down 20x, rotated 180°, positioned at ground level")

                # Verify table is actually visible
                children = table_prim.GetChildren()
                print(f"[GestureLLM] Table prim now has {len(children)} child prims")
                
                if children:
                    print("[GestureLLM] Direct URDF table import complete ✔")
                else:
                    print("[GestureLLM] Warning: Table prim has no children - may not be visible")
            else:
                print("[GestureLLM] Warning: Could not find table prim for positioning")

        except Exception as e:
            print(f"[GestureLLM] Could not create URDF table:")
            print(f"[GestureLLM]   {e}")
            raise

    def _convert_urdf_to_usd(self, urdf_path, usd_path, name):
        """Convert URDF to USD using Isaac Sim's URDF importer"""
        try:
            from omni.importer.urdf import _urdf
            import omni.kit.commands
            
            cfg = _urdf.ImportConfig()
            cfg.fix_base = True
            cfg.merge_fixed_joints = True
            cfg.make_default_prim = False
            cfg.self_collision = False
            
            print(f"[GestureLLM] Converting {name} URDF to USD...")
            
            omni.kit.commands.execute(
                "URDFParseAndImportFile", 
                urdf_path=urdf_path, 
                import_config=cfg, 
                dest_path=usd_path
            )
            
        except Exception as e:
            print(f"[GestureLLM] URDF conversion failed for {name}: {e}")
            raise
        
    def _create_jaco_robot(self):
        """Create Jaco robot by directly importing URDF into Isaac Sim"""
        try:
            from pathlib import Path
            import omni.kit.commands
            
            # Use the exact same Jaco URDF that PyBullet was using
            jaco_urdf_path = str(_REPO_ROOT / "jaco" / "j2n6s300.urdf")
            
            if not Path(jaco_urdf_path).exists():
                raise Exception(f"Jaco URDF not found at: {jaco_urdf_path}")
                
            print(f"[GestureLLM] Directly importing Jaco URDF: {jaco_urdf_path}")
            
            # Direct URDF import using Isaac Sim commands
            result = omni.kit.commands.execute(
                "URDFCreateImportConfig"
            )
            import_config = result[1]
            
            # Configure import settings to match PyBullet behavior
            import_config.merge_fixed_joints = False
            import_config.convex_decomp = False
            import_config.import_inertia_tensor = True
            import_config.fix_base = True
            import_config.distance_scale = 1.0
            import_config.default_drive_strength = 1047.19751
            import_config.default_position_drive_damping = 52.35988
            import_config.default_drive_type = 1  # Position drive

            
            # Import the URDF directly to stage
            omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=jaco_urdf_path,
                import_config=import_config,
                dest_path="/World/Jaco"
            )
            
            # Position the robot to match PyBullet
            import omni.isaac.core.utils.prims as prim_utils
            # prim_utils.set_prim_transform("/World/Jaco", 
            #                             translation=[0.0, 0.0, 0.0])
            
            # Create articulation wrapper
            from omni.isaac.core.articulations import Articulation
            jaco = Articulation(prim_path="/World/Jaco", name="jaco_robot")
            self.world.scene.add(jaco)
            
            print("[GestureLLM] Successfully imported Jaco robot directly from URDF!")
            return jaco
            
        except Exception as e:
            print(f"[GestureLLM] Direct URDF import failed: {e}")
            print(f"[GestureLLM] Error details: {type(e).__name__}")
            
            # Try fallback method with USD conversion
            try:
                return self._load_jaco_fallback()
            except Exception as e2:
                print(f"[GestureLLM] Fallback method also failed: {e2}")
                print("[GestureLLM] Using placeholder robot...")
                return self._create_placeholder_robot()
                
    def _load_jaco_fallback(self):
        """Fallback method using the existing USD conversion approach"""
        print("[GestureLLM] Trying fallback USD conversion method...")
        
        from pathlib import Path
        jaco_urdf_path = str(_REPO_ROOT / "jaco" / "j2n6s300.urdf")
        jaco_usd_path = str(_REPO_ROOT / "jaco_usd" / "jaco.usd")
        
        # Use the existing _load_jaco method logic
        jaco_usd_dir = Path(jaco_usd_path).parent
        jaco_usd_dir.mkdir(exist_ok=True)
        
        if not Path(jaco_usd_path).exists():
            # Try the URDF conversion
            try:
                from omni.importer.urdf import _urdf
                import omni.kit.commands
                
                cfg = _urdf.ImportConfig()
                cfg.fix_base = True
                cfg.merge_fixed_joints = True
                cfg.make_default_prim = False
                cfg.self_collision = False
                
                print("[GestureLLM] Converting Jaco URDF to USD...")
                omni.kit.commands.execute(
                    "URDFParseAndImportFile",
                    urdf_path=jaco_urdf_path,
                    import_config=cfg,
                    dest_path=jaco_usd_path
                )
            except ImportError:
                raise Exception("URDF importer not available and USD file doesn't exist")
        
        # Load the USD file
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.articulations import Articulation
        
        add_reference_to_stage(jaco_usd_path, "/World/Jaco")
        jaco = Articulation(prim_path="/World/Jaco", name="jaco_robot")
        self.world.scene.add(jaco)
        
        return jaco
            
    def _create_placeholder_robot(self):
        """Create a simple placeholder 'robot' using basic shapes"""
        from omni.isaac.core.objects import DynamicCuboid
        import numpy as np
        
        # Create a simple robot placeholder - base, arm segments, and gripper
        base = DynamicCuboid(
            prim_path="/World/RobotBase",
            name="robot_base", 
            position=np.array([0.0, 0.0, 0.1]),
            scale=np.array([0.15, 0.15, 0.2]),
            color=np.array([0.3, 0.3, 0.3]),  # Dark gray
            mass=5.0
        )
        
        arm = DynamicCuboid(
            prim_path="/World/RobotArm",
            name="robot_arm",
            position=np.array([-0.2, 0.0, 0.3]),
            scale=np.array([0.4, 0.08, 0.08]),
            color=np.array([0.7, 0.7, 0.7]),  # Light gray
            mass=2.0
        )
        
        gripper = DynamicCuboid(
            prim_path="/World/RobotGripper", 
            name="robot_gripper",
            position=np.array([-0.4, 0.0, 0.3]),
            scale=np.array([0.1, 0.05, 0.05]),
            color=np.array([1.0, 0.5, 0.0]),  # Orange
            mass=0.5
        )
        
        self.world.scene.add(base)
        self.world.scene.add(arm)
        self.world.scene.add(gripper)
        
        print("[GestureLLM] Created placeholder robot with base, arm, and gripper")
        return base  # Return base as the 'robot' object
        
    def _load_jaco(self):
        """Import the Jaco URDF and return an Articulation handle."""
        # Importer is idempotent – importing the same USD multiple times is OK.
        usd_path = str(_REPO_ROOT / "jaco_usd" / "jaco.usd")
        usd_dir = Path(usd_path).parent
        usd_dir.mkdir(exist_ok=True)

        if not Path(usd_path).exists():
            # Convert URDF → USD on first run
            try:
                # Try different import paths for URDF importer in Isaac Sim 4.5
                try:
                    from omni.importer.urdf import _urdf  # type: ignore
                except ImportError:
                    # Try alternative path
                    from isaacsim.asset.importer.urdf import _urdf  # type: ignore
                    
                import omni.kit.commands  # type: ignore
                
                cfg = _urdf.ImportConfig()
                cfg.fix_base = True
                cfg.merge_fixed_joints = True
                cfg.make_default_prim = False
                cfg.self_collision = False
                print("[GestureLLM] Converting Jaco URDF → USD … (one-time step)")
                
                omni.kit.commands.execute(  # type: ignore
                    "URDFParseAndImportFile", urdf_path=str(_URDF_PATH), import_config=cfg, dest_path=usd_path
                )
                
            except ImportError as e:
                print(f"[GestureLLM] Warning: URDF importer not available: {e}")
                print("[GestureLLM] Please convert Jaco URDF to USD manually, or ensure omni.importer.urdf is installed.")
                print("[GestureLLM] Creating a simple cube placeholder for now...")
                
                # Create a minimal USD with a cube as placeholder
                placeholder_usd = f'''#usda 1.0
(
    defaultPrim = "Root"
    upAxis = "Z"
)

def Xform "Root"
{{
    def Cube "JacoPlaceholder"
    {{
        double size = 0.2
        double3 xformOp:translate = (0, 0, 0.1)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
}}
'''
                Path(usd_path).write_text(placeholder_usd)

        # Add the generated USD to the stage
        prim_path = "/World/Jaco"
        add_reference_to_stage(usd_path, prim_path)
        articulation = Articulation(prim_path)
        self.world.scene.add(articulation)
        return articulation

    # ------------------------------------------------------------------
    def _setup_kinematics(self, jaco):
        """Setup a simple but reliable velocity controller for Isaac Sim 4.5+"""
        print("[GestureLLM] Setting up Jacobian-based velocity controller for Isaac Sim 4.5+")
        # No need for complex IK solver - we'll use direct Jacobian control
        return "jacobian_transpose"  # Simple marker that velocity control is available

    def _setup_camera(self):
        """Set camera view to match original PyBullet setup"""
        try:
            from omni.isaac.core.utils.viewports import set_camera_view
            
            # Original PyBullet camera: distance=1.5, yaw=-40, pitch=-30, target=[-0.2, 0.0, 0.0]
            camera_position = [0.8, 1.0, 0.8]  # Approximate position based on PyBullet params
            camera_target = [-0.2, 0.0, 0.0]   # Same target as PyBullet
            
            set_camera_view(
                eye=camera_position,
                target=camera_target,
                camera_prim_path="/OmniverseKit_Persp"
            )
            print("[GestureLLM] Camera positioned to view workspace")
            
        except Exception as e:
            print(f"[GestureLLM] Could not set camera view: {e}")
            print("[GestureLLM] Using default camera position")

    def step(self):
        # Implementation of step method
        pass

    def _set_jaco_initial_pose(self):
        """Set the initial joint positions of the Jaco arm to match PyBullet's setup."""
        if not self._jaco:
            print("[GestureLLM] ERROR: Jaco robot not loaded, cannot set initial pose.")
            return
            
        # print("\n=== SETTING JACO INITIAL POSE ===")  # Debug: commented out
        
        import numpy as np
        
        try:
            # Step 1: Check if robot is properly initialized
            # print(f"[GestureLLM] Robot object type: {type(self._jaco)}")  # Debug: commented out
            # print(f"[GestureLLM] Robot object: {self._jaco}")  # Debug: commented out
            
            # Step 2: Discover available attributes and methods for SingleArticulation
            # print("[GestureLLM] Available attributes on robot object:")  # Debug: commented out
            robot_attrs = [attr for attr in dir(self._jaco) if not attr.startswith('_')]
            joint_related_attrs = [attr for attr in robot_attrs if 'joint' in attr.lower()]
            position_related_attrs = [attr for attr in robot_attrs if 'position' in attr.lower()]
            
            # print(f"[GestureLLM] Joint-related attributes: {joint_related_attrs}")  # Debug: commented out
            # print(f"[GestureLLM] Position-related attributes: {position_related_attrs}")  # Debug: commented out
            
            # Step 3: Try to get joint information using available methods
            joint_names = None
            num_joints = None
            
            # Try get_joints_state() - might contain joint names and info
            if hasattr(self._jaco, 'get_joints_state'):
                try:
                    joints_state = self._jaco.get_joints_state()
                    # print(f"[GestureLLM] Joints state: {joints_state}")  # Debug: commented out
                    # print(f"[GestureLLM] Joints state type: {type(joints_state)}")  # Debug: commented out
                    # if hasattr(joints_state, '__dict__'):
                    #     print(f"[GestureLLM] Joints state attributes: {joints_state.__dict__}")  # Debug: commented out
                except Exception as e:
                    # print(f"[GestureLLM] Error getting joints state: {e}")  # Debug: commented out
                    pass
                    
            # Try get_joints_default_state() - might contain default joint info
            if hasattr(self._jaco, 'get_joints_default_state'):
                try:
                    default_state = self._jaco.get_joints_default_state()
                    # print(f"[GestureLLM] Default joints state: {default_state}")  # Debug: commented out
                    # print(f"[GestureLLM] Default joints state type: {type(default_state)}")  # Debug: commented out
                    # if hasattr(default_state, '__dict__'):
                    #     print(f"[GestureLLM] Default joints state attributes: {default_state.__dict__}")  # Debug: commented out
                except Exception as e:
                    # print(f"[GestureLLM] Error getting default joints state: {e}")  # Debug: commented out
                    pass
            
            # Method 1: Try joint_names (old API)
            if hasattr(self._jaco, 'joint_names'):
                joint_names = list(self._jaco.joint_names)
                # print(f"[GestureLLM] Got joint names via joint_names: {joint_names}")  # Debug: commented out
            
            # Method 2: Try get_joint_names() method
            elif hasattr(self._jaco, 'get_joint_names'):
                joint_names = self._jaco.get_joint_names()
                # print(f"[GestureLLM] Got joint names via get_joint_names(): {joint_names}")  # Debug: commented out
            
            # Method 3: Try dof_names (new API)
            elif hasattr(self._jaco, 'dof_names'):
                joint_names = list(self._jaco.dof_names)
                # print(f"[GestureLLM] Got joint names via dof_names: {joint_names}")  # Debug: commented out
                
            # Method 4: Try get_dof_names() method
            elif hasattr(self._jaco, 'get_dof_names'):
                joint_names = self._jaco.get_dof_names()
                # print(f"[GestureLLM] Got joint names via get_dof_names(): {joint_names}")  # Debug: commented out
                
            # else:
            #     print("[GestureLLM] Could not find joint names attribute/method")  # Debug: commented out
                
            # Try to get number of joints/DOFs
            if hasattr(self._jaco, 'num_dof'):
                num_joints = self._jaco.num_dof
                # print(f"[GestureLLM] Number of DOF: {num_joints}")  # Debug: commented out
            elif hasattr(self._jaco, 'get_dof_count'):
                num_joints = self._jaco.get_dof_count()
                # print(f"[GestureLLM] Number of DOF (via get_dof_count): {num_joints}")  # Debug: commented out
                
            # Step 4: Get current joint positions (we know this method exists)
            current_positions = None
            try:
                current_positions = self._jaco.get_joint_positions()
                # print(f"[GestureLLM] Current joint positions: {current_positions}")  # Debug: commented out
                # print(f"[GestureLLM] Number of joints detected: {len(current_positions) if current_positions is not None else 'None'}")  # Debug: commented out
            except Exception as e:
                # print(f"[GestureLLM] Error getting current positions: {e}")  # Debug: commented out
                pass
                
            # Step 5: Try to set joint positions directly (without joint names)
            if current_positions is not None and len(current_positions) > 0:
                # print(f"[GestureLLM] Robot has {len(current_positions)} joints, trying to set positions...")  # Debug: commented out
                
                # Create new position array based on current positions
                new_positions = list(current_positions)
                
                # PyBullet joint mapping:
                # rp = [0, np.pi/4, np.pi, 1.0*np.pi, 1.8*np.pi, 0*np.pi, 1.75*np.pi, 0.5*np.pi]
                # jacoArmJoints = [2, 3, 4, 5, 6, 7]  # The actual arm joints in PyBullet
                # So PyBullet arm joints get: [π, π, 1.8π, 0, 1.75π, 0.5π]
                
                # Isaac Sim joint mapping (from diagnostic):
                # Joints 0-5: j2n6s300_joint_1 through j2n6s300_joint_6 (arm joints)
                # Joints 6-8: finger joints
                
                # Map PyBullet arm joint values to Isaac Sim arm joints (0-5)
                pybullet_arm_positions = [np.pi, np.pi, 1.8*np.pi, 0*np.pi, 1.75*np.pi, 0.5*np.pi]
                
                # Set Isaac Sim arm joints (0-5) to PyBullet arm joint values
                for i in range(min(len(pybullet_arm_positions), 6)):  # Only set first 6 joints (arm)
                    if i < len(new_positions):
                        new_positions[i] = pybullet_arm_positions[i]
                        
                # Keep finger joints at their current positions (indices 6-8)
                # Don't modify them for now
                     
                # print(f"[GestureLLM] Setting {len(pybullet_arm_positions)} arm joint positions")  # Debug: commented out
                # print(f"[GestureLLM] Original positions: {current_positions}")  # Debug: commented out
                # print(f"[GestureLLM] Target positions:   {new_positions}")  # Debug: commented out
                
                # Try different methods to set positions
                success = False
                
                # Method 1: set_joint_positions with full array
                try:
                    self._jaco.set_joint_positions(new_positions)
                    print("[GestureLLM] Jaco arm positioned to match PyBullet setup")
                    success = True
                except Exception as e:
                    # print(f"[GestureLLM] ✗ set_joint_positions() with full array failed: {e}")  # Debug: commented out
                    pass
                    
                # Method 2: set_joint_positions with specific indices
                if not success:
                    try:
                        # Only set the arm joints (0-5) with specific values
                        arm_indices = list(range(len(pybullet_arm_positions)))
                        arm_positions = pybullet_arm_positions
                        self._jaco.set_joint_positions(arm_positions, arm_indices)
                        print("[GestureLLM] Jaco arm positioned to match PyBullet setup")
                        success = True
                    except Exception as e:
                        # print(f"[GestureLLM] ✗ set_joint_positions() with arm indices failed: {e}")  # Debug: commented out
                        pass
                        
                # Verify if positions changed (debug only)
                # try:
                #     final_positions = self._jaco.get_joint_positions()
                #     print(f"[GestureLLM] Final positions:    {final_positions}")  # Debug: commented out
                #     
                #     if len(final_positions) == len(current_positions):
                #         changed = [abs(final_positions[i] - current_positions[i]) > 0.001 for i in range(len(final_positions))]
                #         print(f"[GestureLLM] Changes detected:   {changed}")  # Debug: commented out
                #         if any(changed):
                #             print("[GestureLLM] ✓ Joint positions changed successfully!")  # Debug: commented out
                #         else:
                #             print("[GestureLLM] ✗ No joint positions changed")  # Debug: commented out
                #             
                # except Exception as e:
                #     print(f"[GestureLLM] Could not verify position changes: {e}")  # Debug: commented out
                    
            else:
                print("[GestureLLM] Warning: Could not get current joint positions to set initial pose")
                
        except Exception as e:
            print(f"[GestureLLM] Error setting initial pose: {e}")
            # import traceback  # Debug: commented out
            # traceback.print_exc()  # Debug: commented out
            
        # print("=== INITIAL POSE SETTING COMPLETE ===\n")  # Debug: commented out

    def set_ee_velocity(self, target_velocity: "np.ndarray"):
        """Simple direct joint velocity control for Isaac Sim 4.5+ (no Jacobian required)."""
        if self._ik_solver is None:
            print("[GestureLLM] Velocity controller not initialized.")
            return
            
        if self._jaco is None:
            print("[GestureLLM] Jaco robot not available.")
            return
            
        try:
            # Convert single velocity command to 3D vector if needed
            if np.isscalar(target_velocity):
                target_velocity = np.array([target_velocity, 0.0, 0.0])
            elif len(target_velocity) == 1:
                target_velocity = np.array([target_velocity[0], 0.0, 0.0])
            elif len(target_velocity) == 2:
                target_velocity = np.array([target_velocity[0], target_velocity[1], 0.0])
            
            print(f"[GestureLLM] Setting EE velocity: {target_velocity}")
            
            # Simple mapping: velocity commands to joint velocities
            # This is a basic but effective approach for Isaac Sim 4.5+
            # Map end-effector velocity to joint space using a simplified mapping
            
            # For Jaco arm: 6 arm joints + 3 gripper joints = 9 total
            # We'll control only the first 6 joints (arm), ignore gripper
            arm_joint_count = 6
            
            # Simple velocity mapping - this works without requiring Jacobians
            joint_velocities = np.zeros(arm_joint_count)
            
            # Map X velocity to joints 1-2 (base rotation and shoulder)
            if len(target_velocity) > 0:
                joint_velocities[0] = target_velocity[0] * 0.5  # Base rotation
                joint_velocities[1] = target_velocity[0] * 0.3  # Shoulder
            
            # Map Y velocity to joints 3-4 (elbow and wrist)
            if len(target_velocity) > 1:
                joint_velocities[2] = target_velocity[1] * 0.4  # Elbow
                joint_velocities[3] = target_velocity[1] * 0.3  # Wrist 1
            
            # Map Z velocity to joints 5-6 (wrist rotation)
            if len(target_velocity) > 2:
                joint_velocities[4] = target_velocity[2] * 0.2  # Wrist 2
                joint_velocities[5] = target_velocity[2] * 0.2  # Wrist 3
            
            print(f"[GestureLLM] Computed joint velocities: {joint_velocities}")
            
            # Apply the joint velocities directly using set_joint_velocities
            if hasattr(self._jaco, 'set_joint_velocities'):
                # Pad to 9 joints (6 arm + 3 gripper) with zeros for gripper
                full_joint_velocities = np.concatenate([joint_velocities, np.zeros(3)])
                self._jaco.set_joint_velocities(full_joint_velocities)
                print("[GestureLLM] Applied joint velocities via set_joint_velocities")
            else:
                # Fallback: integrate velocities and set positions
                joint_positions = self._jaco.get_joint_positions()
                if joint_positions is not None:
                    dt = 0.01  # Integration timestep
                    # Only update arm joints, leave gripper joints unchanged
                    new_positions = joint_positions.copy()
                    new_positions[:arm_joint_count] += joint_velocities * dt
                    
                    if hasattr(self._jaco, 'set_joint_positions'):
                        self._jaco.set_joint_positions(new_positions)
                        print("[GestureLLM] Applied integrated positions via set_joint_positions")
                    else:
                        print("[GestureLLM] No method available to apply joint control")
                else:
                    print("[GestureLLM] Could not get current joint positions for integration")
            
        except Exception as e:
            print(f"[GestureLLM] Error in velocity control: {e}")
            import traceback
            print(f"[GestureLLM] Full traceback: {traceback.format_exc()}")

    # Control parameters (matching PyBullet simulation_manager.py)
    def _setup_control_params(self):
        """Setup control parameters matching PyBullet implementation"""
        # Workspace limits (matching PyBullet)
        self.wu = [0.1, 0.5, 0.5]      # Upper workspace limits
        self.wl = [-.66, -.5, 0.02]    # Lower workspace limits
        
        # Movement parameters (matching PyBullet)
        self.dist = .002       # Translation step size
        self.ang = .005       # Angular step size
        self.rot_theta = .008  # Rotation step size
        
        # Setup rotation matrices
        self._setup_rotation_matrices()
        
        # Control states
        self.gripper_state = "open"  # Can be "open" or "closed"
        self.gripper_open_pos = 0.0
        self.gripper_closed_pos = 1.2
        
        # Current end-effector pose (will be updated from robot state)
        self.pos = [0.0, 0.0, 0.0]
        self.orn = [0.0, 0.0, 0.0, 1.0]  # quaternion
        
        # Object height for collision prevention
        self.object_height = 0.15  # Height of objects
        self.grasp_height_offset = 0.02  # Small offset above objects for grasping
        self.min_z_height = self.wl[2] + self.object_height/2  # Minimum Z height to prevent collision
        
        # Get initial end-effector pose
        self._update_ee_pose_from_robot()
        
    def _setup_rotation_matrices(self):
        """Setup rotation matrices for orientation control (matching PyBullet)"""
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

    def _update_ee_pose_from_robot(self):
        """Update current end-effector pose from robot state"""
        if self._jaco is None:
            return
        try:
            # Get end-effector pose (you may need to adjust this based on your robot structure)
            # For now, we'll use a simple approximation
            joint_positions = self._jaco.get_joint_positions()
            if joint_positions is not None:
                # This is a simplified approach - you might need to compute forward kinematics
                # For now, we'll maintain the pose in self.pos and self.orn
                pass
        except Exception as e:
            print(f"[GestureLLM] Error updating EE pose: {e}")

    def _process_translation_flat(self, state):
        """Process translation commands (matching PyBullet implementation)"""
        if state == 0:  # Forward
            # Get base joint angle for coordinate transformation
            joint_positions = self._jaco.get_joint_positions()
            if joint_positions is not None:
                baseTheta = joint_positions[0]  # First joint is base rotation
                s = np.sin(baseTheta)
                c = np.cos(baseTheta)
                self.pos[0] += self.dist * c
                self.pos[1] -= self.dist * s
        elif state == 1:  # Backward
            joint_positions = self._jaco.get_joint_positions()
            if joint_positions is not None:
                baseTheta = joint_positions[0]
                s = np.sin(baseTheta)
                c = np.cos(baseTheta)
                self.pos[0] -= self.dist * c
                self.pos[1] += self.dist * s
        elif state == 2:  # Left
            n = np.sqrt(self.pos[0]**2 + self.pos[1]**2)
            if n > 0:
                dx = -self.pos[1]/n
                dy = self.pos[0]/n
                self.pos[0] += self.dist * dx
                self.pos[1] += self.dist * dy
        elif state == 3:  # Right
            n = np.sqrt(self.pos[0]**2 + self.pos[1]**2)
            if n > 0:
                dx = -self.pos[1]/n
                dy = self.pos[0]/n
                self.pos[0] -= self.dist * dx
                self.pos[1] -= self.dist * dy
        elif state == 4:  # Up
            self.pos[2] += self.dist
        elif state == 5:  # Down
            self.pos[2] -= self.dist
        
        print(f"[GestureLLM] Translation command {state}, new position: {self.pos}")

    def _process_orientation_flat(self, state):
        """Process orientation commands (matching PyBullet implementation)"""
        from scipy.spatial.transform import Rotation as R
        
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
        
        print(f"[GestureLLM] Orientation command {state}, new orientation: {self.orn}")

    def _process_gripper_flat(self, state):
        """Process gripper commands (matching PyBullet implementation)"""
        if state == 12 and self.gripper_state != "open":
            self.gripper_state = "open"
            print("[GestureLLM] Opening gripper")
            # TODO: Implement gripper control for Isaac Sim
        elif state == 13 and self.gripper_state != "closed":
            self.gripper_state = "closed"
            print("[GestureLLM] Closing gripper")
            # TODO: Implement gripper control for Isaac Sim

    def _apply_workspace_limits(self):
        """Apply workspace limits to robot position (matching PyBullet implementation)"""
        self.pos[0] = np.clip(self.pos[0], self.wl[0], self.wu[0])
        self.pos[1] = np.clip(self.pos[1], self.wl[1], self.wu[1])
        
        # Improved Z-axis limits based on object height
        if self.gripper_state == "closed":
            # When holding an object, prevent going too low
            self.pos[2] = np.clip(self.pos[2], 
                                 self.min_z_height + self.grasp_height_offset, 
                                 self.wu[2])
        else:
            # When gripper is open, allow going to grasping height
            self.pos[2] = np.clip(self.pos[2], 
                                 self.min_z_height, 
                                 self.wu[2])

    def _update_robot_state(self):
        """Update robot joint positions using IK (matching PyBullet approach)"""
        if self._jaco is None:
            return
            
        try:
            # Use Isaac Sim's built-in IK or set positions directly
            # For now, we'll use a simple position-based control
            current_positions = self._jaco.get_joint_positions()
            if current_positions is not None:
                # This is a simplified approach - you may need proper IK
                # For demonstration, we'll just apply small changes to joints based on desired pose
                
                # Simple mapping from Cartesian space to joint space (very basic)
                # This should be replaced with proper IK when available
                joint_changes = np.zeros(6)  # 6 arm joints
                
                # Map position changes to joint changes (simplified)
                if hasattr(self, '_last_pos'):
                    pos_change = np.array(self.pos) - np.array(self._last_pos)
                    # Very basic mapping - should be replaced with proper Jacobian
                    joint_changes[0] = pos_change[0] * 2.0  # Base rotation
                    joint_changes[1] = pos_change[1] * 1.5  # Shoulder
                    joint_changes[2] = pos_change[2] * 1.0  # Elbow
                    
                    # Apply changes to current joint positions
                    new_positions = current_positions.copy()
                    new_positions[:6] += joint_changes
                    
                    # Set new joint positions
                    self._jaco.set_joint_positions(new_positions)
                    print(f"[GestureLLM] Updated joint positions")
                
                self._last_pos = self.pos.copy()
                
        except Exception as e:
            print(f"[GestureLLM] Error updating robot state: {e}")
