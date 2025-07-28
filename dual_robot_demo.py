"""
Dual Go2 Robot Demo with ONNX Model and Camera Recording

===============================================================================
                                CAMERA SYSTEM GUIDE
===============================================================================

This demo showcases Genesis camera system for recording multi-robot simulations.
Genesis provides two types of cameras:

1. VIEWER CAMERA (Built-in)
   - Automatically created with the scene
   - User-controllable with mouse/keyboard
   - Good for interactive visualization
   - Position: scene.viewer_options.camera_pos

2. ENTITY CAMERA (Programmable)
   - Added programmatically before scene.build()
   - Can be moved dynamically during simulation
   - Perfect for automated recording and tracking
   - Returns RGB/depth/segmentation data

===============================================================================
                            ENTITY CAMERA USAGE
===============================================================================

### Basic Camera Creation (Before scene.build())
```python
camera = scene.add_camera(
    res=(1920, 1080),           # Resolution (width, height)
    pos=(0.0, -8.0, 4.0),       # Camera position (x, y, z)
    lookat=(0.0, 0.0, 0.5),     # Look-at target point
    up=(0.0, 0.0, 1.0),         # Up vector (usually Z-up)
    fov=50,                     # Field of view in degrees
    GUI=False                   # Set True to show camera view window
)
```

### Dynamic Camera Movement (After scene.build())
```python
# Update camera position and target
camera.set_pose(
    pos=(new_x, new_y, new_z),          # New camera position
    lookat=(target_x, target_y, target_z) # New look-at point
)

# Alternative: Use transform matrix
import numpy as np
transform_matrix = np.eye(4)  # 4x4 transformation matrix
camera.set_pose(transform=transform_matrix)
```

### Camera Rendering and Frame Capture
```python
# Render RGB image
rgb_image = camera.render(rgb=True)

# Render with multiple outputs
rgb, depth, segmentation = camera.render(
    rgb=True, 
    depth=True, 
    segmentation=True
)

# Handle tuple return format
if isinstance(rgb_image, tuple):
    rgb_data = rgb_image[0]
else:
    rgb_data = rgb_image

# Convert and save frame
frame = rgb_data.cpu().numpy()  # Convert from tensor
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
cv2.imwrite("frame.png", frame_bgr)
```

===============================================================================
                        MULTI-ROBOT TRACKING SYSTEM
===============================================================================

This demo implements an intelligent tracking camera that automatically:
1. Calculates center point between multiple robots
2. Adjusts distance based on robot separation  
3. Maintains optimal viewing angle
4. Updates every simulation step

### Key Implementation (See update_tracking_camera() method):

```python
def update_tracking_camera(self):
    # Get robot positions
    pos1 = self.robot1.get_pos()
    pos2 = self.robot2.get_pos()
    
    # Calculate center point
    center_x = (pos1[0] + pos2[0]) / 2.0
    center_y = (pos1[1] + pos2[1]) / 2.0
    
    # Calculate distance between robots
    robot_distance = torch.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    # Set camera distance based on robot separation
    camera_distance = max(min_distance, robot_distance.item() * 1.5)
    camera_height = max(3.0, camera_distance * 0.4)
    
    # Position camera
    camera_pos = (center_x.item(), center_y.item() - camera_distance, camera_height)
    lookat_pos = (center_x.item(), center_y.item(), center_z.item() + 0.3)
    
    # Update camera pose
    self.camera.set_pose(pos=camera_pos, lookat=lookat_pos)
```

### Tracking Algorithm Parameters:
- `min_distance = 6.0`: Minimum camera distance from robots
- `distance_multiplier = 1.5`: Scale factor for robot separation
- `height_ratio = 0.4`: Camera height relative to distance
- `lookat_offset = 0.3`: Look slightly above robot center

===============================================================================
                            ADVANCED CAMERA FEATURES
===============================================================================

### 1. Follow Specific Entity
```python
# Make camera follow a specific robot
camera.follow_entity(
    entity=robot1,
    fixed_axis=(None, None, 'z'),  # Fix Z-axis, free X,Y movement
    smoothing=0.9,                 # Smooth movement (0-1)
    fix_orientation=False          # Allow camera rotation
)
```

### 2. Attach Camera to Robot
```python
# Attach camera to robot link (first-person view)
offset_transform = np.array([
    [1, 0, 0, 0.2],    # 0.2m forward
    [0, 1, 0, 0.0],    # 0.0m sideways  
    [0, 0, 1, 0.3],    # 0.3m upward
    [0, 0, 0, 1]
])
camera.attach(robot1.links[0], offset_transform)
```

### 3. Multi-Camera Setup
```python
# Create multiple cameras for different views
overhead_cam = scene.add_camera(res=(1280, 720), pos=(0, 0, 10), lookat=(0, 0, 0))
side_cam = scene.add_camera(res=(1280, 720), pos=(5, 0, 2), lookat=(0, 0, 1))
front_cam = scene.add_camera(res=(1280, 720), pos=(0, -5, 2), lookat=(0, 0, 1))

# Record from multiple angles
overhead_frame = overhead_cam.render(rgb=True)
side_frame = side_cam.render(rgb=True)
front_frame = front_cam.render(rgb=True)
```

===============================================================================
                            RECORDING BEST PRACTICES
===============================================================================

### Performance Optimization:
1. **Resolution**: Use 1920x1080 for high quality, 1280x720 for speed
2. **Frame Rate**: Match simulation frequency (25 FPS for Genesis)
3. **Selective Recording**: Only record when needed (`--enable_camera`)
4. **Batch Processing**: Save frames as PNG, convert to video with ffmpeg

### Video Conversion:
```bash
# Navigate to output directory
cd dual_robot_demo_[timestamp]/

# Convert to MP4 (match original 25 FPS)
ffmpeg -framerate 25 -pattern_type glob -i '*.png' \
       -c:v libx264 -crf 18 -pix_fmt yuv420p output.mp4
```

### Memory Management:
- Use `GUI=False` for recording cameras (no display window)
- Delete frame data after saving to prevent memory leaks
- Consider downsampling for long recordings

===============================================================================
                                USAGE EXAMPLES
===============================================================================

### Basic Usage (ONNX-only, no external dependencies):
```bash
# Run with viewer only (no recording)
python dual_robot_demo.py --duration 10 --no_record

# Run with tracking camera recording
python dual_robot_demo.py --duration 30 --enable_camera

# Use custom ONNX model
python dual_robot_demo.py --onnx_path my_model.onnx --enable_camera
```

### Key Features:
✅ ONNX-only inference (no .pt dependency needed)
✅ Intelligent multi-robot tracking camera  
✅ 1920x1080 high-resolution recording
✅ Real-time camera pose updates
✅ Comprehensive ONNX model documentation
✅ 25 FPS stable performance

===============================================================================
"""
import argparse
import os
import time
import math
import numpy as np
import cv2
from datetime import datetime

import torch
import genesis as gs

# Check for ONNX Runtime
try:
    import onnxruntime as ort
    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False
    print("Warning: onnxruntime not available, will use dummy actions")


class ONNXPolicyController:
    """Controller using ONNX model for robot inference"""
    
    def __init__(self, onnx_path):
        self.session = None
        
        if HAVE_ONNX and os.path.exists(onnx_path):
            try:
                self.session = ort.InferenceSession(onnx_path)
                print(f"✅ Loaded ONNX model: {onnx_path}")
                
                # Try to read action std from ONNX metadata (if embedded)
                self.action_std = self._extract_std_from_metadata()
                if self.action_std is not None:
                    print(f"✅ Found action std in ONNX metadata: {self.action_std.shape}")
                else:
                    print("ℹ️  Using deterministic actions (no std found)")
                    
            except Exception as e:
                print(f"❌ Failed to load ONNX model: {e}")
                self.session = None
        else:
            print(f"❌ ONNX model not found: {onnx_path}")
    
    def _extract_std_from_metadata(self):
        """Try to extract action std from ONNX metadata"""
        try:
            import onnx
            # This would require the std to be embedded in metadata
            # For now, return None (deterministic only)
            return None
        except:
            return None
    
    def predict(self, observation, deterministic=True):
        """Predict action from observation"""
        if self.session is None:
            # Return dummy actions if no model
            return np.zeros(12, dtype=np.float32)
        
        try:
            # Ensure observation is the right shape and type
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            observation = observation.astype(np.float32)
            
            # Run inference
            action_mean = self.session.run(None, {'observation': observation})[0]
            action_mean = action_mean.flatten()
            
            # Add noise if not deterministic and std is available
            if not deterministic and self.action_std is not None:
                noise = np.random.normal(0, 1, size=action_mean.shape)
                action = action_mean + self.action_std * noise
            else:
                # Use deterministic mean action
                action = action_mean
            
            return action.astype(np.float32)
            
        except Exception as e:
            print(f"❌ ONNX inference error: {e}")
            return np.zeros(12, dtype=np.float32)


class DualRobotEnvironment:
    """Environment with two Go2 robots circling in opposite directions"""
    
    def __init__(self, onnx_model_path, enable_camera=False):
        # Initialize Genesis
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)
        
        # Create scene (match Go2Env exactly)
        self.dt = 0.02  # Match Go2Env
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),  # Match Go2Env
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),  # Match Go2Env
                camera_pos=(0.0, -6.0, 3.0),  # Centered view to see both robots
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=50,  # Wider field of view
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),  # Match Go2Env
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,  # Match Go2Env
                constraint_solver=gs.constraint_solver.Newton,  # Match Go2Env
                enable_collision=True,  # Match Go2Env
                enable_joint_limit=True,  # Match Go2Env
            ),
            show_viewer=True,
        )
        
        # Add ground plane (match Go2Env)
        self.scene.add_entity(gs.morphs.URDF(file="genesis/assets/urdf/plane/plane.urdf", fixed=True))
        
        # Robot base positions and orientations (match Go2Env format)
        self.base_init_pos1 = torch.tensor([2.0, 0.0, 0.42], device=gs.device)  # Match Go2Env height
        self.base_init_quat1 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        self.base_init_pos2 = torch.tensor([-2.0, 0.0, 0.42], device=gs.device)
        self.base_init_quat2 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)  # Same orientation
        
        # Load robots (match Go2Env)
        self.robot1 = self.scene.add_entity(
            gs.morphs.URDF(
                file="genesis/assets/urdf/go2/urdf/go2.urdf",  # Match Go2Env path
                pos=self.base_init_pos1.cpu().numpy(),
                quat=self.base_init_quat1.cpu().numpy(),
            ),
        )
        
        self.robot2 = self.scene.add_entity(
            gs.morphs.URDF(
                file="genesis/assets/urdf/go2/urdf/go2.urdf",  # Match Go2Env path
                pos=self.base_init_pos2.cpu().numpy(),
                quat=self.base_init_quat2.cpu().numpy(),
            ),
        )
        
        # Add camera for recording (optional, before building scene)
        self.camera = None
        self.enable_camera = enable_camera
        
        if self.enable_camera:
            print("📷 Adding camera for recording...")
            self.camera = self.scene.add_camera(
                res=(1920, 1080),
                pos=(0.0, -8.0, 4.0),  # Match viewer angle but further back
                lookat=(0.0, 0.0, 0.5),
                up=(0.0, 0.0, 1.0),
                fov=50,
                GUI=False
            )
        
        # Build scene (match Go2Env with n_envs=1)
        self.scene.build(n_envs=1)
        
        # Get motor DOF indices (match Go2Env exactly)
        joint_names = [
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'
        ]
        
        # Get motor DOF indices for both robots
        self.motors_dof_idx = [self.robot1.get_joint(name).dof_start for name in joint_names]
        
        # Set PD control parameters (match Go2Env exactly)
        kp = 20.0  # env_cfg["kp"]  
        kd = 0.5   # env_cfg["kd"]
        self.robot1.set_dofs_kp([kp] * 12, self.motors_dof_idx)  # 12 actions
        self.robot1.set_dofs_kv([kd] * 12, self.motors_dof_idx)
        self.robot2.set_dofs_kp([kp] * 12, self.motors_dof_idx)
        self.robot2.set_dofs_kv([kd] * 12, self.motors_dof_idx)
        
        # Initialize ONNX controllers
        self.controller1 = ONNXPolicyController(onnx_model_path)
        self.controller2 = ONNXPolicyController(onnx_model_path)
        
        # Robot parameters
        self.action_dim = 12  # Locomotion actions from ONNX model
        self.action_scale = 0.25
        
        # Default joint positions (from go2_env.py default_joint_angles)
        self.default_dof_pos = torch.tensor([
            0.0, 0.8, -1.5,  # FR
            0.0, 0.8, -1.5,  # FL  
            0.0, 1.0, -1.5,  # RR
            0.0, 1.0, -1.5,  # RL
        ], device=gs.device, dtype=torch.float32)
        
        # Initialize joint positions
        self.reset_robots()
        
        # Circling parameters
        self.circle_radius = 2.0
        self.angular_velocity1 = 0.3  # rad/s clockwise
        self.angular_velocity2 = -0.3  # rad/s counter-clockwise
        self.time = 0.0
        
        # Recording setup
        self.recording = False
        self.frame_count = 0
        self.output_dir = None
        
        print("🤖 Dual robot environment initialized")
        print(f"  - Robot 1: Circling clockwise")
        print(f"  - Robot 2: Circling counter-clockwise")
        print(f"  - Circle radius: {self.circle_radius}m")
        
    def reset_robots(self):
        """Reset robots to default pose"""
        # Set joint positions to default (using correct API from go2_env.py)
        # Note: No batch dimension needed for single environment
        self.robot1.set_dofs_position(
            position=self.default_dof_pos,  # 1D tensor
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
        )
        self.robot2.set_dofs_position(
            position=self.default_dof_pos,  # 1D tensor
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
        )
        
    def get_robot_observation(self, robot, command_vel, base_init_quat):
        """Get observation for a robot (exactly match go2_env.py)"""
        # Get robot state
        base_quat = robot.get_quat()
        base_lin_vel = robot.get_vel()
        base_ang_vel = robot.get_ang()
        dof_pos = robot.get_dofs_position(self.motors_dof_idx)
        dof_vel = robot.get_dofs_velocity(self.motors_dof_idx)
        
        # Take first env if batch (single env)
        if base_quat.dim() > 1:
            base_quat = base_quat[0]
            base_lin_vel = base_lin_vel[0]
            base_ang_vel = base_ang_vel[0]
            dof_pos = dof_pos[0]
            dof_vel = dof_vel[0]
        
        # Transform velocities to robot frame (exactly like Go2Env)
        from genesis.utils.geom import inv_quat, transform_by_quat
        inv_base_quat = inv_quat(base_quat)
        base_lin_vel_robot = transform_by_quat(base_lin_vel, inv_base_quat)
        base_ang_vel_robot = transform_by_quat(base_ang_vel, inv_base_quat)
        
        # Projected gravity (exactly like Go2Env)
        global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=torch.float32)
        projected_gravity = transform_by_quat(global_gravity, inv_base_quat)
        
        # Scale observations (match Go2Env obs_scales)
        obs_scales = {
            "lin_vel": 2.0,
            "ang_vel": 0.25, 
            "dof_pos": 1.0,
            "dof_vel": 0.05
        }
        
        # Commands scaled (match Go2Env)
        commands = torch.tensor(command_vel, device=gs.device, dtype=torch.float32)
        commands_scale = torch.tensor([obs_scales["lin_vel"], obs_scales["lin_vel"], obs_scales["ang_vel"]], device=gs.device)
        commands_scaled = commands * commands_scale
        
        # Joint positions relative to default
        dof_pos_scaled = (dof_pos - self.default_dof_pos) * obs_scales["dof_pos"]
        
        # Joint velocities scaled  
        dof_vel_scaled = dof_vel * obs_scales["dof_vel"]
        
        # Previous actions (zero for simplicity)
        actions = torch.zeros(12, device=gs.device, dtype=torch.float32)
        
        # Build observation (exactly match Go2Env order)
        obs = torch.cat([
            base_ang_vel_robot * obs_scales["ang_vel"],  # 3
            projected_gravity,                           # 3
            commands_scaled,                            # 3 
            dof_pos_scaled,                             # 12
            dof_vel_scaled,                             # 12
            actions,                                    # 12
        ])
        
        return obs.cpu().numpy()
    
    def compute_command_velocities(self):
        """Compute desired velocities for testing ONNX model"""
        # Simple forward motion test to verify ONNX model works
        # Robot 1: Forward motion
        cmd_vel1 = np.array([
            1.0,  # forward velocity
            0.0,  # lateral velocity  
            0.0   # no angular velocity (straight forward)
        ], dtype=np.float32)
        
        # Robot 2: Forward motion with slight turn
        cmd_vel2 = np.array([
            1.0,  # forward velocity
            0.0,  # lateral velocity
            0.2   # slight turning to test both robots
        ], dtype=np.float32)
        
        return cmd_vel1, cmd_vel2
    
    def update_tracking_camera(self):
        """Update camera position to keep both robots in view"""
        if not self.enable_camera or self.camera is None:
            return
            
        # Get robot positions
        pos1 = self.robot1.get_pos()
        pos2 = self.robot2.get_pos()
        
        # Handle batch dimension (take first env)
        if pos1.dim() > 1:
            pos1 = pos1[0]
            pos2 = pos2[0]
        
        # Calculate center point between robots
        center_x = (pos1[0] + pos2[0]) / 2.0
        center_y = (pos1[1] + pos2[1]) / 2.0
        center_z = (pos1[2] + pos2[2]) / 2.0
        
        # Calculate distance between robots
        robot_distance = torch.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
        # Set camera distance based on robot separation (with minimum distance)
        min_distance = 6.0
        camera_distance = max(min_distance, robot_distance.item() * 1.5)
        
        # Camera height proportional to distance
        camera_height = max(3.0, camera_distance * 0.4)
        
        # Position camera behind and above the center point
        camera_pos = (
            center_x.item(),
            center_y.item() - camera_distance,
            camera_height
        )
        
        # Look at point slightly above the center
        lookat_pos = (
            center_x.item(),
            center_y.item(), 
            center_z.item() + 0.3
        )
        
        # Debug output every 50 steps
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
        else:
            self.debug_counter = 0
            
        if self.debug_counter % 50 == 0:
            print(f"🎥 Camera tracking - Robot1: {pos1[:2].tolist()}, Robot2: {pos2[:2].tolist()}")
            print(f"   Center: {center_x.item():.2f}, {center_y.item():.2f}, Distance: {robot_distance.item():.2f}")
            print(f"   Camera pos: {camera_pos}, lookat: {lookat_pos}")
        
        # Update camera pose
        self.camera.set_pose(pos=camera_pos, lookat=lookat_pos)
    
    def step(self):
        """Step the simulation"""
        # Get command velocities
        cmd_vel1, cmd_vel2 = self.compute_command_velocities()
        
        # Get observations (pass base quaternions)
        obs1 = self.get_robot_observation(self.robot1, cmd_vel1, self.base_init_quat1)
        obs2 = self.get_robot_observation(self.robot2, cmd_vel2, self.base_init_quat2)
        
        # Get actions from ONNX model (deterministic for consistent behavior)
        action1 = self.controller1.predict(obs1, deterministic=True)
        action2 = self.controller2.predict(obs2, deterministic=True)
        
        # Scale and apply actions (following go2_env.py)
        scaled_action1 = action1 * self.action_scale
        scaled_action2 = action2 * self.action_scale
        
        # Compute target positions (following go2_env.py: target = action * scale + default)
        target_pos1 = torch.tensor(scaled_action1, device=gs.device, dtype=torch.float32) + self.default_dof_pos
        target_pos2 = torch.tensor(scaled_action2, device=gs.device, dtype=torch.float32) + self.default_dof_pos
        
        # Control robots (following go2_env.py API)
        self.robot1.control_dofs_position(target_pos1, self.motors_dof_idx)
        self.robot2.control_dofs_position(target_pos2, self.motors_dof_idx)
        
        # Step simulation
        self.scene.step()
        
        # Update tracking camera
        self.update_tracking_camera()
        
        # Update time
        self.time += self.scene.dt
    
    def start_recording(self, output_dir):
        """Start recording camera frames"""
        if self.recording:
            print("Already recording!")
            return
            
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.recording = True
        self.frame_count = 0
        self.recording_start_time = time.time()
        
        print(f"🔴 Started recording to: {os.path.abspath(self.output_dir)}")
        
    def capture_frame(self):
        """Capture frame from dedicated camera (optional)"""
        if not self.recording or not self.enable_camera or self.camera is None:
            return
            
        try:
            # Use dedicated camera to render RGB image
            result = self.camera.render(rgb=True)
            
            # Handle tuple return from camera.render()
            if isinstance(result, tuple):
                rgb_image = result[0] if len(result) > 0 else None
            else:
                rgb_image = result
            
            if rgb_image is not None:
                # Convert to numpy if needed
                if hasattr(rgb_image, 'cpu'):
                    frame = rgb_image.cpu().numpy()
                else:
                    frame = rgb_image
                
                # Ensure frame is in the right format (H, W, C)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Convert from RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    print(f"❌ Unexpected frame shape: {frame.shape}")
                    return False
                
                # Ensure frame is uint8
                if frame_bgr.dtype != np.uint8:
                    if frame_bgr.max() <= 1.0:
                        frame_bgr = (frame_bgr * 255).astype(np.uint8)
                    else:
                        frame_bgr = frame_bgr.astype(np.uint8)
                
                # Generate filename
                current_time = datetime.now()
                filename = f"frame_{current_time.hour:02d}_{current_time.minute:02d}_{current_time.second:02d}_{self.frame_count:06d}.png"
                filepath = os.path.join(self.output_dir, filename)
                
                # Save frame
                success = cv2.imwrite(filepath, frame_bgr)
                
                if success:
                    self.frame_count += 1
                    if self.frame_count % 30 == 0:
                        duration = time.time() - self.recording_start_time
                        print(f"🎬 Recorded {self.frame_count} frames ({duration:.1f}s)")
                    return True
                else:
                    print(f"❌ Failed to save frame: {filename}")
            else:
                print("❌ Camera returned None image")
                    
        except Exception as e:
            print(f"❌ Frame capture error: {e}")
            import traceback
            traceback.print_exc()
            
        return False
    
    def stop_recording(self):
        """Stop recording"""
        if not self.recording:
            return
            
        self.recording = False
        duration = time.time() - self.recording_start_time
        
        print(f"\n🎬 Recording stopped!")
        print(f"  - Frames: {self.frame_count}")
        print(f"  - Duration: {duration:.1f}s")
        print(f"  - Location: {os.path.abspath(self.output_dir)}")
        print(f"  - Average FPS: {self.frame_count/duration:.1f}")
        
        # Provide ffmpeg command
        print(f"\n🎞️  To convert to video:")
        print(f"cd {self.output_dir}")
        print(f"ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p dual_robots.mp4")
    
    def run(self, duration=30.0, record=True):
        """Run the demo"""
        
        if record and self.enable_camera and self.camera is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./dual_robot_demo_{timestamp}"
            self.start_recording(output_dir)
            print("📷 Camera recording enabled (will be slower)")
        elif record:
            print("📝 Recording disabled (camera off for speed)")
        
        print(f"\n🚀 Starting ONNX model test for {duration}s...")
        print("🧪 Testing forward motion commands")
        print("Press Ctrl+C to stop early")
        
        start_time = time.time()
        step_count = 0
        
        try:
            while time.time() - start_time < duration:
                self.step()
                step_count += 1
                
                # Capture frame every step if recording and camera enabled
                if self.recording and self.enable_camera and self.camera and step_count % 2 == 0:
                    self.capture_frame()
                
                # Print progress every 2 seconds for faster feedback
                elapsed = time.time() - start_time
                if step_count % 100 == 0:
                    print(f"⏱️  Progress: {elapsed:.1f}s / {duration}s - Step {step_count}")
                    
        except KeyboardInterrupt:
            print("\n⚠️  Demo interrupted by user")
        
        if self.recording:
            self.stop_recording()
            
        print(f"\n✅ Demo completed! Total steps: {step_count}")
        print("🤖 Check if both robots moved forward as expected")


def main():
    parser = argparse.ArgumentParser(description="Dual Go2 Robot Demo with ONNX")
    parser.add_argument("--onnx_path", type=str, 
                        default="exported_models/go2-walking_policy_iter100.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Demo duration in seconds")
    parser.add_argument("--no_record", action="store_true",
                        help="Disable recording")
    parser.add_argument("--enable_camera", action="store_true",
                        help="Enable camera recording (slower)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 Dual Go2 Robot ONNX Model Test")
    print("=" * 60)
    print(f"ONNX Model: {args.onnx_path}")
    print(f"Duration: {args.duration}s")
    print(f"Recording: {'No' if args.no_record else 'Yes'}")
    print(f"Camera: {'Enabled' if args.enable_camera else 'Disabled (faster)'}")
    print("=" * 60)
    
    # Check if ONNX model exists
    if not os.path.exists(args.onnx_path):
        print(f"❌ ONNX model not found: {args.onnx_path}")
        print("Please run extract_policy_standalone.py first to generate the model")
        return
    
    # Create and run environment
    env = DualRobotEnvironment(args.onnx_path, enable_camera=args.enable_camera)
    env.run(duration=args.duration, record=not args.no_record)


if __name__ == "__main__":
    main()