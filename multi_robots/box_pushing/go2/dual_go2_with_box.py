import argparse
import os
import time
from datetime import datetime

import torch
import numpy as np
import cv2
import onnxruntime as ort
import genesis as gs
from scipy.spatial.transform import Rotation as R

class ONNXPolicyController:
    """Controller using an ONNX locomotion policy for Go2"""
    def __init__(self, onnx_path: str):
        self.session = None
        if os.path.exists(onnx_path):
            try:
                self.session = ort.InferenceSession(onnx_path)
                print(f"✅ Loaded ONNX model: {onnx_path}")
            except Exception as e:
                print(f"❌ Failed to load ONNX model: {e}")
        else:
            print(f"❌ ONNX model not found at: {onnx_path}")

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self.session is None:
            return np.zeros(12, dtype=np.float32)
        # ensure correct shape
        obs = observation.astype(np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        try:
            action = self.session.run(None, {'observation': obs})[0]
            action = action.flatten().astype(np.float32)
            return action
        except Exception as e:
            print(f"❌ ONNX inference error: {e}")
            return np.zeros(12, dtype=np.float32)


class DualRobotEnvironment:
    """Two Go2 robots walking side by side with a box in front"""
    def __init__(self, onnx_model_path: str, enable_camera: bool = False):
        # Initialize Genesis
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)

        # Simulation parameters
        self.dt = 0.02
        self.time = 0.0
        self.circle_radius = 2.0  # for reference, not used here

        # Create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                substeps=10,
                gravity=(0, 0, -9.8),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, 0, 0.8),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=40,
            ),
            mpm_options=gs.options.MPMOptions(
                dt=5e-4,
                lower_bound=(-1.0, -1.0, -0.2),
                upper_bound=(1.0, 1.0, 1.0),
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                visualize_mpm_boundary=True,
            ),
            show_viewer=True,
        )


        # Ground plane
        self.scene.add_entity(
            morph=gs.morphs.Plane()
        )
               
        def look_at_2d(from_xy, to_xy):
            dx = to_xy[0] - from_xy[0]
            dy = to_xy[1] - from_xy[1]
            yaw = np.arctan2(dy, dx)
            quat = R.from_euler('z', yaw).as_quat()  
            return [quat[3], quat[0], quat[1], quat[2]]  

        # Base poses
        self.base_init_pos1 = torch.tensor([-1.0, -0.6, 0.42], device=gs.device)
        # self.base_init_quat1 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        self.base_init_quat1 = torch.tensor(look_at_2d([-1.0, -0.5], [0.0, 0.0]), device=gs.device)
        self.base_init_pos2 = torch.tensor([-1.0, 0.6, 0.42], device=gs.device)
        # self.base_init_quat2 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        self.base_init_quat2 = torch.tensor(look_at_2d([-1.0,  0.3], [0.0, 0.0]), device=gs.device)

        # Add two Go2 robots
        self.robot1 = self.scene.add_entity(
            gs.morphs.URDF(
                file="genesis/assets/urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos1.cpu().numpy(),
                quat=self.base_init_quat1.cpu().numpy(),
            )
        )
        self.robot2 = self.scene.add_entity(
            gs.morphs.URDF(
                file="genesis/assets/urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos2.cpu().numpy(),
                quat=self.base_init_quat2.cpu().numpy(),
            )
        )
        
        self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/box/t_shaped_box/t_shaped_box.urdf",
                pos=(0.0, 0.0, 0.6),
                quat=(1.0, 0.0, 0.0, 0.0),
                
            )
        )
        
        self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/room/source/Archive/Room-A.obj",
                pos=(0.3, 0.3, 4.0),
                scale=0.1,
                euler=(0, 0, 0),
            ),
            # surface=gs.surfaces.Default(
            #     diffuse_texture=gs.textures.ImageTexture(
            #         image_path="meshes/worm/bdy_Base_Color.png",
            #     ),
            # ),
        )
        
        # box_pos = [0.0, 0.0, 0.6]
        
        # box_size = [0.4, 1.6, 0.5]
        # self.scene.add_entity(
        #     morph=gs.morphs.Box(
        #         size=box_size, 
        #         pos=box_pos, 
        #     ),
        #     material=gs.materials.Rigid(
        #         rho=10, 
        #         friction=0.2, 
        #         sdf_cell_size=0.005, 
        #         sdf_min_res=32, 
        #         sdf_max_res=128
        #     ),
        # )
        
        # self.scene.add_entity(
        #     morph=gs.morphs.Sphere(
        #         pos=box_pos,
        #         radius=0.6,
        #     ),
        #     material=gs.materials.Rigid(
        #         rho=10, 
        #         friction=1.0, 
        #         sdf_cell_size=0.005,
        #         sdf_min_res=32,
        #         sdf_max_res=128
        #     ),
        # )
        
        # self.scene.add_entity(
        #     morph=gs.morphs.Cylinder(
        #         pos=box_pos,
        #         radius=0.6,
        #         height=0.5,
        #     ),
        #     material=gs.materials.Rigid(
        #         rho=5,
        #         friction=1.0,
        #         sdf_cell_size=0.005,
        #         sdf_min_res=32,
        #         sdf_max_res=128
        #     ),
        # )


        # Optional camera
        self.enable_camera = enable_camera
        if self.enable_camera:
            self.camera = self.scene.add_camera(
                res=(1920, 1080),
                pos=(0.0, -8.0, 4.0),
                lookat=(0.0, 0.0, 0.5),
                up=(0.0, 0.0, 1.0),
                fov=50,
                GUI=False,
            )
        else:
            self.camera = None

        # Build scene
        self.scene.build(n_envs=1)

        # Joint setup
        joint_names = [
            'FR_hip_joint','FR_thigh_joint','FR_calf_joint',
            'FL_hip_joint','FL_thigh_joint','FL_calf_joint',
            'RR_hip_joint','RR_thigh_joint','RR_calf_joint',
            'RL_hip_joint','RL_thigh_joint','RL_calf_joint'
        ]
        self.motors_dof_idx = [
            self.robot1.get_joint(name).dof_start for name in joint_names
        ]

        # PD gains
        kp, kd = 60.0, 1.5
        self.robot1.set_dofs_kp([kp]*12, self.motors_dof_idx)
        self.robot1.set_dofs_kv([kd]*12, self.motors_dof_idx)
        self.robot2.set_dofs_kp([kp]*12, self.motors_dof_idx)
        self.robot2.set_dofs_kv([kd]*12, self.motors_dof_idx)

        # Default joint angles
        self.default_dof_pos = torch.tensor([
            0.0,0.8,-1.5,
            0.0,0.8,-1.5,
            0.0,1.0,-1.5,
            0.0,1.0,-1.5
        ], device=gs.device, dtype=torch.float32)
        # Reset to default
        self.reset_robots()

        # ONNX controllers
        self.controller1 = ONNXPolicyController(onnx_model_path)
        self.controller2 = ONNXPolicyController(onnx_model_path)

        # Action scale
        self.action_scale = 0.25

        print("🤖 DualRobotEnvironment initialized with ONNX locomotion.")
        
    def reset_robots(self):
        # Zero velocity + default positions
        self.robot1.set_dofs_position(self.default_dof_pos, self.motors_dof_idx, zero_velocity=True)
        self.robot2.set_dofs_position(self.default_dof_pos, self.motors_dof_idx, zero_velocity=True)

    def get_robot_observation(self, robot, command_vel, base_init_quat):
        from genesis.utils.geom import inv_quat, transform_by_quat
        base_quat = robot.get_quat()
        base_vel = robot.get_vel()
        base_ang = robot.get_ang()
        dof_pos = robot.get_dofs_position(self.motors_dof_idx)
        dof_vel = robot.get_dofs_velocity(self.motors_dof_idx)
        # single-env
        if base_quat.dim()>1:
            base_quat, base_vel, base_ang, dof_pos, dof_vel = (
                base_quat[0], base_vel[0], base_ang[0], dof_pos[0], dof_vel[0]
            )
        # transform to body frame
        inv_q = inv_quat(base_quat)
        lin_body = transform_by_quat(base_vel, inv_q)
        ang_body = transform_by_quat(base_ang, inv_q)
        # gravity
        grav = torch.tensor([0.0,0.0,-1.0], device=gs.device)
        proj_grav = transform_by_quat(grav, inv_q)
        # scales
        scales = dict(lin_vel=2.0, ang_vel=0.25, dof_pos=1.0, dof_vel=0.05)
        cmd = torch.tensor(command_vel, device=gs.device)
        cmd_scaled = cmd * torch.tensor([scales['lin_vel'], scales['lin_vel'], scales['ang_vel']], device=gs.device)
        pos_scaled = (dof_pos - self.default_dof_pos)*scales['dof_pos']
        vel_scaled = dof_vel*scales['dof_vel']
        prev_act = torch.zeros(12, device=gs.device)
        obs = torch.cat([
            ang_body*scales['ang_vel'], proj_grav,
            cmd_scaled, pos_scaled, vel_scaled, prev_act
        ])
        return obs.cpu().numpy()

    def compute_command_velocities(self):
        # forward for both, slight turn on second
        return (
            np.array([1.0,0.0,0.0],dtype=np.float32),
            np.array([1.0,0.0,0.0],dtype=np.float32)
        )

    def update_tracking_camera(self):
        if not self.enable_camera or self.camera is None:
            return
        p1 = self.robot1.get_pos()[0] if self.robot1.get_pos().dim()>1 else self.robot1.get_pos()
        p2 = self.robot2.get_pos()[0] if self.robot2.get_pos().dim()>1 else self.robot2.get_pos()
        cx = (p1[0]+p2[0])/2.0; cy = (p1[1]+p2[1])/2.0
        dist = 6.0; height=3.0
        self.camera.set_pose(pos=(cx.item(), cy.item()-dist, height), lookat=(cx.item(), cy.item(),0.5))

    def step(self):
        # 1) get commands
        cmd1, cmd2 = self.compute_command_velocities()
        # 2) collect obs + predict
        obs1 = self.get_robot_observation(self.robot1, cmd1, self.base_init_quat1)
        obs2 = self.get_robot_observation(self.robot2, cmd2, self.base_init_quat2)
        act1 = self.controller1.predict(obs1)
        act2 = self.controller2.predict(obs2)
        # 3) scale & offset
        tgt1 = torch.tensor(act1, device=gs.device)*self.action_scale + self.default_dof_pos
        tgt2 = torch.tensor(act2, device=gs.device)*self.action_scale + self.default_dof_pos
        # 4) apply
        self.robot1.control_dofs_position(tgt1, self.motors_dof_idx)
        self.robot2.control_dofs_position(tgt2, self.motors_dof_idx)
        # 5) step sim & camera
        self.scene.step()
        self.update_tracking_camera()
        self.time += self.dt

    def run(self, duration: float = 30.0):
        print(f"🚀 Running ONNX-driven demo for {duration}s...")
        start = time.time()
        try:
            while time.time() - start < duration:
                self.step()
        except KeyboardInterrupt:
            print("⚠️ Demo interrupted")
        print("✅ Demo completed.")


def main():
    parser = argparse.ArgumentParser(description="Dual Go2 with Box (ONNX-driven)")
    parser.add_argument("--onnx_path", type=str,
                        default="exported_models/go2-walking_policy_iter80000.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--duration", type=float, default=200.0,
                        help="Run duration in seconds")
    parser.add_argument("--enable_camera", action="store_true",
                        help="Enable off-screen tracking camera")
    args = parser.parse_args()

    env = DualRobotEnvironment(args.onnx_path, enable_camera=args.enable_camera)
    env.run(duration=args.duration)


if __name__ == "__main__":
    main()
    
'''
    python multi_robots/box_pushing/go2/dual_go2_with_box.py
'''
