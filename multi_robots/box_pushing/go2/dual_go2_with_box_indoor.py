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
        obs = observation.astype(np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        try:
            action = self.session.run(None, {'observation': obs})[0]
            return action.flatten().astype(np.float32)
        except Exception as e:
            print(f"❌ ONNX inference error: {e}")
            return np.zeros(12, dtype=np.float32)

class DualRobotEnvironment:
    def __init__(self, onnx_model_path: str):
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)

        self.dt = 0.02
        self.time = 0.0

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(substeps=10, gravity=(0, 0, -9.8)),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, 0, 1.0),
                camera_lookat=(0.0, 0.0, 0.3),
                camera_fov=50,
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
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                enable_collision=True,
                enable_self_collision=True,
            ),
            show_viewer=True,
        )

        def look_at_2d(from_xy, to_xy):
            dx, dy = to_xy[0] - from_xy[0], to_xy[1] - from_xy[1]
            yaw = np.arctan2(dy, dx)
            quat = R.from_euler('z', yaw).as_quat()
            return [quat[3], quat[0], quat[1], quat[2]]


        room_origin = np.array([43.0, 40.0, 0.0], dtype=np.float32)
        self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="assets/meshes/premium_room/premium_room.glb",
                pos=tuple(room_origin),
                scale=1.0,
                euler=(90.0, 0.0, 0.0),
                fixed=True,
            ),
            surface=gs.surfaces.Collision(), 
        )
        
        self.base_init_pos1 = torch.tensor([-0.3, -0.6, 0.52], device=gs.device)
        self.base_init_quat1 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        # self.base_init_quat1 = torch.tensor(look_at_2d([-0.3, -0.6], [0.0, 0.0]), device=gs.device)
        self.base_init_pos2 = torch.tensor([-0.3,  0.6, 0.52], device=gs.device)
        self.base_init_quat2 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        # self.base_init_quat2 = torch.tensor(look_at_2d([-0.3, 0.6], [0.0, 0.0]), device=gs.device)
        
        self.robot1 = self.scene.add_entity(
            gs.morphs.URDF(
                file="assets/urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos1.cpu().numpy(),
                quat=self.base_init_quat1.cpu().numpy(),
            )
        )
        self.robot2 = self.scene.add_entity(
            gs.morphs.URDF(
                file="assets/urdf/go2/urdf/go2.urdf",
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

        self.enable_camera = False
        self.camera = None

        self.scene.build(n_envs=1)

        joint_names = [
            'FR_hip_joint','FR_thigh_joint','FR_calf_joint',
            'FL_hip_joint','FL_thigh_joint','FL_calf_joint',
            'RR_hip_joint','RR_thigh_joint','RR_calf_joint',
            'RL_hip_joint','RL_thigh_joint','RL_calf_joint'
        ]
        self.motors_dof_idx = [self.robot1.get_joint(name).dof_start for name in joint_names]

        kp, kd = 60.0, 1.5
        self.robot1.set_dofs_kp([kp]*12, self.motors_dof_idx)
        self.robot1.set_dofs_kv([kd]*12, self.motors_dof_idx)
        self.robot2.set_dofs_kp([kp]*12, self.motors_dof_idx)
        self.robot2.set_dofs_kv([kd]*12, self.motors_dof_idx)

        self.default_dof_pos = torch.tensor([
            0.0,0.8,-1.5, 0.0,0.8,-1.5,
            0.0,1.0,-1.5, 0.0,1.0,-1.5
        ], device=gs.device, dtype=torch.float32)

        self.reset_robots()
        self.controller1 = ONNXPolicyController(onnx_model_path)
        self.controller2 = ONNXPolicyController(onnx_model_path)
        self.action_scale = 0.25

        print("🤖 DualRobotEnvironment initialized with ONNX locomotion.")

    def reset_robots(self):
        self.robot1.set_dofs_position(self.default_dof_pos, self.motors_dof_idx, zero_velocity=True)
        self.robot2.set_dofs_position(self.default_dof_pos, self.motors_dof_idx, zero_velocity=True)

    def get_robot_observation(self, robot, command_vel, base_init_quat):
        from genesis.utils.geom import inv_quat, transform_by_quat
        base_quat = robot.get_quat()
        base_vel = robot.get_vel()
        base_ang = robot.get_ang()
        dof_pos = robot.get_dofs_position(self.motors_dof_idx)
        dof_vel = robot.get_dofs_velocity(self.motors_dof_idx)
        if base_quat.dim()>1:
            base_quat, base_vel, base_ang, dof_pos, dof_vel = (
                base_quat[0], base_vel[0], base_ang[0], dof_pos[0], dof_vel[0]
            )
        inv_q = inv_quat(base_quat)
        lin_body = transform_by_quat(base_vel, inv_q)
        ang_body = transform_by_quat(base_ang, inv_q)
        grav = torch.tensor([0.0,0.0,-1.0], device=gs.device)
        proj_grav = transform_by_quat(grav, inv_q)
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
        return (
            np.array([1.0,0.0,0.0],dtype=np.float32),
            np.array([1.0,0.0,0.0],dtype=np.float32)
        )

    def step(self):
        cmd1, cmd2 = self.compute_command_velocities()
        obs1 = self.get_robot_observation(self.robot1, cmd1, self.base_init_quat1)
        obs2 = self.get_robot_observation(self.robot2, cmd2, self.base_init_quat2)
        act1 = self.controller1.predict(obs1)
        act2 = self.controller2.predict(obs2)
        tgt1 = torch.tensor(act1, device=gs.device)*self.action_scale + self.default_dof_pos
        tgt2 = torch.tensor(act2, device=gs.device)*self.action_scale + self.default_dof_pos
        self.robot1.control_dofs_position(tgt1, self.motors_dof_idx)
        self.robot2.control_dofs_position(tgt2, self.motors_dof_idx)
        self.scene.step()
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
    parser = argparse.ArgumentParser(description="Dual Go2 inside Room")
    parser.add_argument("--onnx_path", type=str, default="exported_models/go2-walking_policy_iter80000.onnx")
    parser.add_argument("--duration", type=float, default=200.0)
    args = parser.parse_args()

    env = DualRobotEnvironment(args.onnx_path)
    env.run(duration=args.duration)

if __name__ == "__main__":
    main()
