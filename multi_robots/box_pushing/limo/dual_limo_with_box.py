import argparse
import os
import time
import numpy as np
import torch
import cv2
import genesis as gs
from scipy.spatial.transform import Rotation as R

class DualLimoEnvironment:
    def __init__(self, forward_speed=100.0, enable_video=False, video_path="limo_demo.mp4"):
        # Initialize Genesis backend
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)
        self.dt = 0.02
        self.time = 0.0
        self.forward_speed = forward_speed
        self.enable_video = enable_video
        self.video_writer = None
        self.video_path = video_path

        # Setup scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=5),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1.0 / self.dt),
                camera_pos=(0.0, -3.0, 1.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=50,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=True,
        )
        
        default_material = gs.materials.Rigid(rho=1000.0, friction=1.0)

        # Ground plane
        self.scene.add_entity(
            gs.morphs.URDF(file="genesis/assets/urdf/plane/plane.urdf", fixed=True),
            material=default_material
        )

        # Helper for robot orientation
        def look_at_2d(from_xy, to_xy):
            dx, dy = to_xy[0] - from_xy[0], to_xy[1] - from_xy[1]
            yaw = np.arctan2(dy, dx)
            quat = R.from_euler('z', yaw).as_quat()
            return [quat[3], quat[0], quat[1], quat[2]]

        # Robot start poses
        self.robot_positions = [(-0.5, -0.3, 0.1), (-0.5, 0.3, 0.1)]
        self.robot_quats = [
            look_at_2d(self.robot_positions[0][:2], (0.0, 0.0)),
            look_at_2d(self.robot_positions[1][:2], (0.0, 0.0)),
        ]
        # self.robot_quats = [
        #     [1.0, 0.0, 0.0, 0.0],
        #     [1.0, 0.0, 0.0, 0.0],  # Both robots facing forward
        # ]
        self.robots = []
        for pos, quat in zip(self.robot_positions, self.robot_quats):
            r = self.scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/limo_ros2/urdf/limo.urdf",
                    pos=pos,
                    quat=quat,
                ),
                material=default_material
            )
            self.robots.append(r)
            
        mass = 0.3
        friction = 0.8

        # # Add a rectangle box
        # box_size = [0.4, 0.8, 0.2]
        # box_pos = [0.0, 0.0, box_size[2] / 2]
        # box_volume = np.prod(box_size)
        # rho = mass / box_volume
        # self.scene.add_entity(
        #     morph=gs.morphs.Box(
        #         size=box_size, 
        #         pos=box_pos
        #     ),
        #     material=gs.materials.Rigid(
        #         rho=rho, 
        #         friction=friction
        #     ),
        # )
        
        # # Add a sphere
        # radius = 0.4
        # box_pos = [0.0, 0.0, radius + 0.01]
        # sphere_volume = (4/3) * np.pi * radius**3
        # rho = mass / sphere_volume
        # self.scene.add_entity(
        #     morph=gs.morphs.Sphere(
        #         pos=box_pos,
        #         radius=radius,
        #     ),
        #     material=gs.materials.Rigid(
        #         rho=rho, 
        #         friction=friction,
        #     ),
        # )
        
        # Add a cylinder
        radius = 0.4
        height = 0.3
        box_pos = [0.0, 0.0, height + 0.01]
        cyl_volume = np.pi * radius**2 * height
        rho = mass / cyl_volume
        self.scene.add_entity(
            morph=gs.morphs.Cylinder(
                pos=box_pos,
                radius=radius,
                height=height,
            ),
            material=gs.materials.Rigid(
                rho=rho,
                friction=friction,
            ),
        )

        # Offscreen camera for recording
        self.camera = None
        if enable_video:
            self.camera = self.scene.add_camera(
                res=(1280, 720),
                pos=(0.0, -6.0, 3.0),
                lookat=(0.0, 0.0, 0.5),
                up=(0.0, 0.0, 1.0),
                fov=50,
                GUI=False,
            )

        # Build scene
        self.scene.build(n_envs=1)
        self.camera.start_recording()

        # Wheel joints and PD gains
        kp, kd = 100.0, 50.0
        self.robot_wheel_dofs = []
        # for robot in self.robots:
            # get DOF indices for wheel joints
        dofs = [self.robots[0].get_joint(name).dof_start for name in [
            'hinten_left_wheel', 
            'hinten_right_wheel', 
            'front_left_wheel', 
            'front_right_wheel'
        ]]
        # apply PD gains only to wheel DOFs
        self.robots[0].set_dofs_kp([kp] * len(dofs), dofs)
        self.robots[0].set_dofs_kv([kd] * len(dofs), dofs)
        self.robots[1].set_dofs_kp([kp] * len(dofs), dofs)
        self.robots[1].set_dofs_kv([kd] * len(dofs), dofs)
        self.robot_wheel_dofs.append(dofs)

        # Setup video writer if needed
        if self.enable_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_path, fourcc, int(1.0 / self.dt), (1280, 720)
            )

        print(
            "🎥 DualLimoEnvironment initialized with video recording." if self.enable_video else
            "🤖 DualLimoEnvironment initialized."
        )

    def step(self):
        # drive only the first robot
        dofs = self.robot_wheel_dofs[0]
        robot = self.robots[0]
        t = min(self.time, 1.0)
        speed = 10.0 * (t / 1.0)
        robot.control_dofs_velocity([speed] * len(dofs), dofs)
        
        robot1 = self.robots[1]
        robot1.control_dofs_velocity([speed] * len(dofs), dofs)

        self.scene.step()
        self.update_camera_and_video()
        self.time += self.dt

    def update_camera_and_video(self):
        if not self.enable_video or self.camera is None:
            return
        # center view at the moving robot
        p = self.robots[0].get_pos()[0]
        cx, cy = p[0], p[1]
        self.camera.set_pose(pos=(cx, cy - 6.0, 3.0), lookat=(cx, cy, 0.5))

    def run(self, duration=30.0):
        print(f"🚗 Running LIMO demo for {duration}s...")
        start = time.time()
        try:
            while time.time() - start < duration:
                self.step()
        except KeyboardInterrupt:
            print("⚠️ Demo interrupted")
        print("✅ Demo finished.")
        if self.enable_video:
            self.camera.stop_recording()
            print(f"📹 Video saved to {self.video_path}")

def main():
    parser = argparse.ArgumentParser(description="Single LIMO Robot Demo")
    parser.add_argument("--speed", type=float, default=3.0, help="Wheel forward speed")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration (s)")
    parser.add_argument("--record_video", action="store_true", help="Enable video recording")
    parser.add_argument("--video_path", type=str, default="limo_demo.mp4", help="Output video path")
    args = parser.parse_args()

    env = DualLimoEnvironment(
        forward_speed=args.speed,
        enable_video=args.record_video,
        video_path=args.video_path
    )
    env.run(duration=args.duration)

if __name__ == "__main__":
    main()
    
'''
    python multi_robots/box_pushing/limo/dual_limo_with_box.py --record_video --duration 90.0 --video_path output.mp4
'''