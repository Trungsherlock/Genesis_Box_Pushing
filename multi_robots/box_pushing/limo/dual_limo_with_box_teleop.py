import sys
import os
import time
import argparse
import numpy as np
import genesis as gs
from scipy.spatial.transform import Rotation as R
import torch

# Non-blocking key input
def get_key():
    if os.name == 'nt':
        import msvcrt
        return msvcrt.getch().decode('utf-8') if msvcrt.kbhit() else None
    else:
        import select, termios, tty
        dr,_,_ = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1)
        return None

class DualLimoEnvironment:
    """Dual LIMO teleop demo without any camera or video recording."""
    def __init__(self):
        # Initialize Genesis backend
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)
        self.dt = 0.02
        self.time = 0.0

        # Scene setup
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

        # Ground plane
        self.scene.add_entity(
            gs.morphs.URDF(file="genesis/assets/urdf/plane/plane.urdf", fixed=True)
        )

        # Robot start poses
        self.robot_positions = [(-0.5, -0.3, 0.1), (-0.5, 0.3, 0.1)]
        self.robot_quats = [[1.0, 0.0, 0.0, 0.0]] * 2
        self.robots = []
        for pos, quat in zip(self.robot_positions, self.robot_quats):
            r = self.scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/limo_ros2/urdf/limo.urdf",
                    pos=pos,
                    quat=quat,
                )
            )
            self.robots.append(r)

        # Box in front
        box_size = [0.4, 0.8, 0.2]
        box_pos = [0.0, 0.0, box_size[2] / 2 + 0.01]
        self.scene.add_entity(
            morph=gs.morphs.Box(size=box_size, pos=box_pos),
            material=gs.materials.Rigid(rho=3000.0, friction=0.8),
        )

        # Build simulation
        self.scene.build(n_envs=1)

        # Wheel parameters
        self.wheel_base = 0.14   # m
        self.wheel_radius = 0.045  # m
        
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

        # Teleop state
        self.lin_vel = 0.0       # m/s
        self.ang_vel = 0.0       # rad/s
        self.lin_step = 0.2      # increment m/s
        self.ang_step = 0.5      # increment rad/s

        print("🎮 Teleop ready: W/S forward/back, A/D rotate, X stop, Q quit.")

    def step(self):
        # Differential drive kinematics
        v, w = self.lin_vel, self.ang_vel
        L, r = self.wheel_base, self.wheel_radius
        vl = (v - w * (L / 2)) / r
        vr = (v + w * (L / 2)) / r
        speeds = [vl, vr, vl, vr]

        for robot, dofs in zip(self.robots, self.robot_wheel_dofs):
            robot.control_dofs_velocity(speeds, dofs)

        self.scene.step()
        self.time += self.dt

    def run_teleop(self):
        # Setup non-blocking stdin on Unix
        if os.name != 'nt':
            import tty, termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)

        try:
            while True:
                key = get_key()
                if key:
                    k = key.lower()
                    if k == 'w': self.lin_vel += self.lin_step
                    elif k == 's': self.lin_vel -= self.lin_step
                    elif k == 'a': self.ang_vel += self.ang_step
                    elif k == 'd': self.ang_vel -= self.ang_step
                    elif k in ['x', ' ']: self.lin_vel = 0.0; self.ang_vel = 0.0
                    elif k == 'q': print("🛑 Quitting."); break

                    # Clamp speeds
                    self.lin_vel = np.clip(self.lin_vel, -2.0, 2.0)
                    self.ang_vel = np.clip(self.ang_vel, -3.14, 3.14)
                    print(f"cmd → v={self.lin_vel:.2f} m/s, ω={self.ang_vel:.2f} rad/s")

                self.step()
                time.sleep(self.dt)
        finally:
            # Restore terminal on Unix
            if os.name != 'nt':
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

if __name__ == '__main__':
    # No camera or video flags
    env = DualLimoEnvironment()
    env.run_teleop()