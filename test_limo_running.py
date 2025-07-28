import numpy as np
np.float = float  # restore deprecated alias

import genesis as gs

# 1) Initialize Genesis
gs.init(backend=gs.cpu)

# 2) Define and build the scene
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=4e-3,
        substeps=10,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_fov=30,
        res=(1280, 720),
        max_FPS=60,
    ),
    show_viewer=True,
)

# 3) Add a ground plane
plane = scene.add_entity(gs.morphs.Plane())

# 4) Spawn four limos slightly above ground to avoid initial clipping
robot_positions = [
    (-0.5, -0.5, 0.3),
    ( 0.5, -0.5, 0.3),
    ( 0.5,  0.5, 0.3),
    (-0.5,  0.5, 0.3),
]
# Use alternative identity quaternion ordering (w, x, y, z)
identity_quat = (0.0, 0.0, 0.0, 1.0)

robots = []
for pos in robot_positions:
    r = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/limo_ros2/urdf/limo.urdf",
            pos=pos,
            quat=identity_quat,
        )
    )
    robots.append(r)

scene.build()

# 5) Let the physics settle before moving
scene.reset()
for _ in range(50):
    scene.step()

# 6) Configure wheel PD gains
wheel_joints = ['hinten_left_wheel', 'hinten_right_wheel']
wheel_dofs = [robots[0].get_joint(name).dof_start for name in wheel_joints]
kp, kv = 50.0, 5.0
for r in robots:
    r.set_dofs_kp([kp] * len(wheel_dofs), wheel_dofs)
    r.set_dofs_kv([kv] * len(wheel_dofs), wheel_dofs)

# 7) Sample a fixed random velocity for each robot
np.random.seed(42)
robot_vels = np.random.uniform(-3.0, 3.0, (len(robots), len(wheel_dofs)))

# 8) Main simulation loop: drive robots at their sampled velocities
for step in range(1000):
    # (Optional) re-sample every 100 steps for dynamic motion:
    # if step % 100 == 0:
    #     robot_vels = np.random.uniform(-3.0, 3.0, robot_vels.shape)
    for i, r in enumerate(robots):
        r.control_dofs_velocity(robot_vels[i], wheel_dofs)
    scene.step()
