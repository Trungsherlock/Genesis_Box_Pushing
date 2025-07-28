import numpy as np
np.float = float

import os
from xacrodoc import XacroDoc
import genesis as gs

# src = "limo_ros2/limo_car/urdf/limo_ackerman_base.xacro"
# dst = "urdf/limo_ros2/urdf/limo.urdf"

# if not os.path.exists(dst):
#     os.makedirs(os.path.dirname(dst), exist_ok=True)
#     print(f"URDF not found at {dst}, generating from XACRO...")
#     XacroDoc.from_file(src).to_urdf_file(dst)
#     print(f"Generated URDF at {dst}")
# with open(dst, 'r+', encoding='utf-8') as f:
#     content = f.read()
#     if 'file://' in content:
#         content = content.replace('file://', '')
#         f.seek(0)
#         f.write(content)
#         f.truncate()
#         print("Cleaned URDF mesh paths.")

gs.init(backend=gs.cpu)

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

plane = scene.add_entity(gs.morphs.Plane())
robot_positions = [
    (-0.5, -0.5, 0.1),
    (0.5, -0.5, 0.1),
    (0.5, 0.5, 0.1),
    (-0.5, 0.5, 0.1)
]
robots = []
for pos in robot_positions:
    robot = scene.add_entity(
        gs.morphs.URDF(file="urdf/limo_ros2/urdf/limo.urdf", pos=pos, quat=(1, 0, 0, 0))
    )
    robots.append(robot)

cloth = scene.add_entity(
    material=gs.materials.PBD.Cloth(),
    morph=gs.morphs.Mesh(
        file="meshes/cloth.obj",
        scale=2.0,
        pos=(0, 0, 0.8),
        euler=(0.0, 0, 0.0),
    ),
    surface=gs.surfaces.Default(
        color=(0.2, 0.4, 0.8, 1.0),
        vis_mode="visual",
    ),
)

dummy_bodies = []
for robot in robots:
    # make sure stick_link exists in robot.links
    stick_pos = robot.get_link("base_footprint").pos
    dummy = scene.add_entity(
        morph=gs.morphs.Sphere(radius=0.02, pos=stick_pos),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0, 1.0)),
    )
    dummy_bodies.append(dummy)

# link each dummy to its robot's stick_link
for robot, dummy in zip(robots, dummy_bodies):
    scene.link_entities(
        parent_entity=robot,
        child_entity=dummy,
        parent_link_name="base_footprint",
        child_link_name=""  # dummy root
    )


scene.build()
# scene.reset()

# 7) Find the four corner‐particles on the cloth
tracked_pids = [
    cloth.find_closest_particle(d.get_links_pos()[0])
    for d in dummy_bodies
]

# 8) Setup wheel PD gains
wheel_joints = ['hinten_left_wheel','hinten_right_wheel']
wheel_dofs   = [robots[0].get_joint(j).dof_start for j in wheel_joints]
for r in robots:
    r.set_dofs_kp([50.0]*len(wheel_dofs), wheel_dofs)
    r.set_dofs_kv([ 5.0]*len(wheel_dofs), wheel_dofs)
np.random.seed(42)
robot_vels = np.random.uniform(-3, 3, (len(robots), len(wheel_dofs)))

# 9) Main sim loop: teleport corner particles each frame
for _ in range(1000):
    for d, pid in zip(dummy_bodies, tracked_pids):
        target_pos = d.get_links_pos()[0]
        cloth.set_particle_position(particle_idx=pid, pos=target_pos)

    # b) drive the wheels
    for i, r in enumerate(robots):
        r.control_dofs_velocity(robot_vels[i], wheel_dofs)

    # c) step sim
    scene.step()

# wheel_joints = ['hinten_left_wheel', 'hinten_right_wheel']
# wheel_dofs = [robots[0].get_joint(n).dof_start for n in wheel_joints]
# kp, kd = 50.0, 5.0
# for r in robots:
#     r.set_dofs_kp([kp]*len(wheel_dofs), wheel_dofs)
#     r.set_dofs_kv([kd]*len(wheel_dofs), wheel_dofs)

# num_robots = len(robots)
# num_dofs = len(wheel_dofs)
# np.random.seed(42)
# robot_velocities = np.random.uniform(low=-3.0, high=3.0, size=(num_robots, num_dofs))

# for _ in range(1000):
#     for i, robot in enumerate(robots):
#         robot.control_dofs_velocity(robot_velocities[i], wheel_dofs)
#     scene.step()
