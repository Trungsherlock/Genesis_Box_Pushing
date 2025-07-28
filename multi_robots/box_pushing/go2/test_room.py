import genesis as gs
gs.init(backend=gs.cpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.5, 2.5, 2.0),  # Camera position in world coordinates
        camera_lookat=(0.0, 0.0, 1.25),     # Focus point: center of room or object
        camera_fov=60,
        res=(1280, 720)
    ),
    show_viewer=True
)

# room = scene.add_entity(
#     gs.morphs.URDF(file="urdf/indoor/room/room.urdf", fixed=True),
# )

room = scene.add_entity(
    morph=gs.morphs.Mesh(
        file="meshes/room/source/Archive/Room-A.obj",
        pos=(0.0, 0.0, 0.0),
        scale=1,
        euler=(90, 0, 0),
        fixed=True,
    ),
    # material=gs.materials.MPM.Muscle(
    #     E=5e5,
    #     nu=0.45,
    #     rho=10000.0,
    #     model="neohooken",
    #     n_groups=4,
    # ),
    surface=gs.surfaces.Default(
        diffuse_texture=gs.textures.ImageTexture(
            image_path="meshes/room/textures/Room-Color-A01VRayTotalLightingMap.jpg",
        ),
    ),
)

scene.build()

for i in range(3000):
    scene.step()
