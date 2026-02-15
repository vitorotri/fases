import trimesh
import numpy as np
import hashlib
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
OBJ_PATH = os.path.join(script_dir, "..", "input", "scene", "LaboerAtelierv12.obj") # noise signal
#OBJ_PATH = "G:\Meine Ablage\FreqLab Acoustics\Development\Room Scanning\Laboer Atelier v12.obj"   # <-- change path here


# --------------------------------------------------
# Deterministic color generator from object name
# --------------------------------------------------
def color_from_name(name):
    h = hashlib.md5(name.encode()).hexdigest()
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return np.array([r, g, b, 1.0])


# --------------------------------------------------
# Load OBJ robustly as Scene
# --------------------------------------------------
print("Loading OBJ:", OBJ_PATH)

loaded = trimesh.load(
    OBJ_PATH,
    force="scene",
    process=False,   # IMPORTANT: don't merge meshes
    maintain_order=True
)

# Convert single mesh → scene
if isinstance(loaded, trimesh.Trimesh):
    scene = trimesh.Scene()
    scene.add_geometry(loaded, node_name="mesh")
else:
    scene = loaded

print("\n--- Scene summary ---")
print(scene)
print("\nGeometry found:")
for name in scene.geometry.keys():
    print("   ", name)

# --------------------------------------------------
# If exporter merged everything into ONE mesh,
# we split it by connected components (lifesaver!)
# --------------------------------------------------
if len(scene.geometry) == 1:
    print("\n⚠ Only one mesh detected — splitting components...")
    single_mesh = list(scene.geometry.values())[0]
    pieces = single_mesh.split(only_watertight=False)

    scene = trimesh.Scene()
    for i, part in enumerate(pieces):
        scene.add_geometry(part, node_name=f"part_{i}")

    print(f"Split into {len(pieces)} components.")


# --------------------------------------------------
# Colorize each object
# --------------------------------------------------
for name, geom in scene.geometry.items():

    # Ensure vertex colors exist
    if not hasattr(geom.visual, "vertex_colors"):
        geom.visual.vertex_colors = np.ones((len(geom.vertices), 4))

    color = color_from_name(name)
    geom.visual.vertex_colors[:] = (color * 255).astype(np.uint8)

    print(f"Color assigned to {name}: {color[:3]}")


# --------------------------------------------------
# Auto camera placement
# --------------------------------------------------
bbox = scene.bounds
size = np.max(bbox[1] - bbox[0])
center = scene.centroid

camera_distance = size * 2.5

camera = trimesh.scene.cameras.Camera(
    resolution=(1200, 800),
    fov=(60, 45)
)

camera_transform = trimesh.transformations.translation_matrix(
    [center[0], center[1] - camera_distance, center[2] + camera_distance / 2]
)

scene.camera = camera
scene.camera_transform = camera_transform

print("\nLaunching viewer...")
scene.show()

