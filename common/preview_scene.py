
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def load_obj(path):
    verts = []
    faces = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith("f "):
                ids = []
                for p in line.split()[1:]:
                    ids.append(int(p.split("/")[0]) - 1)
                faces.append(ids)

    return np.array(verts, dtype=float), faces


def face_edges(face):
    edges = []
    for i in range(len(face)):
        a = face[i]
        b = face[(i + 1) % len(face)]
        edges.append((a, b))
    return edges


def plot_sphere(ax, center, radius, color):
    u = np.linspace(0.0, 2.0 * np.pi, 24)
    v = np.linspace(0.0, np.pi, 16)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, linewidth=0, antialiased=True, alpha=0.95, color=color)


def set_equal_axes(ax, verts):
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_scene(obj_path, src, rec):
    verts, faces = load_obj(obj_path)
    if verts.size == 0:
        raise RuntimeError("No vertices found in OBJ.")

    segments = []
    seen = set()
    for face in faces:
        for a, b in face_edges(face):
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            segments.append([verts[a], verts[b]])

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if segments:
        lc = Line3DCollection(segments, linewidths=0.7)
        ax.add_collection3d(lc)

    bbox = verts.max(axis=0) - verts.min(axis=0)
    base = np.max(bbox)
    if base <= 0:
        base = 1.0
    radius = 0.03 * base

    plot_sphere(ax, np.array(src), radius, "red")
    plot_sphere(ax, np.array(rec), radius, "blue")

    ax.text(src[0], src[1], src[2], "  Source", color="red")
    ax.text(rec[0], rec[1], rec[2], "  Receiver", color="blue")

    set_equal_axes(ax, np.vstack([verts, np.array(src), np.array(rec)]))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Scene preview")
    #plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    obj = sys.argv[1]
    src = list(map(float, sys.argv[2:5]))
    rec = list(map(float, sys.argv[5:8]))
    plot_scene(obj, src, rec)
