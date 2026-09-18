"""
gui.py  —  PySide6 GUI for the 3D Acoustic FVM Solver  (v2)
==========================================================
Requirements:
    pip install PySide6 PyOpenGL PyOpenGL_accelerate numpy

Run from project root:
    python common/gui.py

Key bindings (GL viewport):
    LMB (normal)   : rotate camera
    MMB / drag     : pan
    Scroll         : zoom
    T              : reset camera to the initial view
    S              : enter "place SOURCE" mode
    R              : enter "place RECEIVER" mode
    X / Y / Z      : (in placement mode) lock to that axis
    Click          : (in placement mode) confirm placement
    Escape         : cancel placement mode / deselect layer
    LMB click      : (normal mode) ray-pick layer
    Shift + LMB drag: inside-box look mode; lets you look straight up/down
    Ctrl + LMB drag : roll around view axis
    Ctrl+Z          : undo last source/receiver placement, move, or removal
"""

import sys, os, math, glob
from shift_obj import shift_to_positive
from pathlib import Path
from datetime import datetime
import subprocess

import numpy as np

from PySide6.QtCore    import (Qt, QThread, Signal, Slot, QPoint, QPointF, QEvent,
                              QSettings, QTimer, QSize, QVariantAnimation, QUrl)
from PySide6.QtGui     import (QFont, QColor, QPalette, QSurfaceFormat,
                              QPainter, QCursor, QKeySequence, QPolygon,
                              QPixmap, QIcon, QPen, QShortcut)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QDockWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox,
    QListWidget, QListWidgetItem, QTabWidget, QTabBar, QTextEdit,
    QFileDialog, QMessageBox, QDoubleSpinBox, QSpinBox, QAbstractSpinBox,
    QGroupBox, QScrollArea, QAbstractItemView, QInputDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QStatusBar, QSlider, QMenu, QProgressBar, QDialog,
    QStyledItemDelegate, QStyle, QRadioButton, QButtonGroup
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL  import *
from OpenGL.GLU import *

# The absorption curve is now drawn by the in-house AlphaCurveEditor widget,
# so matplotlib is no longer required. We keep an optional import only so any
# future matplotlib-based view still works if present; the app runs fine without.
os.environ.setdefault("QT_API", "pyside6")
try:
    import matplotlib
    from matplotlib.figure import Figure
    try:
        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except Exception:
        matplotlib.use("Qt5Agg")
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# ─────────────────────────────────────────────────────────────
#  Palette
# ─────────────────────────────────────────────────────────────
PAL = {
    "bg":        "#1a1d23", "panel":   "#20242c", "border":  "#2e3340",
    "accent":    "#7c5cfc", "accent2": "#a48bff", "warn":    "#ffb74d",
    "ok":        "#a5d6a7", "err":     "#ef9a9a", "text":    "#cfd8dc",
    "subtext":   "#78909c", "highlight":"#263238","btn":     "#343d4c",
    "btn_hover": "#434e60", "layer_sel":"#251a55",
}

# Line colours for the Results overlay/param plots, in order:
# blue-purple (accent), cyan-green (layers), pink-red, fiery orange-red.
PLOT_COLORS = ["#7c5cfc", "#0ff091", "#ff5c7a", "#ff5a1f"]


def _plot_color(i):
    """Colour for the i-th receiver: the hand-picked palette first, then distinct colours beyond it
    by hopping hue with the golden ratio (well-spread for any count)."""
    if i < len(PLOT_COLORS):
        return PLOT_COLORS[i]
    import matplotlib.pyplot as plt
    return plt.get_cmap('hsv')(((i - len(PLOT_COLORS)) * 0.61803) % 1.0)

LAYER_COLORS = [
	(15/255,  240/255, 145/255)] #, aqua-cyan
'''
    (45/255,   2/255, 249/255),   # blue-violet
    (100/255, 249/255,  2/255),   # green
    (240/255,  25/255,  2/255),   # red
    (249/255, 225/255,  2/255),   # yellow
    (15/255,  240/255, 150/255),  # cyan
    (249/255,   2/255, 237/255),  # magenta
    (249/255, 130/255,   2/255),  # orange
    (160/255,  2/255,  249/255),  # purple
    (2/255,   130/255, 249/255),  # azure
    (200/255, 249/255,   2/255),  # lime
    (249/255,   2/255, 120/255),  # pink
    (249/255, 160/255, 200/255),  # rose
]
'''
BBOX_COLOR = (0.35, 0.65, 1.0)
SRC_COLOR  = (138/255, 130/255, 255/255)
REC_COLOR  = (252/255, 118/255, 78/255)
HL_COLOR   = (0.2, 1.0, 0.5)   # highlight tint for a selected/hovered point
# Tint for sources/receivers placed OUTSIDE the domain box (single place to
# edit; used for placed points and the placement preview). RGB in 0..1.
OUTSIDE_COLOR = (252/255, 245/255, 35/255)

# ─────────────────────────────────────────────────────────────
#  Scene colour scheme (code-level toggle, not in the GUI)
# ─────────────────────────────────────────────────────────────
# Set True for an inverted "light" viewport: white background with black
# wireframes/box. False = the default dark background with white wireframes.
SCENE_LIGHT_MODE = False

# These are mutable at runtime: the Dark/Light toggle button calls
# set_scene_mode() to flip them, and every draw call reads the current values.
def _scene_palette(light):
    if light:
        return {"bg":   (1.00, 1.00, 1.00, 1.0),   # white background (GL clear)
                "wire": (0.0, 0.0, 0.0, 0.9),       # black wireframe + domain box
                "grey": (0.45, 0.45, 0.45)}          # unselected fill on white
    return {"bg":   (0.05, 0.055, 0.07, 1.0),       # dark background (GL clear)
            "wire": (1.0, 1.0, 1.0, 0.9),           # white wireframe + domain box
            "grey": (0.62, 0.62, 0.62)}              # unselected fill on dark

_pal = _scene_palette(SCENE_LIGHT_MODE)
SCENE_BG   = _pal["bg"]
WIRE_COLOR = _pal["wire"]
GREY_FILL  = _pal["grey"]

def set_scene_mode(light):
    """Flip the viewport colour scheme (background / wireframe / fill)."""
    global SCENE_LIGHT_MODE, SCENE_BG, WIRE_COLOR, GREY_FILL
    SCENE_LIGHT_MODE = bool(light)
    p = _scene_palette(SCENE_LIGHT_MODE)
    SCENE_BG, WIRE_COLOR, GREY_FILL = p["bg"], p["wire"], p["grey"]
# Transparency of the SELECTED (highlighted) layer's fill, 0.0 (invisible) ..
# 1.0 (fully opaque). Edit this value to taste.
SEL_ALPHA  = 0.9

# Single source of truth for the X/Y/Z axis colours, shared by the corner
# gizmo, the placement cursor, and the drag-to-move axis guides so they always
# match. (RGB 0..1.)
AXIS_COLORS = {
    'X': (0.55, 0.20, 0.95),          # royal purple
    'Y': (252/255, 5/255, 95/255),    # vivid pink-magenta
    'Z': (105/255, 210/255, 45/255),          # bright lime
}

def pressure_lut(palette='color', dark=False):
    """256x3 RGB LUT for the animation.
      'color' -> the 5-colour diverging map; the CENTRE (zero pressure) blends with the
                 background: white in Light, near-black in Dark.
      'greys' -> linear black->white ramp (sign preserved: negative dark, positive light)."""
    if palette == 'greys':
        g = np.linspace(0, 255, 256).astype(np.uint8)
        return np.stack([g, g, g], axis=1)
    centre = (18, 18, 22) if dark else (255, 255, 255)
    stops = [(0.00, (220,  20,  60)),   # crimson
             (0.25, (  0, 255, 127)),   # springgreen
             (0.50, centre),            # white (Light) / near-black (Dark) = zero pressure
             (0.75, (124, 252,   0)),   # lawngreen
             (1.00, (138,  43, 226))]   # blueviolet
    lut = np.zeros((256, 3), np.uint8)
    for i in range(256):
        t = i / 255.0
        for k in range(len(stops) - 1):
            t0, c0 = stops[k]; t1, c1 = stops[k + 1]
            if t0 <= t <= t1:
                f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                lut[i] = [int(round(c0[j] + (c1[j] - c0[j]) * f)) for j in range(3)]
                break
    return lut

# Opacity of the gizmo's opposite-side (negative-axis) balls, 0.0..1.0.
# Set to 0.0 to hide them completely (fully transparent); 1.0 = fully opaque.
GIZMO_NEG_ALPHA = 0.55

OCTAVE_BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000]

# ─────────────────────────────────────────────────────────────
#  Solver executable location
# ─────────────────────────────────────────────────────────────
# Absolute path to the compiled solver. This is the single place to edit if the
# program is installed elsewhere; it is the default shown in the Run tab's
# "Binary:" field and is NOT overwritten by a project's saved manifest unless
# that manifest points to a file that actually exists.
SOLVER_BINARY = "/home/vitorotri/Documents/Github/fases/3D-solver/build/phases3D"

# Predefined absorption-material catalogue used by the material table RMB menu.
# Expected line format inside the text file:
#   material_name  a63  a125  a250  a500  a1000  a2000  a4000  a8000
# Example:
#   rough_concrete 0.01 0.01 0.02 0.04 0.06 0.08 0.10 0.10
# The GUI shows material_name as "material name" in the selector, but keeps the
# file format underscore-based and human-editable.
MATERIALS_FILE = "/home/vitorotri/Documents/Github/fases/common/materialsREF.txt"

# ─────────────────────────────────────────────────────────────
#  Project layout
# ─────────────────────────────────────────────────────────────
# Relative subdirectories scaffolded by "New project". A phases project is a
# self-contained CASE directory: the solver is launched from its root and reads
# everything through the fixed file contract (config/, input/*). Edit this list
# if your layout changes.
#
# NOTE: shared *source/build* code — core/, include/, common/, scripts/ — is
# deliberately NOT scaffolded per project, since those are shared across cases.
# If you instead want each project to be a full solver-tree copy, add them here.
PROJECT_SUBDIRS = [
    "config",
    "input/mat", "input/fit", "input/src", "input/rec",
    "output",
]
# Manifest file at the project root: the single source of truth that ties a
# directory together (scene path, binary, layer set, …). #-commented key=value,
# same style as config.txt so it stays human-readable and hand-editable.
MANIFEST_NAME = "project.txt"

# ─────────────────────────────────────────────────────────────
#  OBJ parser  (mirrors parsing3D.h)
# ─────────────────────────────────────────────────────────────
def _is_bbox(name: str) -> bool:
    """The reserved domain layer (names[0] in the solver, used for the PML shell,
    never a material). Match both spellings: 'BoundingBox' and legacy 'boundary'."""
    n = name.lower()
    return "bounding" in n or "boundary" in n


def _sibling_base(name: str):
    """'<base>_<int>' -> '<base>' (the scratch pool of exploded pieces, e.g. 'surface_12' -> 'surface');
    None if the name isn't an exploded piece. Used to sweep leftover pieces back into one layer."""
    b, sep, tail = name.rpartition("_")
    return b if (sep and b and tail.isdigit()) else None


def triangulate_face(face, vertices):
    """Triangulate a planar polygon face (indices into `vertices`) by ear
    clipping. Handles concave faces (e.g. an L-shaped floor) that a naive
    triangle fan would tear across reflex corners.
    ponytail: O(n^2) ear clip; faces here are a handful of verts, fine.
    Falls back to a fan if the polygon is degenerate/self-touching."""
    n = len(face)
    if n == 3:
        return [(face[0], face[1], face[2])]
    P = [vertices[i] for i in face]
    # Newell normal -> pick projection plane by dominant component
    nx = ny = nz = 0.0
    for i in range(n):
        a, b = P[i], P[(i+1) % n]
        nx += (a[1]-b[1])*(a[2]+b[2])
        ny += (a[2]-b[2])*(a[0]+b[0])
        nz += (a[0]-b[0])*(a[1]+b[1])
    ax, ay, az = abs(nx), abs(ny), abs(nz)
    if ax >= ay and ax >= az:
        pts = [(p[1], p[2]) for p in P]
    elif ay >= az:
        pts = [(p[2], p[0]) for p in P]
    else:
        pts = [(p[0], p[1]) for p in P]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    def in_tri(p, a, b, c):
        d1 = cross(a, b, p); d2 = cross(b, c, p); d3 = cross(c, a, p)
        neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (neg and pos)

    # signed area -> orient CCW so ears have positive cross
    area = sum(pts[i][0]*pts[(i+1) % n][1] - pts[(i+1) % n][0]*pts[i][1]
               for i in range(n))
    order = list(range(n))
    if area < 0:
        order.reverse()

    tris, V, guard = [], order[:], 0
    while len(V) > 3 and guard < 10*n:
        guard += 1
        m = len(V); clipped = False
        for k in range(m):
            i0, i1, i2 = V[(k-1) % m], V[k], V[(k+1) % m]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if cross(a, b, c) <= 0:            # reflex/degenerate: not an ear
                continue
            if any(j not in (i0, i1, i2) and in_tri(pts[j], a, b, c) for j in V):
                continue
            tris.append((face[i0], face[i1], face[i2]))
            V.pop(k); clipped = True; break
        if not clipped:
            break                              # degenerate: bail to fan below
    if len(V) == 3:
        tris.append((face[V[0]], face[V[1]], face[V[2]]))
    elif len(V) > 3:                            # fallback fan on leftovers
        for i in range(1, len(V)-1):
            tris.append((face[V[0]], face[V[i]], face[V[i+1]]))
    return tris


def parse_obj(path: str, return_split: bool = False):
    """
    Parse vertices and grouped faces from an OBJ file.

    Material split: exporters (SketchUp, Blender) often emit ONE object carrying
    several `usemtl` blocks -- the whole room arrives as a single layer, and the
    per-surface materials are invisible. When a file has more distinct `usemtl`
    names than `o`/`g` names we therefore treat each `usemtl` as a group boundary
    and name the group after the material. Files that already group per surface
    are untouched (the count test fails), so this cannot rename existing layers.
    Repeats of a material merge back into one group, as the solver does.
    `return_split` -> (verts, groups, split_used); the caller MUST force the
    derived _merged OBJ to disk when it is True, because main3D reads only `g`/`o`
    and would otherwise see one group where the GUI shows several.

    Group handling: an object may be introduced by an ``o`` line, a ``g`` line,
    or BOTH (some exporters emit ``o chair1`` immediately followed by
    ``g chair1``). We therefore do NOT start a fresh group on every o/g line —
    that would create a duplicate (often empty) group for each o+g pair. Instead
    we only open a new group when the RESOLVED name actually changes, and we
    drop any group that ends up with no faces. Group names that carry several
    tokens (e.g. ``g Mesh1 ceiling Model``) resolve to the second token, which
    is the human-readable layer name used elsewhere in the GUI.
    """
    # pre-scan: does this file group by material rather than by object?
    n_mtl, n_grp = set(), set()
    try:
        with open(path) as _f:
            for _l in _f:
                _t = _l.split()
                if not _t:
                    continue
                if _t[0] == "usemtl" and len(_t) > 1:
                    n_mtl.add(_t[1])
                elif _t[0] in ("o", "g") and len(_t) > 1:
                    n_grp.add(_t[1])
    except OSError:
        pass
    split_mtl = len(n_mtl) > len(n_grp)

    vertices, groups = [], []
    cur_name, cur_tris = None, []

    def resolve_name(parts):
        # "g Mesh1 ceiling Model" -> "ceiling"; "o chair1"/"g chair1" -> "chair1"
        if len(parts) >= 2:
            return parts[1]
        if parts:
            return parts[0]
        return "unnamed"

    def flush():
        # only keep groups that actually contain geometry
        if cur_name is not None and cur_tris:
            groups.append({"name": cur_name,
                           "tris": np.array(cur_tris, np.int32)})

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()
            if tok[0] == "v":
                vertices.append((float(tok[1]), float(tok[2]), float(tok[3])))
            elif tok[0] in ("o", "g"):
                new_name = resolve_name(tok[1:])
                # Only begin a new group if the name truly changes. An o/g pair
                # naming the same object (o chair1 then g chair1) is treated as
                # one group, so it is not duplicated.
                if new_name != cur_name:
                    flush()
                    cur_tris = []
                    cur_name = new_name
            elif tok[0] == "usemtl" and split_mtl and len(tok) > 1:
                if tok[1] != cur_name:
                    flush()
                    cur_tris = []
                    cur_name = tok[1]
            elif tok[0] == "f":
                face = []
                for t in tok[1:]:
                    vi = int(t.split("/")[0])
                    vi = vi-1 if vi > 0 else len(vertices)+vi
                    face.append(vi)
                if len(face) < 3: continue
                cur_tris.extend(triangulate_face(face, vertices))
    flush()
    if split_mtl:                      # one material = one layer, however scattered
        merged = {}          # dicts keep insertion order (3.7+)
        for g in groups:
            if g["name"] in merged:
                merged[g["name"]] = {"name": g["name"],
                                     "tris": np.vstack([merged[g["name"]]["tris"], g["tris"]])}
            else:
                merged[g["name"]] = g
        groups = list(merged.values())
    verts = (np.array(vertices, np.float32) if vertices
             else np.zeros((0,3), np.float32))
    return (verts, groups, split_mtl) if return_split else (verts, groups)


# GENERIC (non-axis-aligned) ray directions for the point-in-mesh test. Room walls are axis-aligned, so
# an axis ray (+x/+y/+z) runs coplanar with wall triangles and passes through shared edges/vertices
# constantly -> parity miscounts, and it does so differently at every grid spacing (the split bug the
# user hit). These irrational-ish dirs almost never graze an edge or lie in an axis-aligned triangle.
MESH_RAY_DIRS = np.array([(0.531, 0.673, 0.514),
                          (0.271, -0.638, 0.721),
                          (-0.802, 0.301, 0.515)], np.float64)


def mesh_ray_geom(verts, tris, ray_dirs=MESH_RAY_DIRS):
    """Precompute Möller–Trumbore constants for `tris` (Nx3 vertex indices) over each ray direction.
    Returns (a, e1, e2, dirs) where dirs is a list of (rd, pv, inv, ok) per normalized direction."""
    verts = np.asarray(verts, np.float64)
    T = np.asarray(tris)
    a = verts[T[:, 0]]; e1 = verts[T[:, 1]] - a; e2 = verts[T[:, 2]] - a
    dirs = []
    for rd in ray_dirs:
        rd = np.asarray(rd, np.float64); rd = rd / np.linalg.norm(rd)
        pv = np.cross(rd, e2)
        det = np.einsum('ij,ij->i', e1, pv)
        ok = np.abs(det) > 1e-9
        inv = np.zeros_like(det); inv[ok] = 1.0 / det[ok]
        dirs.append((rd, pv, inv, ok))
    return a, e1, e2, dirs


def mesh_inside_vote(pts, geom):
    """Point-in-mesh by ray parity (odd forward crossings = inside), MAJORITY-VOTED over the generic ray
    directions in `geom` (>=2 => inside). Generic dirs dodge the axis-aligned-mesh coplanarity/edge-graze
    that split the room at some grid spacings; the vote covers the rare remaining graze. `geom` is the
    tuple from mesh_ray_geom. ponytail: O(pts*tris*ndirs) Python loop, fine at load."""
    a, e1, e2, dirs = geom
    out = np.zeros(len(pts), bool)
    for i, ro in enumerate(np.asarray(pts, np.float64)):
        tv = ro - a
        qv = np.cross(tv, e1)                     # depends on point only -> reuse across directions
        votes = 0
        for rd, pv, inv, ok in dirs:
            u = np.einsum('ij,ij->i', tv, pv) * inv
            v = (qv @ rd) * inv
            t = np.einsum('ij,ij->i', e2, qv) * inv
            hit = ok & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
            votes += int(hit.sum()) & 1
        out[i] = votes >= 2
    return out


def write_obj(path, verts, groups):
    """Write (verts, groups) back as a plain OBJ: one `v` per vertex, then `g name`
    + triangle `f` (1-based) per group. Used to persist the coplanar-MERGED group set
    so the solver reads the SAME layers the GUI shows (main3D opens <layer>.fit per OBJ
    group; a merged-away group would have no .fit -> crash)."""
    with open(path, "w") as f:
        f.write("# merged scene — written by gui.py (groups match GUI layers)\n")
        for x, y, z in verts:
            f.write(f"v {x:.6g} {y:.6g} {z:.6g}\n")
        for g in groups:
            f.write(f"g {g['name']}\n")
            for a, b, c in np.asarray(g["tris"]):
                f.write(f"f {a+1} {b+1} {c+1}\n")


def face_normals(verts, tris):
    if len(tris) == 0:
        return np.zeros((0,3), np.float32)
    a, b, c = verts[tris[:,0]], verts[tris[:,1]], verts[tris[:,2]]
    n = np.cross(b-a, c-a).astype(np.float32)
    nlen = np.linalg.norm(n, axis=1, keepdims=True)
    nlen = np.where(nlen < 1e-12, 1.0, nlen)
    return n / nlen


def feature_edges(verts, tris, angle_thresh_deg=5.0, weld_tol=1e-5):
    """
    Return a list of (i, j) vertex-index pairs for the 'real' edges of a mesh:
      • boundary edges  -> used by exactly one triangle
      • crease edges    -> shared by two triangles whose normals differ by more
                           than angle_thresh_deg
    Coplanar interior diagonals (the triangulation lines inside a flat plane)
    are skipped. Vertices are welded by position so duplicated coincident
    vertices still register shared edges.
    """
    if len(tris) == 0:
        return np.zeros((0, 2), np.int32)

    # weld: map every vertex to a representative index by quantised position
    q = np.round(verts / weld_tol).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)   # inv[v] = welded id
    wtris = inv[tris]                                     # tris in welded ids

    fn = face_normals(verts, tris)

    edge_map = {}   # (a,b) sorted welded ids -> list of triangle indices
    for ti, (a, b, c) in enumerate(wtris):
        for e in ((a, b), (b, c), (c, a)):
            key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
            edge_map.setdefault(key, []).append(ti)

    cos_thr = math.cos(math.radians(angle_thresh_deg))
    # remember one original (unwelded) vertex pair per welded edge for drawing
    rep = {}
    for ti, tri in enumerate(tris):
        wa, wb, wc = wtris[ti]
        for (oa, ob), (a, b) in zip(((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])),
                                    ((wa,wb),(wb,wc),(wc,wa))):
            key = (a, b) if a < b else (b, a)
            rep.setdefault(key, (oa, ob))

    out = []
    for key, tlist in edge_map.items():
        if len(tlist) == 1:
            out.append(rep[key])                 # boundary edge
        elif len(tlist) >= 2:
            n0, n1 = fn[tlist[0]], fn[tlist[1]]
            if abs(float(np.dot(n0, n1))) < cos_thr:
                out.append(rep[key])             # crease edge (real corner)
            # else: coplanar -> internal diagonal, skip
    return np.array(out, np.int32) if out else np.zeros((0, 2), np.int32)


def merge_coplanar_groups(verts, groups, angle_thresh_deg=5.0, weld_tol=1e-5,
                          respect_names=False):
    """Union non-bbox groups joined by a HIDDEN (non-crease) shared edge -> a surface split across
    per-triangle groups becomes ONE selectable layer. Uses the SAME angle as the wireframe crease test
    (feature_edges), so the invariant holds: if the diagonal line is hidden, the layer is merged. Two
    surfaces meeting at a real crease (> angle) stay separate. Returns a new [{name, tris}] list;
    bbox groups pass through unchanged.

    respect_names: never fuse two groups with DIFFERENT names. Set when the grouping carries
    meaning -- e.g. groups split from `usemtl`, where each name is a real material. Without it
    a floor of linoleum meeting parquet at a hidden coplanar edge silently collapses into one
    layer under whichever name came first, and you lose the material you need to assign
    absorption to. Auto-generated groupings (obj1..objN) still want the blanket merge."""
    from collections import defaultdict
    q = np.round(verts / weld_tol).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    cos_thr = math.cos(math.radians(angle_thresh_deg))
    n = len(groups)
    gname = [g["name"] for g in groups]
    gi = [i for i, g in enumerate(groups) if not _is_bbox(g["name"]) and len(g["tris"])]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    if gi:
        all_tris = np.concatenate([np.asarray(groups[i]["tris"]) for i in gi])
        tri_grp = np.concatenate([np.full(len(groups[i]["tris"]), i, np.int32) for i in gi])
        fn = face_normals(verts, all_tris)
        wt = inv[all_tris]
        edge_tris = defaultdict(list)
        for ti, (a, b, c) in enumerate(wt):
            for e in ((a, b), (b, c), (c, a)):
                edge_tris[frozenset(e)].append(ti)
        for tl in edge_tris.values():
            for k in range(1, len(tl)):                  # any hidden (non-crease) edge fuses the groups
                t0, t1 = tl[0], tl[k]
                if tri_grp[t0] != tri_grp[t1] and abs(float(fn[t0] @ fn[t1])) >= cos_thr \
                        and not (respect_names
                                 and gname[int(tri_grp[t0])] != gname[int(tri_grp[t1])]):
                    parent[find(int(tri_grp[t0]))] = find(int(tri_grp[t1]))
    order, merged = [], {}
    for i, g in enumerate(groups):
        r = find(i)
        if r not in merged:
            merged[r] = {"name": g["name"], "tris": [np.asarray(g["tris"])]}
            order.append(r)
        else:
            merged[r]["tris"].append(np.asarray(g["tris"]))
    return [{"name": merged[r]["name"],
             "tris": np.concatenate(merged[r]["tris"]) if merged[r]["tris"] else np.zeros((0, 3), np.int32)}
            for r in order]


def _tri_area2(verts, tris):
    """2*area (= |edge×edge|) per triangle."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return np.linalg.norm(np.cross(b - a, c - a), axis=1)


def drop_degenerate_faces(verts, groups, rel_tol=1e-4):
    """Remove sliver triangles (area tiny RELATIVE to the mesh, not just exactly zero) and any group left
    empty by that. A triangulated ceiling carries Delaunay slivers whose unstable normal made explode
    spawn phantom `surface_N` layers with nothing you can see or pick — this cleans them (and their
    leftover layers) at load. Threshold = rel_tol × median triangle area, so real facets (never 10000×
    below the median) are safe. bbox groups pass through; dropping faces never removes vertices, so shared
    geometry is unaffected."""
    verts = np.asarray(verts, np.float64)
    areas = [_tri_area2(verts, np.asarray(g["tris"])) for g in groups
             if not _is_bbox(g["name"]) and len(g["tris"])]
    if not areas:
        return list(groups)
    thr = max(1e-12, rel_tol * float(np.median(np.concatenate(areas))))
    out = []
    for g in groups:
        if _is_bbox(g["name"]):
            out.append(g); continue
        tris = np.asarray(g["tris"])
        if len(tris) == 0:
            continue
        keep = tris[_tri_area2(verts, tris) > thr]
        if len(keep):
            out.append({"name": g["name"], "tris": keep})
    return out


def split_coplanar_components(verts, tris, angle_thresh_deg=5.0, weld_tol=1e-5):
    """Split one triangle set into surfaces = connected components joined by a HIDDEN (coplanar) shared
    edge — the inverse of merge_coplanar_groups for a single group. Used by 'Explode' to break a
    name-merged layer back into its separate coplanar surfaces. Returns a list of tri arrays (one per
    surface); a single flat surface returns one component (nothing to explode)."""
    from collections import defaultdict
    tris = np.asarray(tris)
    n = len(tris)
    if n <= 1:
        return [tris]
    q = np.round(verts / weld_tol).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    fn = face_normals(verts, tris)
    # sliver triangles (area tiny relative to the mesh) get an unstable normal, so each would become its
    # own component -> a phantom surface_N with no area to show or pick. Absorb them into any edge-neighbour.
    area2 = _tri_area2(np.asarray(verts, np.float64), tris)
    med = float(np.median(area2)) if len(area2) else 0.0
    degen = area2 < max(1e-12, 1e-4 * med)
    wt = inv[tris]
    cos_thr = math.cos(math.radians(angle_thresh_deg))
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    edge_tris = defaultdict(list)
    for ti, (a, b, c) in enumerate(wt):
        for e in ((a, b), (b, c), (c, a)):
            edge_tris[frozenset(e)].append(ti)
    for tl in edge_tris.values():
        for k in range(1, len(tl)):                      # coplanar neighbours (or a sliver) stay in one component
            t0, t1 = tl[0], tl[k]
            if degen[t0] or degen[t1] or abs(float(fn[t0] @ fn[t1])) >= cos_thr:
                parent[find(t0)] = find(t1)
    comps = defaultdict(list)
    for ti in range(n):
        comps[find(ti)].append(ti)
    return [tris[idx] for idx in comps.values()]


def global_feature_edges(verts, groups, angle_thresh_deg=5.0, weld_tol=1e-5):
    """Per-group feature edges computed with GLOBAL adjacency across every non-bbox group. This hides a
    coplanar triangulation diagonal even when its two triangles live in DIFFERENT groups (e.g. an OBJ
    grouped one-triangle-per-group) — per-group feature_edges can't, since each group sees the diagonal
    as its own boundary. Returns a list parallel to `groups` of (K,2) original vertex-index arrays."""
    out = [np.zeros((0, 2), np.int32) for _ in groups]
    gi_list = [i for i, g in enumerate(groups) if not _is_bbox(g["name"]) and len(g["tris"])]
    if not gi_list:
        return out
    all_tris = np.concatenate([groups[i]["tris"] for i in gi_list])
    tri_grp = np.concatenate([np.full(len(groups[i]["tris"]), i, np.int32) for i in gi_list])
    q = np.round(verts / weld_tol).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    wtris = inv[all_tris]
    fn = face_normals(verts, all_tris)
    cos_thr = math.cos(math.radians(angle_thresh_deg))
    edge_tris, rep = {}, {}
    for ti, (a, b, c) in enumerate(wtris):
        tri = all_tris[ti]
        for (oa, ob), (x, y) in zip(((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])),
                                    ((a, b), (b, c), (c, a))):
            key = (x, y) if x < y else (y, x)
            edge_tris.setdefault(key, []).append(ti)
            rep.setdefault(key, (int(oa), int(ob)))
    per = {i: [] for i in gi_list}
    for key, tl in edge_tris.items():
        feat = len(tl) == 1 or any(abs(float(np.dot(fn[tl[0]], fn[tl[k]]))) < cos_thr
                                   for k in range(1, len(tl)))
        if not feat:
            continue                                     # coplanar interior diagonal -> hide
        for ti in set(tl):
            per[int(tri_grp[ti])].append(rep[key])
    for i, elist in per.items():
        if elist:
            out[i] = np.unique(np.array(elist, np.int32), axis=0)
    return out


# ─────────────────────────────────────────────────────────────
#  Ray / triangle math
# ─────────────────────────────────────────────────────────────
def ray_from_screen(mx, my, W, H, modelview, projection):
    """Returns (origin, direction) in world space."""
    viewport = [int(0), int(0), int(W), int(H)]
    # two depths to get a ray segment
    near = gluUnProject(float(mx), float(H-my), 0.0, modelview, projection, viewport)
    far  = gluUnProject(float(mx), float(H-my), 1.0, modelview, projection, viewport)
    near, far = np.array(near, np.float32), np.array(far, np.float32)
    d = far - near
    dn = np.linalg.norm(d)
    return near, (d/dn if dn > 1e-12 else d)


def ray_tri_intersect(ro, rd, a, b, c):
    """Möller–Trumbore. Returns t or None."""
    EPS = 1e-8
    e1, e2 = b-a, c-a
    pv = np.cross(rd, e2)
    det = np.dot(e1, pv)
    if abs(det) < EPS: return None
    inv = 1.0/det
    tv = ro - a
    u = np.dot(tv, pv) * inv
    if u < 0 or u > 1: return None
    qv = np.cross(tv, e1)
    v = np.dot(rd, qv) * inv
    if v < 0 or u+v > 1: return None
    t = np.dot(e2, qv) * inv
    return t if t > 1e-5 else None


def ray_group_hit(ro, rd, verts, tris):
    """Returns min t hitting any triangle in group, or None."""
    best = None
    for tri in tris:
        a,b,c = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        t = ray_tri_intersect(ro, rd,
                              a.astype(np.float64),
                              b.astype(np.float64),
                              c.astype(np.float64))
        if t is not None and (best is None or t < best):
            best = t
    return best


# ─────────────────────────────────────────────────────────────
#  GL viewport
# ─────────────────────────────────────────────────────────────
class GLViewer(QOpenGLWidget):
    layerPicked   = Signal(str)    # emitted on ray-pick
    pointPlaced   = Signal(str, float, float, float)  # mode, x, y, z
    pointRemoved  = Signal(str, int)                   # kind, index
    pointMoved    = Signal(str, int, float, float, float)  # kind, index, x, y, z
    pointHovered  = Signal(str, int)                   # kind, index (G-mode hover)
    receiverPicked = Signal(int)                        # receiver clicked in Results mode (-1 = none)
    pointDragStarted = Signal()                        # a grab/drag just began
    coordsChanged = Signal(float, float, float)        # live HUD coords
    themeToggle   = Signal()                            # RMB "Dark/Light scene" chosen
    sliceChanged  = Signal(float, float, float)         # x0,y0,z0 world slice positions
    frameChanged  = Signal(int)                         # current anim frame -> drives the scrubber
    fpvRequested     = Signal()                          # RMB "Enter receiver view" (first-person)
    fpvExitRequested = Signal()                          # RMB "Go back" from receiver view
    fpvPoseChanged   = Signal(float, float, float)       # FPV mouselook -> az,el,roll (deg) to the dials
    propsChanged     = Signal()                          # a prop transform confirmed/cancelled (refresh panel)
    surfacesPicked   = Signal(list)                       # RMB "Group selected" -> list of picked group names
    explodeSurface   = Signal(int)                         # RMB "Explode" -> split ONE picked group (by index)
    mergeSiblings    = Signal(str)                          # RMB -> collapse '<base>_N' exploded pieces back to '<base>'
    cleanseLayers    = Signal()                             # RMB -> drop zero/sliver-area layers + dump layer areas
    origFrameToggled = Signal(bool)                         # RMB -> show src/rec coords in the original OBJ frame
    shiftPolicyToggled = Signal(bool)                       # RMB -> rewrite the OBJ to the positive octant, or load as-is
    groupXfDone      = Signal(list)                         # point group move/rotate committed -> [(kind, idx), …] to sync tables
    modeLabelChanged = Signal(str)                          # current interaction mode -> status-bar badge
    pointsDeleted    = Signal(list)                         # edit-mode multi-delete -> [(kind, idx), …] to drop rows

    # placement modes
    MODE_NONE  = 0
    MODE_SRC   = 1
    MODE_REC   = 2
    MODE_SLICE = 3

    # default/reset camera = isometric: looking down the body diagonal so x, y, z are symmetric on
    # screen. yaw 225 (into the +x+y corner), pitch -54.736 = -(90 - 35.264), elevation 35.264 =
    # arcsin(1/sqrt3). Used by __init__ and the fit/reset (T) so they never drift apart.
    ISO_YAW = 225.0
    ISO_PITCH = -54.7356

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 500)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # camera
        self._yaw   = self.ISO_YAW      # starting view only — free to orbit after (isometric x/y/z)
        self._pitch = self.ISO_PITCH
        self._gxf = None          # active point group transform (move/rotate) or None
        self._pt_sel = []         # edit-mode multi-selection of points: [(kind, idx), …] (highlighted)
        self._pt_box = False      # True while an edit-mode rubber-band box-select is in progress
        self._nav_mode = False    # RMB-toggled: WAXD fly navigation active (else keys inert)
        self._show_help = False   # '?' toggles the shortcuts overlay
        self._last_mode_label = None   # last emitted mode string (status-bar badge)
        self._fly_keys = set()    # WAXD keys currently held for CadnaA-style fly navigation
        self._fly_timer = QTimer(self); self._fly_timer.setInterval(16)  # ~60 fps continuous move
        self._fly_timer.timeout.connect(self._fly_tick)
        self._free_x = 0.0        # Alt+LMB free tumble around screen X
        self._free_y = 0.0        # Alt+LMB free tumble around screen Y
        self._roll  =  0.0        # Ctrl+LMB roll around view axis
        self._dist  =   6.0
        self._fpv   = None          # first-person "inside a receiver" view: {eye,fwd,up} or None
        self._axes_visible = True   # grey infinite axes + origin 'o' (RMB menu toggles)
        self._pan   = np.zeros(3, np.float32)
        self._last_pos  = None
        self._mouse_btn = None
        # Drag-to-move state for already-placed sources/receivers.
        # A point is grabbed only when the user holds the "edit" key (E) at
        # mouse-press time, so the (cheap) closest-point search runs once per
        # press rather than on every mouse-move.
        self._drag_kind = None    # "source"/"receiver" while dragging, else None
        self._drag_idx  = -1      # index into the corresponding list
        self._grab_held = False   # True while "edit positions" (E) mode is armed
        # Axis constraint while dragging a point (mirrors placement's X/Y/Z lock).
        self._drag_axis      = None            # None / 'X' / 'Y' / 'Z'
        self._drag_axis_base = None            # point position when axis was locked
        self._drag_anchor_px = None            # mouse global pos when axis was locked

        # scene
        self._verts  = np.zeros((0,3), np.float32)
        self._groups = []   # {name, tris, color, visible, is_bbox}
        self._ray_paths = {}     # ray-tracer overlay: {ray_id: [(p0,p1,order), ...]} world coords
        self._ray_src   = {}     # ray_id -> source index it was launched from
        self._selected_ray = None
        self._show_rays = True
        self._bbox_visible  = False
        self._selected_layer = None
        self._surf_sel  = set()    # group names multi-selected for grouping; all highlighted
        self._surf_pick = False    # armed (RMB) -> next LMB drag rubber-bands surfaces into _surf_sel
        self._hover_world = None   # world coords under mouse for HUD


        # points
        self._sources   = []  # list of [x,y,z]
        self._receivers = []  # list of [x,y,z]
        # Index of the highlighted source/receiver (-1 = none). Driven either by
        # table selection or by G-mode hover, so the user can tell which point
        # in the scene corresponds to which table row.
        self._hl_src = -1
        self._hl_rec = -1
        self._sel_recs = set()          # receivers multi-selected for Results overlay
        self._results_pick = False      # LMB click picks/toggles a receiver instead of ray-pick
        self._glyph_queue = []   # per-frame source/receiver glyph overlay list
        self._grid_h = 0.0       # user grid spacing [m]; drives the i,j,k HUD indices
        self._hud_left = 0       # left inset [px] so the bottom-left HUD clears the tab panel

        # placement state
        self._mode      = self.MODE_NONE
        self._props     = []          # props3d.Prop list (placed primitives/assets); drawn + exported
        self._prop_sel  = -1          # selected prop index (panel highlight), or -1
        self._props_mode = False      # Draw mode: P/R/S transform props (and src/rec placement locked)
        self._xform     = None        # active modal transform: {mode,axis,idxs,bases,anchor}
        self._copy_arm  = False       # C pressed: next LMB drag draws a box to pick props to copy
        self._rect      = None        # [QPointF start, QPointF cur] rubber-band box while selecting
        self._props_undo_stack = []   # pre-op snapshots (list of Prop dicts) for Ctrl+Z on prop ops
        self._ortho     = False       # orthographic top-down projection (plan view for placement)
        self._axis_lock = None          # 'X','Y','Z' or None
        self._place_pos = np.zeros(3, np.float32)  # frozen base position
        self._cursor_world = np.zeros(3, np.float32)
        self._slices = None            # np(3,) world x0,y0,z0 of the animation slice planes
        self._plane_vis = {'x': True, 'y': True, 'z': True}  # which slice planes show / animate
        self._anim = None          # {'x':(F,h,w), ...} loaded pressure-slice frames, or None
        self._anim_frame = 0
        self._anim_min = -0.01
        self._anim_max = 0.01
        self._anim_tex = {}        # axis -> GL texture id
        self._anim_palette = 'color'   # 'color' | 'greys' | 'particles'
        self._sim2d = {}           # axis -> slice2d.Slice2DSim (live 2D FDTD, didactic particle anim)
        self._part_idx = {}        # axis -> (rows,cols) strided display sample indices
        self._part_gain = 1.0      # auto display gain (no knob)
        self._part_tex = None      # lazy round-ball point-sprite texture
        self._lut = None; self._lut_key = None

        # pivot (orbit centre = mesh centre)
        self._pivot      = np.zeros(3, np.float32)
        self._scene_span = 1.0   # updated by load_mesh
        self._domain_min = np.zeros(3, np.float32)  # scene AABB min
        self._domain_max = np.ones(3,  np.float32)  # scene AABB max
        # cached GL matrices (for unprojection)
        self._mv  = np.eye(4, dtype=np.float64)
        self._proj= np.eye(4, dtype=np.float64)

    # ── public ──────────────────────────────────────────────
    def load_mesh(self, verts, groups):
        self._verts  = verts
        self._obj_geom_cache = None      # invalidate cached triangles for the inside-mesh test
        self._groups = []
        color_idx = 0   # counts only non-bbox groups so colors are never skipped
        for g in groups:
            is_bbox = _is_bbox(g["name"])
            if is_bbox:
                color = BBOX_COLOR
            else:
                color = LAYER_COLORS[color_idx % len(LAYER_COLORS)]
                color_idx += 1
            self._groups.append({
                "name":    g["name"],
                "tris":    g["tris"],
                "normals": face_normals(verts, g["tris"]) if len(g["tris"]) else None,
                "edges":   None,                         # filled globally below
                "color":   color,
                "visible": True,
                "is_bbox": is_bbox,
            })
        # feature edges with GLOBAL adjacency: hides coplanar diagonals even across per-triangle groups.
        # The bbox is its own thing (keep its box outline via per-group edges).
        gedges = global_feature_edges(verts, self._groups)
        for gi, grp in enumerate(self._groups):
            grp["edges"] = (feature_edges(verts, grp["tris"]) if grp["is_bbox"] and len(grp["tris"])
                            else gedges[gi])
        if len(verts):
            mn, mx = verts.min(0), verts.max(0)
            ctr  = (mn + mx) * 0.5
            span = (mx - mn).max()
            self._pivot      = ctr.copy()
            self._scene_span = span if span > 0 else 1.0
            self._domain_min = mn.copy()
            self._domain_max = mx.copy()
            self._dist       = span * 1.8 if span > 0 else 6.0
            # nudge the scene right (~10% of view width) so the sliding left panel doesn't cover it
            self._pan        = np.array([0.12 * self._dist, 0.0, 0.0], np.float32)
            self._yaw   = self.ISO_YAW      # starting/reset view only — orbit freely after
            self._pitch = self.ISO_PITCH
            self._free_x = 0.0
            self._free_y = 0.0
            self._roll   = 0.0
            # snapshot the exact initial view so reset can restore it precisely
            self._save_home_view()
            # default placement position at mesh centre
            self._place_pos = ctr.copy()
        self.update()

    def set_layer_visible(self, name, v):
        for g in self._groups:
            if g["name"] == name: g["visible"] = v
        self.update()

    def is_layer_visible(self, name):
        for g in self._groups:
            if g["name"] == name:
                return bool(g["visible"])
        return False

    def set_bbox_visible(self, v):
        self._bbox_visible = v; self.update()

    def select_layer(self, name):
        self._selected_layer = name if name else None
        self.update()

    def clear_selection(self):
        self._selected_layer = None
        self.update()

    def clear_surf_sel(self):
        self._surf_sel = set(); self.update()

    def _prop_layers(self):
        """Layer names owned by placed props (columns/boxes/…). The surface tools must never group/explode/
        cleanse these — they only apply to the domain enclosure. Props are baked into the OBJ under their
        layer name, so excluding these names keeps prop geometry intact."""
        return {p.layer for p in self._props}

    def arm_surface_rect(self):
        """Arm a one-shot rubber-band surface select: the next LMB drag boxes surfaces (front-most
        only) into _surf_sel. RMB then groups/explodes them."""
        self._surf_pick = True
        self.setCursor(Qt.CursorShape.CrossCursor)

    def _find_surfaces_in_rect(self, rect):
        """Front-most visible surfaces inside the screen rectangle. Samples a grid of pixels and keeps
        the nearest ray hit at each (occluded back walls are NOT grabbed — the random-pick bug of the old
        any-vertex-inside test). Inner surfaces show up only when a dollhouse cut exposes them. One
        numpy ray-vs-ALL-triangles pass per sample (per-triangle Python loops froze the UI)."""
        W, H = self.width(), self.height()
        x0, x1 = sorted((rect[0].x(), rect[1].x())); y0, y1 = sorted((rect[0].y(), rect[1].y()))
        x0 = max(0.0, x0); y0 = max(0.0, y0); x1 = min(float(W), x1); y1 = min(float(H), y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return set()
        # concat pickable groups' tris once + remember which group each triangle came from.
        # Props are excluded — surface grouping is only for the domain enclosure.
        pl = self._prop_layers()
        cand = [g for g in self._groups
                if g["visible"] and not g["is_bbox"] and not self._dollhouse_cut(g)
                and len(g["tris"]) and g["name"] not in pl]
        if not cand:
            return set()
        tris = np.concatenate([np.asarray(g["tris"]) for g in cand])
        tri_grp = np.concatenate([np.full(len(g["tris"]), i, np.int32) for i, g in enumerate(cand)])
        verts = np.asarray(self._verts, np.float64)
        v0 = verts[tris[:, 0]]; e1 = verts[tris[:, 1]] - v0; e2 = verts[tris[:, 2]] - v0

        def nearest(ro, rd):
            p = np.cross(rd, e2)
            det = np.einsum('ij,ij->i', e1, p)
            ok = np.abs(det) > 1e-9
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            tv = ro - v0
            u = np.einsum('ij,ij->i', tv, p) * inv
            q = np.cross(tv, e1)
            v = np.einsum('j,ij->i', rd, q) * inv
            tt = np.einsum('ij,ij->i', e2, q) * inv
            hit = ok & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (tt > 1e-6)
            if not hit.any():
                return None
            return int(tri_grp[np.argmin(np.where(hit, tt, np.inf))])

        step = max(8.0, max(x1 - x0, y1 - y0) / 25.0)   # cap ~25 samples/axis
        names, yy = set(), y0
        while yy <= y1:
            xx = x0
            while xx <= x1:
                ro, rd = ray_from_screen(xx, yy, W, H, self._mv, self._proj)
                gi = nearest(ro.astype(np.float64), rd.astype(np.float64))
                if gi is not None:
                    names.add(cand[gi]["name"])
                xx += step
            yy += step
        return names

    def _rot_all(self):
        """The arcball's composite rotation (world→eye), matching the glRotatef order in _setup_camera:
        Rz(roll)·Rx(free_x)·Ry(free_y)·Rx(pitch)·Rz(yaw)."""
        r = math.radians
        def Rx(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        def Ry(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        def Rz(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return (Rz(r(self._roll)) @ Rx(r(self._free_x)) @ Ry(r(self._free_y))
                @ Rx(r(self._pitch)) @ Rz(r(self._yaw)))

    def _fly_steer(self, dx, dy):
        """Cockpit mouselook: turn about the EYE, not the pivot. Rotate, then move the pivot so the eye
        position is unchanged (eye = pivot + R.T·[-pan_x,-pan_y,dist]). Without this, yaw/pitch orbit
        the scene around a point ahead of you."""
        off = np.array([-self._pan[0], -self._pan[1], self._dist], float)
        eye = np.asarray(self._pivot, float) + self._rot_all().T @ off
        self._yaw   += dx * 0.2
        self._pitch += dy * 0.2
        self._pivot = (eye - self._rot_all().T @ off).astype(np.float32)

    def set_nav_mode(self, on):
        """Toggle CadnaA-style fly navigation (WAXD move, mouse steers while W/X held). Off = keys inert."""
        self._nav_mode = bool(on)
        if not on:
            self._fly_keys.clear(); self._fly_timer.stop()
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)
        self.update()

    def _fly_tick(self):
        """CadnaA-style fly: move the camera along its view dir / strafe while WAXD are held. Translating
        the pivot translates the eye (arcball), so _dist/angles stay put."""
        if not self._fly_keys or self._fpv is not None or self._view_dir is None:
            return
        vd = np.asarray(self._view_dir, np.float64)
        up = np.array([0, 0, 1.0])                              # Q/Z change height along world up
        right = np.cross(vd, up); rn = np.linalg.norm(right)
        right = right / rn if rn > 1e-6 else np.array([1.0, 0, 0])
        K = Qt.Key
        move = (vd    * (K.Key_W in self._fly_keys) - vd    * (K.Key_X in self._fly_keys)
                + right * (K.Key_D in self._fly_keys) - right * (K.Key_A in self._fly_keys)
                + up    * (K.Key_Q in self._fly_keys) - up    * (K.Key_Z in self._fly_keys))
        n = np.linalg.norm(move)
        if n > 1e-9:
            step = max(self._scene_span, 1e-3) * 0.02          # ~1.2 spans/sec at 60 fps
            self._pivot = (np.asarray(self._pivot, np.float64) + move / n * step).astype(np.float32)
            self.update()

    def _pick_group_idx(self, qpos):
        """Index into self._groups of the nearest surface hit by the ray through qpos, or -1. Same
        filters as _do_ray_pick (visible, non-bbox unless shown, not dollhouse-cut). Index (not name)
        so explode acts on the ONE clicked surface — several groups can share a name (e.g. props)."""
        W, H = self.width(), self.height()
        ro, rd = ray_from_screen(qpos.x(), qpos.y(), W, H, self._mv, self._proj)
        best_t, best_i = None, -1
        for i, g in enumerate(self._groups):
            if not g["visible"]: continue
            if g["is_bbox"] and not self._bbox_visible: continue
            if self._dollhouse_cut(g): continue
            if len(g["tris"]) == 0: continue
            t = ray_group_hit(ro.astype(np.float64), rd.astype(np.float64), self._verts, g["tris"])
            if t is not None and (best_t is None or t < best_t):
                best_t, best_i = t, i
        return best_i

    def _pick_group_name(self, qpos):
        i = self._pick_group_idx(qpos)
        return self._groups[i]["name"] if i >= 0 else None

    def set_sources(self, pts):
        # keep x,y,z and the SPL (Lp,1m [dB], 4th col) so the particle sim can size its injection
        self._sources = [list(p[:3]) + ([float(p[3])] if len(p) > 3 else []) for p in pts]
        self.update()

    def set_receivers(self, pts):
        self._receivers = [list(p[:3]) for p in pts]; self.update()

    def set_highlight(self, kind, idx):
        """Highlight a source/receiver by index in the 3D scene. kind is
        'source' / 'receiver' / None (None clears both)."""
        self._hl_src = idx if kind == "source"   else -1
        self._hl_rec = idx if kind == "receiver" else -1
        self.update()

    def set_selected_receivers(self, indices):
        """Receivers highlighted as the Results multi-selection."""
        self._sel_recs = set(int(i) for i in indices); self.update()

    def set_results_pick(self, on):
        self._results_pick = bool(on)

    def enter_placement_mode(self, mode):
        self._mode      = mode
        self._axis_lock = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def enter_slice_mode(self):
        """Slice-plane picking: hold X/Y/Z to slide the x/y/z plane, LMB commits.
        Slides from the domain centre; the three planes stay drawn."""
        c = (self._domain_min + self._domain_max) * 0.5
        if self._slices is None:
            self._slices = c.astype(np.float32).copy()
        self._place_pos = c.astype(np.float32).copy()
        self.enter_placement_mode(self.MODE_SLICE)

    def set_slices(self, x, y, z):
        self._slices = np.array([x, y, z], np.float32)
        self.update()

    def set_plane_visible(self, axis, on):
        self._plane_vis[axis] = bool(on)
        self.update()

    # ── slice-pressure animation ──
    def load_animation(self, out_dir):
        """Load pressureX/Y/Z.bin + header.txt from out_dir. Returns frame count (0 if none).
        Layouts (from main3D): X = F×(Ny·Nz) idx j+Ny·k -> (F,Nz,Ny); Y = F×(Nx·Nz) -> (F,Nz,Nx);
        Z = F×(Nx·Ny) -> (F,Ny,Nx)."""
        try:
            hp = open(os.path.join(out_dir, "header.txt")).read().split()
            Nx, Ny, Nz = int(hp[0]), int(hp[1]), int(hp[2])
        except Exception:
            return 0
        planes = {'x': ("pressureX.bin", Nz, Ny),
                  'y': ("pressureY.bin", Nz, Nx),
                  'z': ("pressureZ.bin", Ny, Nx)}
        anim, frames = {}, 0
        for ax, (fn, hgt, wid) in planes.items():
            p = os.path.join(out_dir, fn)
            if not os.path.isfile(p):
                continue
            a = np.fromfile(p, np.float32)
            per = hgt * wid
            if per == 0 or a.size < per:
                continue
            F = a.size // per
            anim[ax] = a[:F * per].reshape(F, hgt, wid)
            frames = max(frames, F)
        if not anim:
            return 0
        self._anim = anim
        # Particles: a LIVE 2D FDTD per visible plane (slice2d.Slice2DSim), seeded by the scene's
        # sources projected onto the plane. This is a didactic 2D re-sim, NOT the 3D field. The GL
        # particle renderer reads each sim's `disp`; step() is driven per tick from step_frame().
        from slice2d import Slice2DSim
        try:
            cfg = np.atleast_1d(np.loadtxt(os.path.join(out_dir, os.pardir, "config", "config.txt")))
            src_opt = int(cfg[15]) if len(cfg) > 15 else 0       # 0/1 pulse, 2 sine
            g = lambda i: (len(cfg) > i and float(cfg[i]) > 0.0)  # PML face flags -> absorbing edges
            U, D, R, L, F, B = g(6), g(7), g(8), g(9), g(10), g(11)
        except Exception:
            src_opt = 0; U = D = R = L = F = B = False
        # box face -> plane edge (from functions3D.cuh): L/R=x-/x+, B/F=y-/y+, D/U=z-/z+.
        # edges are (top, bottom, left, right) = (row-min, row-max, col-min, col-max)
        absorb_by_ax = {'x': (D, U, B, F),      # YZ: col=y, row=z
                        'y': (D, U, L, R),      # XZ: col=x, row=z
                        'z': (B, F, L, R)}      # XY: col=x, row=y
        planes2d = {'x': (Nz, Ny), 'y': (Nz, Nx), 'z': (Ny, Nx)}   # (hgt, wid) per plane at solver res
        self._sim2d = {}; self._part_idx = {}; self._part_gain = 1.0
        PAD = 4
        CAP = 100                                  # cap the live-sim grid: keeps the mask (point-in-mesh,
        if self._slices is None or not len(self._verts):   # a Python per-point loop) and the FDTD fast
            self._anim_frame = 0; self.update(); return frames
        for ax, (h0, w0) in planes2d.items():
            s = max(1, (max(h0, w0) + CAP - 1) // CAP)
            hgt, wid = max(8, h0 // s), max(8, w0 // s)
            stride = max(1, max(hgt, wid) // 60)   # uniform particle spacing on the capped grid
            # room cross-section mask on this plane (True = inside room), via the point-in-mesh test
            jj, ii = np.meshgrid(np.arange(hgt), np.arange(wid), indexing='ij')
            wcen = self._cells_world(ax, wid, hgt, ii.ravel() + 0.5, jj.ravel() + 0.5)
            inside = self._inside_obj(wcen).reshape(hgt, wid)
            if not inside.any():
                continue
            # strided, jittered particle lattice trimmed to the room interior
            rr, cc = np.meshgrid(np.arange(0, hgt, stride), np.arange(0, wid, stride), indexing='ij')
            rr = rr.ravel(); cc = cc.ravel()
            rng = np.random.default_rng(ord(ax))                        # distinct, stable per plane
            pr = (rng.random(len(rr)) * stride).astype(np.float32)      # jitter [0,stride)
            pc = (rng.random(len(cc)) * stride).astype(np.float32)
            keep = (rr >= PAD) & (rr < hgt - PAD) & (cc >= PAD) & (cc < wid - PAD)
            keep &= inside[np.clip(rr, 0, hgt - 1), np.clip(cc, 0, wid - 1)]
            self._part_idx[ax] = (rr[keep], cc[keep], pr[keep], pc[keep])
            # project each 3D source onto this plane -> (row, col, SPL), then build the live sim
            srcs = [self._world_to_plane(ax, s, wid, hgt) + (s[3] if len(s) > 3 else 94.0,)
                    for s in self._sources]
            self._sim2d[ax] = Slice2DSim(inside, srcs, src_opt=src_opt, absorb=absorb_by_ax[ax])
        # Gain: warm up to measure the grid's displacement-per-unit-amplitude, then anchor the gain
        # to a REFERENCE SPL (not the live field). This way ABSOLUTE source SPL and SPL_AMP_CALIB DO
        # change the on-screen pulse: displayed ~ 1.2 * CALIB * 10^((SPL - 94dB)/20) cells. (An
        # auto-normalised gain would cancel amplitude entirely.) Reset after so playback starts at t=0.
        from slice2d import P_REF
        REF_SPL = 94.0
        peaks = []; rep_amp = 0.0
        for sim in self._sim2d.values():
            rep_amp = max(rep_amp, float(sim.amp))
            for _ in range(50):
                sim.step(1)
                a = np.abs(sim.disp).ravel()
                if a.size:
                    peaks.append(float(np.percentile(a, 99)))
            sim.reset()
        peaks = [p for p in peaks if p > 1e-12]
        if peaks and rep_amp > 0:
            k = float(np.median(peaks)) / rep_amp              # displacement per unit source amplitude
            amp_ref = P_REF * 10.0 ** (REF_SPL / 20.0)         # reference pressure (no calib)
            self._part_gain = 1.2 / (k * amp_ref)              # ~1.2*CALIB cells at 94 dB, scales w/ SPL
        self._anim_frame = 0
        self.update()
        return frames

    def _world_to_plane(self, ax, world, W, H):
        """3D world point -> (row, col) cell on plane `ax` (inverse of _cells_world); the out-of-plane
        coordinate is dropped (projection). Clamped to the grid."""
        mn, mx = self._domain_min, self._domain_max
        def frac(val, a, b):
            return 0.0 if b <= a else (float(val) - a) / (b - a)
        if ax == 'x':      # plane YZ: col<-y, row<-z
            u = frac(world[1], mn[1], mx[1]); v = frac(world[2], mn[2], mx[2])
        elif ax == 'y':    # plane XZ: col<-x, row<-z
            u = frac(world[0], mn[0], mx[0]); v = frac(world[2], mn[2], mx[2])
        else:              # plane XY: col<-x, row<-y
            u = frac(world[0], mn[0], mx[0]); v = frac(world[1], mn[1], mx[1])
        return (int(np.clip(v * H, 0, H - 1)), int(np.clip(u * W, 0, W - 1)))

    def clear_animation(self):
        self._anim = None
        self.update()

    def set_pressure_range(self, pmin, pmax):
        self._anim_min = float(pmin); self._anim_max = float(pmax)
        self.update()

    def set_anim_palette(self, name):
        self._anim_palette = name
        self.update()

    def step_frame(self):
        if self._anim is None:
            return
        if self._anim_palette == 'particles':          # advance the live 2D sims each tick
            for ax, sim in self._sim2d.items():
                if self._plane_vis[ax]:
                    sim.step()
        F = max(len(v) for v in self._anim.values())
        self._anim_frame = (self._anim_frame + 1) % F
        # loop the live particle sim in sync with the frame cycle so it repeats like color/greys
        # (otherwise it just rings on forever and never restarts)
        if self._anim_palette == 'particles' and self._anim_frame == 0:
            for sim in self._sim2d.values():
                sim.reset()
        self.frameChanged.emit(self._anim_frame)
        self.update()

    def seek(self, k):
        """Scrubber jump to frame k. color/greys index stored frames; the live particle sim has no
        stored past, so it replays from reset to k (cheap; F is small)."""
        if self._anim is None:
            return
        F = max((len(v) for v in self._anim.values()), default=0)
        if F == 0:
            return
        k = max(0, min(F - 1, int(k)))
        if self._anim_palette == 'particles':
            for sim in self._sim2d.values():
                sim.reset()
            for _ in range(k):
                for ax, sim in self._sim2d.items():
                    if self._plane_vis[ax]:
                        sim.step()
        self._anim_frame = k
        self.update()

    def _colormap(self, img):
        """2-D pressure slice -> (h,w,4) uint8 RGBA via the pressure LUT + [min,max] range."""
        key = (self._anim_palette, not SCENE_LIGHT_MODE)   # rebuild LUT on palette/scheme change
        if key != self._lut_key:
            self._lut = pressure_lut(self._anim_palette, not SCENE_LIGHT_MODE)
            self._lut_key = key
        lo, hi = self._anim_min, self._anim_max
        span = (hi - lo) if hi > lo else 1.0
        idx = np.clip((img - lo) / span, 0.0, 1.0) * 255.0
        idx = idx.astype(np.uint8)
        rgb = self._lut[idx]                            # (h,w,3)
        h, w = img.shape
        rgba = np.empty((h, w, 4), np.uint8)
        rgba[..., :3] = rgb
        # alpha follows |p|: zero pressure (ghost/PML border, silence) -> transparent, so the
        # dead frame around the domain drops out and only the wavefronts show.
        amp = max(abs(lo), abs(hi), 1e-12)
        rgba[..., 3] = (np.clip(np.abs(img) / amp, 0.0, 1.0) * 235).astype(np.uint8)
        return rgba

    def _draw_anim(self):
        """Texture the visible slice quads with the current animation frame."""
        if self._anim is None or self._slices is None or self._anim_palette == 'particles':
            return
        mn, mx = self._domain_min, self._domain_max
        x0, y0, z0 = float(self._slices[0]), float(self._slices[1]), float(self._slices[2])
        # quad corners in draw order matching texcoords (0,0)(1,0)(1,1)(0,1)
        quads = {
            'x': [(x0, mn[1], mn[2]), (x0, mx[1], mn[2]), (x0, mx[1], mx[2]), (x0, mn[1], mx[2])],
            'y': [(mn[0], y0, mn[2]), (mx[0], y0, mn[2]), (mx[0], y0, mx[2]), (mn[0], y0, mx[2])],
            'z': [(mn[0], mn[1], z0), (mx[0], mn[1], z0), (mx[0], mx[1], z0), (mn[0], mx[1], z0)],
        }
        tc = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        # sort planes BACK-TO-FRONT by projected depth of their centre, so the nearer plane
        # composites over the farther one (fixed x,y,z order made z always look in front).
        cen = {'x': (x0, (mn[1]+mx[1])*0.5, (mn[2]+mx[2])*0.5),
               'y': ((mn[0]+mx[0])*0.5, y0, (mn[2]+mx[2])*0.5),
               'z': ((mn[0]+mx[0])*0.5, (mn[1]+mx[1])*0.5, z0)}
        vp = [0, 0, int(self.width()), int(self.height())]
        def _depth(ax):
            try:
                return gluProject(cen[ax][0], cen[ax][1], cen[ax][2], self._mv, self._proj, vp)[2]
            except Exception:
                return 0.0
        order = sorted(('x', 'y', 'z'), key=_depth, reverse=True)   # far first, near last
        glDisable(GL_LIGHTING); glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # depth-WRITE the visible wavefronts so intersecting planes occlude each other
        # per-pixel; alpha-test drops the transparent silence so it doesn't write depth and
        # hide the plane behind it. This is what actually fixes front/back at the crossings.
        glEnable(GL_ALPHA_TEST); glAlphaFunc(GL_GREATER, 0.06)
        glDepthMask(GL_TRUE)
        glColor4f(1, 1, 1, 1)
        for ax in order:
            if not self._plane_vis[ax] or ax not in self._anim:
                continue
            frame = self._anim[ax][self._anim_frame % len(self._anim[ax])]
            rgba = self._colormap(frame)
            h, w = frame.shape
            tex = self._anim_tex.get(ax)
            if tex is None:
                tex = glGenTextures(1); self._anim_tex[ax] = tex
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba)
            glBegin(GL_QUADS)
            for v, (s, t) in zip(quads[ax], tc):
                glTexCoord2f(s, t); glVertex3f(*v)
            glEnd()
        glDisable(GL_TEXTURE_2D); glDisable(GL_ALPHA_TEST)
        glDepthMask(GL_TRUE); glDisable(GL_BLEND); glEnable(GL_LIGHTING)

    def _cells_world(self, ax, W, H, col, row):
        """Map particle (col,row) cell coords on plane `ax` to world points (N,3)."""
        mn, mx = self._domain_min, self._domain_max
        x0, y0, z0 = float(self._slices[0]), float(self._slices[1]), float(self._slices[2])
        u = col / W; v = row / H
        w = np.empty((len(col), 3), np.float32)
        if ax == 'x':
            w[:, 0] = x0; w[:, 1] = mn[1] + u*(mx[1]-mn[1]); w[:, 2] = mn[2] + v*(mx[2]-mn[2])
        elif ax == 'y':
            w[:, 1] = y0; w[:, 0] = mn[0] + u*(mx[0]-mn[0]); w[:, 2] = mn[2] + v*(mx[2]-mn[2])
        else:
            w[:, 2] = z0; w[:, 0] = mn[0] + u*(mx[0]-mn[0]); w[:, 1] = mn[1] + v*(mx[1]-mn[1])
        return w

    def _obj_geom(self):
        """Cached Möller constants for all non-bbox triangles, over the generic ray directions
        (module-level mesh_ray_geom / MESH_RAY_DIRS). Generic dirs dodge the axis-aligned-mesh degeneracy
        that made a single +x (or +x/+y/+z) ray miscount and split the room at some grid spacings."""
        if getattr(self, "_obj_geom_cache", None) is not None:
            return self._obj_geom_cache
        verts = np.asarray(self._verts, np.float64)
        T = [np.asarray(g["tris"]) for g in self._groups
             if not g.get("is_bbox") and len(g["tris"])]
        if not T or len(verts) == 0:
            self._obj_geom_cache = None; return None
        self._obj_geom_cache = mesh_ray_geom(verts, np.concatenate(T, 0))
        return self._obj_geom_cache

    def _inside_obj(self, pts):
        """Point-in-mesh, majority-voted over generic rays — see mesh_inside_vote. Returns all-True when
        there is no mesh geometry (nothing to mask against)."""
        G = self._obj_geom()
        if G is None:
            return np.ones(len(pts), bool)
        return mesh_inside_vote(pts, G)

    def _particle_tex(self):
        """Lazy radial-alpha disc so points render as round balls (POINT_SMOOTH is square on
        some drivers). White texture -> glColor tints it."""
        if self._part_tex:
            return self._part_tex
        n = 32
        yy, xx = np.mgrid[0:n, 0:n]
        d = np.sqrt((xx - (n - 1) / 2.0) ** 2 + (yy - (n - 1) / 2.0) ** 2) / ((n - 1) / 2.0)
        a = (np.clip(1.0 - d, 0.0, 1.0) ** 0.6 * 255).astype(np.uint8)
        rgba = np.dstack([np.full((n, n), 255, np.uint8)] * 3 + [a])
        tid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, n, n, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba)
        self._part_tex = tid
        return tid

    def _draw_particles(self):
        """Plain-blue balls displaced from their rest cell by the pressure-derived displacement;
        they return to rest once the wavefront has passed (displacement -> 0)."""
        if self._anim_palette != 'particles' or self._slices is None or not self._part_idx:
            return
        mn, mx = self._domain_min, self._domain_max
        x0, y0, z0 = float(self._slices[0]), float(self._slices[1]), float(self._slices[2])
        G = self._part_gain                     # movement untouched
        clamp = 4.0
        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, self._particle_tex())
        glEnable(GL_POINT_SPRITE); glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
        glPointSize(5.0); glColor4f(0.34, 0.30, 1.0, 1.0)     # bright purple-blue
        glEnableClientState(GL_VERTEX_ARRAY)
        for ax, sim in self._sim2d.items():
            if not self._plane_vis[ax] or ax not in self._part_idx:
                continue
            H, W = sim.H, sim.W
            rr, cc, pr, pc = self._part_idx[ax]
            disp = sim.disp[rr, cc]                            # (Nsel,2) live displacement in cells
            col = cc + 0.5 + pc + np.clip(G * disp[:, 0], -clamp, clamp)
            row = rr + 0.5 + pr + np.clip(G * disp[:, 1], -clamp, clamp)
            u = col / W; v = row / H
            world = np.empty((len(col), 3), np.float32)
            if ax == 'x':
                world[:, 0] = x0
                world[:, 1] = mn[1] + u * (mx[1] - mn[1]); world[:, 2] = mn[2] + v * (mx[2] - mn[2])
            elif ax == 'y':
                world[:, 1] = y0
                world[:, 0] = mn[0] + u * (mx[0] - mn[0]); world[:, 2] = mn[2] + v * (mx[2] - mn[2])
            else:
                world[:, 2] = z0
                world[:, 0] = mn[0] + u * (mx[0] - mn[0]); world[:, 1] = mn[1] + v * (mx[1] - mn[1])
            world = np.ascontiguousarray(world)
            glVertexPointer(3, GL_FLOAT, 0, world)
            glDrawArrays(GL_POINTS, 0, len(world))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisable(GL_POINT_SPRITE); glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)

    def exit_placement_mode(self):
        self._mode      = self.MODE_NONE
        self._axis_lock = None
        # also cancel any armed/in-progress point-edit (G) mode
        self._grab_held = False
        self._drag_kind = None
        self._drag_idx  = -1
        self._drag_axis = None
        self._drag_axis_base = None
        self._drag_anchor_px = None
        self._hl_src = -1
        self._hl_rec = -1
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    # ── GL callbacks ────────────────────────────────────────
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glDisable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        # TWO_SIDE: OpenGL flips the normal of back-facing polygons toward the
        # viewer before lighting, so a wall lit by a headlight looks the same
        # whether its OBJ winding points in or out. This is the key to making
        # color winding-independent.
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glDisable(GL_CULL_FACE)
        # Modest global ambient floor.
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.35, 0.35, 0.35, 1.0])
        # Single "headlight": diffuse + ambient. Its POSITION is set every
        # frame in _setup_camera (in eye space) so it always shines from the
        # camera toward whatever you are looking at.
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.7, 0.7, 0.7, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
        glClearColor(*SCENE_BG)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def enter_fpv(self, eye, az, el, roll):
        """Teleport the camera INTO a receiver: eye at its position, aimed by the head pose (az/el/roll
        deg, same world head frame the auralization uses). Position/zoom frozen; LMB looks, Ctrl+LMB
        rolls (see mouseMoveEvent), feeding the angles back to the dials."""
        self._fpv = {"eye": np.asarray(eye, np.float32),
                     "az": float(az), "el": float(el), "roll": float(roll)}
        self.update()

    def set_fpv_pose(self, az, el, roll):
        """Re-aim the frozen camera from outside (dials / Restore), without re-emitting fpvPoseChanged."""
        if self._fpv is None:
            return
        self._fpv["az"], self._fpv["el"], self._fpv["roll"] = float(az), float(el), float(roll)
        self.update()

    def exit_fpv(self):
        self._fpv = None
        self.update()

    def _fpv_basis(self):
        """(eye, fwd, up) world vectors from the stored az/el/roll — matches head_basis in binauralize."""
        f = self._fpv
        az, el, roll = math.radians(f["az"]), math.radians(f["el"]), math.radians(f["roll"])
        ca, sa, ce, se = math.cos(az), math.sin(az), math.cos(el), math.sin(el)
        fwd = np.array([ce * ca, ce * sa, se])
        left = np.cross([0, 0, 1.0], fwd); n = np.linalg.norm(left)
        left = left / n if n > 1e-6 else np.array([0, 1.0, 0])
        u0 = np.cross(fwd, left)
        cr, sr = math.cos(roll), math.sin(roll)
        return f["eye"], fwd, sr * left + cr * u0

    def _setup_camera(self):
        fpv = self._fpv
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        asp = self.width() / max(self.height(), 1)
        near = 0.02 if fpv else max(self._dist * 0.001, 0.01)
        far  = 2000.0 if fpv else self._dist * 500.0
        if self._ortho and not fpv:                             # orthographic plan/placement view
            half = max(self._dist * 0.5, 1e-3)
            glOrtho(-half * asp, half * asp, -half, half, -self._dist * 500.0, self._dist * 500.0)
        else:
            gluPerspective(70.0 if fpv else 45.0, asp, near, far)   # wider FOV inside a head
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Headlight: with the modelview at identity (eye space), position the
        # light just behind the eye looking down -Z. It stays glued to the
        # camera regardless of how the scene is rotated below.
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 1.0, 0.0])
        if fpv:                                                 # first-person: look from the receiver
            e, f, u = self._fpv_basis(); c = e + f
            gluLookAt(float(e[0]), float(e[1]), float(e[2]),
                      float(c[0]), float(c[1]), float(c[2]),
                      float(u[0]), float(u[1]), float(u[2]))
            self._mv   = glGetDoublev(GL_MODELVIEW_MATRIX)
            self._proj = glGetDoublev(GL_PROJECTION_MATRIX)
            self._view_dir = np.asarray(f, float) / (np.linalg.norm(f) or 1.0)
            return
        # Standard arcball:
        #   1. pull camera back along Z
        #   2. apply view-space pan  (pan moves the scene left/right/up in screen space)
        #   3. rotate around pivot
        #   4. translate world so pivot sits at origin
        glTranslatef(0.0, 0.0, -self._dist)
        glTranslatef(self._pan[0], self._pan[1], 0.0)   # pan in view/screen space
        # Camera rotation is intentionally split into two modes:
        #   - normal LMB orbit keeps OBJ Z as the main/up axis
        #   - Alt+LMB adds free tumble around screen axes
        #   - Ctrl+LMB rolls around the view axis
        # This keeps the room easy to navigate, but still allows total freedom
        # when inspecting hard-to-see surfaces such as the ceiling.
        glRotatef(self._roll,   0, 0, 1)
        glRotatef(self._free_x, 1, 0, 0)
        glRotatef(self._free_y, 0, 1, 0)
        glRotatef(self._pitch,  1, 0, 0)
        glRotatef(self._yaw,    0, 0, 1)
        glTranslatef(-self._pivot[0], -self._pivot[1], -self._pivot[2])
        self._mv   = glGetDoublev(GL_MODELVIEW_MATRIX)
        self._proj = glGetDoublev(GL_PROJECTION_MATRIX)
        # camera view direction in world (into the scene), for the dollhouse facing test. Built
        # by unprojecting screen-centre at depth 0 and 1 -> no matrix-layout guessing, correct
        # for ortho AND perspective.
        try:
            vp = [0, 0, int(self.width()), int(self.height())]
            cx, cy = self.width() * 0.5, self.height() * 0.5
            p0 = np.array(gluUnProject(cx, cy, 0.0, self._mv, self._proj, vp))
            p1 = np.array(gluUnProject(cx, cy, 1.0, self._mv, self._proj, vp))
            vd = p1 - p0
            self._view_dir = vd / (np.linalg.norm(vd) or 1.0)
        except Exception:
            self._view_dir = None

    def paintGL(self):
        glClearColor(*SCENE_BG)        # re-applied each frame so Dark/Light toggles take effect
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._setup_camera()
        self._glyph_queue = []            # collected by _draw_sources/_receivers
        self._draw_gizmo()   # replaces grid + world axes
        self._draw_axes()    # before the mesh so the wireframe paints on top (less z-fighting)
        self._draw_mesh()
        self._draw_props()
        self._draw_slices()
        self._draw_anim()
        self._draw_particles()
        self._draw_rays()
        self._draw_sources()
        self._draw_receivers()
        if self._mode != self.MODE_NONE:
            self._draw_placement_cursor()
        elif self._drag_kind is not None and self._drag_idx >= 0:
            # show the same axis guide + coordinate readout while moving a point
            self._draw_drag_guide()
        elif self._xform is not None:
            self._draw_xform_guide()               # crosshair + locked-axis line, like the drag guide
        if self._gxf is not None:
            self._draw_gxf_guide()                 # point group transform: centroid crosshair + axis line
        self._draw_hud()
        # origin marker + source/receiver glyphs, drawn last as a QPainter overlay
        gp = QPainter(self)
        self._paint_origin(gp)
        if self._glyph_queue:
            self._paint_point_glyphs(gp)
        self._paint_colorbar(gp)
        if self._rect is not None:
            self._paint_copy_rect(gp)
        if self._show_help:
            self._paint_help(gp)                   # '?' shortcuts overlay (on-demand, centred)
        gp.end()
        lbl = self._mode_label()                   # report mode to the status-bar badge (bottom-right)
        if lbl != self._last_mode_label:
            self._last_mode_label = lbl
            self.modeLabelChanged.emit(lbl)

    def _prop_center_world(self, xf):
        pr = self._props[xf["idxs"][0]]; c = pr.center()           # primary prop of the group
        return [pr.pos[0] + c[0], pr.pos[1] + c[1], pr.pos[2] + c[2]]

    def _draw_xform_guide(self):
        """Crosshair at the prop center + a locked-axis line, mirroring _draw_drag_guide so a prop
        transform and a source/receiver drag look/feel the same."""
        xf = self._xform
        if xf is None or not xf["idxs"] or not (0 <= xf["idxs"][0] < len(self._props)):
            return
        p = self._prop_center_world(xf)
        glDisable(GL_LIGHTING)
        col = (1.0, 1.0, 1.0) if xf["axis"] is None else AXIS_COLORS[xf["axis"]]
        glColor4f(col[0], col[1], col[2], 1.0); glLineWidth(1.5)
        d = self._dist * 0.04
        glBegin(GL_LINES)
        glVertex3f(p[0]-d, p[1], p[2]); glVertex3f(p[0]+d, p[1], p[2])
        glVertex3f(p[0], p[1]-d, p[2]); glVertex3f(p[0], p[1]+d, p[2])
        glVertex3f(p[0], p[1], p[2]-d); glVertex3f(p[0], p[1], p[2]+d)
        glEnd()
        if xf["axis"]:
            ext = self._dist * 1.5
            ax = {'X': (1, 0, 0), 'Y': (0, 1, 0), 'Z': (0, 0, 1)}[xf["axis"]]
            glLineWidth(2.0)
            glBegin(GL_LINES)
            glVertex3f(p[0]-ax[0]*ext, p[1]-ax[1]*ext, p[2]-ax[2]*ext)
            glVertex3f(p[0]+ax[0]*ext, p[1]+ax[1]*ext, p[2]+ax[2]*ext)
            glEnd()
        glEnable(GL_LIGHTING)

    def _draw_gxf_guide(self):
        """Point group move/rotate: crosshair at the group centroid + the locked-axis line, so the axis
        is visible while transforming."""
        gx = self._gxf
        if gx is None:
            return
        p = np.asarray(gx["centroid"], float)
        glDisable(GL_LIGHTING)
        # neutral white crosshair marks the centroid; only the LOCKED axis gets a long coloured line.
        glColor4f(1.0, 1.0, 1.0, 1.0); glLineWidth(1.5)
        d = self._dist * 0.04
        glBegin(GL_LINES)
        glVertex3f(p[0]-d, p[1], p[2]); glVertex3f(p[0]+d, p[1], p[2])
        glVertex3f(p[0], p[1]-d, p[2]); glVertex3f(p[0], p[1]+d, p[2])
        glVertex3f(p[0], p[1], p[2]-d); glVertex3f(p[0], p[1], p[2]+d)
        glEnd()
        if gx.get("axis"):
            ext = self._dist * 1.5
            ax = {'X': (1, 0, 0), 'Y': (0, 1, 0), 'Z': (0, 0, 1)}[gx["axis"]]
            col = AXIS_COLORS[gx["axis"]]
            glColor4f(col[0], col[1], col[2], 1.0); glLineWidth(2.5)
            glBegin(GL_LINES)
            glVertex3f(p[0]-ax[0]*ext, p[1]-ax[1]*ext, p[2]-ax[2]*ext)
            glVertex3f(p[0]+ax[0]*ext, p[1]+ax[1]*ext, p[2]+ax[2]*ext)
            glEnd()
        glEnable(GL_LIGHTING)

    def _paint_copy_rect(self, p):
        """Rubber-band box while copy-selecting props (C then LMB-drag)."""
        from PySide6.QtCore import QRectF
        a, b = self._rect
        r = QRectF(a, b).normalized()
        col = QColor(PAL["accent"])
        p.fillRect(r, QColor(col.red(), col.green(), col.blue(), 40))
        p.setPen(col); p.drawRect(r)

    def _paint_colorbar(self, p):
        """Vertical LUT scale bar on the middle-right edge while a colour/greys animation runs,
        labelled with the user's Pressure range [min,max]."""
        if (self._anim is None or self._anim_palette not in ('color', 'greys')
                or not getattr(self, "_anim_playing", False)):
            return
        from PySide6.QtGui import QImage
        from PySide6.QtCore import QRect
        lut = pressure_lut(self._anim_palette, not SCENE_LIGHT_MODE)      # (256,3) uint8
        strip = np.ascontiguousarray(lut[::-1])                          # row 0 = max at top
        img = QImage(strip.data, 1, 256, 3, QImage.Format.Format_RGB888)
        W, H = self.width(), self.height()
        bw = 14; bh = max(120, int(H * 0.4)); bx = W - bw - 56; by = (H - bh) // 2
        p.drawImage(QRect(bx, by, bw, bh), img)
        fg = QColor(20, 20, 20) if SCENE_LIGHT_MODE else QColor(235, 235, 235)
        p.setPen(fg); p.drawRect(QRect(bx, by, bw, bh))
        f = QFont(); f.setPointSize(8); p.setFont(f)
        lo, hi = self._anim_min, self._anim_max
        tx = bx + bw + 5
        p.drawText(bx - 2, by - 6, "Pa")
        p.drawText(tx, by + 9, f"{hi:+.3g}")
        p.drawText(tx, by + bh // 2 + 4, f"{(lo + hi) * 0.5:+.3g}")
        p.drawText(tx, by + bh, f"{lo:+.3g}")

    # ── drawing helpers ──────────────────────────────────────
    def _draw_gizmo(self):
        """
        Fixed top-left axis gizmo with transparency.
        """
        W, H = self.width(), self.height()
        size  = 80
        pad   = 14
        box   = pad*2 + size
        cx    = pad + size // 2
        cy_gl = pad + size // 2
        vp_x  = W - box
        vp_y  = 0
    
        L  = size * 0.36
        br = size * 0.10
    
        # Add alpha (0.5 = 50% transparency) to each color
        axes = [            
            ((1,0,0), AXIS_COLORS['X'] + (0.99,), 'X'),
            ((0,1,0), AXIS_COLORS['Y'] + (0.99,), 'Y'),
            ((0,0,1), AXIS_COLORS['Z'] + (0.99,), 'Z'),
        ]
    
        glViewport(vp_x, vp_y, box, box)
        glScissor (vp_x, vp_y, box, box)
        glEnable(GL_SCISSOR_TEST)
    
        glMatrixMode(GL_PROJECTION)
        glPushMatrix(); glLoadIdentity()
        glOrtho(0, box, 0, box, -200, 200)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix(); glLoadIdentity()
    
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # ── ENABLE BLENDING FOR TRANSPARENCY ────────────────
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Standard alpha blending
        # ───────────────────────────────────────────────────
    
        glLineWidth(2.0)
    
        yaw_r   = math.radians(self._yaw)
        pitch_r = math.radians(self._pitch)
    
        def project_axis(ax, ay, az):
            # returns (screen_x, screen_y, depth) — depth is the view-space z
            # (larger = nearer the camera) used only to paint-sort the 3 axes.
            x1 = ax * math.cos(yaw_r) - ay * math.sin(yaw_r)
            y1 = ax * math.sin(yaw_r) + ay * math.cos(yaw_r)
            z1 = az
            x2 = x1
            y2 = y1 * math.cos(pitch_r) - z1 * math.sin(pitch_r)
            depth = y1 * math.sin(pitch_r) + z1 * math.cos(pitch_r)
            return x2, y2, depth

        # Project all three tips once. depth = view-space z (larger = nearer).
        proj = []
        for (ax,ay,az), col, label in axes:
            px, py, dz = project_axis(ax, ay, az)
            proj.append((px, py, dz, col, label))

        # ── build ONE draw list of sticks + balls, sorted by depth ──
        # Every primitive (each stick and each ball) carries a representative
        # depth and is painted back-to-front together, so a stick correctly
        # passes in front of / behind another stick or a ball (painter's
        # algorithm over ~9 items — no depth buffer, negligible cost).
        #   item = (depth, kind, data...)
        #     kind "stick" -> (px, py, col)
        #     kind "ball"  -> (tx, ty, rad, col)
        br_neg = br * 0.8
        items = []
        for px, py, dz, col, label in proj:
            # stick centre->+tip: sort by its midpoint depth (centre is depth 0)
            items.append((dz * 0.5, "stick", px, py, col))
            # positive ball: full size, opaque
            items.append((dz, "ball", cx + px*L, cy_gl + py*L, br, col))
            # negative ball: smaller, opacity from the editable GIZMO_NEG_ALPHA
            ncol = (col[0], col[1], col[2],
                    (col[3] if len(col) > 3 else 1.0) * GIZMO_NEG_ALPHA)
            items.append((-dz, "ball", cx - px*L, cy_gl - py*L, br_neg, ncol))
        items.sort(key=lambda t: t[0])           # farthest first, nearest last

        segs = 20
        for it in items:
            if it[1] == "stick":
                _, _, px, py, col = it
                glColor4f(*col)
                glBegin(GL_LINES)
                glVertex2f(cx, cy_gl)
                glVertex2f(cx + px*L, cy_gl + py*L)
                glEnd()
            else:
                _, _, tx, ty, rad, col = it
                glColor4f(*col)
                glBegin(GL_TRIANGLE_FAN)
                glVertex2f(tx, ty)
                for k in range(segs + 1):
                    a = 2 * math.pi * k / segs
                    glVertex2f(tx + rad * math.cos(a), ty + rad * math.sin(a))
                glEnd()
    
        # ── DISABLE BLENDING AFTER GIZMO ──────────────────
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW);  glPopMatrix()
        glDisable(GL_SCISSOR_TEST)
        glViewport(0, 0, W, H)
    
        # ── labels via QPainter ───────────────────────────
        # The gizmo is rendered in a sub-viewport at (vp_x, 0) sized box×box with
        # glOrtho(0,box,0,box), so a gizmo-local point (gx, gy) maps to window
        # coords (vp_x + gx, H - gy). Each label sits just BEYOND its ball, along
        # the axis direction, so the letter lines up with the axis it names.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setFont(QFont("Monospace", 9, QFont.Weight.Bold))
        fm = painter.fontMetrics()
        for (ax,ay,az), col, label in axes:
            px, py, _dz = project_axis(ax, ay, az)
            mag = math.hypot(px, py) or 1.0
            ux, uy = px / mag, py / mag
            gx = cx + px * L + ux * (br + 8)     # tip + just past the ball
            gy = cy_gl + py * L + uy * (br + 8)
            wx = vp_x + gx - fm.horizontalAdvance(label) / 2.0
            wy = (H - gy) + fm.ascent() / 2.0 - 1
            painter.setPen(QColor(int(col[0]*255), int(col[1]*255), int(col[2]*255)))
            painter.drawText(int(wx), int(wy), label)
        painter.end()

    def _dollhouse_cut(self, g):
        """True if group g is a near, camera-facing wall to hide in dollhouse mode. Geometric:
        outward direction (centroid - room centre) vs view direction; grazing bias keeps floor/
        ceiling. Shared by rendering AND picking so you can't select a hidden near wall."""
        if not getattr(self, "_dollhouse", False) or g.get("is_bbox"):
            return False
        vdir = getattr(self, "_view_dir", None)
        if vdir is None:
            return False
        cen = g.get("_cen")
        if cen is None:
            tris = np.asarray(g["tris"])
            if len(tris) == 0:
                return False
            cen = np.asarray(self._verts, np.float32)[np.unique(tris.ravel())].mean(axis=0)
            g["_cen"] = cen
        out = cen - np.asarray(self._pivot, np.float32)
        n = float(np.linalg.norm(out))
        if n < 1e-9:
            return False
        return float(np.dot(out / n, vdir)) < -0.2

    def _draw_mesh(self):
        if len(self._verts) == 0:
            return
        verts = np.asarray(self._verts, np.float32)
        # dollhouse: hide wall groups facing the camera (fills+wires) so the outside view matches
        # the inside view. Selected wall is kept so you always see what you're editing.
        def _cut(g):
            return self._dollhouse_cut(g) and g["name"] != self._selected_layer

        # Final visualization rule:
        #   1. OBJ groups keep their own assigned colors. No grouping by wall,
        #      floor, ceiling, normal direction, or shell detection.
        #   2. Surfaces are real opaque, flat, and two-sided, so the ceiling color
        #      is the same from outside and inside.
        #   3. A non-occluding wire overlay is drawn for every visible group. This
        #      is the honest way to inspect a closed room: the solid box remains
        #      physically correct, but interior objects such as a sphere are still
        #      readable through their wire silhouette.
        #   4. The boundary object remains wire-only.
        glDisable(GL_CULL_FACE)
        glDisable(GL_LIGHTING)          # flat, stable layer colors

        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        def draw_triangles(g, alpha, color=None):
            tris = g["tris"]
            if len(tris) == 0:
                return
            r, gr, b = color if color else g["color"]
            glColor4f(r, gr, b, alpha)
            glBegin(GL_TRIANGLES)
            for tri in tris:
                for vi in tri:
                    glVertex3fv(verts[vi])
            glEnd()

        # Semi-transparent faces, NO edges. Walls are see-through so interior
        # layers (sphere, inner panels) are visible from outside. Selected
        # layer is drawn opaque and vivid in its own hue.
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Depth-write off for the whole mesh overlay so nothing hard-occludes
        # anything else and draw order alone controls layering.
        GREY = GREY_FILL
        glDepthMask(GL_FALSE)
        fill_a = 0.22

        # Pass A: FILLS first. Unselected layers get a transparent grey plane;
        # the selected layer gets its own vivid hue at SEL_ALPHA. Drawing every
        # fill BEFORE the wireframe is what keeps the highlighted layer BEHIND
        # the white wireframe instead of painting over it.
        for g in self._groups:
            if g["is_bbox"] or not g["visible"] or _cut(g):
                continue
            if self._selected_layer == g["name"] or g["name"] in self._surf_sel:
                draw_triangles(g, alpha=SEL_ALPHA)            # selected / surface-picked: own colour
            else:
                draw_triangles(g, alpha=fill_a, color=GREY)   # others: grey

        # Pass B: feature-edge wireframe for ALL visible layers (the selected
        # one included), drawn last so the edges always sit on top of every
        # fill. Colour follows the scene scheme (white on dark / black on light).
        glColor4f(*WIRE_COLOR)
        glLineWidth(1.3)
        for g in self._groups:
            if g["is_bbox"] or not g["visible"] or _cut(g):
                continue
            edges = g.get("edges")
            if edges is None or len(edges) == 0:
                continue
            glBegin(GL_LINES)
            for i, j in edges:
                glVertex3fv(verts[i])
                glVertex3fv(verts[j])
            glEnd()
        glDepthMask(GL_TRUE)

        glDisable(GL_BLEND)

        # Pass 4: domain bounding box — explicit hexahedron from the origin
        # (0,0,0) to (Lx, Ly, Lz), drawn as a white wireframe when enabled.
        if self._bbox_visible and not getattr(self, "_dollhouse", False):
            self._draw_domain_box()
            glDisable(GL_BLEND)

        glEnable(GL_LIGHTING)

    _TOP_PITCH = 0.0    # ortho snap: pitch 0 / yaw 0 = look down +Z onto the XY plane (X right, Y up)

    def set_ortho(self, on, top_down=True):
        """Orthographic projection toggle. top_down also snaps the camera to a plan view (look down z)."""
        self._ortho = bool(on)
        if on and top_down:
            self._free_x = self._free_y = self._roll = 0.0
            self._pitch = self._TOP_PITCH; self._yaw = 0.0
        self.update()

    def _layer_color(self, name):
        """Per-layer color, stable within a session. Hues assigned by ORDER of first appearance via the
        golden angle (137.5 deg) so they spread across the WHOLE wheel — reds/oranges/pinks/blues/greens
        — and consecutive layers stay far apart (name-hashing clustered them in blue/green)."""
        import colorsys
        m = getattr(self, "_layer_hue", None)
        if m is None:
            m = self._layer_hue = {}
        if name not in m:
            m[name] = (len(m) * 137.508) % 360.0
        return colorsys.hsv_to_rgb(m[name] / 360.0, 0.55, 0.95)

    def _draw_props(self):
        """Filled + wire pass for placed props (props3d.Prop), colored per layer — ONLY in Draw mode
        (placement). In Simulation the props are the baked scene groups, drawn transparent/grey by
        _draw_mesh with normal layer highlighting; the colorful overlay would fight that."""
        if not self._props or not self._props_mode:
            return
        glDisable(GL_LIGHTING); glDisable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST); glDepthMask(GL_TRUE)
        for i, p in enumerate(self._props):
            verts, tris = p.world()
            sel = (i == self._prop_sel)
            r, g, b = self._layer_color(p.layer)
            glColor4f(r, g, b, 1.0 if sel else 0.9)
            glBegin(GL_TRIANGLES)
            for a, bb, c in tris:
                glVertex3f(*verts[a]); glVertex3f(*verts[bb]); glVertex3f(*verts[c])
            glEnd()
            glColor4f(*WIRE_COLOR); glLineWidth(2.0 if sel else 1.0)
            glBegin(GL_LINES)
            for u, v in p.edges():                # feature edges only -> no triangulation shown
                glVertex3f(*verts[u]); glVertex3f(*verts[v])
            glEnd()
        glEnable(GL_LIGHTING)

    def set_props_mode(self, on):
        """Draw mode: enable P/R/S modal prop transforms (and the caller locks src/rec placement)."""
        self._props_mode = bool(on)
        if not on and self._xform is not None:
            self._xform = None; self.update()

    def _find_nearest_prop(self, qpos, radius=60.0):
        """Index of the prop whose center projects nearest the cursor (within radius px), else -1."""
        if not self._props:
            return -1
        self.makeCurrent()
        W, H = self.width(), self.height(); vp = [0, 0, int(W), int(H)]
        best, bi = radius, -1
        for i, p in enumerate(self._props):
            try:
                sx, sy, _ = gluProject(float(p.pos[0]), float(p.pos[1]), float(p.pos[2]),
                                       self._mv, self._proj, vp)
                d = math.hypot(sx - qpos.x(), (H - sy) - qpos.y())
                if d < best:
                    best, bi = d, i
            except Exception:
                pass
        return bi

    def _snapshot(self, idxs):
        return [{"pos": list(self._props[i].pos), "rot": list(self._props[i].rot),
                 "scale": list(self._props[i].scale)} for i in idxs]

    def _group_pivot(self, idxs):
        """Centroid (world) of the props' centers — the pivot for group rotate/scale. For one prop this
        is its own center, so single-prop R/S stay in place (identical to before)."""
        pts = []
        for i in idxs:
            c = self._props[i].center()
            pts.append([self._props[i].pos[k] + c[k] for k in range(3)])
        n = max(1, len(pts))
        return [sum(p[k] for p in pts) / n for k in range(3)]

    @staticmethod
    def _rot_axis(v, axis, deg):
        """Rotate vector v about world axis X/Y/Z by `deg`."""
        a = math.radians(deg); ca, sa = math.cos(a), math.sin(a); x, y, z = v
        if axis == "X":
            return (x, ca * y - sa * z, sa * y + ca * z)
        if axis == "Y":
            return (ca * x + sa * z, y, -sa * x + ca * z)
        return (ca * x - sa * y, sa * x + ca * y, z)

    def _xform_start(self, mode, idxs=None):
        """Begin a modal transform on a GROUP of props (idxs). Default idxs = the selected prop, else
        the nearest to the cursor. mode 'p'=move (whole group), 'r'/'s' act on the primary prop."""
        if not self._props:
            return False
        cpos = self.mapFromGlobal(QCursor.pos())
        if idxs is None:
            idx = self._prop_sel if 0 <= self._prop_sel < len(self._props) else self._find_nearest_prop(cpos)
            if idx < 0:
                return False
            idxs = [idx]; self._prop_sel = idx
        self._xform = {"mode": mode, "axis": None, "idxs": list(idxs), "anchor": cpos,
                       "bases": self._snapshot(idxs), "pivot": self._group_pivot(idxs)}
        self.update()
        return True

    def _scene_tris_ABC(self, exclude=()):
        """Triangle vertex arrays (A,B,C) of the room mesh + every OTHER prop — the surfaces a prop can
        rest on. None if the scene is empty."""
        A, B, C = [], [], []
        v = np.asarray(self._verts, float)
        room = [np.asarray(g["tris"]) for g in self._groups
                if not g.get("is_bbox") and len(g["tris"]) and "ceiling" not in g["name"].lower()]
        if len(v) and room:
            rt = np.concatenate(room); A.append(v[rt[:, 0]]); B.append(v[rt[:, 1]]); C.append(v[rt[:, 2]])
        for j, p in enumerate(self._props):
            if j in exclude:
                continue
            pv, pt = p.world(); pv = np.asarray(pv, float); pt = np.asarray(pt)
            if len(pt):
                A.append(pv[pt[:, 0]]); B.append(pv[pt[:, 1]]); C.append(pv[pt[:, 2]])
        if not A:
            return None
        return np.concatenate(A), np.concatenate(B), np.concatenate(C)

    @staticmethod
    def _top_z_at(x, y, A, B, C):
        """Highest surface z under the vertical line (x,y) among triangles ABC (ceilings already excluded
        by _scene_tris_ABC, vertical walls skip via zero xy-area). None if the column hits nothing."""
        a2, v0, v1 = A[:, :2], B[:, :2] - A[:, :2], C[:, :2] - A[:, :2]
        px, py = x - a2[:, 0], y - a2[:, 1]
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        ok = np.abs(den) > 1e-12
        ds = np.where(ok, den, 1.0)
        u = (px * v1[:, 1] - v1[:, 0] * py) / ds
        w = (v0[:, 0] * py - px * v0[:, 1]) / ds
        inside = ok & (u >= -1e-9) & (w >= -1e-9) & (u + w <= 1 + 1e-9)
        z = A[:, 2] + u * (B[:, 2] - A[:, 2]) + w * (C[:, 2] - A[:, 2])
        z = np.where(inside, z, -np.inf)
        m = float(z.max()) if z.size else -np.inf
        return None if m == -np.inf else m

    def _drop_prop(self, i, exclude=None):
        """Rest prop i's bottom on the highest surface (stage/floor/other prop) directly beneath its
        footprint; fall back to the domain floor. Sampled at the footprint centre + corners."""
        if not (0 <= i < len(self._props)):
            return
        p = self._props[i]
        v = np.asarray(p.world()[0], float)
        amn, amx = v.min(0), v.max(0)
        top = 0.0                                              # fallback: rest base at z=0 (room floor)
        ABC = self._scene_tris_ABC(exclude if exclude is not None else {i})
        if ABC is not None:
            A, B, C = ABC
            ex = amn[0] + (amx[0] - amn[0]) * 0.02; ex1 = amx[0] - (amx[0] - amn[0]) * 0.02
            ey = amn[1] + (amx[1] - amn[1]) * 0.02; ey1 = amx[1] - (amx[1] - amn[1]) * 0.02
            samples = [((amn[0] + amx[0]) / 2, (amn[1] + amx[1]) / 2),
                       (ex, ey), (ex1, ey), (ex1, ey1), (ex, ey1)]
            hits = [self._top_z_at(x, y, A, B, C) for x, y in samples]
            hits = [h for h in hits if h is not None]
            if hits:
                top = max(hits)
        p.pos = [p.pos[0], p.pos[1], p.pos[2] + (top - amn[2])]

    def _xform_reanchor(self):
        """Re-snapshot the base transforms + mouse anchor to the CURRENT state, so toggling an axis
        lock mid-drag continues from what the user sees (no jump)."""
        xf = self._xform
        if xf is None:
            return
        xf["bases"] = self._snapshot(xf["idxs"])
        xf["anchor"] = self.mapFromGlobal(QCursor.pos())
        xf["pivot"] = self._group_pivot(xf["idxs"])

    def _xform_end(self, cancel):
        xf = self._xform
        if xf is None:
            return
        if cancel:                                                 # restore every prop's snapshot
            for i, b in zip(xf["idxs"], xf["bases"]):
                if 0 <= i < len(self._props):
                    p = self._props[i]
                    p.pos, p.rot, p.scale = list(b["pos"]), list(b["rot"]), list(b["scale"])
        else:                                                      # settle each prop onto the surface below
            ex = set(xf["idxs"])
            for i in xf["idxs"]:
                self._drop_prop(i, exclude=ex)
        self._xform = None
        self.propsChanged.emit(); self.update()

    def _xform_apply(self, qpos):
        """Drive the transform from the mouse (no button). P moves the WHOLE group by one delta (screen-
        plane or along the locked axis); R/S act on the primary prop (about its center)."""
        xf = self._xform
        if xf is None:
            return
        dx = qpos.x() - xf["anchor"].x(); dy = -(qpos.y() - xf["anchor"].y())
        ai = {"X": 0, "Y": 1, "Z": 2}
        if xf["mode"] == "p":
            self.makeCurrent()
            W, H = self.width(), self.height(); vp = [0, 0, int(W), int(H)]
            base0 = np.array(xf["bases"][0]["pos"], float)         # primary drives the shared delta
            delta = np.zeros(3)
            if xf["axis"] is None:                                 # free: screen-plane at the primary depth
                try:
                    _, _, sz = gluProject(*base0, self._mv, self._proj, vp)
                    w = gluUnProject(float(qpos.x()), float(H) - float(qpos.y()), float(sz),
                                     self._mv, self._proj, vp)
                    delta = np.array(w, float) - base0
                except Exception:
                    pass
            else:                                                  # along the locked world axis
                axv = np.array({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[xf["axis"]], float)
                try:
                    s0 = np.array(gluProject(*base0, self._mv, self._proj, vp)[:2])
                    s1 = np.array(gluProject(*(base0 + axv * 0.1), self._mv, self._proj, vp)[:2])
                    sdir = s1 - s0; L = np.linalg.norm(sdir)
                    if L > 1e-6:
                        delta = axv * (float(np.dot([dx, dy], sdir / L)) * self._dist * 0.0015)
                except Exception:
                    pass
            for i, b in zip(xf["idxs"], xf["bases"]):
                bp = b["pos"]; self._props[i].pos = [bp[0] + delta[0], bp[1] + delta[1], bp[2] + delta[2]]
            if xf["axis"] != "Z":                    # ride the surface below LIVE (so you see where it lands);
                ex = set(xf["idxs"])                 # skip when Z-locked so you can set height by hand
                for i in xf["idxs"]:
                    self._drop_prop(i, exclude=ex)
        elif xf["mode"] == "r":
            axis = xf["axis"] or "Z"; k = ai[axis]; ang = dx * 0.5; piv = xf["pivot"]  # 0.5 deg/px
            for i, b in zip(xf["idxs"], xf["bases"]):
                p = self._props[i]; cm = p.center()
                off = [b["pos"][m] + cm[m] - piv[m] for m in range(3)]     # base center - pivot
                ro = self._rot_axis(off, axis, ang)                       # orbit about the group pivot
                p.pos = [piv[m] + ro[m] - cm[m] for m in range(3)]
                p.rot = list(b["rot"]); p.rot[k] = b["rot"][k] + ang      # + spin each in place
        elif xf["mode"] == "s":
            f = max(0.01, 1.0 + dx * 0.01); piv = xf["pivot"]
            for i, b in zip(xf["idxs"], xf["bases"]):
                p = self._props[i]; cm = p.center()
                off = [b["pos"][m] + cm[m] - piv[m] for m in range(3)]
                if xf["axis"] is None:
                    noff = [o * f for o in off]; ns = [c * f for c in b["scale"]]
                else:
                    k = ai[xf["axis"]]; noff = list(off); noff[k] = off[k] * f
                    ns = list(b["scale"]); ns[k] = b["scale"][k] * f
                p.pos = [piv[m] + noff[m] - cm[m] for m in range(3)]      # move from pivot + scale each
                p.scale = ns
        self.update()

    def _find_props_in_rect(self, rect):
        """Indices of props whose center projects inside the screen rectangle (two QPoints)."""
        self.makeCurrent()
        W, H = self.width(), self.height(); vp = [0, 0, int(W), int(H)]
        x0, x1 = sorted((rect[0].x(), rect[1].x())); y0, y1 = sorted((rect[0].y(), rect[1].y()))
        out = []
        for i, p in enumerate(self._props):
            c = p.center(); wc = (p.pos[0] + c[0], p.pos[1] + c[1], p.pos[2] + c[2])
            try:
                sx, sy, _ = gluProject(float(wc[0]), float(wc[1]), float(wc[2]), self._mv, self._proj, vp)
                sy = H - sy
                if x0 <= sx <= x1 and y0 <= sy <= y1:
                    out.append(i)
            except Exception:
                pass
        return out

    def _copy_start(self, idxs):
        """Duplicate props idxs, then begin a group MOVE of the copies (drag to place, X/Y/Z lock)."""
        import props3d
        self.push_props_undo()                                     # so Ctrl+Z removes the copies
        new = []
        for i in idxs:
            if 0 <= i < len(self._props):
                self._props.append(props3d.Prop.from_dict(self._props[i].to_dict()))
                new.append(len(self._props) - 1)
        if not new:
            return
        self._prop_sel = new[-1]
        self._xform_start("p", idxs=new)
        self.propsChanged.emit()

    def push_props_undo(self):
        """Snapshot props BEFORE a mutating op (move/rotate/scale/copy/add/delete/assign) for Ctrl+Z."""
        snap = [p.to_dict() for p in self._props]
        if self._props_undo_stack and self._props_undo_stack[-1] == snap:
            return
        self._props_undo_stack.append(snap)
        self._props_undo_stack = self._props_undo_stack[-50:]

    def props_undo(self):
        """Restore the last pre-op prop snapshot. Returns True if something was undone."""
        if not self._props_undo_stack:
            return False
        import props3d
        self._props = [props3d.Prop.from_dict(d) for d in self._props_undo_stack.pop()]
        self._prop_sel = -1; self._xform = None
        self.propsChanged.emit(); self.update()
        return True

    def _view_ground_xy(self):
        """(x,y) where the centre of the view meets the z=0 ground plane — so a new prop lands where the
        user is looking. Falls back to the orbit pivot when the view is near-horizontal."""
        try:
            self.makeCurrent()
            W, H = self.width(), self.height()
            ro, rd = ray_from_screen(W / 2.0, H / 2.0, W, H, self._mv, self._proj)
            ro = np.asarray(ro, float); rd = np.asarray(rd, float)
            if abs(rd[2]) > 1e-6:
                p = ro + rd * (-ro[2] / rd[2])
                if np.all(np.isfinite(p)):
                    return float(p[0]), float(p[1])
        except Exception:
            pass
        return float(self._pivot[0]), float(self._pivot[1])

    def add_prop(self, prop):
        cx, cy = self._view_ground_xy()
        prop.pos = [cx, cy, 0.0]                          # rough; then centre the footprint on (cx,cy)
        v = np.asarray(prop.world()[0], float); mn, mx = v.min(0), v.max(0)
        prop.pos[0] += cx - (mn[0] + mx[0]) / 2.0
        prop.pos[1] += cy - (mn[1] + mx[1]) / 2.0
        self._props.append(prop); self._prop_sel = len(self._props) - 1
        self._drop_prop(self._prop_sel)                  # base rests on stage/floor there (fallback z=0)
        self.propsChanged.emit(); self.update()          # refresh panel + 2D footprints live

    def remove_prop(self, i):
        if 0 <= i < len(self._props):
            del self._props[i]; self._prop_sel = min(self._prop_sel, len(self._props) - 1)
            self.propsChanged.emit(); self.update()

    def select_prop(self, i):
        self._prop_sel = i if 0 <= i < len(self._props) else -1; self.update()

    def _draw_domain_box(self):
        """
        White wireframe hexahedron of the simulation domain, spanning from the
        origin (0,0,0) to (Lx, Ly, Lz). Lx/Ly/Lz are the scene extents taken
        from the loaded geometry's bounding box (domain_max). Non-occluding.
        """
        Lx, Ly, Lz = (float(self._domain_max[0]),
                      float(self._domain_max[1]),
                      float(self._domain_max[2]))
        # eight corners of the box (0..L on each axis)
        c = [
            (0,  0,  0),  (Lx, 0,  0),  (Lx, Ly, 0),  (0,  Ly, 0),   # bottom z=0
            (0,  0,  Lz), (Lx, 0,  Lz), (Lx, Ly, Lz), (0,  Ly, Lz),  # top z=Lz
        ]
        edges = [
            (0,1),(1,2),(2,3),(3,0),   # bottom square
            (4,5),(5,6),(6,7),(7,4),   # top square
            (0,4),(1,5),(2,6),(3,7),   # vertical pillars
        ]
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)   # always visible, never occluded
        glDepthMask(GL_FALSE)
        glLineWidth(1.6)
        glColor4f(*WIRE_COLOR)          # wireframe (scheme-dependent)
        glBegin(GL_LINES)
        for a, b in edges:
            glVertex3f(*c[a])
            glVertex3f(*c[b])
        glEnd()
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def _draw_axes(self):
        """Thin grey positive x/y/z axes as GL lines from the origin, length scaled by the
        camera distance so they stay very long and keep expanding as you zoom out. The
        parser keeps the scene on the positive side, so axes are positive-only."""
        if not self._axes_visible:
            return
        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE)
        glLineWidth(1.0)                 # thinner than the wireframe
        glColor4f(0.5, 0.5, 0.5, 1.0)    # mid-grey, scheme-independent
        BIG = self._dist * 1e4           # grows with zoom-out -> effectively "very long"
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0); glVertex3f(BIG, 0, 0)
        glVertex3f(0, 0, 0); glVertex3f(0, BIG, 0)
        glVertex3f(0, 0, 0); glVertex3f(0, 0, BIG)
        glEnd()
        glDepthMask(GL_TRUE); glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)

    def _draw_slices(self):
        """Translucent axis-aligned slice planes, shown ONLY while in slice mode. Holding
        an axis shows just that plane (the one being moved); with no axis held all three
        show faintly as a reference. Each spans the scene AABB, tinted by axis colour."""
        if self._slices is None or self._mode != self.MODE_SLICE:
            return
        mn, mx = self._domain_min, self._domain_max
        x0, y0, z0 = float(self._slices[0]), float(self._slices[1]), float(self._slices[2])
        quads = {
            'X': [(x0, mn[1], mn[2]), (x0, mx[1], mn[2]), (x0, mx[1], mx[2]), (x0, mn[1], mx[2])],
            'Y': [(mn[0], y0, mn[2]), (mx[0], y0, mn[2]), (mx[0], y0, mx[2]), (mn[0], y0, mx[2])],
            'Z': [(mn[0], mn[1], z0), (mx[0], mn[1], z0), (mx[0], mx[1], z0), (mn[0], mx[1], z0)],
        }
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)             # translucent: don't occlude the mesh behind
        for ax, q in quads.items():
            if not self._plane_vis[ax.lower()]:   # quads/AXIS_COLORS keyed 'X'/'Y'/'Z'; _plane_vis 'x'/'y'/'z'
                continue                  # plane toggled off in the Scene tab
            if self._axis_lock and self._axis_lock != ax:
                continue                  # editing one axis -> only that plane
            alpha = 0.50 if self._axis_lock == ax else 0.30   # brighter/opaquer: readable in Light mode
            c = AXIS_COLORS[ax]
            # nudge toward white so the tint reads on a light background too
            glColor4f(min(1.0, c[0]+0.15), min(1.0, c[1]+0.15), min(1.0, c[2]+0.15), alpha)
            glBegin(GL_QUADS)
            for v in q:
                glVertex3f(float(v[0]), float(v[1]), float(v[2]))
            glEnd()
        glDepthMask(GL_TRUE); glDisable(GL_BLEND); glEnable(GL_LIGHTING)

    def _outside_domain(self, x, y, z):
        """Return True if (x,y,z) lies outside the scene bounding box."""
        p = np.array([x, y, z], np.float32)
        return bool(np.any(p < self._domain_min) or np.any(p > self._domain_max))

    def _point_glyph(self, x, y, z, color, kind, hl=False):
        """
        Queue a source/receiver to be drawn as its real Unicode glyph at the
        EXACT world coordinate (no offset icon beside it):
          • source   -> U+25C9 ◉
          • receiver -> U+25CD ◍
        The glyph is rendered with QPainter as a 2D overlay after the GL scene
        (see _paint_point_glyphs), so it uses the true font character, stays a
        constant screen size and always faces the camera. Colour is preserved
        (red when outside the domain, highlight tint when selected/hovered).
        """
        W, H = self.width(), self.height()
        viewport = [int(0), int(0), int(W), int(H)]
        try:
            sx, sy, sz = gluProject(float(x), float(y), float(z),
                                    self._mv, self._proj, viewport)
        except Exception:
            return
        if sz < 0.0 or sz > 1.0:      # behind the camera / clipped
            return
        # GL y is bottom-up; QPainter y is top-down
        self._glyph_queue.append((float(sx), float(H - sy), tuple(color),
                                  kind, bool(hl)))

    def _paint_origin(self, painter):
        """Origin marker at world (0,0,0): three positive-axis lines, each 0.1 of the
        domain length on its axis (world size, scales with zoom), thicker than the
        wireframe, white in Dark / black in Light, with an 'o' at the origin."""
        if not self._axes_visible:
            return
        W, H = self.width(), self.height()
        vp = [0, 0, int(W), int(H)]
        try:
            x0, y0, z0 = gluProject(0.0, 0.0, 0.0, self._mv, self._proj, vp)
        except Exception:
            return
        if z0 < 0.0 or z0 > 1.0:      # origin behind camera / clipped
            return
        col = QColor(0, 0, 0) if SCENE_LIGHT_MODE else QColor(255, 255, 255)
        pen = QPen(col); pen.setWidthF(5.0); pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        p0 = QPointF(x0, H - y0)
        g = 0.1   # fixed gizmo length [m] on all 3 axes
        for end in ((g, 0, 0), (0, g, 0), (0, 0, g)):
            try:
                sx, sy, sz = gluProject(end[0], end[1], end[2], self._mv, self._proj, vp)
            except Exception:
                continue
            if sz < 0.0 or sz > 1.0:
                continue
            painter.drawLine(p0, QPointF(sx, H - sy))
        painter.setFont(QFont("Sans", 11))
        painter.drawText(QPointF(x0 + 6, H - y0 - 6), "o")

    def _paint_point_glyphs(self, painter):
        """Render all queued source/receiver glyphs with QPainter."""
        if not self._glyph_queue:
            return
        from PySide6.QtGui import QFont, QColor
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        for sx, sy, col, kind, hl in self._glyph_queue:
            ch   = "\u25C9" if kind == "source" else "\u25CD"
            size = 20 if hl else 16
            f = QFont("Noto Sans Symbols2", size)
            painter.setFont(f)
            painter.setPen(QColor(int(col[0]*255), int(col[1]*255), int(col[2]*255)))
            fm = painter.fontMetrics()
            # centre the glyph on the projected point
            wx = sx - fm.horizontalAdvance(ch) / 2.0
            wy = sy + (fm.ascent() - fm.descent()) / 2.0
            painter.drawText(int(round(wx)), int(round(wy)), ch)

    RAY_COLOR    = (0.55, 0.45, 0.95)   # purplish blue (all rays)
    RAY_ESCAPE   = (0.42, 0.38, 0.55)   # dimmer, for escaped (leaked) stubs
    RAY_SELECTED = (0.10, 1.00, 0.70)   # cyan-green (the ray picked in the table)

    def load_ray_paths(self, path):
        """Load ray_paths.txt (segment/line: src_id ray_id x0 y0 z0 x1 y1 z1 order) from
        the solver's --dump-rays pass. World coords, so it overlays the raw OBJ. Groups
        segments by ray_id (unique across sources); records each ray's source index."""
        rays = {}; srcs = {}
        try:
            with open(path) as f:
                for ln in f:
                    p = ln.split()
                    if len(p) < 9:
                        continue
                    sid = int(float(p[0])); rid = int(float(p[1]))
                    x0,y0,z0,x1,y1,z1 = (float(v) for v in p[2:8])
                    rays.setdefault(rid, []).append(((x0,y0,z0), (x1,y1,z1), int(float(p[8]))))
                    srcs[rid] = sid
        except OSError:
            return {}
        self._ray_paths = rays
        self._ray_src = srcs
        self._selected_ray = None
        self._show_rays = True
        self.update()
        return rays

    def clear_ray_paths(self):
        self._ray_paths = {}
        self._ray_src = {}
        self._selected_ray = None
        self.update()

    def set_selected_ray(self, rid):
        self._selected_ray = rid
        self.update()

    def _draw_rays(self):
        if not self._show_rays or not self._ray_paths:
            return
        glDisable(GL_LIGHTING)
        # all non-selected rays: purplish blue (escaped stubs dimmer)
        glLineWidth(1.4)
        glBegin(GL_LINES)
        for rid, segs in self._ray_paths.items():
            if rid == self._selected_ray:
                continue
            for p0, p1, order in segs:
                glColor4f(*(self.RAY_ESCAPE if order < 0 else self.RAY_COLOR),
                          0.35 if order < 0 else 0.7)
                glVertex3f(*p0); glVertex3f(*p1)
        glEnd()
        # selected ray: cyan-green, thicker, drawn on top (depth test off)
        sel = self._ray_paths.get(self._selected_ray)
        if sel:
            glLineWidth(2.6)
            glDisable(GL_DEPTH_TEST)
            glColor4f(*self.RAY_SELECTED, 1.0)
            glBegin(GL_LINES)
            for p0, p1, order in sel:
                glVertex3f(*p0); glVertex3f(*p1)
            glEnd()
            glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def _draw_sources(self):
        for i, pt in enumerate(self._sources):
            sel = ("source", i) in self._pt_sel
            hl  = (i == self._hl_src) or sel
            if sel:
                col = HL_COLOR
            elif self._outside_domain(pt[0], pt[1], pt[2]):
                col = OUTSIDE_COLOR
            else:
                col = HL_COLOR if hl else SRC_COLOR
            self._point_glyph(pt[0], pt[1], pt[2], col, "source", hl=hl)

    def _draw_receivers(self):
        for i, pt in enumerate(self._receivers):
            sel = ("receiver", i) in self._pt_sel
            hl  = (i == self._hl_rec) or (i in self._sel_recs) or sel
            if sel:
                col = HL_COLOR
            elif self._outside_domain(pt[0], pt[1], pt[2]):
                col = OUTSIDE_COLOR
            else:
                col = HL_COLOR if hl else REC_COLOR
            self._point_glyph(pt[0], pt[1], pt[2], col, "receiver", hl=hl)

    def _draw_placement_cursor(self):
        """Axis arrow + crosshair at _cursor_world."""
        p = self._cursor_world
        glDisable(GL_LIGHTING)

        col = (1.0, 1.0, 1.0) if self._axis_lock is None \
              else AXIS_COLORS[self._axis_lock]

        glColor4f(*col, 1.0)
        glLineWidth(1.5)
        d = self._dist * 0.04

        # crosshair
        glBegin(GL_LINES)
        glVertex3f(p[0]-d, p[1], p[2]); glVertex3f(p[0]+d, p[1], p[2])
        glVertex3f(p[0], p[1]-d, p[2]); glVertex3f(p[0], p[1]+d, p[2])
        glVertex3f(p[0], p[1], p[2]-d); glVertex3f(p[0], p[1], p[2]+d)
        glEnd()

        # axis ray
        if self._axis_lock:
            ext = self._dist * 1.5
            ax = {'X':(1,0,0),'Y':(0,1,0),'Z':(0,0,1)}[self._axis_lock]
            glLineWidth(2.0)
            glColor4f(*col, 1.0)
            glBegin(GL_LINES)
            glVertex3f(p[0]-ax[0]*ext, p[1]-ax[1]*ext, p[2]-ax[2]*ext)
            glVertex3f(p[0]+ax[0]*ext, p[1]+ax[1]*ext, p[2]+ax[2]*ext)
            glEnd()

        # preview: draw the SAME glyph as a placed point, at the exact centre -- ONLY for
        # source/receiver placement. Slice mode shows just the crosshair (no symbol).
        if self._mode in (self.MODE_SRC, self.MODE_REC):
            col_override = OUTSIDE_COLOR \
                           if self._outside_domain(p[0], p[1], p[2]) else None
            icon_col = col_override if col_override else \
                       (SRC_COLOR if self._mode == self.MODE_SRC else REC_COLOR)
            kind = "source" if self._mode == self.MODE_SRC else "receiver"
            self._point_glyph(p[0], p[1], p[2], icon_col, kind)

        glEnable(GL_LIGHTING)

    def _draw_drag_guide(self):
        """
        While moving an existing point with E + LMB, draw the same crosshair and
        (when an axis is locked) the axis guide line, using the shared axis
        colours. Mirrors the placement cursor so the two interactions feel alike.
        """
        lst = self._sources if self._drag_kind == "source" else self._receivers
        if not (0 <= self._drag_idx < len(lst)):
            return
        p = lst[self._drag_idx]
        glDisable(GL_LIGHTING)

        col = (1.0, 1.0, 1.0) if self._drag_axis is None \
              else AXIS_COLORS[self._drag_axis]
        glColor4f(col[0], col[1], col[2], 1.0)
        glLineWidth(1.5)
        d = self._dist * 0.04

        # crosshair at the point
        glBegin(GL_LINES)
        glVertex3f(p[0]-d, p[1], p[2]); glVertex3f(p[0]+d, p[1], p[2])
        glVertex3f(p[0], p[1]-d, p[2]); glVertex3f(p[0], p[1]+d, p[2])
        glVertex3f(p[0], p[1], p[2]-d); glVertex3f(p[0], p[1], p[2]+d)
        glEnd()

        # axis guide line when a drag axis is locked
        if self._drag_axis:
            ext = self._dist * 1.5
            ax = {'X':(1,0,0), 'Y':(0,1,0), 'Z':(0,0,1)}[self._drag_axis]
            glLineWidth(2.0)
            glBegin(GL_LINES)
            glVertex3f(p[0]-ax[0]*ext, p[1]-ax[1]*ext, p[2]-ax[2]*ext)
            glVertex3f(p[0]+ax[0]*ext, p[1]+ax[1]*ext, p[2]+ax[2]*ext)
            glEnd()

        glEnable(GL_LIGHTING)

    def _unproject_hover(self, qpos):
        """Unproject the mouse pixel onto the pivot-depth plane for a HUD readout."""
        self.makeCurrent()
        W, H = self.width(), self.height()
        viewport = [int(0), int(0), int(W), int(H)]
        try:
            sx, sy, sz = gluProject(float(self._pivot[0]), float(self._pivot[1]),
                                    float(self._pivot[2]), self._mv, self._proj, viewport)
            world = gluUnProject(float(qpos.x()), float(H) - float(qpos.y()), float(sz),
                                 self._mv, self._proj, viewport)
            return np.array(world, np.float32)
        except Exception:
            return None

    def _draw_hud(self):
        """Overlay text using QPainter on top of GL."""
        if self._mode == self.MODE_NONE:
            # Modal prop transform: same top banner as a point drag (mode, locked axis, center coords).
            if self._xform is not None and self._xform["idxs"] and 0 <= self._xform["idxs"][0] < len(self._props):
                cw = self._prop_center_world(self._xform)
                mode_str = {"p": "MOVE", "r": "ROTATE", "s": "SCALE"}[self._xform["mode"]] + " PROP"
                axis_str = (f"  locked: {self._xform['axis']}" if self._xform["axis"]
                            else "  (hold X / Y / Z to lock an axis)")
                ijk = self._ijk_str(cw)
                coord_str = (f"  center x={cw[0]:.3f}  y={cw[1]:.3f}  z={cw[2]:.3f} m"
                             + (f"   [{ijk}]" if ijk else ""))
                painter = QPainter(self)
                painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
                from PySide6.QtCore import QRect
                painter.fillRect(QRect(0, 0, self.width(), 28), QColor(20, 25, 35, 210))
                painter.setFont(QFont("Monospace", 10, QFont.Weight.Bold))
                painter.setPen(QColor(PAL["accent"]))
                painter.drawText(8, 18, f"● {mode_str}")
                painter.setFont(QFont("Monospace", 10)); painter.setPen(QColor(PAL["text"]))
                painter.drawText(150, 18, axis_str + coord_str)
                painter.end()
                return
            # While moving an existing point with G, show a top banner with the
            # live coordinates and the locked axis, mirroring placement.
            if self._drag_kind is not None and self._drag_idx >= 0:
                lst = self._sources if self._drag_kind == "source" else self._receivers
                if 0 <= self._drag_idx < len(lst):
                    p = lst[self._drag_idx]
                    mode_str = "MOVE " + ("SOURCE" if self._drag_kind == "source"
                                          else "RECEIVER")
                    axis_str = (f"  locked: {self._drag_axis}" if self._drag_axis
                                else "  (hold X / Y / Z to lock an axis)")
                    ijk = self._ijk_str(p)
                    coord_str = (f"  x={p[0]:.3f}  y={p[1]:.3f}  z={p[2]:.3f} m"
                                 + (f"   [{ijk}]" if ijk else ""))
                    painter = QPainter(self)
                    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
                    from PySide6.QtCore import QRect
                    painter.fillRect(QRect(0, 0, self.width(), 28),
                                     QColor(20, 25, 35, 210))
                    painter.setFont(QFont("Monospace", 10, QFont.Weight.Bold))
                    painter.setPen(QColor(PAL["accent"]))
                    painter.drawText(8, 18, f"● {mode_str}")
                    painter.setFont(QFont("Monospace", 10))
                    painter.setPen(QColor(PAL["text"]))
                    painter.drawText(140, 18, axis_str + coord_str)
                    painter.end()
                return
            # otherwise: show hover coordinates in the bottom-left corner
            if self._hover_world is not None:
                p = self._hover_world
                painter = QPainter(self)
                painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
                painter.setFont(QFont("Monospace", 9))
                # this text has no banner behind it, so flip it with the scheme
                painter.setPen(QColor(20, 20, 20) if SCENE_LIGHT_MODE
                               else QColor(PAL["subtext"]))
                x0 = 10 + self._hud_left       # clear the tab panel when it is open
                ijk = self._ijk_str(p)
                if ijk:
                    painter.drawText(x0, self.height() - 24, ijk)      # grid indices, stacked above
                txt = f"x={p[0]:.3f}  y={p[1]:.3f}  z={p[2]:.3f} m"
                painter.drawText(x0, self.height() - 10, txt)
                painter.end()
            return
        p = self._cursor_world
        mode_str = ("SLICE"    if self._mode == self.MODE_SLICE else
                    "SOURCE"   if self._mode == self.MODE_SRC   else "RECEIVER")
        mode_col = ((1.0, 1.0, 1.0) if self._mode == self.MODE_SLICE else
                    SRC_COLOR if self._mode == self.MODE_SRC else REC_COLOR)
        axis_str = f"  locked: {self._axis_lock}" if self._axis_lock else "  (hold X / Y / Z + drag LMB to lock an axis)"
        ijk = self._ijk_str(p)
        coord_str = (f"  x={p[0]:.3f}  y={p[1]:.3f}  z={p[2]:.3f} m"
                     + (f"   [{ijk}]" if ijk else ""))

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # semi-transparent banner at top
        from PySide6.QtGui import QBrush
        from PySide6.QtCore import QRect
        banner_h = 28
        painter.fillRect(QRect(0, 0, self.width(), banner_h),
                         QColor(20, 25, 35, 210))

        mc = QColor(int(mode_col[0]*255), int(mode_col[1]*255), int(mode_col[2]*255))
        painter.setFont(QFont("Monospace", 10, QFont.Weight.Bold))
        painter.setPen(mc)
        painter.drawText(8, 18, f"● {mode_str}")

        painter.setFont(QFont("Monospace", 10))
        painter.setPen(QColor(PAL["text"]))
        painter.drawText(110, 18, axis_str + coord_str)

        # coord label near cursor (project 3-D point to screen)
        try:
            viewport = [int(0), int(0), int(self.width()), int(self.height())]
            sx, sy, sz = gluProject(float(p[0]), float(p[1]), float(p[2]),
                                    self._mv, self._proj, viewport)
            sy_qt = self.height() - int(sy)
            sx_i  = int(sx)
            if 10 < sx_i < self.width()-10 and 30 < sy_qt < self.height()-10:
                painter.setFont(QFont("Monospace", 9))
                painter.setPen(mc)
                label = f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
                painter.drawText(sx_i + 14, sy_qt - 10, label)
        except Exception:
            pass

        painter.end()

    def _mode_label(self):
        """Short label for the current interaction mode -> the persistent corner chip."""
        if self._fpv is not None:            return "RECEIVER VIEW"
        if self._gxf is not None:            return "GROUP ROTATE" if self._gxf["mode"] == "r" else "GROUP MOVE"
        if self._xform is not None:          return {"p": "PROP MOVE", "r": "PROP ROTATE", "s": "PROP SCALE"}[self._xform["mode"]]
        if self._mode == self.MODE_SRC:      return "PLACE SOURCE"
        if self._mode == self.MODE_REC:      return "PLACE RECEIVER"
        if self._mode == self.MODE_SLICE:    return "SLICE"
        if self._surf_pick:                  return "SURFACE SELECT"
        if self._nav_mode:                   return "NAVIGATE"
        if self._grab_held:                  return "EDIT POINTS"
        if self._props_mode:                 return "DRAW / PROPS"
        return "VIEW"

    _HELP = [
        ("VIEW / NAVIGATE", ["LMB drag orbit · MMB pan · wheel zoom",
                             "Space  fly nav (W/X fwd·back, A/D strafe, Q/Z height, mouse looks)",
                             "T  reset view · RMB  scene menu"]),
        ("POINTS (Simulation)", ["S / R  place source / receiver  (+ G  make a grid/shell array)",
                                 "E  edit: drag a point · box-select many · P move · R rotate · Del delete",
                                 "hold X / Y / Z  lock an axis"]),
        ("PROPS (Draw mode)", ["P / R / S  move / rotate / scale · X/Y/Z lock",
                               "C  copy-select · props auto-rest on the surface below (stage/floor)",
                               "Ctrl+Z  undo"]),
        ("SURFACES (RMB menu)", ["rectangle-select → Group / Explode / Cleanse layers"]),
    ]

    def _paint_help(self, p):
        """Semi-transparent shortcuts panel, toggled with '?'."""
        from PySide6.QtCore import QRect
        W, H = self.width(), self.height()
        bw, bh = min(560, W - 40), min(320, H - 40)
        x = (W - bw) // 2; y = (H - bh) // 2
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor(PAL["accent"])); p.setBrush(QColor(15, 18, 24, 236))
        p.drawRoundedRect(x, y, bw, bh, 8, 8)
        cy = y + 24
        p.setPen(QColor(PAL["accent2"])); p.setFont(QFont("Monospace", 11, QFont.Weight.Bold))
        p.drawText(QRect(x, cy - 16, bw, 20), Qt.AlignmentFlag.AlignHCenter, "Keyboard & mouse — press ? to close")
        cy += 14
        for title, rows in self._HELP:
            cy += 14
            p.setPen(QColor(PAL["accent"])); p.setFont(QFont("Monospace", 9, QFont.Weight.Bold))
            p.drawText(x + 16, cy, title); cy += 4
            p.setPen(QColor(PAL["text"])); p.setFont(QFont("Monospace", 8))
            for r in rows:
                cy += 13; p.drawText(x + 26, cy, r)

    # ── mouse / keyboard ────────────────────────────────────
    def leaveEvent(self, e):
        self._hover_world = None
        self.update()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        self._last_pos  = e.position()
        self._mouse_btn = e.button()
        self._drag_moved = False
        # Point group transform in progress: consume the press; the release commits (LMB) / cancels (RMB)
        # so it can't fall through to place a spurious point in S/R mode.
        if self._gxf is not None:
            return
        # A modal prop transform in progress: LMB confirms, RMB cancels; consume so the camera stays put.
        if self._xform is not None:
            if e.button() == Qt.MouseButton.LeftButton:
                self._xform_end(cancel=False); return
            if e.button() == Qt.MouseButton.RightButton:
                self._xform_end(cancel=True); return
        # Surface rect-select armed (RMB menu): LMB starts the rubber-band box; consume so it doesn't orbit.
        if self._surf_pick and e.button() == Qt.MouseButton.LeftButton:
            self._rect = [e.position(), e.position()]; return
        # Copy-select armed (C): LMB starts the rubber-band box; consume so it doesn't orbit.
        if self._copy_arm and e.button() == Qt.MouseButton.LeftButton:
            self._rect = [e.position(), e.position()]; self._copy_arm = False; return
        # Edit mode (E) + LMB: grab the nearest point to drag it, else start a rubber-band box-select
        # of multiple points (release fills _pt_sel; then G/R transform, Delete removes).
        if (self._fpv is None and self._grab_held and e.button() == Qt.MouseButton.LeftButton
                and self._mode == self.MODE_NONE):
            self._begin_point_drag(e.position())
            if self._drag_kind is None:
                self._rect = [e.position(), e.position()]; self._pt_box = True
            return

    def mouseMoveEvent(self, e):
        if self._fpv is not None:      # receiver view: LMB looks, Ctrl+LMB rolls, position frozen
            if self._mouse_btn == Qt.MouseButton.LeftButton and self._last_pos is not None:
                dx = e.position().x() - self._last_pos.x()
                dy = e.position().y() - self._last_pos.y()
                if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self._fpv["roll"] = max(-180.0, min(180.0, self._fpv["roll"] + dx * 0.3))
                else:
                    self._fpv["az"] = (self._fpv["az"] - dx * 0.3) % 360.0      # drag right -> look right
                    self._fpv["el"] = max(-89.0, min(89.0, self._fpv["el"] - dy * 0.3))
                self.fpvPoseChanged.emit(self._fpv["az"], self._fpv["el"], self._fpv["roll"])
                self._last_pos = e.position()
                self.update()
            return
        # ── grid group-move: mouse translates the new group (no button needed), eye/scene fixed ──
        if self._gxf is not None:
            self._gxf_apply(e.position()); self._last_pos = e.position(); return
        # ── fly steering: while moving forward/back (W/X held), the mouse guides the view direction ──
        if self._nav_mode and self._fpv is None:            # free-look: mouse turns the view in place (eye fixed)
            if self._last_pos is not None:
                self._fly_steer(e.position().x() - self._last_pos.x(),
                                e.position().y() - self._last_pos.y())
            self._last_pos = e.position(); self.update(); return
        # ── rubber-band box for copy-select: extend it, no orbit ──
        if self._rect is not None:
            self._rect[1] = e.position(); self.update(); return
        # ── modal prop transform: mouse drives it, no button held ──
        if self._xform is not None:
            self._xform_apply(e.position()); self._last_pos = e.position(); return
        # ── placement mode: mouse moves the cursor with NO button needed ──
        if self._mode != self.MODE_NONE:
            if self._axis_lock:
                self._slide_axis(e.position())     # slide along locked axis
            else:
                self._update_cursor_world(e.position())
            if self._mode == self.MODE_SLICE and self._axis_lock:
                i = 'XYZ'.index(self._axis_lock)   # move only that plane, live
                self._slices[i] = float(self._cursor_world[i])
            self._last_pos = e.position()
            self.update()
            return

        # ── dragging an already-placed point: move it, do not rotate camera ──
        if self._drag_kind is not None and self._drag_idx >= 0:
            self._drag_point_to(e.position())
            self._drag_moved = True
            self._last_pos = e.position()
            self.update()
            return

        # ── G-mode hover (no button held): highlight nearest point + table row.
        # Must run on plain mouse-tracking moves, so it is gated on button state
        # rather than on _last_pos (which persists between interactions).
        if (self._grab_held and self._mode == self.MODE_NONE
                and self._mouse_btn is None and self._drag_kind is None):
            kind, idx = self._find_nearest_point(e.position(), radius=30.0)
            self.set_highlight(kind, idx)
            self.pointHovered.emit(kind if kind else "", idx)
            self._hover_world = self._unproject_hover(e.position())
            self.update()
            return

        if self._last_pos is None:
            self._hover_world = self._unproject_hover(e.position())
            self.update()
            return

        dx = e.position().x() - self._last_pos.x()
        dy = e.position().y() - self._last_pos.y()
        if abs(dx) > 1 or abs(dy) > 1:
            self._drag_moved = True

        if True:
            if self._mouse_btn == Qt.MouseButton.LeftButton:
                mods = e.modifiers()
                if self._ortho:
                    # Top-down plan view stays FIXED: LMB pans instead of orbiting (switch to
                    # Perspective via RMB to rotate). Keeps the XY view locked for placement.
                    sc = self._dist * 0.002
                    self._pan[0] += dx * sc
                    self._pan[1] -= dy * sc
                elif mods & Qt.KeyboardModifier.AltModifier:
                    # Free tumble: hold Alt while dragging with LMB.
                    # This deliberately does NOT clamp, so you can inspect the
                    # room from any orientation, including over/under the ceiling.
                    self._free_y += dx * 0.45
                    self._free_x += dy * 0.45
                elif mods & Qt.KeyboardModifier.ControlModifier:
                    # Roll: hold Ctrl while dragging horizontally with LMB.
                    self._roll += dx * 0.45
                elif mods & Qt.KeyboardModifier.ShiftModifier:
                    # Inside-box look mode: keep Z as the main yaw axis, but do
                    # NOT clamp pitch. This lets you look straight up at the
                    # ceiling or straight down at the floor after zooming inside.
                    self._yaw   += dx * 0.35
                    self._pitch += dy * 0.35
                else:
                    # Default stable architectural orbit: Z remains the main/up axis.
                    # Allow full pitch instead of stopping at +/-89 degrees.
                    self._yaw   += dx * 0.4
                    self._pitch += dy * 0.4
            elif self._mouse_btn == Qt.MouseButton.MiddleButton:
                sc = self._dist * 0.002
                self._pan[0] += dx * sc
                self._pan[1] -= dy * sc

        self._last_pos = e.position()
        self.update()

    def mouseReleaseEvent(self, e):
        moved = getattr(self, '_drag_moved', False)
        self._mouse_btn = None
        self._last_pos  = None

        # Point group transform: LMB drops it, RMB cancels. Consumed here so S/R mode doesn't also place.
        if self._gxf is not None and e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._gxf_commit(cancel=(e.button() == Qt.MouseButton.RightButton)); return

        # Finish an edit-mode point box-select: fill the multi-selection (highlighted; then G/R/Delete).
        if self._rect is not None and self._pt_box and e.button() == Qt.MouseButton.LeftButton:
            self._pt_sel = self._find_points_in_rect(self._rect)
            self._rect = None; self._pt_box = False
            self.update(); return

        # Finish a surface rect-select: add the front-most enclosed surfaces to the selection (highlight).
        if self._rect is not None and self._surf_pick and e.button() == Qt.MouseButton.LeftButton:
            self._surf_sel |= self._find_surfaces_in_rect(self._rect)
            self._rect = None; self._surf_pick = False
            self.setCursor(Qt.CursorShape.ArrowCursor); self.update(); return

        # Finish the copy box: duplicate props whose center is inside it, then grab the copies (move).
        if self._rect is not None and e.button() == Qt.MouseButton.LeftButton:
            idxs = self._find_props_in_rect(self._rect); self._rect = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if idxs:
                self._copy_start(idxs)                              # copies enter a group MOVE
            self.update(); return

        if self._fpv is not None:      # receiver view: no layer pick / point pick / placement — only
            if not moved and e.button() == Qt.MouseButton.RightButton:  # the RMB Go-back menu
                self._scene_menu(e.position())
            return

        # Finish a point drag: emit the final position so the table updates,
        # then auto-disarm "edit positions" mode (one move per E activation).
        if self._drag_kind is not None and self._drag_idx >= 0:
            lst = self._sources if self._drag_kind == "source" else self._receivers
            if 0 <= self._drag_idx < len(lst):
                p = lst[self._drag_idx]
                self.pointMoved.emit(self._drag_kind, self._drag_idx,
                                     float(p[0]), float(p[1]), float(p[2]))
            self._drag_kind = None
            self._drag_idx  = -1
            self._drag_axis      = None
            self._drag_axis_base = None
            self._drag_anchor_px = None
            self._grab_held = False                      # edit mode off after a move
            self._hl_src = -1
            self._hl_rec = -1
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        if self._mode != self.MODE_NONE:
            # Place / commit on LMB release (after an axis drag or a plain click).
            if e.button() == Qt.MouseButton.LeftButton:
                if self._mode == self.MODE_SLICE:
                    s = self._slices
                    self.sliceChanged.emit(float(s[0]), float(s[1]), float(s[2]))
                else:
                    p = self._cursor_world.copy()
                    mode_name = "source" if self._mode == self.MODE_SRC else "receiver"
                    self.pointPlaced.emit(mode_name, float(p[0]), float(p[1]), float(p[2]))
            return

        if not moved and e.button() == Qt.MouseButton.LeftButton:
            if self._results_pick:                       # Results mode: click toggles a receiver
                kind, idx = self._find_nearest_point(e.position(), radius=30.0)
                self.receiverPicked.emit(idx if kind == "receiver" else -1)
                return
            if self._props_mode or self._grab_held:      # positioning props / editing points: don't
                return                                   # pick+highlight domain surfaces (annoying)
            self._do_ray_pick(e.position())
        if not moved and e.button() == Qt.MouseButton.RightButton:
            if self._fpv is not None or not self._do_point_remove(e.position()):  # empty -> scene menu
                self._scene_menu(e.position())

    def wheelEvent(self, e):
        if self._fpv is not None:      # receiver view: zoom disabled
            return
        steps = e.angleDelta().y() / 120.0   # 1 notch = 120 units
        f = 0.85 ** steps                     # 15% per notch

        # ── dolly toward cursor, not toward fixed pivot ──────────────
        # Unproject the pixel under the mouse at mid-scene depth to get
        # a world-space target, then shift the pivot a fraction toward it.
        self.makeCurrent()
        W, H = self.width(), self.height()
        pos = e.position()
        mx, my = float(pos.x()), float(pos.y())
        viewport = [int(0), int(0), int(W), int(H)]
        try:
            # sample depth at the pivot itself to get a stable reference depth
            sx, sy, sz = gluProject(
                float(self._pivot[0]),
                float(self._pivot[1]),
                float(self._pivot[2]),
                self._mv, self._proj, viewport)
            # unproject mouse at that same depth
            world = gluUnProject(mx, float(H) - my, float(sz),
                                 self._mv, self._proj, viewport)
            target = np.array(world, np.float32)
            # shift pivot toward/away from cursor proportional to zoom step
            shift_frac = 1.0 - f          # positive = zoom in, negative = out
            self._pivot += (target - self._pivot) * shift_frac
        except Exception:
            pass  # fallback: plain zoom with no pivot shift

        self._dist = max(0.001, self._dist * f)
        self.update()

    def _save_home_view(self):
        """Capture the current camera as the 'home' view that reset restores."""
        self._home_view = {
            "pivot":  self._pivot.copy(),
            "pan":    self._pan.copy(),
            "dist":   float(self._dist),
            "yaw":    float(self._yaw),
            "pitch":  float(self._pitch),
            "free_x": float(getattr(self, "_free_x", 0.0)),
            "free_y": float(getattr(self, "_free_y", 0.0)),
            "roll":   float(getattr(self, "_roll", 0.0)),
        }

    def _reset_view(self):
        """Restore the exact initial view captured at load time (pivot, pan,
        distance/zoom and every orientation angle), so the scene returns to
        precisely the same visualization each time."""
        hv = getattr(self, "_home_view", None)
        if hv is None:
            return
        self._pivot  = np.array(hv["pivot"], np.float32)
        self._pan    = np.array(hv["pan"],   np.float32)
        self._dist   = hv["dist"]
        self._yaw    = hv["yaw"]
        self._pitch  = hv["pitch"]
        self._free_x = hv["free_x"]
        self._free_y = hv["free_y"]
        self._roll   = hv["roll"]
        self.update()

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key.Key_Question:          # '?' toggles the shortcuts overlay in any mode
            self._show_help = not self._show_help; self.update(); e.accept(); return
        # Point group transform modal: G/R switch move/rotate, X/Y/Z pick axis, Enter commits, Esc cancels.
        if self._gxf is not None:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._gxf_commit(); e.accept(); return
            if key == Qt.Key.Key_Escape:
                self._gxf_commit(cancel=True); e.accept(); return
            if key in (Qt.Key.Key_P, Qt.Key.Key_R):        # P move / R rotate (consistent with props)
                self._gxf["mode"] = 'g' if key == Qt.Key.Key_P else 'r'; self._gxf_reanchor()
                e.accept(); return
            if key in (Qt.Key.Key_X, Qt.Key.Key_Y, Qt.Key.Key_Z):   # HELD lock (release unlocks)
                if not e.isAutoRepeat():
                    self._gxf["axis"] = {Qt.Key.Key_X: 'X', Qt.Key.Key_Y: 'Y', Qt.Key.Key_Z: 'Z'}[key]
                    self._gxf_reanchor()
                e.accept(); return
            e.accept(); return                              # swallow everything else during the modal
        # Edit mode with a multi-selection: P move / R rotate the group; Delete removes it.
        if (self._grab_held and self._pt_sel and self._mode == self.MODE_NONE and not self._props_mode):
            if key == Qt.Key.Key_P:
                self.start_group_xform(self._pt_sel, "g"); e.accept(); return
            if key == Qt.Key.Key_R:
                self.start_group_xform(self._pt_sel, "r"); e.accept(); return
            if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                self.pointsDeleted.emit(list(self._pt_sel)); self._pt_sel = []
                self.update(); e.accept(); return
        # ── Draw mode: modal prop transform. P move, R rotate, S scale; X/Y/Z lock axis (toggle);
        #    Enter confirms, Esc cancels. Blender-style: after the key, the mouse drives it (no button). ──
        if self._props_mode and self._mode == self.MODE_NONE:
            if self._xform is None and self._rect is None and e.matches(QKeySequence.StandardKey.Undo):
                if self.props_undo():
                    e.accept(); return
            if key == Qt.Key.Key_Escape and (self._copy_arm or self._rect is not None):
                self._copy_arm = False; self._rect = None; self.update(); e.accept(); return
            if self._xform is None and self._rect is None and key == Qt.Key.Key_C and self._props:
                self._copy_arm = True                                # next LMB drag = box-select to copy
                self.setCursor(Qt.CursorShape.CrossCursor); e.accept(); return
            if self._xform is not None:
                if key == Qt.Key.Key_Escape:
                    self._xform_end(cancel=True); e.accept(); return
                if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    self._xform_end(cancel=False); e.accept(); return
                if key in (Qt.Key.Key_X, Qt.Key.Key_Y, Qt.Key.Key_Z):    # HELD (like placement/drag locks)
                    if not e.isAutoRepeat():
                        self._xform_reanchor()                            # continue from here (no jump)
                        self._xform['axis'] = {Qt.Key.Key_X: 'X', Qt.Key.Key_Y: 'Y', Qt.Key.Key_Z: 'Z'}[key]
                        self.update()
                    e.accept(); return
                if not e.isAutoRepeat() and key in (Qt.Key.Key_P, Qt.Key.Key_R, Qt.Key.Key_S):
                    self._xform_reanchor()                                # switch mode mid-transform
                    self._xform['mode'] = {Qt.Key.Key_P: 'p', Qt.Key.Key_R: 'r', Qt.Key.Key_S: 's'}[key]
                    self._xform['axis'] = None
                    self.update(); e.accept(); return
            if self._xform is None and key in (Qt.Key.Key_P, Qt.Key.Key_R, Qt.Key.Key_S):
                self.push_props_undo()                            # so Ctrl+Z reverts this transform
                if self._xform_start({Qt.Key.Key_P: 'p', Qt.Key.Key_R: 'r', Qt.Key.Key_S: 's'}[key]):
                    e.accept(); return
                elif self._props_undo_stack:
                    self._props_undo_stack.pop()                  # no prop to grab -> drop the snapshot
        if key == Qt.Key.Key_Escape:
            self.exit_placement_mode()
            e.accept(); return
        if key == Qt.Key.Key_T and self._mode == self.MODE_NONE:
            # reset camera to the exact initial view captured at load time
            self._reset_view()
            e.accept(); return
        # "E" toggles "edit positions" (Edit) mode: press once to arm, again to
        # disarm. While armed, LMB grabs and drags the nearest placed point.
        # The mode also self-deactivates after a completed move (see release).
        if key == Qt.Key.Key_E and self._mode == self.MODE_NONE:
            self._grab_held = not self._grab_held
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._grab_held
                           else Qt.CursorShape.ArrowCursor)
            if not self._grab_held:
                # leaving edit mode clears the hover highlight + multi-selection
                self.set_highlight(None, -1)
                self.pointHovered.emit("", -1)
                self._pt_sel = []
            e.accept(); return
        # WAXD fly navigation (idle only) — hold to move; mouse steers while W/X held. X = back (S is source).
        if (self._nav_mode and self._fpv is None and self._mode == self.MODE_NONE and self._xform is None
                and not self._props_mode and self._drag_kind is None
                and key in (Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_X, Qt.Key.Key_D,
                            Qt.Key.Key_Q, Qt.Key.Key_Z)):
            if not e.isAutoRepeat():
                self._fly_keys.add(key)
                if not self._fly_timer.isActive():
                    self._fly_timer.start()
            e.accept(); return
        # X/Y/Z while dragging a point -> constrain movement to that axis.
        if self._drag_kind is not None and key in (
                Qt.Key.Key_X, Qt.Key.Key_Y, Qt.Key.Key_Z):
            self._set_drag_axis({Qt.Key.Key_X:'X', Qt.Key.Key_Y:'Y',
                                 Qt.Key.Key_Z:'Z'}[key])
            e.accept(); return
        # X/Y/Z during placement -> placement axis lock (unchanged).
        if self._mode != self.MODE_NONE:
            if key == Qt.Key.Key_X:
                self._set_axis_lock('X'); e.accept(); return
            elif key == Qt.Key.Key_Y:
                self._set_axis_lock('Y'); e.accept(); return
            elif key == Qt.Key.Key_Z:
                self._set_axis_lock('Z'); e.accept(); return
        e.ignore()  # let MainWindow handle S / T for mode switching

    def keyReleaseEvent(self, e):
        # Axis lock behaves like a held modifier: releasing X/Y/Z unlocks.
        if e.isAutoRepeat():
            return
        key = e.key()
        # Release a held fly key -> stop that direction (timer stops when none left).
        if key in self._fly_keys and not e.isAutoRepeat():
            self._fly_keys.discard(key)
            if not self._fly_keys:
                self._fly_timer.stop()
            e.accept(); return
        # Releasing the held X/Y/Z during a point group transform -> back to free.
        if self._gxf is not None and key in (Qt.Key.Key_X, Qt.Key.Key_Y, Qt.Key.Key_Z):
            if self._gxf.get("axis") == {Qt.Key.Key_X: 'X', Qt.Key.Key_Y: 'Y', Qt.Key.Key_Z: 'Z'}[key]:
                self._gxf["axis"] = None; self._gxf_reanchor()
            e.accept(); return
        # Releasing X/Y/Z during a modal prop transform -> back to free (held, like everywhere else).
        if self._xform is not None and key in (Qt.Key.Key_X, Qt.Key.Key_Y, Qt.Key.Key_Z):
            self._xform_reanchor()                                # continue from here (no jump)
            self._xform['axis'] = None
            self.update()
            e.accept(); return
        # Releasing X/Y/Z while dragging a point -> back to free movement.
        if self._drag_kind is not None and key in (
                Qt.Key.Key_X, Qt.Key.Key_Y, Qt.Key.Key_Z):
            held = {Qt.Key.Key_X:'X', Qt.Key.Key_Y:'Y', Qt.Key.Key_Z:'Z'}[key]
            if self._drag_axis == held:
                self._drag_axis      = None
                self._drag_axis_base = None
                self._drag_anchor_px = None
            e.accept(); return
        if self._mode != self.MODE_NONE and key in (
                Qt.Key.Key_X, Qt.Key.Key_Y, Qt.Key.Key_Z):
            held = {Qt.Key.Key_X: 'X', Qt.Key.Key_Y: 'Y', Qt.Key.Key_Z: 'Z'}[key]
            if self._axis_lock == held:
                self._axis_lock = None
                self.update()
            e.accept(); return
        e.ignore()

    def _do_point_remove(self, qpos):
        """Right-click: remove nearest source or receiver within 30 px."""
        self.makeCurrent()
        W, H = self.width(), self.height()
        viewport = [int(0), int(0), int(W), int(H)]
        best_dist, best_list, best_idx = 30.0, None, -1
        for lst, kind in [(self._sources, "source"), (self._receivers, "receiver")]:
            for i, pt in enumerate(lst):
                try:
                    sx, sy, _ = gluProject(float(pt[0]), float(pt[1]), float(pt[2]),
                                           self._mv, self._proj, viewport)
                    d = math.hypot(sx - qpos.x(), (H - sy) - qpos.y())
                    if d < best_dist:
                        best_dist, best_list, best_idx, best_kind = d, lst, i, kind
                except Exception:
                    pass
        if best_list is not None and best_idx >= 0:
            best_list.pop(best_idx)
            self.pointRemoved.emit(best_kind, best_idx)
            self.update()
            return True
        return False

    def set_grid_h(self, h):
        self._grid_h = float(h)
        self.update()

    def set_hud_left(self, px):
        # store only -- no repaint. During the panel slide this is called every frame;
        # forcing a full GL redraw each time is the slide lag. The HUD re-lays out on the
        # next natural repaint (and once at slide end, see _panel_anim.finished).
        self._hud_left = int(px)

    def _ijk_str(self, p):
        """Integer grid indices i,j,k at world point p for the user's grid spacing
        (origin at 0). '' when no spacing is set yet."""
        h = self._grid_h
        if not h or h <= 0.0:
            return ""
        return f"i={int(round(p[0]/h))}  j={int(round(p[1]/h))}  k={int(round(p[2]/h))}"

    def _scene_menu(self, qpos):
        """RMB on empty space -> scene options (stacked). Inside a receiver view, ONLY 'Go back'."""
        m = QMenu(self)
        gpos = self.mapToGlobal(QPoint(int(qpos.x()), int(qpos.y())))
        if self._fpv is not None:
            gb = m.addAction("↩  Go back")
            if m.exec(gpos) is gb:
                self.fpvExitRequested.emit()
            return
        # ── surface grouping: rectangle-select surfaces, then group / explode ──
        hit_idx = self._pick_group_idx(qpos)
        hit = self._groups[hit_idx]["name"] if hit_idx >= 0 else None
        sel_s = m.addAction("Select surfaces by rectangle")
        exp_s = grp_s = clr_s = mrg_s = None
        # Explode/merge only on enclosure surfaces — never props (all props share one layer name, so
        # acting by name would shatter every prop at once).
        if hit and not _is_bbox(hit) and hit not in self._prop_layers():
            exp_s = m.addAction(f"Explode this surface ('{hit}')")
            base = _sibling_base(hit)                     # 'surface_12' -> 'surface' when part of an exploded pool
            if base and sum(1 for g in self._groups if _sibling_base(g["name"]) == base) > 1:
                mrg_s = m.addAction(f"Merge all '{base}_*' → layer…")
        if self._surf_sel:
            grp_s = m.addAction(f"Group {len(self._surf_sel)} selected → layer…")
            clr_s = m.addAction("Clear surface selection")
        cln_s = m.addAction("Cleanse phantom layers")
        # Only meaningful when the OBJ was actually shifted on import (centred-origin scenes
        # like the BRAS auditorium); otherwise the two frames are identical.
        # Two ALTERNATING options in one exclusive group: the scene is read either in the
        # shifted (solver) frame or in the file's own frame -- never both, never neither.
        from PySide6.QtGui import QActionGroup
        sh = getattr(self, "_obj_shift", (0, 0, 0))
        on = bool(getattr(self, "_shift_enabled", True))
        cm = m.addMenu("Scene coordinates")
        grp = QActionGroup(cm); grp.setExclusive(True)
        sp_s = cm.addAction("Shifted to positive octant"
                            + (f"  (by {sh[0]:g}, {sh[1]:g}, {sh[2]:g})" if any(sh) else ""))
        of_s = cm.addAction("Original OBJ frame")
        for a, checked in ((sp_s, on), (of_s, not on)):
            a.setCheckable(True); a.setChecked(checked); grp.addAction(a)
        m.addSeparator()

        ax = m.addAction("Hide axes" if self._axes_visible else "Show axes")
        th = m.addAction("Light scene" if not SCENE_LIGHT_MODE else "Dark scene")
        dh = m.addAction("Exit dollhouse view" if getattr(self, "_dollhouse", False)
                         else "Dollhouse view")
        oo = m.addAction("Perspective view" if self._ortho else "Top-down (ortho) view")
        nv = m.addAction("Exit navigation mode" if self._nav_mode else "Navigation mode (WAXD fly)")
        m.addSeparator()
        ev = m.addAction("🎧︎  Enter receiver view")
        chosen = m.exec(gpos)
        if chosen is None:
            return
        if chosen is sel_s:
            self.arm_surface_rect(); return
        if chosen is exp_s:
            self.explodeSurface.emit(hit_idx); return
        if chosen is mrg_s:
            self.mergeSiblings.emit(_sibling_base(hit)); return
        if chosen is grp_s:
            self.surfacesPicked.emit(sorted(self._surf_sel)); return
        if chosen is clr_s:
            self._surf_sel = set(); self.update(); return
        if chosen is cln_s:
            self.cleanseLayers.emit(); return
        if chosen in (sp_s, of_s):
            want = (chosen is sp_s)
            if want != bool(getattr(self, "_shift_enabled", True)):
                self._shift_enabled = want
                self.shiftPolicyToggled.emit(want)
            return
        if chosen is ax:
            self._axes_visible = not self._axes_visible
            self.update()
        elif chosen is th:
            self.themeToggle.emit()
        elif chosen is dh:
            self._dollhouse = not getattr(self, "_dollhouse", False)
            self.update()
        elif chosen is oo:                                  # scene-wide plan view (props + src/rec later)
            self.set_ortho(not self._ortho)
        elif chosen is nv:
            self.set_nav_mode(not self._nav_mode)
        elif chosen is ev:
            self.fpvRequested.emit()

    def _find_nearest_point(self, qpos, radius=30.0):
        """Return (kind, idx) of the nearest source/receiver within `radius`
        screen pixels of qpos, or (None, -1). Shared by grab and G-mode hover."""
        self.makeCurrent()
        W, H = self.width(), self.height()
        viewport = [int(0), int(0), int(W), int(H)]
        best_dist, best_kind, best_idx = radius, None, -1
        for lst, kind in [(self._sources, "source"), (self._receivers, "receiver")]:
            for i, pt in enumerate(lst):
                try:
                    sx, sy, _ = gluProject(float(pt[0]), float(pt[1]), float(pt[2]),
                                           self._mv, self._proj, viewport)
                    d = math.hypot(sx - qpos.x(), (H - sy) - qpos.y())
                    if d < best_dist:
                        best_dist, best_kind, best_idx = d, kind, i
                except Exception:
                    pass
        return best_kind, best_idx

    def _lst(self, kind):
        return self._sources if kind == "source" else self._receivers

    def _find_points_in_rect(self, rect):
        """[(kind, idx), …] of sources+receivers whose glyph projects inside the screen rectangle."""
        self.makeCurrent()
        W, H = self.width(), self.height(); vp = [0, 0, int(W), int(H)]
        x0, x1 = sorted((rect[0].x(), rect[1].x())); y0, y1 = sorted((rect[0].y(), rect[1].y()))
        out = []
        for kind in ("source", "receiver"):
            for i, pt in enumerate(self._lst(kind)):
                try:
                    sx, sy, sz = gluProject(float(pt[0]), float(pt[1]), float(pt[2]), self._mv, self._proj, vp)
                    if sz < 0 or sz > 1:
                        continue
                    if x0 <= sx <= x1 and y0 <= (H - sy) <= y1:
                        out.append((kind, i))
                except Exception:
                    pass
        return out

    def start_group_xform(self, items, mode):
        """Modal move ('g') / rotate ('r') of a set of points (items = [(kind, idx), …]). Mouse drives it,
        click/Enter commits, Esc/RMB cancels. X/Y/Z pick the axis. Out-of-domain points render yellow.
        Used by grid creation (move) and edit-mode multi-select (move/rotate)."""
        items = [(k, i) for (k, i) in items if 0 <= i < len(self._lst(k))]
        if not items:
            return
        self._gxf = {"items": items, "mode": mode,
                     "orig": [list(self._lst(k)[i][:3]) for (k, i) in items]}
        self._gxf_reanchor()
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.update()

    def start_group_move(self, kind, idxs):          # grid-creation entry point (translate)
        self.start_group_xform([(kind, i) for i in idxs], "g")

    def _gxf_reanchor(self):
        """Re-base the transform on the CURRENT coords (start, or after a mode/axis switch) so continuing
        never jumps and the current result is preserved."""
        gx = self._gxf
        gx["base"] = [list(self._lst(k)[i][:3]) for (k, i) in gx["items"]]
        gx["centroid"] = np.mean(np.asarray(gx["base"], float), axis=0)
        gx["anchor"] = None; gx["anchor_ang"] = None

    def _gxf_apply(self, qpos):
        (self._gxf_move if self._gxf["mode"] == "g" else self._gxf_rotate)(qpos)

    def _gxf_move(self, qpos):
        gx = self._gxf; self.makeCurrent()
        W, H = self.width(), self.height(); vp = [0, 0, int(W), int(H)]
        c = np.asarray(gx["centroid"], float)
        mx, myf = float(qpos.x()), float(H) - float(qpos.y())
        try:
            if gx.get("axis"):
                # slide the group centroid along the world axis to the point closest to the mouse RAY
                # (view-independent — same trick as single-point drag, so it tracks the cursor exactly).
                if gx["anchor"] is None:
                    gx["anchor"] = c.copy(); return
                a0 = gx["anchor"]
                ax = {'X': np.array([1., 0, 0]), 'Y': np.array([0, 1., 0]), 'Z': np.array([0, 0, 1.])}[gx["axis"]]
                ro = np.array(gluUnProject(mx, myf, 0.0, self._mv, self._proj, vp), float)
                rd = np.array(gluUnProject(mx, myf, 1.0, self._mv, self._proj, vp), float) - ro
                w0 = a0 - ro; b = ax @ rd; cc = rd @ rd; d0 = ax @ w0; e0 = rd @ w0
                denom = cc - b * b
                if abs(denom) < 1e-9:
                    return
                d = ax * ((b * e0 - cc * d0) / denom)            # displacement of centroid along the axis
            else:
                _, _, sz = gluProject(float(c[0]), float(c[1]), float(c[2]), self._mv, self._proj, vp)
                world = np.array(gluUnProject(mx, myf, float(sz), self._mv, self._proj, vp), float)
                if gx["anchor"] is None:
                    gx["anchor"] = world; return
                d = world - gx["anchor"]
        except Exception:
            return
        for n, (kind, idx) in enumerate(gx["items"]):
            b = gx["base"][n]; self._lst(kind)[idx] = [b[0] + d[0], b[1] + d[1], b[2] + d[2]]
        self.update()

    def _gxf_rotate(self, qpos):
        gx = self._gxf; self.makeCurrent()
        W, H = self.width(), self.height(); vp = [0, 0, int(W), int(H)]
        c = gx["centroid"]
        try:
            cxs, cys, _ = gluProject(float(c[0]), float(c[1]), float(c[2]), self._mv, self._proj, vp)
        except Exception:
            return
        ang = math.atan2((float(H) - float(qpos.y())) - cys, float(qpos.x()) - cxs)
        if gx["anchor_ang"] is None:
            gx["anchor_ang"] = ang; return
        R = self._axis_rot_np(gx.get("axis") or 'Z', ang - gx["anchor_ang"])   # default vertical (Z)
        for n, (kind, idx) in enumerate(gx["items"]):
            p = c + R @ (np.asarray(gx["base"][n], float) - c)
            self._lst(kind)[idx] = [float(p[0]), float(p[1]), float(p[2])]
        self.update()

    @staticmethod
    def _axis_rot_np(axis, th):
        c, s = math.cos(th), math.sin(th)
        if axis == 'X': return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == 'Y': return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _gxf_commit(self, cancel=False):
        gx = self._gxf; self._gxf = None
        self._pt_sel = []                                # drop the highlight once the group op is done
        self.setCursor(Qt.CursorShape.ArrowCursor)
        if gx and cancel:
            for n, (kind, idx) in enumerate(gx["items"]):
                if 0 <= idx < len(self._lst(kind)):
                    self._lst(kind)[idx] = list(gx["orig"][n])
        if gx and not cancel:
            self.groupXfDone.emit(gx["items"])
        self.update()

    def _begin_point_drag(self, qpos):
        """
        Grab the nearest placed source/receiver within a screen-space radius so
        it can be dragged. Called once on LMB press while edit mode is armed, so the
        closest-point search is not repeated on every mouse-move (keeps it light).
        """
        kind, idx = self._find_nearest_point(qpos, radius=30.0)
        if kind is not None:
            # snapshot for undo BEFORE any drag motion mutates the point
            self.pointDragStarted.emit()
            self._drag_kind = kind
            self._drag_idx  = idx
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _set_drag_axis(self, axis):
        """Lock the in-progress point drag to a single world axis. Freezes the
        other two coordinates at the point's current position and anchors the
        mouse, so movement maps cleanly onto that axis."""
        lst = self._sources if self._drag_kind == "source" else self._receivers
        if not (0 <= self._drag_idx < len(lst)):
            return
        self._drag_axis      = axis
        self._drag_axis_base = list(lst[self._drag_idx])   # frozen base point
        self._drag_anchor_px = QCursor.pos()               # mouse anchor (global)

    def _drag_point_to(self, qpos):
        """
        Move the grabbed point. With no axis constraint, the point follows the
        mouse on the screen-plane at its current depth. With an axis locked
        (X/Y/Z held), it slides only along that world axis from the locked base,
        driven by the total mouse movement since the lock (driftless).
        """
        lst = self._sources if self._drag_kind == "source" else self._receivers
        if not (0 <= self._drag_idx < len(lst)):
            return
        self.makeCurrent()
        W, H = self.width(), self.height()
        viewport = [int(0), int(0), int(W), int(H)]

        # ── axis-constrained movement (view-INDEPENDENT) ──
        # Slide along the world axis to the point on it CLOSEST to the mouse ray, so the point tracks
        # the cursor exactly at any camera angle (no screen-projection heuristic / sensitivity fudge).
        if self._drag_axis is not None and self._drag_axis_base is not None:
            ax = {'X': np.array([1., 0, 0]), 'Y': np.array([0, 1., 0]),
                  'Z': np.array([0, 0, 1.])}[self._drag_axis]
            base = np.array(self._drag_axis_base, np.float64)
            try:
                mx, myf = float(qpos.x()), float(H) - float(qpos.y())
                ro = np.array(gluUnProject(mx, myf, 0.0, self._mv, self._proj, viewport), np.float64)
                rd = np.array(gluUnProject(mx, myf, 1.0, self._mv, self._proj, viewport), np.float64) - ro
                w0 = base - ro                                  # closest points between axis line & ray
                b = float(ax @ rd); c = float(rd @ rd); d = float(ax @ w0); e = float(rd @ w0)
                denom = c - b * b                               # a = ax·ax = 1
                if abs(denom) < 1e-9:
                    return                                      # ray ~parallel to the axis -> ambiguous
                s = (b * e - c * d) / denom                     # axis coord under the cursor
                newp = base + ax * s
                lst[self._drag_idx] = [float(newp[0]), float(newp[1]), float(newp[2])]
            except Exception:
                pass
            return

        # ── free screen-plane movement ──
        cur = lst[self._drag_idx]
        try:
            _, _, sz = gluProject(float(cur[0]), float(cur[1]), float(cur[2]),
                                  self._mv, self._proj, viewport)
            world = gluUnProject(float(qpos.x()), float(H) - float(qpos.y()),
                                 float(sz), self._mv, self._proj, viewport)
            lst[self._drag_idx] = [float(world[0]), float(world[1]), float(world[2])]
        except Exception:
            pass

    def _set_axis_lock(self, axis):
        # Holding X/Y/Z freezes the other two coords at the CURRENT cursor point;
        # the mouse then slides the point only along this axis.
        if self._axis_lock == axis:
            return
        self._axis_lock = axis
        self._place_pos = self._cursor_world.copy()   # frozen base point
        self._lock_anchor_px = QCursor.pos()          # mouse anchor (global)
        self.update()

    def _update_cursor_world(self, qpos):
        """Unproject mouse to a plane at _place_pos depth."""
        self.makeCurrent()
        W, H = self.width(), self.height()
        try:
            viewport = [int(0), int(0), int(W), int(H)]
            mx, my = qpos.x(), qpos.y()
            # depth at current place_pos
            sx, sy, sz = gluProject(
                float(self._place_pos[0]),
                float(self._place_pos[1]),
                float(self._place_pos[2]),
                self._mv, self._proj, viewport)
            world = gluUnProject(float(mx), float(H-my), float(sz),
                                 self._mv, self._proj, viewport)
            self._cursor_world = np.array(world, np.float32)
        except Exception:
            pass

    def _slide_axis(self, qpos):
        """
        Slide the cursor ONLY along the locked axis; the two other coordinates
        stay frozen at _place_pos. Displacement is the TOTAL mouse movement since
        the key was pressed (anchor), so there is no drift and no jump at lock.
        """
        ax = {'X': np.array([1,0,0],np.float32),
              'Y': np.array([0,1,0],np.float32),
              'Z': np.array([0,0,1],np.float32)}[self._axis_lock]
        self.makeCurrent()
        W, H = self.width(), self.height()
        try:
            viewport = [int(0), int(0), int(W), int(H)]
            p0 = self._place_pos
            p1 = p0 + ax * 0.1
            s0 = np.array(gluProject(*p0.astype(float), self._mv, self._proj, viewport)[:2])
            s1 = np.array(gluProject(*p1.astype(float), self._mv, self._proj, viewport)[:2])
            screen_dir = s1 - s0
            slen = np.linalg.norm(screen_dir)
            if slen < 1e-6:
                return
            screen_dir /= slen
            anchor = getattr(self, "_lock_anchor_px", None)
            if anchor is None:
                self._lock_anchor_px = QCursor.pos()
                return
            cur = QCursor.pos()
            mdx = cur.x() - anchor.x()
            mdy = -(cur.y() - anchor.y())
            move = float(np.dot(np.array([mdx, mdy], np.float32), screen_dir))
            scale = self._dist * 0.0015
            self._cursor_world = self._place_pos + ax * move * scale
        except Exception:
            pass

    def _do_ray_pick(self, qpos):
        best_name = self._pick_group_name(qpos)
        if best_name and best_name != self._selected_layer:
            self._selected_layer = best_name
            self.layerPicked.emit(best_name)
        else:
            # Empty space OR clicking the already-selected surface again -> deselect (toggle off).
            self._selected_layer = None
            self.layerPicked.emit("")
            # also clear any source/receiver highlight (scene + table selection)
            if self._hl_src != -1 or self._hl_rec != -1:
                self.set_highlight(None, -1)
                self.pointHovered.emit("", -1)
        self.update()


# ─────────────────────────────────────────────────────────────
#  Material editor
# ─────────────────────────────────────────────────────────────
class _RunStatusBar(QStatusBar):
    """Status bar where the solver's time-left owns the line while a run is active.

    ~25 call sites push transient text (placement hints, 'Placed source at…', mode
    changes). Any of them used to wipe the ETA until the next progress line arrived.
    While `hold` is set only priority messages get through, so 'time left' stays put.
    """
    def __init__(self):
        super().__init__()
        self.hold = False

    def showMessage(self, text, timeout=0, priority=False):
        if self.hold and not priority:
            return                      # a run is active: don't clobber the ETA
        super().showMessage(text, timeout)


class _FitAllThread(QThread):
    """Runs the fitter on all materials in the background. The script pops its
    summary figure; this thread streams the process output line-by-line (so the
    Run-tab Output log fills live, before plt.show() blocks) and reports the
    result when the user closes the plot window."""
    done = Signal(int, str)
    line = Signal(str)          # one line of stdout/stderr as it is produced

    def __init__(self, cmd, cwd):
        super().__init__(); self._cmd = cmd; self._cwd = cwd

    def run(self):
        try:
            proc = subprocess.Popen(
                self._cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", cwd=self._cwd, bufsize=1)
            lines = []
            for raw in proc.stdout:
                ln = raw.rstrip("\n")
                lines.append(ln)
                self.line.emit(ln)
            proc.wait()
            rc = proc.returncode
            if rc != 0:
                print("\n".join(lines))   # full traceback on the terminal too
            tail = next((l for l in reversed(lines) if "error" in l.lower()),
                        lines[-1] if lines else "")
            self.done.emit(rc, tail)
        except Exception as ex:
            self.done.emit(-1, str(ex))


class ToggleSwitch(QCheckBox):
    """
    A small iOS-style on/off slider switch (used for the hard/soft flag).
    Behaves exactly like a QCheckBox (checked / toggled signal) but is painted
    as a sliding pill: grey knob on the left when off, accent-purple track with
    the knob on the right when on.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._w, self._h = 42, 22         # track size in px
        self.setFixedSize(self._w, self._h)

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(self._w, self._h)

    def hitButton(self, pos):
        # A bare QCheckBox only treats its small indicator rect as clickable;
        # accept the WHOLE painted switch so any click toggles it.
        return self.rect().contains(pos)

    def paintEvent(self, _):
        from PySide6.QtGui import QPainter, QColor
        from PySide6.QtCore import QRectF
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        on = self.isChecked()
        r = self._h / 2.0
        # track
        track = QColor(PAL["accent"]) if on else QColor(PAL["btn"])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 0, self._w, self._h), r, r)
        # knob
        knob_d = self._h - 6
        kx = (self._w - knob_d - 3) if on else 3
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(QRectF(kx, 3, knob_d, knob_d))
        p.end()


class AlphaCurveEditor(QWidget):
    """
    Interactive replacement for the matplotlib absorption plot.

    • One control point per octave band, drawn as a filled circle in the GUI
      accent (purple) tone, joined by a smooth (Catmull-Rom) curve of the same
      colour. Vertical position encodes alpha 0..1 (bottom..top).
    • Click-drag a point to set its value; drag sideways across the widget to
      "draw" the whole curve — every band the cursor passes over follows the
      pointer height.
    • Numeric values appear next to the points ONLY while dragging.
    • Shows the material name (top), the 'α' symbol (left) and 'Frequency [Hz]'
      with band numbers (bottom). No axes/ticks otherwise.

    Two-way tied to the table via:
      • set_values(name, vals)   – table  -> editor
      • valuesChanged(list)      – editor -> table  (emitted live during drag)
    """
    valuesChanged = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(240)
        self.setMouseTracking(True)
        self._name = None
        self._vals = None              # list[float] or None when nothing selected
        self._n    = len(OCTAVE_BANDS)
        self._dragging = False
        self._pad_l, self._pad_r = 34, 16
        self._pad_t, self._pad_b = 26, 30

    # ── public API ──
    def set_values(self, name, vals):
        """Load values from the table (no signal emitted)."""
        self._name = name
        self._vals = [min(1.0, max(0.0, float(v))) for v in vals] if vals else None
        self.update()

    def clear(self):
        self._name = None
        self._vals = None
        self.update()

    # ── geometry helpers ──
    def _plot_rect(self):
        return (self._pad_l, self._pad_t,
                self.width()  - self._pad_l - self._pad_r,
                self.height() - self._pad_t - self._pad_b)

    def _point_xy(self, i):
        x0, y0, w, h = self._plot_rect()
        x = x0 + (w * i / (self._n - 1) if self._n > 1 else 0)
        y = y0 + h * (1.0 - self._vals[i])
        return x, y

    def _nearest_index(self, px):
        x0, y0, w, h = self._plot_rect()
        if w <= 0:
            return 0
        frac = (px - x0) / w
        i = round(frac * (self._n - 1))
        return int(min(self._n - 1, max(0, i)))

    def _value_from_y(self, py):
        x0, y0, w, h = self._plot_rect()
        if h <= 0:
            return 0.0
        return min(1.0, max(0.0, 1.0 - (py - y0) / h))

    # ── interaction ──
    def mousePressEvent(self, e):
        if self._vals is None:
            return
        self._dragging = True
        self._apply(e.position().x(), e.position().y())

    def mouseMoveEvent(self, e):
        if self._dragging and self._vals is not None:
            self._apply(e.position().x(), e.position().y())

    def mouseReleaseEvent(self, e):
        self._dragging = False
        self.update()

    def _apply(self, px, py):
        i = self._nearest_index(px)
        self._vals[i] = self._value_from_y(py)
        self.valuesChanged.emit(list(self._vals))
        self.update()

    # ── painting ──
    def paintEvent(self, _):
        from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPainterPath
        from PySide6.QtCore import QRectF, QPointF
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # dark plotting background matching the GUI
        p.fillRect(self.rect(), QColor(PAL["bg"]))
        x0, y0, w, h = self._plot_rect()

        accent = QColor(PAL["accent"])
        grey   = QColor(PAL["subtext"])      # muted light grey for labels/numbers
        light  = QColor(PAL["text"])         # bright text for name / α / Hz
        rail   = QColor(PAL["border"])       # slider-rail colour

        # title (material name) and α / Hz labels
        p.setPen(light)
        p.setFont(QFont("Sans", 9, QFont.Weight.Bold))
        if self._name:
            p.drawText(QRectF(0, 2, self.width(), 18),
                       Qt.AlignmentFlag.AlignCenter, self._name)
        p.setFont(QFont("Sans", 10, QFont.Weight.Bold))
        p.drawText(QRectF(2, y0, 24, h), Qt.AlignmentFlag.AlignCenter, "α")
        p.setFont(QFont("Sans", 8))
        p.setPen(light)
        p.drawText(QRectF(0, self.height() - 16, self.width(), 14),
                   Qt.AlignmentFlag.AlignCenter, "Octave band [Hz]")

        if self._vals is None:
            p.setPen(grey)
            p.setFont(QFont("Sans", 11))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "α per octave band")
            p.end()
            return

        # one VERTICAL slider rail per band (full 0..1 height); no horizontal
        # grid, so each band reads as a small vertical slider.
        p.setPen(QPen(rail, 2))
        for i in range(self._n):
            gx, _ = self._point_xy(i)
            p.drawLine(int(gx), int(y0), int(gx), int(y0 + h))

        # band frequency numbers along the bottom
        p.setPen(grey)
        p.setFont(QFont("Sans", 7))
        for i, b in enumerate(OCTAVE_BANDS):
            x, _ = self._point_xy(i)
            p.drawText(QRectF(x - 20, y0 + h + 2, 40, 12),
                       Qt.AlignmentFlag.AlignCenter, str(b))

        # smooth connecting curve (Catmull-Rom -> cubic Bézier)
        pts = [QPointF(*self._point_xy(i)) for i in range(self._n)]
        path = QPainterPath(pts[0])
        for i in range(len(pts) - 1):
            p0 = pts[i - 1] if i > 0 else pts[i]
            p1 = pts[i]
            p2 = pts[i + 1]
            p3 = pts[i + 2] if i + 2 < len(pts) else pts[i + 1]
            c1 = QPointF(p1.x() + (p2.x() - p0.x()) / 6.0,
                         p1.y() + (p2.y() - p0.y()) / 6.0)
            c2 = QPointF(p2.x() - (p3.x() - p1.x()) / 6.0,
                         p2.y() - (p3.y() - p1.y()) / 6.0)
            path.cubicTo(c1, c2, p2)
        p.setPen(QPen(accent, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        # control points (circles) + live value readout while dragging
        p.setFont(QFont("Sans", 7, QFont.Weight.Bold))
        for i in range(self._n):
            x, y = self._point_xy(i)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(accent)
            p.drawEllipse(QPointF(x, y), 4.5, 4.5)
            if self._dragging:
                p.setPen(accent)
                p.drawText(QRectF(x - 18, y - 18, 36, 12),
                           Qt.AlignmentFlag.AlignCenter, f"{self._vals[i]:.2f}")
        p.end()


class MaterialRefPreview(QWidget):
    """Compact name + sparkline preview for a predefined material entry."""
    def __init__(self, display_name, vals, parent=None):
        super().__init__(parent)
        self._name = display_name
        self._vals = [min(1.0, max(0.0, float(v))) for v in vals[:len(OCTAVE_BANDS)]]
        self._selected = False
        # Let the QListWidget own the mouse handling/selection.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setMinimumHeight(32)

    def sizeHint(self):
        # Width is intentionally close to the end of the curve column: enough
        # room for names, a small controlled gap, and the compact preview.
        return QSize(510, 36)

    def set_selected(self, selected):
        selected = bool(selected)
        if selected != self._selected:
            self._selected = selected
            self.update()

    def paintEvent(self, _):
        from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPainterPath
        from PySide6.QtCore import QRectF, QPointF
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg = QColor(PAL['accent']) if self._selected else QColor(PAL['panel'])
        fg = QColor(PAL['bg']) if self._selected else QColor(PAL['text'])
        curve_col = QColor(PAL['bg']) if self._selected else QColor(PAL['accent2'])

        p.fillRect(self.rect(), bg)

        margin = 10
        gap = 22

        p.setPen(fg)
        p.setFont(QFont('Sans', 9))
        fm = p.fontMetrics()

        # Fixed columns keep every curve aligned. The name column is wide, but
        # not excessive; the curve starts after a modest constant gap instead
        # of jumping left/right with each material-name length.
        curve_w = 46.0
        name_w = min(390.0, max(260.0, self.width() - 2 * margin - gap - curve_w - 14.0))
        curve_left = margin + name_w + gap
        curve_rect = QRectF(curve_left, 8,
                            curve_w, max(12, self.height() - 16))
        name_rect = QRectF(margin, 0, name_w, self.height())

        visible_name = fm.elidedText(self._name, Qt.TextElideMode.ElideRight,
                                     int(name_rect.width()))
        p.drawText(name_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   visible_name)

        if len(self._vals) >= 2 and curve_rect.width() > 10 and curve_rect.height() > 6:
            # The preview must be physically honest: absorption is always scaled
            # against the full 0..1 range. Do NOT normalize each material to its
            # own min/max, otherwise tiny changes such as 0.01 -> 0.02 look like
            # large S-shaped curves even though they are practically flat.
            pts = []
            n = len(self._vals)
            y_pad = curve_rect.height() * 0.14
            y0 = curve_rect.bottom() - y_pad   # alpha = 0 reference
            y1 = curve_rect.top() + y_pad      # alpha = 1 reference
            y_usable = max(1.0, y0 - y1)

            # Minimal 0..1 reference: top and bottom guide lines only, with no
            # labels, values or axes. This preserves the clean list while making
            # the vertical scale unambiguous.
            guide_col = QColor(PAL['bg'] if self._selected else PAL['subtext'])
            guide_col.setAlpha(85 if self._selected else 70)
            p.setPen(QPen(guide_col, 1.0))
            p.drawLine(QPointF(curve_rect.left(), y1), QPointF(curve_rect.right(), y1))
            p.drawLine(QPointF(curve_rect.left(), y0), QPointF(curve_rect.right(), y0))

            for i, v in enumerate(self._vals):
                x = curve_rect.left() + (curve_rect.width() * i / (n - 1))
                y = y0 - y_usable * min(1.0, max(0.0, v))
                pts.append(QPointF(x, y))

            # Piecewise-linear path: continuous, compact, and no cubic overshoot.
            path = QPainterPath(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            p.setPen(QPen(curve_col, 2.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        p.end()


class _CellDelegate(QStyledItemDelegate):
    """Per-column tweaks: `no_select` paints the cell as if never selected (so a
    row-selection highlight skips it); `center` centre-aligns the text; `tint`
    fills the cell background with a colour (unless the cell is selected)."""
    def __init__(self, parent=None, no_select=False, center=False, tint=None):
        super().__init__(parent)
        self._no_select = no_select
        self._center = center
        self._tint = QColor(tint) if tint else None
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if self._center:
            option.displayAlignment = Qt.AlignmentFlag.AlignCenter
        if self._tint is not None:
            from PySide6.QtGui import QBrush
            option.backgroundBrush = QBrush(self._tint)
    def paint(self, painter, option, index):
        if self._no_select:
            option.state &= ~QStyle.StateFlag.State_Selected
        super().paint(painter, option, index)


class MaterialPanel(QWidget):
    """
    Single panel for ALL material layers:
      • matplotlib plot on top (white bg, cyan curve) — populated on selection
      • one table: column 0 = layer name, columns 1..N = alpha per octave band
      • vertical scrollbar when many layers
      • Ctrl+V paste from Excel / space- or tab-separated text
      • Save .mat (per selected row) and Run vectFIT
    """
    rowSelected = Signal(str)   # emitted with layer name when a row is selected
    fitBegan = Signal(str)      # a fit run started; payload = command header line
    fitLog   = Signal(str)      # one line of fitter output -> Run tab Output log
    fitAllDone = Signal(int)    # fit-all finished; payload = return code
    status   = Signal(str)      # status text -> shared main status bar (no per-panel label)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mat_dir = None       # input/mat directory
        self._fit_dir = None       # input/fit directory
        self._sctr_dir = None      # input/sctr directory (single scattering coeff per material)
        self._proj_root = None
        from PySide6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)

        # ── interactive alpha-curve editor (replaces the matplotlib plot) ──
        self._curve = AlphaCurveEditor()
        self._curve.valuesChanged.connect(self._on_curve_changed)
        lay.addWidget(self._curve)

        # ── table ──
        cols = ["Layer"] + [str(b) for b in OCTAVE_BANDS] + ["Scatter"]
        self._tbl = QTableWidget(0, len(cols))
        self._tbl.setHorizontalHeaderLabels(cols)
        for c, b in enumerate(OCTAVE_BANDS, start=1):   # unit on hover
            it = self._tbl.horizontalHeaderItem(c)
            if it is not None:
                it.setToolTip(f"{b} Hz")
        _sit = self._tbl.horizontalHeaderItem(len(OCTAVE_BANDS) + 1)   # trailing Scatter column
        if _sit is not None:
            _sit.setToolTip("single scattering coefficient (0..1) — expanded per octave band by convention in the solver")
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._tbl.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        hh = self._tbl.horizontalHeader()
        # The left panel is widened to 540 px to fit the extra Scatter column. The
        # band centre frequency shows as the number only (unit in the header tooltip),
        # so eight numbers render in full. Layer is fixed+narrow (names elide with the
        # full name on hover); the 8 band columns stretch to share the middle; and the
        # Scatter column is fixed wide enough to read the word + value. No scrollbar as
        # long as usable width >= 90 + 8*44 + Scatter (~500 px).
        self._tbl.setTextElideMode(Qt.TextElideMode.ElideRight)
        hh.setMinimumSectionSize(44)        # fits the widest band number ("8000")
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        for c in range(1, len(OCTAVE_BANDS) + 1):   # band columns stretch to share the width
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
        sctr_c = len(OCTAVE_BANDS) + 1              # Scatter: fixed, just wide enough
        hh.setSectionResizeMode(sctr_c, QHeaderView.ResizeMode.Fixed)
        hh.resizeSection(sctr_c, 50)
        hh.resizeSection(0, 90)
        # Layer + Scatter skip the row-selection highlight; the alpha and Scatter
        # numbers are centred (Layer name stays left-aligned).
        self._tbl.setItemDelegateForColumn(0, _CellDelegate(self._tbl, no_select=True, tint=PAL['panel']))
        for c in range(1, len(OCTAVE_BANDS) + 1):
            self._tbl.setItemDelegateForColumn(c, _CellDelegate(self._tbl, center=True))
        self._tbl.setItemDelegateForColumn(sctr_c, _CellDelegate(self._tbl, no_select=True, center=True))
        self._tbl.setStyleSheet(f"""
            QTableWidget {{background:{PAL['bg']};color:{PAL['text']};
                gridline-color:{PAL['border']};border:1px solid {PAL['border']};}}
            QTableWidget::item:selected:!active {{background:{PAL['bg']};color:{PAL['text']};}}
            QHeaderView::section {{background:{PAL['panel']};color:{PAL['subtext']};
                border:1px solid {PAL['border']};font-size:11px;}}
        """)
        self._tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self._tbl.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._tbl.itemSelectionChanged.connect(self._on_row_selected)
        self._undo_stack = []      # Ctrl+Z: full-table snapshots (small table, cheap)
        self._prev = None          # last settled state, pushed before the next change
        self._tbl.itemChanged.connect(self._on_item_changed)
        # RMB on a layer row: optional predefined-material lookup. Existing
        # manual typing, paste, plotting, save and fit behaviour stays unchanged.
        self._tbl.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tbl.customContextMenuRequested.connect(self._show_material_menu)
        from PySide6.QtWidgets import QSizePolicy
        self._tbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Expanding)
        self._tbl.setMinimumHeight(140)   # table fills remaining space
        lay.addWidget(self._tbl, 1)

        # ── save-all button (placed on the hard/soft row below) ──
        self._save_btn = QPushButton("🖫\uFE0E  Save all .mat")
        self._save_btn.setFixedHeight(24)
        self._save_btn.clicked.connect(self.save_all)
        # The fitting buttons moved to the Run tab ("Fit impedances"); the button
        # objects are kept (hidden, not added to any layout) because run_fit_all /
        # run_fit_selected / _on_fit_all_done enable/disable them as busy flags.
        self._fit_btn = QPushButton("⚙  Run vectFIT");  self._fit_btn.setVisible(False)
        self._fit_btn.clicked.connect(self.run_fit_selected)
        self._fit_all_btn = QPushButton("⚙  Fit all");  self._fit_all_btn.setVisible(False)
        self._fit_all_btn.clicked.connect(self.run_fit_all)

        # ── hard/soft toggle for the selected layer + Save all .mat, one row ──
        hs = QHBoxLayout()
        self._soft_btn = ToggleSwitch()          # small on/off slider switch
        self._soft_btn.toggled.connect(self._on_soft_toggled)
        hs.addWidget(self._soft_btn)
        self._soft_lbl = QLabel("Hard (R > 0)")
        self._soft_lbl.setStyleSheet(f"color:{PAL['text']};font-size:11px;")
        hs.addWidget(self._soft_lbl)
        hs.addStretch(1)
        hs.addWidget(self._save_btn)             # Save all .mat, right of the toggle
        lay.addLayout(hs)

    # ── curve editor bridge ──
    def _clear_plot(self):
        self._curve.clear()

    def _plot_row(self, name, vals):
        self._curve.set_values(name, vals)

    def _on_curve_changed(self, vals):
        """Editor -> table: write dragged values into the selected row's cells."""
        r = self._current_row()
        if r < 0 or self._tbl.item(r, 0) is None:
            return
        self._tbl.blockSignals(True)
        for c, v in enumerate(vals):
            self._tbl.setItem(r, c + 1, QTableWidgetItem(f"{v:.2f}"))
        self._tbl.blockSignals(False)

    # ── predefined material catalogue (RMB on material table row) ──
    def _show_material_menu(self, pos):
        """Right-click row menu: optionally copy values from materialsREF.txt."""
        item = self._tbl.itemAt(pos)
        if item is None:
            return
        row = item.row()
        if row < 0 or self._tbl.item(row, 0) is None:
            return

        self._tbl.setCurrentCell(row, max(item.column(), 0))
        self._tbl.selectRow(row)

        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{ background:{PAL['panel']}; color:{PAL['text']};
                     border:1px solid {PAL['border']}; }}
            QMenu::item {{ padding:5px 18px; border-radius:3px; }}
            QMenu::item:selected {{ background:{PAL['accent']}; color:{PAL['bg']}; }}
        """)
        choose = menu.addAction("\u26C1  Choose material from predefined list…")

        # Fill actions on a numeric cell: copy this value across rows / bands.
        col = item.column()
        bands = len(OCTAVE_BANDS)
        fills = {}
        if 1 <= col <= bands + 1:                     # alpha cols or the Scatter col
            menu.addSeparator()
            fills["rows_above"] = menu.addAction("Apply value to rows above")
            fills["rows_below"] = menu.addAction("Apply value to rows below")
            if 1 <= col <= bands:                     # column fills only across alpha bands
                fills["cols_behind"] = menu.addAction("Apply value to columns behind")
                fills["cols_after"]  = menu.addAction("Apply value to columns after")

        chosen = menu.exec(self._tbl.viewport().mapToGlobal(pos))
        if chosen == choose:
            self._select_material_from_ref(row)
        else:
            for mode, act in fills.items():
                if chosen == act:
                    self._apply_fill(row, col, mode)
                    break

    def _read_materials_ref(self):
        """Return [(file_name, display_name, values)] parsed from MATERIALS_FILE.

        Expected text format per non-comment line:
            rough_concrete 0.01 0.01 0.02 0.04 0.06 0.08 0.10 0.10
        Extra columns are ignored; malformed/incomplete lines are skipped.
        """
        materials = []
        with open(MATERIALS_FILE, encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.split("#")[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 1 + len(OCTAVE_BANDS):
                    continue
                file_name = parts[0]
                try:
                    vals = [float(x) for x in parts[1:1 + len(OCTAVE_BANDS)]]
                except ValueError:
                    continue
                # Absorption coefficients are alpha values; clamp accidental
                # out-of-range catalogue entries instead of poisoning the table.
                vals = [min(1.0, max(0.0, v)) for v in vals]
                display_name = file_name.replace("_", " ")
                materials.append((file_name, display_name, vals))
        return materials

    def _select_material_from_ref(self, row):
        """Show a readable selector window and apply the chosen alpha values."""
        layer_item = self._tbl.item(row, 0)
        if layer_item is None:
            return
        layer_name = layer_item.text()

        if not os.path.isfile(MATERIALS_FILE):
            QMessageBox.warning(
                self, "Predefined material list not found",
                "The predefined material catalogue was not found.\n\n"
                "Create it with one material per line, for example:\n"
                "rough_concrete 0.01 0.01 0.02 0.04 0.06 0.08 0.10 0.10")
            self.status.emit("Missing predefined material catalogue.")
            return

        try:
            materials = self._read_materials_ref()
        except Exception:
            QMessageBox.critical(
                self, "Could not read material list",
                "Could not read the predefined material catalogue. Check that it exists and is readable.")
            self.status.emit("Could not read predefined material catalogue.")
            return

        if not materials:
            QMessageBox.information(
                self, "No predefined materials",
                "No valid materials were found. Each non-comment line must be:\n\n"
                "material_name  a63  a125  a250  a500  a1000  a2000  a4000  a8000")
            self.status.emit("No valid predefined materials found.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Choose predefined material for {layer_name}")
        # Narrow enough that the list ends just after the compact curve column,
        # but with enough room to keep material names readable.
        dlg.resize(550, 480)
        dlg.setStyleSheet(f"""
            QDialog {{ background:{PAL['bg']}; color:{PAL['text']}; }}
            QLabel {{ color:{PAL['text']}; }}
            QLineEdit {{ background:{PAL['panel']}; color:{PAL['text']};
                         border:1px solid {PAL['border']}; padding:5px; }}
            QListWidget {{ background:{PAL['panel']}; color:{PAL['text']};
                           border:1px solid {PAL['border']}; }}
            QListWidget::item {{ padding:5px; }}
            QListWidget::item:selected {{ background:{PAL['accent']}; color:{PAL['bg']}; }}
            QPushButton {{ background:{PAL['btn']}; color:{PAL['text']};
                           border:1px solid {PAL['border']}; padding:6px 10px;
                           border-radius:4px; }}
            QPushButton:hover {{ background:{PAL['btn_hover']}; }}
        """)

        v = QVBoxLayout(dlg)
        title = QLabel(f"Select predefined absorption values for layer: {layer_name}")
        title.setStyleSheet(f"color:{PAL['accent2']}; font-weight:bold;")
        v.addWidget(title)
        filt = QLineEdit()
        filt.setPlaceholderText("Filter materials…")
        v.addWidget(filt)

        lst = QListWidget()
        lst.setAlternatingRowColors(True)
        lst.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        v.addWidget(lst, 1)

        for file_name, display_name, vals in materials:
            item = QListWidgetItem()
            item.setSizeHint(QSize(510, 38))
            item.setToolTip(display_name)
            item.setData(Qt.ItemDataRole.UserRole, (file_name, display_name, vals))
            lst.addItem(item)
            lst.setItemWidget(item, MaterialRefPreview(display_name, vals, lst))

        def refresh_preview_selection():
            for i in range(lst.count()):
                it = lst.item(i)
                widget = lst.itemWidget(it)
                if widget is not None:
                    widget.set_selected(it.isSelected())

        lst.itemSelectionChanged.connect(refresh_preview_selection)
        if lst.count():
            lst.setCurrentRow(0)
            refresh_preview_selection()

        def apply_filter(text):
            q = text.strip().lower()
            first_visible = -1
            for i in range(lst.count()):
                it = lst.item(i)
                file_name, display_name, vals = it.data(Qt.ItemDataRole.UserRole)
                hay = f"{file_name} {display_name}".lower()
                hide = bool(q and q not in hay)
                it.setHidden(hide)
                if not hide and first_visible < 0:
                    first_visible = i
            if first_visible >= 0:
                lst.setCurrentRow(first_visible)
            refresh_preview_selection()

        filt.textChanged.connect(apply_filter)
        lst.itemDoubleClicked.connect(lambda *_: dlg.accept())

        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel = QPushButton("Cancel")
        ok = QPushButton("Apply material")
        ok.setDefault(True)
        cancel.clicked.connect(dlg.reject)
        ok.clicked.connect(dlg.accept)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        v.addLayout(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = lst.currentItem()
        if chosen is None or chosen.isHidden():
            self.status.emit("No predefined material selected.")
            return

        file_name, display_name, vals = chosen.data(Qt.ItemDataRole.UserRole)
        self._apply_material_values(row, vals)
        self.status.emit(
            f"Applied predefined material '{display_name}' to layer '{layer_name}' "
            "(remember to Save all .mat).")

    def _apply_material_values(self, row, vals):
        """Copy alpha values to a row without changing the layer name."""
        self._tbl.blockSignals(True)
        for c, v in enumerate(vals[:len(OCTAVE_BANDS)], start=1):
            self._tbl.setItem(row, c, QTableWidgetItem(f"{float(v):.2f}"))
        self._tbl.blockSignals(False)
        self._tbl.selectRow(row)
        self._on_row_selected()

    # ── population ──
    def set_project(self, proj_root):
        self._proj_root = proj_root
        if proj_root:
            self._mat_dir = os.path.join(proj_root, "input", "mat")
            self._fit_dir = os.path.join(proj_root, "input", "fit")
            self._sctr_dir = os.path.join(proj_root, "input", "sctr")

    def populate(self, layer_names):
        """Fill table rows from layer names, loading existing .mat values."""
        self._tbl.blockSignals(True)
        self._tbl.setRowCount(0)
        for name in layer_names:
            if _is_bbox(name):
                continue
            vals, soft = self._load_mat_values(name)
            r = self._tbl.rowCount(); self._tbl.insertRow(r)
            name_item = QTableWidgetItem(name)
            name_item.setToolTip(name)   # full name on hover (cell may be elided)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._tbl.setItem(r, 0, name_item)
            for c, v in enumerate(vals):
                self._tbl.setItem(r, c+1, QTableWidgetItem(f"{v:.2f}"))
            self._tbl.setItem(r, len(OCTAVE_BANDS)+1,
                              QTableWidgetItem(f"{self._load_sctr_value(name):.2f}"))  # Scatter col
            self._set_row_soft(r, soft)   # after the cells exist
        self._tbl.blockSignals(False)
        # always land on a selected material so value edits attribute to a row
        # (an unselected table silently dropped edits -> stale-check missed them).
        if self._tbl.rowCount() > 0:
            self._tbl.selectRow(0)
        else:
            self._clear_plot()
        self._undo_stack.clear()          # fresh table -> fresh undo history
        self._prev = self._snapshot()

    # ── ALL.mat : central hard/soft registry ─────────────────
    # One file, one line per material: "<name>  <flag>" (0=hard, 1=soft).
    # The hard/soft flag is kept OUT of the per-material .mat files (which stay
    # 8 alpha values only, so every alpha-only reader keeps working) and lives
    # here instead, read by both the GUI and vectFIT.
    def _allmat_path(self):
        return os.path.join(self._mat_dir, "ALL.mat") if self._mat_dir else None

    def _read_all_soft(self):
        """Return {name: bool soft} parsed from ALL.mat (missing -> empty)."""
        d = {}
        p = self._allmat_path()
        if p and os.path.isfile(p):
            try:
                with open(p) as f:
                    for line in f:
                        line = line.split("#")[0].strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            d[parts[0]] = (parts[1] != "0")
            except Exception:
                pass
        return d

    def _write_all_soft(self):
        """Rewrite ALL.mat from the current table (one line per layer row)."""
        p = self._allmat_path()
        if not p:
            return
        os.makedirs(self._mat_dir, exist_ok=True)
        with open(p, "w") as f:
            f.write("# material hard/soft registry  —  <name>  <flag>  (0=hard, 1=soft)\n")
            for r in range(self._tbl.rowCount()):
                item = self._tbl.item(r, 0)
                if item is None:
                    continue
                f.write(f"{item.text()}  {1 if self._row_soft(r) else 0}\n")

    def _load_mat_values(self, name):
        """Return (alphas[8], soft). Alphas come from the per-material .mat
        (8 values, no trailing flag); soft comes from ALL.mat."""
        vals = [0.10] * len(OCTAVE_BANDS)
        if self._mat_dir:
            p = os.path.join(self._mat_dir, name + ".mat")
            if os.path.isfile(p):
                try:
                    got = []
                    with open(p) as f:
                        for line in f:
                            line = line.split("#")[0].strip()
                            if line: got += [float(x) for x in line.split()]
                    for i in range(min(len(got), len(OCTAVE_BANDS))):
                        vals[i] = got[i]
                except Exception:
                    pass
        soft = self._read_all_soft().get(name, False)
        return vals, soft

    # ── scattering (single coeff per material, sibling to .mat) ──
    # Kept in its OWN input/sctr/<name>.sctr file (one value) so the .mat files stay
    # exactly 8 alphas and ism_load_impedances keeps working untouched. The solver
    # expands this single number to a per-octave-band curve by convention.
    def _load_sctr_value(self, name):
        d = self._sctr_dir
        if d:
            p = os.path.join(d, name + ".sctr")
            if os.path.isfile(p):
                try:
                    with open(p) as f:
                        for line in f:
                            line = line.split("#")[0].strip()
                            if line:
                                return min(1.0, max(0.0, float(line.split()[0])))
                except Exception:
                    pass
        return 0.01   # default scattering when none saved yet

    def _sctr_value(self, r):
        item = self._tbl.item(r, len(OCTAVE_BANDS) + 1)
        try:
            return min(1.0, max(0.0, float(item.text()))) if item else 0.01
        except ValueError:
            return 0.01

    def _write_sctr(self, name, r):
        d = self._sctr_dir
        if not d:
            return
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name + ".sctr"), "w") as f:
            f.write("# single scattering coefficient (0..1); expanded per octave band in the solver\n")
            f.write(f"{self._sctr_value(r):.4f}\n")

    # ── selection → plot ──
    def _current_row(self):
        sel = self._tbl.selectionModel().selectedRows()
        return sel[0].row() if sel else self._tbl.currentRow()

    def _replot_current(self):
        r = self._current_row()
        if r < 0 or self._tbl.item(r, 0) is None:
            self._clear_plot(); return None
        name = self._tbl.item(r, 0).text()
        self._plot_row(name, self._row_values(r))
        return name

    def _on_row_selected(self):
        # selection changed: replot, sync the hard/soft toggle, notify viewport
        r = self._current_row()
        if r < 0 and self._tbl.rowCount() > 0 and not getattr(self, "_reselecting", False):
            # a stray clear left no row selected -> reselect so edits keep a target
            self._reselecting = True
            self._tbl.selectRow(min(max(getattr(self, "_last_row", 0), 0),
                                    self._tbl.rowCount() - 1))
            self._reselecting = False
            return
        name = self._replot_current()
        r = self._current_row()
        if r >= 0:
            self._last_row = r
            self._soft_btn.blockSignals(True)
            soft = self._row_soft(r)
            self._soft_btn.setChecked(soft)
            self._soft_lbl.setText("Soft (R < 0)" if soft else "Hard (R > 0)")
            self._soft_btn.blockSignals(False)
        if name:
            self.rowSelected.emit(name)

    def _on_soft_toggled(self, checked):
        r = self._current_row()
        if r < 0 or self._tbl.item(r, 0) is None:
            self._soft_btn.blockSignals(True)
            self._soft_btn.setChecked(False)
            self._soft_btn.blockSignals(False)
            self.status.emit("Select a layer row first."); return
        self._set_row_soft(r, checked)
        self._soft_lbl.setText("Soft (R < 0)" if checked else "Hard (R > 0)")
        name = self._tbl.item(r, 0).text()
        self.status.emit(f"{name}: {'soft' if checked else 'hard'} "
                             "(remember to Save all .mat)")

    def _on_edit(self):
        # a value changed: only replot the curve, no reselection signal
        self._replot_current()

    # ── Ctrl+Z undo (full-table snapshots) ──
    def _snapshot(self):
        """Text of every cell + per-row soft flag. Small table -> whole-state snapshot."""
        rows = []
        for r in range(self._tbl.rowCount()):
            cells = [self._tbl.item(r, c).text() if self._tbl.item(r, c) else ""
                     for c in range(self._tbl.columnCount())]
            rows.append((cells, self._row_soft(r)))
        return rows

    def _push_undo(self):
        if self._prev is not None:
            self._undo_stack.append(self._prev)
            if len(self._undo_stack) > 50:
                self._undo_stack.pop(0)
        self._prev = self._snapshot()

    def _on_item_changed(self, *_):
        # a single manual/programmatic cell edit: record pre-edit state, then replot
        self._push_undo()
        self._on_edit()

    def _restore(self, snap):
        self._tbl.blockSignals(True)
        for r, (cells, soft) in enumerate(snap):
            if r >= self._tbl.rowCount():
                break
            for c, txt in enumerate(cells):
                it = self._tbl.item(r, c)
                if it is None:
                    self._tbl.setItem(r, c, QTableWidgetItem(txt))
                elif it.text() != txt:
                    it.setText(txt)
            self._set_row_soft(r, soft)
        self._tbl.blockSignals(False)
        self._prev = self._snapshot()
        self._replot_current()

    def _undo(self):
        if not self._undo_stack:
            return
        self._restore(self._undo_stack.pop())

    def _apply_fill(self, r, c, mode):
        """RMB fills: copy cell (r,c) value to rows above/below (same col) or
        columns behind/after (same row, alpha cols only). One undo step."""
        src = self._tbl.item(r, c)
        if src is None:
            return
        val = src.text()
        try:
            float(val)
        except ValueError:
            return
        bands = len(OCTAVE_BANDS)
        if mode == "rows_above":    rows, cols = range(0, r), [c]
        elif mode == "rows_below":  rows, cols = range(r + 1, self._tbl.rowCount()), [c]
        elif mode == "cols_behind": rows, cols = [r], range(1, c)
        elif mode == "cols_after":  rows, cols = [r], range(c + 1, bands + 1)
        else:
            return
        self._push_undo()
        self._tbl.blockSignals(True)
        for rr in rows:
            if self._tbl.item(rr, 0) is None:
                continue
            for cc in cols:
                self._tbl.setItem(rr, cc, QTableWidgetItem(val))
        self._tbl.blockSignals(False)
        self._prev = self._snapshot()
        self._replot_current()

    def _row_values(self, r):
        vals = []
        for c in range(1, len(OCTAVE_BANDS) + 1):
            item = self._tbl.item(r, c)
            try: vals.append(float(item.text()) if item else 0.0)
            except ValueError: vals.append(0.0)
        return vals

    # ── hard / soft flag (stored on the layer-name item) ──
    def _row_soft(self, r):
        item = self._tbl.item(r, 0)
        return bool(item.data(Qt.ItemDataRole.UserRole)) if item else False

    def _set_row_soft(self, r, soft):
        """Set the flag and a subtle visual cue: soft layer names show in the
        'warn' accent colour with a tooltip suffix; hard names stay default."""
        item = self._tbl.item(r, 0)
        if item is None:
            return
        item.setData(Qt.ItemDataRole.UserRole, 1 if soft else 0)
        if soft:
            item.setForeground(QColor(PAL["warn"]))
            item.setToolTip(f"{item.text()} — soft (R < 0)")
        else:
            item.setForeground(QColor(PAL["text"]))
            item.setToolTip(item.text())

    # ── Ctrl+V paste (Excel / spaces / tabs) ──
    def keyPressEvent(self, e):
        if e.matches(QKeySequence.StandardKey.Copy):
            self._copy_selection()
            e.accept(); return
        if e.matches(QKeySequence.StandardKey.Paste):
            self._paste_clipboard()
            e.accept(); return
        if e.matches(QKeySequence.StandardKey.Undo):
            self._undo()
            e.accept(); return
        super().keyPressEvent(e)

    def _copy_selection(self):
        _tsv_copy_selection(self._tbl)

    def _paste_clipboard(self):
        text = QApplication.clipboard().text()
        if not text.strip():
            return
        start_row = max(self._tbl.currentRow(), 0)
        start_col = max(self._tbl.currentColumn(), 1)  # never overwrite name col by default
        self._push_undo()                              # paste = one undo step (signals blocked below)
        self._tbl.blockSignals(True)
        for dr, line in enumerate(text.splitlines()):
            if not line.strip():
                continue
            # split on tab OR any whitespace
            parts = line.split("\t") if "\t" in line else line.split()
            rr = start_row + dr
            if rr >= self._tbl.rowCount():
                break
            for dc, val in enumerate(parts):
                cc = start_col + dc
                if cc > len(OCTAVE_BANDS):
                    break
                val = val.strip()
                try:
                    float(val)
                except ValueError:
                    continue
                self._tbl.setItem(rr, cc, QTableWidgetItem(val))
        self._tbl.blockSignals(False)
        self._prev = self._snapshot()                  # post-paste is the new settled state
        self._on_row_selected()
        self.status.emit("Pasted values from clipboard.")

    # ── save / fit ──
    def save_selected(self):
        r = self._tbl.currentRow()
        if r < 0:
            self.status.emit("Select a layer row first."); return
        if not self._mat_dir:
            self.status.emit("Open a project folder first."); return
        name = self._tbl.item(r, 0).text()
        vals = self._row_values(r)
        os.makedirs(self._mat_dir, exist_ok=True)
        p = os.path.join(self._mat_dir, name + ".mat")
        with open(p, "w") as f:
            f.write(f"# alpha per octave band: {OCTAVE_BANDS}\n")
            f.write("  ".join(f"{v:.4f}" for v in vals) + "\n")
        self._write_sctr(name, r)   # single scattering coeff -> input/sctr/<name>.sctr
        self._write_all_soft()      # hard/soft registry lives in ALL.mat
        self.status.emit(f"Saved → {name}.mat")

    def save_all(self):
        """Write a .mat file for EVERY layer row in one pass, independent of the
        current selection (does not move the selected row or change the plot)."""
        if not self._mat_dir:
            self.status.emit("Open a project folder first."); return
        if self._tbl.rowCount() == 0:
            self.status.emit("No materials to save."); return
        os.makedirs(self._mat_dir, exist_ok=True)
        n = 0
        for r in range(self._tbl.rowCount()):
            item = self._tbl.item(r, 0)
            if item is None:
                continue
            name = item.text()
            vals = self._row_values(r)
            p = os.path.join(self._mat_dir, name + ".mat")
            with open(p, "w") as f:
                f.write(f"# alpha per octave band: {OCTAVE_BANDS}\n")
                f.write("  ".join(f"{v:.4f}" for v in vals) + "\n")
            self._write_sctr(name, r)   # single scattering coeff -> input/sctr/<name>.sctr
            n += 1
        self._write_all_soft()      # one central hard/soft registry
        self.status.emit(f"Saved {n} material file(s) → {self._mat_dir}")

    def prune_orphans(self):
        """Delete input/mat|fit|sctr files whose material name is no longer in the table, so renamed or
        removed OBJ groups don't leave files forever. Keeps ALL.mat and any BoundingBox.*. NO-OP on an
        empty table (means the OBJ isn't loaded yet — don't wipe everything)."""
        if not self._mat_dir or self._tbl.rowCount() == 0:
            return
        keep = {self._tbl.item(r, 0).text() for r in range(self._tbl.rowCount()) if self._tbl.item(r, 0)}
        removed = 0
        for d, ext in ((self._mat_dir, ".mat"), (self._fit_dir, ".fit"), (self._sctr_dir, ".sctr")):
            if not d:
                continue
            for f in glob.glob(os.path.join(d, "*" + ext)):
                base = os.path.basename(f); stem = base[:-len(ext)]
                if base == "ALL.mat" or _is_bbox(stem) or stem in keep:
                    continue
                try:
                    os.remove(f); removed += 1
                except OSError:
                    pass
        if removed:
            self.status.emit(f"Pruned {removed} orphaned material file(s).")

    def stale_fits(self, config_path):
        """Material names whose current values no longer match their .fit — recomputes
        the same signature fminconFIT stamps as '# sig <hex>' (alphas + soft + the config
        values that change the fit) and compares. Missing .fit = stale."""
        import hashlib
        if not (self._mat_dir and self._fit_dir) or not self._tbl.rowCount():
            return []
        try:
            # config.txt = one value per line with a trailing '# comment' -> strip it,
            # take the first token per line (same as _load_config), else float() chokes.
            v = [float(l.split("#")[0].split()[0])
                 for l in open(config_path) if l.split("#")[0].strip()]
            csnd = (1.4 * 287.05 * v[3]) ** 0.5     # matches fitter c0 (Celsius, no +273.15)
            dt   = v[4] * v[0] / csnd
            tail = f"|dt={dt:.10g}|fc={v[2]:.10g}|N={int(v[5])}"
        except Exception:
            return []                      # no config -> can't judge, don't block
        soft = self._read_all_soft()
        stale = []
        for r in range(self._tbl.rowCount()):
            it = self._tbl.item(r, 0)
            if it is None or _is_bbox(it.text()):   # reserved shell never fitted
                continue
            name = it.text()
            # hash the .mat (what the fitter fits, :.4f) not the table (:.2f display) -> match
            mp = os.path.join(self._mat_dir, name + ".mat")
            try:
                with open(mp) as f:
                    nums = [x for ln in f if not ln.lstrip().startswith("#") for x in ln.split()]
                alphas = [float(x) for x in nums[:8]]
            except Exception:
                stale.append(name); continue        # no .mat -> must fit
            a = "|".join(f"{round(v,8):.8g}" for v in alphas)
            want = hashlib.sha1((a + f"|soft={int(bool(soft.get(name,False)))}" + tail)
                                .encode()).hexdigest()[:16]
            fit = os.path.join(self._fit_dir, name + ".fit")
            got = next((l.split()[-1] for l in open(fit)
                        if l.lower().startswith("# sig")), None) if os.path.exists(fit) else None
            if got != want:
                stale.append(name)
        return stale

    def run_fit_selected(self):
        r = self._tbl.currentRow()
        if r < 0:
            self.status.emit("Select a layer row first."); return
        if not self._proj_root:
            self.status.emit("Open a project folder first."); return
        self.save_selected()
        name = self._tbl.item(r, 0).text()
        # The fitting scripts live NEXT TO gui.py (both are in common/), so we
        # locate them relative to THIS file — never relative to the project,
        # which can live anywhere on disk.
        script = self._find_fit_script()
        if script is None:
            self.status.emit("vectFIT.py / fminconFIT.py not found next to gui.py"); return
        os.makedirs(self._fit_dir, exist_ok=True)
        mat = os.path.join(self._mat_dir, name + ".mat")
        out = os.path.join(self._fit_dir, name + ".fit")
        # Explicit project paths; --no-plot because output is captured (the
        # script's plt.show() would block the subprocess otherwise).
        proj = self._proj_root
        cmd = [sys.executable, script,
               "--mat",    mat,
               "--out",    out,
               "--config", os.path.join(proj, "config", "config.txt"),
               "--poles",  os.path.join(proj, "config", "poles.txt"),
               "--no-plot"]
        self.status.emit(f"Running {os.path.basename(script)} on {name}…")
        self.fitBegan.emit("$ " + " ".join(cmd))
        QApplication.processEvents()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  cwd=proj, timeout=600)
            out = (proc.stdout or "") + (proc.stderr or "")
            for ln in out.splitlines():
                self.fitLog.emit(ln)
            if proc.returncode == 0:
                self.status.emit(f"Fit done → {name}.fit")
            else:
                msg = out.strip().splitlines()
                tail = msg[-1] if msg else "unknown error"
                self.status.emit(f"Fit failed (rc={proc.returncode}): {tail[:80]}")
        except subprocess.TimeoutExpired:
            self.fitLog.emit("[!] Fit timed out (600 s).")
            self.status.emit("Fit timed out (600 s).")
        except Exception as ex:
            self.fitLog.emit(f"[!] Fit error: {ex}")
            self.status.emit(f"Fit error: {ex}")

    def _find_fit_script(self):
        """The fitting scripts live NEXT TO gui.py (both in common/)."""
        here = os.path.dirname(os.path.abspath(__file__))
        for sn in ("fminconFIT.py", "vectFIT.py"):
            cand = os.path.join(here, sn)
            if os.path.isfile(cand):
                return cand
        return None

    def run_fit_all(self):
        """Fit EVERY layer's material in one vectFIT run. The script shows a
        single summary figure (grid of subplots, one per material), so this is
        run in a background thread: the GUI stays responsive while the user
        inspects and then closes the plot window."""
        if not self._proj_root:
            self.status.emit("Open a project folder first."); return
        if self._tbl.rowCount() == 0:
            self.status.emit("No materials to fit."); return
        script = self._find_fit_script()
        if script is None:
            self.status.emit("vectFIT.py / fminconFIT.py not found next to gui.py"); return
        # Compatibility probe: multi-material fitting needs the updated script.
        # gui.py and vectFIT.py ship TOGETHER — if the copy next to gui.py is an
        # older one, say so explicitly instead of failing with an argparse error.
        try:
            if "--fitdir" not in open(script, encoding="utf-8", errors="ignore").read():
                self.status.emit(
                    f"{os.path.basename(script)} is outdated (no --fitdir support) — "
                    "replace it with the updated version."); return
        except OSError:
            pass
        self.save_all()                      # ensure every .mat is on disk
        os.makedirs(self._fit_dir, exist_ok=True)
        mats = [os.path.join(self._mat_dir, self._tbl.item(r, 0).text() + ".mat")
                for r in range(self._tbl.rowCount()) if self._tbl.item(r, 0)]
        proj = self._proj_root
        cmd = [sys.executable, script,
               "--mat", *mats,
               "--fitdir", self._fit_dir,
               "--config", os.path.join(proj, "config", "config.txt"),
               "--poles",  os.path.join(proj, "config", "poles.txt"),
               # Fit-all covers the WHOLE scene, so poles.txt must reflect THIS
               # material set exactly — not a stale running max from an earlier
               # scene. --refit clears poles.txt first so the count is rebuilt
               # authoritatively (and can shrink, e.g. an all-rigid scene -> 0 0,
               # which the solver floors to 1 1). Single-material run_fit_selected
               # deliberately omits --refit so it merges instead of clobbering.
               "--refit"]
        # NOTE: no --no-plot — the summary grid is the point. The figure opens
        # in its own window; the run finishes when the user closes it.
        self._fit_all_btn.setEnabled(False)
        self._fit_btn.setEnabled(False)
        self.status.emit(f"Fitting {len(mats)} material(s)… close the plot window to finish.")
        self.fitBegan.emit("$ " + " ".join(cmd))
        self._fit_thread = _FitAllThread(cmd, proj)
        self._fit_thread.line.connect(self.fitLog.emit)
        self._fit_thread.done.connect(self._on_fit_all_done)
        self._fit_thread.start()

    def _on_fit_all_done(self, rc, tail):
        self._fit_all_btn.setEnabled(True)
        self._fit_btn.setEnabled(True)
        if rc == 0:
            self.status.emit(f"Fit all done → {self._fit_dir}")
        else:
            self.status.emit(f"Fit all failed (rc={rc}): {tail[:160]}")
        self.fitAllDone.emit(rc)


# ─────────────────────────────────────────────────────────────
#  Point table  (sources / receivers)
# ─────────────────────────────────────────────────────────────
def make_pm_icon(sign, size=18, color=None):
    """Draw a crisp '+' or '−' as a QIcon so the symbol renders at any size and
    never depends on font glyph coverage (avoids the tofu boxes). The button
    text ('Add' / 'Remove') stays at the normal size next to it."""
    from PySide6.QtCore import QSize
    col = QColor(color) if color else QColor(PAL["text"])
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(col); pen.setWidthF(2.4); pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    m = size * 0.22                       # margin
    cy = size / 2.0
    p.drawLine(int(m), int(cy), int(size - m), int(cy))      # horizontal bar
    if sign.startswith("+"):     # '+' may carry a U+FE0E suffix; match loosely
        cx = size / 2.0
        p.drawLine(int(cx), int(m), int(cx), int(size - m))  # vertical bar
    p.end()
    return QIcon(pm)


def make_pulse_icon(kind, w=46, h=28, color=None):
    """Tiny didactic curve of each source pulse shape (just the waveform, no axis), for
    the source-type radio buttons. kind: 'pcs' | 'ricker' | 'sinusoid'."""
    col = QColor(color) if color else QColor(PAL["text"])
    pm = QPixmap(w, h)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    mid = h / 2.0
    # sample the analytical shape over x in [-3,3]
    N = 48
    ys = []
    for i in range(N):
        s = i / (N - 1)                                           # 0..1 across the icon
        x = -3.0 + 6.0 * s
        if kind == "sinusoid":
            y = math.sin(math.pi * x)                              # steady sine (3 cycles)
        elif kind == "ricker":
            y = (1.0 - x * x) * math.exp(-0.5 * x * x)             # mexican hat: peak + 2 dips
        else:  # pcs: causal damped oscillation -> +peak, smaller -peak, tiny +peak, flat
            s0 = 0.12                                              # short flat lead-in
            if s < s0:
                y = 0.0
            else:
                tau = (s - s0) / (1.0 - s0)
                y = math.exp(-3.2 * tau) * math.sin(2.0 * math.pi * 1.35 * tau)
        ys.append(y)
    ymax = max(abs(v) for v in ys) or 1.0
    amp = (h / 2.0 - 3.0)
    pen = QPen(col); pen.setWidthF(1.6)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin); pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    prev = None
    for i, y in enumerate(ys):
        X = 2.0 + (w - 4.0) * i / (N - 1)
        Y = mid - (y / ymax) * amp
        if prev is not None:
            p.drawLine(QPointF(*prev), QPointF(X, Y))
        prev = (X, Y)
    p.end()
    return QIcon(pm)


def _tsv_copy_selection(tbl):
    """Selected cells -> TSV on the clipboard (blank for unselected within the rect)."""
    rngs = tbl.selectedRanges()
    if not rngs:
        return
    r0 = min(x.topRow() for x in rngs);     r1 = max(x.bottomRow() for x in rngs)
    c0 = min(x.leftColumn() for x in rngs); c1 = max(x.rightColumn() for x in rngs)
    rows = []
    for r in range(r0, r1 + 1):
        rows.append("\t".join(
            tbl.item(r, c).text() if tbl.item(r, c) and tbl.item(r, c).isSelected() else ""
            for c in range(c0, c1 + 1)))
    QApplication.clipboard().setText("\n".join(rows))


class _PointTableWidget(QTableWidget):
    """Name column (0) opens its editor ONLY on a double-click; the numeric columns keep single-click
    editing. Selection/highlight is untouched (that's independent of editing in Qt)."""
    def edit(self, index, trigger, event):
        if (index.isValid() and index.column() == 0
                and trigger != QAbstractItemView.EditTrigger.DoubleClicked):
            return False        # Name cell: block single-click/selected-click edits, allow selection
        return super().edit(index, trigger, event)


class PointTable(QWidget):
    changed = Signal()

    def __init__(self, has_amplitude=False, has_freq=False, has_pose=False, parent=None):
        super().__init__(parent)
        self._has_amp = has_amplitude
        self._has_freq = has_freq         # per-source characteristic frequency f0 (sources only)
        self._has_pose = has_pose         # per-receiver head pose [az,el,roll] deg (receivers only)
        self.poses = []                   # row-matched [az,el,roll]; saved to a .poses sidecar
        self._freq_max = None             # optional clamp on the f0 column (set to f_cutoff)
        # Column 0 is an editable Name (defaults to the row number). The numeric
        # columns follow; get_points()/save_file deliberately skip the name so
        # the solver file format (x y z [p0] [f0]) is unchanged.
        # Unicode subscripts (QTableWidget headers are plain text, no HTML): L_p,1m
        cols = (["Name", "x (m)", "y (m)", "z (m)"]
                + (["Lₚ,₁ₘ (dB)"] if has_amplitude else [])
                + (["f₀ (Hz)"] if has_freq else []))
        self._freq_col = (len(cols) - 1) if has_freq else None   # last column when present
        self._coff = 1            # numeric coordinates start at this column
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self._tbl = _PointTableWidget(0, len(cols))
        self._tbl.setHorizontalHeaderLabels(cols)
        hh = self._tbl.horizontalHeader()
        # Name column a bit wider (interactive), coordinate columns share the rest.
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        for c in range(1, len(cols)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
        self._tbl.setColumnWidth(0, 90)
        self._tbl.setItemDelegateForColumn(0, _CellDelegate(self._tbl, tint=PAL['panel']))  # Name col
        for c in range(self._coff, len(cols)):   # centre the numeric columns (Name stays left)
            self._tbl.setItemDelegateForColumn(c, _CellDelegate(self._tbl, center=True))
        self._tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self._tbl.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._tbl.verticalHeader().setVisible(False)   # drop the left row-number gutter
        # Cells are edited ONLY by a deliberate double-click (or clicking an
        # already-selected cell). Keyboard triggers are excluded so stray
        # keystrokes — e.g. while moving a point in edit mode — can never
        # overwrite coordinates by accident.
        self._tbl.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked |
            QAbstractItemView.EditTrigger.SelectedClicked |
            QAbstractItemView.EditTrigger.CurrentChanged)   # single click -> edit
        self._tbl.setStyleSheet(f"""
            QTableWidget {{background:{PAL['bg']};color:{PAL['text']};
                gridline-color:{PAL['border']};border:1px solid {PAL['border']};}}
            QTableWidget::item:selected:!active {{background:{PAL['bg']};color:{PAL['text']};}}
            QHeaderView::section {{background:{PAL['panel']};color:{PAL['subtext']};
                border:1px solid {PAL['border']};}}
        """)
        self._tbl.itemChanged.connect(self._on_item_changed)
        # Right-click a cell -> fill the same value into rows above/below in
        # that column (handy for setting a common amplitude or coordinate).
        self._tbl.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tbl.customContextMenuRequested.connect(self._show_cell_menu)
        lay.addWidget(self._tbl)
        btn_row = QHBoxLayout()
        # '+' / '−' are DRAWN as icons (always render, larger than the text)
        # while the labels stay at the normal button font size.
        for sign, label, fn in [("+\uFE0E", "Add", self._add),
                                 ("-\uFE0E", "Remove", self._remove)]:
            b = QPushButton(label); b.clicked.connect(fn)
            b.setIcon(make_pm_icon(sign))
            b.setIconSize(QSize(16, 16))
            b.setFixedHeight(24)
            btn_row.addWidget(b)
        lay.addLayout(btn_row)

    # Coordinates are shown/typed in the ORIGINAL OBJ frame; the solver needs the
    # shifted (positive-octant) frame. world = displayed + _off, where _off is the
    # shift gui.py applied on import. Zero by default -> identical to the old behaviour.
    # This exists because main3D converts src/rec with "(int)(val/h)+3" and applies NO
    # -bmin term, so the table MUST hand it shifted coordinates even when the user is
    # reading BRAS's centred-origin numbers.
    _off = (0.0, 0.0, 0.0)

    # Added when WRITING src.txt/rec.txt, subtracted when reading them back.
    # main3D converts with "(int)(val/h)+3", i.e. it assumes the mesh's bmin already
    # sits at zero (the +3 is only the ghost pad) -- but parse_obj_3d shifts geometry
    # by -bmin on EVERY axis, negative or not. So the file must carry (world - bmin).
    # Zero whenever the OBJ was rewritten to start at the origin.
    _solver_off = (0.0, 0.0, 0.0)

    def set_solver_offset(self, off):
        """Offset applied on save/load only -- the table and the 3D view keep the
        loaded OBJ's own coordinates."""
        self._solver_off = tuple(float(v) for v in off)

    def set_origin_offset(self, off):
        """Re-display every row in a new frame. `off` is world-minus-displayed."""
        off = tuple(float(v) for v in off)
        if off == tuple(self._off):
            return
        pts = self.get_points()          # -> world, using the OLD offset
        self._off = off
        self.set_points(pts, emit_changed=False)   # -> display, using the NEW offset
        self.changed.emit()

    def _add(self, x=1.0, y=1.0, z=1.0, amp=1.0, freq=100.0, name=None):
        self._tbl.blockSignals(True)
        r = self._tbl.rowCount(); self._tbl.insertRow(r)
        # column 0: editable name, default to the 1-based row number
        self._tbl.setItem(r, 0, QTableWidgetItem(name if name else str(r + 1)))
        x, y, z = (x - self._off[0], y - self._off[1], z - self._off[2])   # world -> displayed
        vals = [x, y, z] + ([amp] if self._has_amp else []) + ([freq] if self._has_freq else [])
        for c, v in enumerate(vals):
            self._tbl.setItem(r, self._coff + c, QTableWidgetItem(f"{v:.4g}"))
        self._tbl.blockSignals(False)
        if self._has_pose:
            self.poses.insert(r, [0, 0, 0])     # new row -> default head pose
        self.changed.emit()

    def add_point(self, x, y, z, amp=1.0):
        self._add(x, y, z, amp)

    def _show_cell_menu(self, pos):
        """Right-click context menu on a cell: offer to copy this cell's value
        to all rows above or below it, in the same column."""
        item = self._tbl.itemAt(pos)
        if item is None:
            return
        row, col = item.row(), item.column()
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{ background:{PAL['panel']}; color:{PAL['text']};
                     border:1px solid {PAL['border']}; padding:4px; }}
            QMenu::item {{ padding:5px 18px; border-radius:3px; }}
            QMenu::item:selected {{ background:{PAL['accent']}; color:{PAL['bg']}; }}
        """)
        act_above = menu.addAction("Apply value to rows above")
        act_below = menu.addAction("Apply value to rows below")
        chosen = menu.exec(self._tbl.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is act_above:
            self._fill_column(row, col, direction="above")
        elif chosen is act_below:
            self._fill_column(row, col, direction="below")

    def _fill_column(self, row, col, direction):
        """Copy the value at (row, col) into every cell above or below it in the
        same column. Emits one change at the end."""
        src_item = self._tbl.item(row, col)
        if src_item is None:
            return
        value = src_item.text()
        if direction == "above":
            rows = range(0, row)
        else:  # below
            rows = range(row + 1, self._tbl.rowCount())
        self._tbl.blockSignals(True)
        for r in rows:
            self._tbl.setItem(r, col, QTableWidgetItem(value))
        self._tbl.blockSignals(False)
        self.changed.emit()

    def update_point(self, idx, x, y, z):
        """Overwrite the x/y/z of an existing row (used by drag-to-move).
        The name and amplitude columns are left unchanged."""
        if not (0 <= idx < self._tbl.rowCount()):
            return
        self._tbl.blockSignals(True)
        for c, v in enumerate((x - self._off[0], y - self._off[1], z - self._off[2])):
            self._tbl.setItem(idx, self._coff + c, QTableWidgetItem(f"{v:.4g}"))
        self._tbl.blockSignals(False)
        self.changed.emit()

    def _remove(self):
        rows = sorted({i.row() for i in self._tbl.selectedItems()}, reverse=True)
        for r in rows:
            self._tbl.removeRow(r)
            if self._has_pose and r < len(self.poses):    # keep poses row-matched
                del self.poses[r]
        self.changed.emit()

    # ── copy / paste (spreadsheet + whitespace txt, both ways) ──
    def keyPressEvent(self, e):
        if e.matches(QKeySequence.StandardKey.Copy):
            _tsv_copy_selection(self._tbl); e.accept(); return
        if e.matches(QKeySequence.StandardKey.Paste):
            self._paste_clipboard(); e.accept(); return
        super().keyPressEvent(e)

    def _paste_clipboard(self):
        text = QApplication.clipboard().text()
        if not text.strip():
            return
        start_row = max(self._tbl.currentRow(), 0)
        start_col = max(self._tbl.currentColumn(), self._coff)   # never the Name col
        ncol = self._tbl.columnCount()
        # numeric-column defaults for rows added beyond the current count
        defs = [0.0, 0.0, 0.0] + ([1.0] if self._has_amp else []) + ([100.0] if self._has_freq else [])
        self._tbl.blockSignals(True)
        for dr, line in enumerate(l for l in text.splitlines() if l.strip()):
            parts = line.split("\t") if "\t" in line else line.split()   # txt -> whitespace cols
            rr = start_row + dr
            if rr >= self._tbl.rowCount():                    # grow: extra rows -> scene + table
                self._tbl.insertRow(rr)
                self._tbl.setItem(rr, 0, QTableWidgetItem(str(rr + 1)))
                for c, dv in enumerate(defs):
                    self._tbl.setItem(rr, self._coff + c, QTableWidgetItem(f"{dv:.4g}"))
            for dc, val in enumerate(parts):
                cc = start_col + dc
                if cc >= ncol:
                    break
                val = val.strip()
                try: float(val)
                except ValueError: continue                   # names / non-numeric -> unchanged
                self._tbl.setItem(rr, cc, QTableWidgetItem(val))
        self._tbl.blockSignals(False)
        self.changed.emit()                                   # -> _refresh_gl_points (scene) + save

    def _on_item_changed(self, item):
        # Clamp the f0 column to [1, freq_max] (freq_max = f_cutoff) if a max is set.
        if (self._freq_col is not None and self._freq_max is not None
                and item.column() == self._freq_col):
            try:
                v = float(item.text())
            except ValueError:
                self.changed.emit(); return
            cl = min(max(v, 1.0), float(self._freq_max))
            if cl != v:
                self._tbl.blockSignals(True)
                item.setText(f"{cl:.4g}")
                self._tbl.blockSignals(False)
        self.changed.emit()

    def set_freq_max(self, fmax):
        """Upper clamp for the f0 column (call with f_cutoff). Also re-clamps existing rows."""
        self._freq_max = fmax
        if self._freq_col is None or fmax is None:
            return
        self._tbl.blockSignals(True)
        for r in range(self._tbl.rowCount()):
            it = self._tbl.item(r, self._freq_col)
            if it is None:
                continue
            try:
                v = float(it.text())
            except ValueError:
                continue
            cl = min(max(v, 1.0), float(fmax))
            if cl != v:
                it.setText(f"{cl:.4g}")
        self._tbl.blockSignals(False)

    def set_freq_column_visible(self, visible):
        """Show/hide the f0 column (shown only for the Sinusoid source type)."""
        if self._freq_col is not None:
            self._tbl.setColumnHidden(self._freq_col, not visible)

    def get_points(self):
        """Return numeric rows (x, y, z[, p0]); the Name column is skipped so
        the 3D view, undo, and saved solver files stay purely numeric."""
        pts = []
        for r in range(self._tbl.rowCount()):
            try:
                row = []
                for c in range(self._coff, self._tbl.columnCount()):
                    item = self._tbl.item(r, c)
                    if item is None: raise ValueError("empty cell")
                    row.append(float(item.text()))
                for _i in range(min(3, len(row))):       # displayed -> world
                    row[_i] += self._off[_i]
                pts.append(tuple(row))
            except (TypeError, ValueError): pass
        return pts

    def get_names(self):
        """Name (col 0) of every row, in table order. Row index matches the
        source/receiver id the solver assigns (both come from this table)."""
        return [(self._tbl.item(r, 0).text() if self._tbl.item(r, 0) else str(r + 1))
                for r in range(self._tbl.rowCount())]

    def get_pose(self, row):
        """[az,el,roll] deg for a row (default [0,0,0] if unset). Padded to rowCount on access."""
        return list(self.poses[row]) if 0 <= row < len(self.poses) else [0, 0, 0]

    def set_pose(self, row, triplet):
        while len(self.poses) <= row:          # grow to reach `row`
            self.poses.append([0, 0, 0])
        self.poses[row] = [float(triplet[0]), float(triplet[1]), float(triplet[2])]

    def set_points(self, pts, emit_changed=True):
        """Replace all rows from numeric tuples (x, y, z[, p0]). Names reset to
        the row number. Used by Ctrl+Z point undo."""
        ncoord = self._tbl.columnCount() - self._coff   # numeric columns
        # trailing numeric-column defaults, in order: [x y z] [amp?] [f0?]
        full_def = [0.0, 0.0, 0.0] + ([1.0] if self._has_amp else []) + ([100.0] if self._has_freq else [])
        self._tbl.blockSignals(True)
        self._tbl.setRowCount(0)
        for pt in pts:
            vals = list(pt[:ncoord])
            for _i in range(min(3, len(vals))):          # world -> displayed
                vals[_i] = float(vals[_i]) - self._off[_i]
            if len(vals) < ncoord:
                vals += full_def[len(vals):ncoord]
            r = self._tbl.rowCount(); self._tbl.insertRow(r)
            self._tbl.setItem(r, 0, QTableWidgetItem(str(r + 1)))
            for c, v in enumerate(vals[:ncoord]):
                self._tbl.setItem(r, self._coff + c, QTableWidgetItem(f"{float(v):.4g}"))
        self._tbl.blockSignals(False)
        if emit_changed:
            self.changed.emit()

    def load_file(self, path):
        self._tbl.setRowCount(0)
        if not os.path.isfile(path): return
        ncoord = self._tbl.columnCount() - self._coff
        with open(path) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if not line: continue
                vals = line.split()
                r = self._tbl.rowCount(); self._tbl.insertRow(r)
                self._tbl.setItem(r, 0, QTableWidgetItem(str(r + 1)))  # default name
                for c, v in enumerate(vals[:ncoord]):
                    if c < 3:                            # file -> world -> displayed
                        try: v = f"{float(v) - self._solver_off[c] - self._off[c]:.6g}"
                        except ValueError: pass
                    self._tbl.setItem(r, self._coff + c, QTableWidgetItem(v))
        npath = path + ".names"                         # restore names from the sidecar, if present
        if os.path.isfile(npath):
            with open(npath) as fn:
                for r, nm in enumerate(fn):
                    nm = nm.rstrip("\n")
                    if r < self._tbl.rowCount() and nm:
                        self._tbl.setItem(r, 0, QTableWidgetItem(nm))
        if self._has_pose:                              # head poses from the .poses sidecar
            self.poses = [[0, 0, 0] for _ in range(self._tbl.rowCount())]
            ppath = path + ".poses"
            if os.path.isfile(ppath):
                with open(ppath) as fp:
                    for r, ln in enumerate(fp):
                        v = ln.split()
                        if r < len(self.poses) and len(v) >= 3:
                            self.poses[r] = [float(v[0]), float(v[1]), float(v[2])]

    def save_file(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            hdr = "x y z" + (" p0" if self._has_amp else "") + (" f0" if self._has_freq else "")
            f.write("# " + hdr + "\n")
            for pt in self.get_points():
                pt = tuple(pt[i] + self._solver_off[i] if i < 3 else pt[i]
                           for i in range(len(pt)))       # world -> solver frame
                f.write("  ".join(f"{v:.6g}" for v in pt) + "\n")
        # names go in a sidecar (one per line, row-matched) so the numeric file main3D parses
        # stays untouched. spaces in names are fine — one name per line.
        with open(path + ".names", "w") as fn:
            fn.write("".join(nm + "\n" for nm in self.get_names()))
        if self._has_pose:                              # head poses, row-matched (az el roll deg)
            with open(path + ".poses", "w") as fp:
                fp.write("".join(f"{a:g} {e:g} {r:g}\n"
                                 for a, e, r in (self.get_pose(i) for i in range(self._tbl.rowCount()))))


# ─────────────────────────────────────────────────────────────
#  Solver runner thread
# ─────────────────────────────────────────────────────────────
class SolverThread(QThread):
    output   = Signal(str)
    progress = Signal(int)          # parsed NN% from the solver's progress bar
    done     = Signal(int)          # NOT 'finished' (that shadows QThread's own)

    def __init__(self, cmd, cwd, stdin=False):
        super().__init__(); self._cmd = cmd; self._cwd = cwd
        self._proc = None
        self._stdin = stdin                 # pipe stdin: the live auralization reads pose lines from it

    def send(self, line):
        """Write one line to the child's stdin (no-op unless started with stdin=True and alive)."""
        p = self._proc
        if self._stdin and p is not None and p.poll() is None and p.stdin:
            try:
                p.stdin.write(line.rstrip("\n") + "\n"); p.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass

    def run(self):
        import re, shutil
        pct_re = re.compile(r"(\d+)\s*%")
        # C stdout is block-buffered when piped (not a TTY), so printf() lines arrive in
        # bursts and are LOST if the solver aborts. Force line-buffering via stdbuf so every
        # line reaches the log live and survives a crash. (stderr is already unbuffered.)
        cmd = self._cmd
        if shutil.which("stdbuf"):
            cmd = ["stdbuf", "-oL", "-eL"] + list(cmd)
        try:
            self._proc = subprocess.Popen(cmd, cwd=self._cwd,
                stdin=subprocess.PIPE if self._stdin else None,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                # own process group, so stop() can signal the solver AND anything it spawns
                start_new_session=(os.name != "nt"))
            for line in self._proc.stdout:
                line = line.rstrip()
                self.output.emit(line)
                m = pct_re.search(line)        # e.g. "[====   ] 39%"
                if m:
                    try: self.progress.emit(int(m.group(1)))
                    except ValueError: pass
            self._proc.wait(); self.done.emit(self._proc.returncode)
        except Exception as ex:
            self.output.emit(f"[ERROR] {ex}"); self.done.emit(-1)

    def stop(self):
        """SIGTERM the process group, then SIGKILL whatever survives.

        A CUDA binary inside a kernel launch routinely ignores SIGTERM. The old code
        sent one and returned, so quitting/force-quitting the GUI left the solver
        running and still holding its GPU context -- the next run then dies with
        CUDA error 999 and the orphan sits unkillable in D state until a reboot.
        Killing the whole group also catches anything the solver spawned.
        """
        p = self._proc
        if not p or p.poll() is not None:
            return
        import signal

        def _sig(s):
            try:
                os.killpg(os.getpgid(p.pid), s)        # whole group
            except (ProcessLookupError, PermissionError, AttributeError, OSError):
                try: p.send_signal(s)                  # Windows / no group: just the child
                except Exception: pass

        _sig(signal.SIGTERM)
        try:
            p.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            pass
        _sig(getattr(signal, "SIGKILL", signal.SIGTERM))
        try:
            p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            # SIGKILL can't touch a task blocked in the NVIDIA driver (D state).
            self.output.emit("[!] solver ignored SIGKILL -- stuck in the GPU driver (D state). "
                             "The GPU context will not free until reboot.")


class ResultsView(QWidget):
    """Floating plot window: a metric selector on top (IR+Spectrum, or an ISO-3382 param per band)
    and the rendered PNG below. Image-based on purpose: setPixmap always repaints."""
    metricChanged = Signal()
    addSimRequested = Signal(str)      # path to another project's params.json (Summary compare)
    clearSimRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proj_root = None
        self._pix = None
        self._export = None                        # tabular data of the current param view, or None
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)   # click -> keyboard focus for Ctrl+C
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        from PySide6.QtWidgets import QComboBox, QSizePolicy, QPushButton, QHBoxLayout
        top = QHBoxLayout()
        self._metric = QComboBox()
        self._metric.addItems(["IR + Spectrum", "T20", "T30", "EDT", "C50", "C80", "D50", "Summary (avg)"])
        self._metric.currentIndexChanged.connect(self._on_metric_changed)
        top.addWidget(self._metric, 1)
        self._add_sim_btn = QPushButton("➕ Add simulation…"); self._add_sim_btn.clicked.connect(self._add_sim)
        self._clear_sim_btn = QPushButton("Clear compared"); self._clear_sim_btn.clicked.connect(self.clearSimRequested.emit)
        self._add_sim_btn.hide(); self._clear_sim_btn.hide()     # only shown in Summary view
        top.addWidget(self._add_sim_btn); top.addWidget(self._clear_sim_btn)
        lay.addLayout(top)
        self._label = QLabel("Select receiver(s) in the Receivers tab or the scene.")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(f"color:{PAL['text']};background:{PAL['bg']};")
        self._label.setMinimumSize(1, 1)           # don't let the pixmap's sizeHint trap the
        self._label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)  # window min after maximize
        self._label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lay.addWidget(self._label, 1)
        from PySide6.QtGui import QShortcut, QKeySequence   # Ctrl+C works even when the combo has focus
        QShortcut(QKeySequence.StandardKey.Copy, self, activated=self._copy_csv)

    def current_metric(self):
        return self._metric.currentText()

    def set_export(self, data):
        """Tabular data behind the current param plot (or None for IR/Spectrum -> export disabled)."""
        self._export = data

    def _csv_text(self, sep=","):
        """Generic table export: {rowheader, columns:[...], rows:[(label,[vals]), ...]}."""
        d = self._export
        if not d:
            return ""
        def fmt(v):
            return "" if v is None else f"{v:.5g}"
        lines = [[d["rowheader"]] + list(d["columns"])]
        for label, vals in d["rows"]:
            lines.append([str(label)] + [fmt(v) for v in vals])
        return "\n".join(sep.join(c) for c in lines)

    def _copy_csv(self):
        from PySide6.QtWidgets import QApplication
        if self._export:                               # TAB-separated: spreadsheets split paste on tabs
            QApplication.clipboard().setText(self._csv_text(sep="\t"))

    def contextMenuEvent(self, e):
        from PySide6.QtWidgets import QMenu
        m = QMenu(self)
        if self.current_metric() == "Summary (avg)":      # cross-simulation compare actions
            m.addAction("Add simulation to compare…", self._add_sim)
            m.addAction("Clear compared", self.clearSimRequested.emit)
            m.addSeparator()
        if self._export:
            m.addAction("Copy as CSV", self._copy_csv)     # slots connected directly
            m.addAction("Export CSV…", self._export_csv_file)
            m.addAction("Export JSON…", self._export_json_file)
        elif self.current_metric() != "Summary (avg)":
            a = m.addAction("Export: switch to a parameter view (T20/T30/…)")
            a.setEnabled(False)
        m.exec(e.globalPos())
        e.accept()

    def _on_metric_changed(self, *_):
        summary = self.current_metric() == "Summary (avg)"
        self._add_sim_btn.setVisible(summary); self._clear_sim_btn.setVisible(summary)
        self.metricChanged.emit()

    def _add_sim(self):
        """Pick a PROJECT FOLDER; resolve its params.json from the standard template path."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        d = QFileDialog.getExistingDirectory(self, "Add simulation — pick the project folder")
        if not d:
            return
        for cand in (os.path.join(d, "output", "params.json"), os.path.join(d, "params.json")):
            if os.path.isfile(cand):
                self.addSimRequested.emit(cand); return
        QMessageBox.warning(self, "No parameters",
                            f"No output/params.json under:\n{d}\nOpen that project's Results first.")

    def _export_csv_file(self):
        from PySide6.QtWidgets import QFileDialog
        if not self._export:
            return
        title = self._export.get("title", "results")
        p, _ = QFileDialog.getSaveFileName(self, "Export CSV", f"{title}.csv", "CSV (*.csv)")
        if p:
            with open(p, "w") as f:
                f.write(self._csv_text())

    def _export_json_file(self):
        from PySide6.QtWidgets import QFileDialog
        import json
        if not self._export:
            return
        title = self._export.get("title", "results")
        p, _ = QFileDialog.getSaveFileName(self, "Export JSON", f"{title}.json", "JSON (*.json)")
        if p:
            with open(p, "w") as f:
                json.dump(self._export, f, indent=1)

    def set_project(self, root):
        self._proj_root = root

    def show_receiver(self, rec):
        from PySide6.QtGui import QPixmap
        if self._proj_root is None or rec is None or rec < 0:
            return
        path = os.path.join(self._proj_root, "output", f"IR_rec{rec}.png")
        if not os.path.isfile(path):
            self._pix = None
            self._label.setText(f"Receiver {rec}: no plot yet — run a full simulation, reopen Results.")
            return
        self._pix = QPixmap(path)
        self._rescale()

    def show_png(self, path):
        """Show an arbitrary PNG (e.g. the on-the-fly receiver-overlay plot)."""
        from PySide6.QtGui import QPixmap
        if not path or not os.path.isfile(path):
            self._pix = None; self._label.setText("No overlay plot."); return
        self._pix = QPixmap(path); self._rescale()

    def _rescale(self):
        if self._pix is not None and not self._pix.isNull():
            self._label.setPixmap(self._pix.scaled(
                self._label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e); self._rescale()


# ─────────────────────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────────────────────
class FillTabBar(QTabBar):
    """
    A tab bar that stretches its tabs to fill the FULL width of the tab widget
    so the last tab ends flush with the panel's right edge (no trailing gap),
    WITHOUT ever shrinking a tab below the width its label needs (so text and
    emoji are never clipped). Only surplus space is distributed; if the tabs
    already need more than the panel width, each keeps its natural width.
    This is done explicitly because QTabBar.setExpanding(True) is style-dependent
    and does not stretch the tabs on every platform.
    """
    def tabSizeHint(self, index):
        hint = super().tabSizeHint(index)
        n = self.count()
        if n <= 0:
            return hint
        # natural widths needed by each tab's content
        nat = [super(FillTabBar, self).tabSizeHint(i).width() for i in range(n)]
        total = self.parentWidget().width() if self.parentWidget() else self.width()
        surplus = total - sum(nat)
        if surplus > 0:
            # spread surplus evenly; the last tab absorbs the rounding remainder
            add = surplus // n
            extra = (surplus - add * (n - 1)) if index == n - 1 else add
            hint.setWidth(nat[index] + extra)
        else:
            # not enough room: keep natural width so nothing is clipped
            hint.setWidth(nat[index])
        return hint

    def resizeEvent(self, e):
        # re-evaluate tab widths whenever the bar is resized
        super().resizeEvent(e)
        self.updateGeometry()


class ChamferTabBar(FillTabBar):
    """FillTabBar (stretches tabs to fill the width) but each tab is painted with only
    its TWO TOP corners cut off diagonally (bottom corners stay square). CUT = cut [px]."""
    CUT = 7

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(self.font())
        c = self.CUT
        for i in range(self.count()):
            r = self.tabRect(i).adjusted(1, 1, -1, -1)
            sel = (i == self.currentIndex())
            poly = QPolygon([
                QPoint(r.left(),    r.top()),      # top-left square
                QPoint(r.right()-c, r.top()),      # top edge, before the right cut
                QPoint(r.right(),   r.top()+c),    # top-right diagonal (the only cut)
                QPoint(r.right(),   r.bottom()),   # bottom-right square
                QPoint(r.left(),    r.bottom()),   # bottom-left square
            ])
            p.setPen(QPen(QColor(PAL["border"]), 1))
            p.setBrush(QColor(PAL["accent"] if sel else PAL["panel"]))
            p.drawPolygon(poly)
            p.setPen(QColor(PAL["bg"] if sel else PAL["text"]))
            p.drawText(r, int(Qt.AlignmentFlag.AlignCenter), self.tabText(i))
        p.end()


class DraggableProgress(QWidget):
    """
    A small progress bar pinned to the top-left corner of the 3D viewport.
    It is a child of the viewport container (absolute-positioned at a fixed
    offset) and is hidden until a run starts.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"background:{PAL['bg']};border:1px solid {PAL['border']};border-radius:5px;")
        self.setFixedSize(220, 30)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4); lay.setSpacing(6)
        self._bar = QProgressBar()
        self._bar.setRange(0, 100); self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFixedHeight(16)
        self._bar.setStyleSheet(f"""
            QProgressBar {{ background:{PAL['panel']}; border:1px solid {PAL['border']};
                border-radius:3px; color:{PAL['text']}; font-size:10px; text-align:center; }}
            QProgressBar::chunk {{ background:{PAL['accent']}; border-radius:2px; }}
        """)
        lay.addWidget(self._bar, 1)
        self.hide()

    def setValue(self, v):
        self._bar.setValue(max(0, min(100, int(v))))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("F.A.S.E.S - Faithful Acoustic Solver for Extended Spaces")
        self.resize(1440, 860)
        self._install_suspend_hook()
        self._proj_root  = None
        self._obj_path   = None
        self._groups     = []
        # Lightweight point undo history.  Each entry stores only the numeric
        # source/receiver rows before a change, so undoing back to the first
        # placed point is cheap even for many points.
        self._point_undo_stack = []
        self._apply_theme()
        self._build_ui()
        # Tab / Shift+Tab cycle through the tabs (with wrap-around). Installed as
        # an application-wide filter so the keys are caught regardless of which
        # child widget has focus; see eventFilter for the guard that preserves
        # normal Tab behaviour while editing text/number fields.
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, e):
        """Cycle tabs with Tab / Shift+Tab (periodic wrap-around).

        Guard: while the user is editing a text or number field (a QLineEdit,
        spin box, or text edit currently has focus), Tab keeps its normal
        meaning (move between fields / table cells). Everywhere else, Tab moves
        to the next tab and Shift+Tab (delivered as Key_Backtab) to the previous.
        """
        if e.type() == QEvent.Type.KeyPress and e.key() in (
                Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            fw = QApplication.focusWidget()
            editing = isinstance(fw, (QLineEdit, QAbstractSpinBox, QTextEdit))
            if not editing and hasattr(self, "_tabs"):
                n = self._tabs.count()
                if n > 0:
                    cur  = self._tabs.currentIndex()
                    step = -1 if e.key() == Qt.Key.Key_Backtab else 1
                    self._tabs.setCurrentIndex((cur + step) % n)
                return True   # consume so focus traversal does not also fire
        # Auto-hide panel: any interaction with the tabs restarts the inactivity countdown;
        # scene-only work lets it run down. A central resize re-centres the grip.
        if hasattr(self, "_panel_idle"):
            t = e.type()
            if t in (QEvent.Type.MouseButtonPress, QEvent.Type.KeyPress, QEvent.Type.Wheel):
                if self._panel_open and self._widget_in_tabs(obj):
                    self._panel_idle.start()
            elif t == QEvent.Type.Resize and obj is self._central:
                # defer: at Resize time the layout hasn't repositioned gl_wrap yet, so its
                # geometry is still stale (this caused the panel to keep the old size on maximize).
                QTimer.singleShot(0, self._layout_panel)
        return super().eventFilter(obj, e)

    def _widget_in_tabs(self, w):
        """True if QObject w is the tab panel or a descendant of it."""
        while w is not None:
            if w is self._tabs:
                return True
            w = w.parent()
        return False

    def _apply_theme(self):
        pal = QPalette()
        for role, key in [
            (QPalette.ColorRole.Window,          "panel"),   # tab content etc. sits on panel, not the darker bg
            (QPalette.ColorRole.WindowText,      "text"),
            (QPalette.ColorRole.Base,            "panel"),
            (QPalette.ColorRole.AlternateBase,   "highlight"),
            (QPalette.ColorRole.Text,            "text"),
            (QPalette.ColorRole.Button,          "btn"),
            (QPalette.ColorRole.ButtonText,      "text"),
            (QPalette.ColorRole.Highlight,       "accent"),
            (QPalette.ColorRole.HighlightedText, "bg"),
        ]:
            pal.setColor(role, QColor(PAL[key]))
        QApplication.setPalette(pal)
        self.setStyleSheet(f"""
            QPushButton {{background:{PAL['btn']};color:{PAL['text']};
                border:1px solid {PAL['border']};border-radius:4px;
                padding:4px 10px;font-size:12px;}}
            QPushButton:hover{{background:{PAL['btn_hover']};border-color:{PAL['accent']};}}
            QPushButton:pressed{{background:{PAL['highlight']};}}
            QPushButton:disabled{{color:{PAL['subtext']};}}
            QDoubleSpinBox,QSpinBox{{background:{PAL['bg']};color:{PAL['text']};
                border:1px solid {PAL['border']};border-radius:3px;padding:2px 4px;}}
            QLineEdit{{background:{PAL['bg']};color:{PAL['text']};
                border:1px solid {PAL['border']};border-radius:3px;padding:2px 6px;}}
            QLineEdit:focus{{border-color:{PAL['accent']};}}
            QCheckBox{{color:{PAL['text']};spacing:6px;}}
            QCheckBox::indicator{{width:14px;height:14px;
                border:1px solid {PAL['border']};border-radius:3px;background:{PAL['bg']};}}
            QCheckBox::indicator:checked{{background:{PAL['accent']};border-color:{PAL['accent']};}}
            QTabWidget::pane{{border:none;background:{PAL['panel']};}}
            QTabBar::tab{{background:{PAL['btn']};color:{PAL['subtext']};
                border:1px solid {PAL['border']};padding:6px 8px;
                border-bottom:none;border-top-left-radius:4px;border-top-right-radius:4px;}}
            QTabBar::tab:selected{{background:{PAL['panel']};color:{PAL['accent']};}}
            QTabBar::tab:hover{{color:{PAL['text']};}}
            QListWidget{{background:{PAL['bg']};border:1px solid {PAL['border']};color:{PAL['text']};}}
            QListWidget::item:selected{{background:{PAL['layer_sel']};color:{PAL['accent2']};}}
            QListWidget::item:hover{{background:{PAL['highlight']};}}
            QLabel{{color:{PAL['text']};}}
            QGroupBox{{color:{PAL['subtext']};border:1px solid {PAL['border']};
                border-radius:4px;margin-top:8px;font-size:11px;}}
            QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 4px;}}
            QScrollArea{{border:none;}}
            QTextEdit{{background:{PAL['bg']};color:{PAL['text']};
                border:1px solid {PAL['border']};font-family:monospace;font-size:11px;}}
            QComboBox{{background:{PAL['btn']};color:{PAL['text']};
                border:1px solid {PAL['border']};border-radius:3px;padding:2px 6px;}}
            QComboBox QAbstractItemView{{background:{PAL['panel']};color:{PAL['text']};}}
            QSplitter::handle{{width:0px;height:0px;background:transparent;border:none;margin:0;padding:0;}}
            QStatusBar{{background:{PAL['panel']};color:{PAL['subtext']};font-size:11px;}}
        """)

    def _build_ui(self):
        central = QWidget()
        self._central = central
        self.setCentralWidget(central)
        root_lay = QHBoxLayout(central)
        root_lay.setContentsMargins(6,6,6,6); root_lay.setSpacing(6)

        # left — GL + status
        gl_wrap = QWidget()
        self._gl_wrap = gl_wrap          # panel overlay is sized to this (inside the margin)
        gl_vlay = QVBoxLayout(gl_wrap)
        gl_vlay.setContentsMargins(0,0,0,0); gl_vlay.setSpacing(3)

        self._gl = GLViewer()
        self._gl.layerPicked.connect(self._on_layer_picked)
        self._gl.surfacesPicked.connect(self._on_surfaces_picked)
        self._gl.explodeSurface.connect(self._on_explode_surface)
        self._gl.mergeSiblings.connect(self._on_merge_siblings)
        self._gl.cleanseLayers.connect(self._on_cleanse_layers)
        self._gl.origFrameToggled.connect(self._apply_orig_frame)
        self._gl.shiftPolicyToggled.connect(self._on_shift_policy)
        self._gl.pointPlaced.connect(self._on_point_placed)
        self._gl.pointMoved.connect(self._on_point_moved)
        self._gl.pointDragStarted.connect(self._remember_points_for_undo)
        self._gl.pointHovered.connect(self._on_point_hovered)
        self._gl.receiverPicked.connect(self._on_scene_receiver_pick)
        self._gl.fpvRequested.connect(self._enter_receiver_view)
        self._gl.fpvExitRequested.connect(self._gl.exit_fpv)
        self._gl.fpvPoseChanged.connect(self._on_fpv_pose)
        self._results_sel = set(); self._sel_sync = False   # shared Results receiver selection
        self._gl.pointRemoved.connect(self._on_point_removed)
        self._gl.groupXfDone.connect(self._on_group_xf)
        self._gl.pointsDeleted.connect(self._on_points_deleted)
        gl_vlay.addWidget(self._gl, stretch=1)      # GL stays in the layout (never re-parented)
        # Results plots are an OVERLAY on top of the scene (QOpenGLWidget crashes if stacked),
        # shown/hidden by the mode bar and positioned clear of the left panel in _layout_panel.
        self._results = ResultsView(self)          # top-level floating window (native resize/maximize)
        self._results.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)   # stays visible while
        self._results.setWindowTitle("Results")
        self._results.resize(720, 560)
        self._results.hide()
        self._results.metricChanged.connect(lambda: self._apply_results_selection(mirror_table=False))
        self._results.addSimRequested.connect(self._add_compare_sim)
        self._results.clearSimRequested.connect(self._clear_compare_sims)
        self._compare = []                         # [(label, params_dict)] extra sims in the Summary view
        self._view_mode = "sim"
        self._panel_locked = False

        # placement toolbar under the viewer
        place_bar = QHBoxLayout()
        place_bar.setSpacing(4)
        place_bar.setContentsMargins(2, 1, 2, 8)    # bottom pad == tab rows' 8px (buttons bottom-aligned)

        self._src_btn = QPushButton("Src [S]")
        self._src_btn.setCheckable(True)
        self._src_btn.setFixedSize(74, 22)
        self._src_btn.clicked.connect(lambda: self._toggle_placement("source"))
        place_bar.addWidget(self._src_btn)

        self._rec_btn = QPushButton("Rec [R]")
        self._rec_btn.setCheckable(True)
        self._rec_btn.setFixedSize(74, 22)
        self._rec_btn.clicked.connect(lambda: self._toggle_placement("receiver"))
        place_bar.addWidget(self._rec_btn)

        self._slice_btn = QPushButton("Slice [L]")
        self._slice_btn.setCheckable(True)
        self._slice_btn.setFixedSize(74, 22)
        self._slice_btn.clicked.connect(lambda: self._toggle_placement("slice"))
        place_bar.addWidget(self._slice_btn)

        self._hint_lbl = QLabel("press  ?  in the 3D view for shortcuts")   # full cheatsheet is the ? overlay now
        place_bar.addWidget(self._hint_lbl)
        place_bar.addStretch()
        # bottom-align the buttons so the bottom margin (not layout centering) sets their height,
        # landing them 8px above the bar bottom == the tab rows' 8px bottom margin.
        for _b in (self._src_btn, self._rec_btn, self._slice_btn):
            place_bar.setAlignment(_b, Qt.AlignmentFlag.AlignBottom)
        bar_wrap = QWidget()
        bar_wrap.setFixedHeight(34)
        bar_wrap.setLayout(place_bar)
        gl_vlay.addWidget(bar_wrap)
        self._place_bar = place_bar     # left margin follows the panel so it isn't covered

        # tabs panel on the LEFT, GL viewport on the RIGHT
        self._tabs = QTabWidget()
        # Custom tab bar that divides the full width across the tabs so the last
        # tab ends flush with the panel edge (set BEFORE adding any tabs).
        self._tabs.setTabBar(ChamferTabBar())
        self._tabs.setUsesScrollButtons(False)        # no arrows
        self._tabs.setElideMode(Qt.TextElideMode.ElideNone)
        # Auto-hide sliding panel: the panel is a FIXED-width overlay on top of the (full-
        # width) viewport. It is TRANSLATED left off-screen to hide (so the tabs keep their
        # proportions, no squish) and slid back when the left grip strip is clicked.
        self._PANEL_W = 540   # fixed panel width (fits the Material tab's Scatter column)
        self._tabs.setParent(self._central)
        self._tabs.setFixedWidth(self._PANEL_W)
        # Only the GL viewport goes in the layout; the panel floats over its left side.
        root_lay.addWidget(gl_wrap, stretch=1)         # fills; panel overlays its left

		# append "\uFE0E" in front of symbol to stay monochrome
        self._tabs.addTab(self._build_scene_tab(),   "🞕\uFE0E  Scene")
        self._tabs.addTab(self._build_src_tab(),     "🔊\uFE0E  Sources")
        self._tabs.addTab(self._build_rec_tab(),     "🎙\uFE0E  Receivers")
        self._tabs.addTab(self._build_config_tab(),  "⚙\uFE0E  Setup")
        self._tabs.addTab(self._build_run_tab(),     "▶\uFE0E  Run")

        # Floating, draggable progress bar overlaying the 3D scene. Parented to
        # the GL container so it sits on top of the viewport; starts hidden and
        # is positioned near the top-left, the user can drag it anywhere.
        self._float_prog = DraggableProgress(gl_wrap)
        self._float_prog.move(16, 16)

        # transparent purple-blue mode selector (accent), top-right of the scene, options STACKED
        _ac = QColor(PAL['accent'])
        self._mode_bar = QWidget(gl_wrap)
        self._mode_bar.setStyleSheet(
            f"background:rgba({_ac.red()},{_ac.green()},{_ac.blue()},150);"
            f"border:1px solid {PAL['border']};border-radius:6px;")
        _mb = QVBoxLayout(self._mode_bar); _mb.setContentsMargins(8, 6, 8, 6); _mb.setSpacing(2)
        self._mode_grp = QButtonGroup(self)          # exclusive: acts like radios, no circle
        _mode_css = (f"QPushButton{{border:none;background:transparent;color:{PAL['text']};"
                     f"text-align:left;padding:2px 6px;}}"
                     f"QPushButton:checked{{color:{PAL['bg']};font-weight:bold;}}")
        from PySide6.QtGui import QFontMetrics
        for key, label in (("sim", "Simulation"), ("results", "Results"), ("draw", "Draw")):
            b = QPushButton(label); b.setCheckable(True); b.setChecked(key == "sim"); b.setFlat(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor); b.setStyleSheet(_mode_css)
            bf = b.font(); bf.setBold(True)              # reserve BOLD width so checked isn't clipped
            b.setMinimumWidth(QFontMetrics(bf).horizontalAdvance(label) + 22)
            b.toggled.connect(lambda on, k=key: on and self._set_mode(k))
            self._mode_grp.addButton(b); _mb.addWidget(b)
        self._mode_bar.adjustSize(); self._mode_bar.raise_()

        # Dark/Light lives in the RMB scene menu now (stacked with Show/Hide axes);
        # the viewport emits themeToggle when it's picked. Also feed the grid spacing
        # to the viewport so the HUD shows i,j,k next to x,y,z.
        self._gl.themeToggle.connect(self._toggle_scene_theme)
        self._gl.sliceChanged.connect(self._on_slice_changed)
        self._cfg['h'].valueChanged.connect(self._gl.set_grid_h)
        self._gl.set_grid_h(self._cfg['h'].value())
        self._restyle_scene_theme()        # initial hint styling
        self._dirty = False; self._wire_dirty()   # track unsaved edits for the close prompt

        # ── auto-hide sliding tab panel (translate the fixed-width overlay) ──
        self._panel_open = True
        self._panel_x = 0                    # current left edge; 0 = shown, -_PANEL_W = hidden
        self._panel_anim = QVariantAnimation(self)
        self._panel_anim.setDuration(150)
        self._panel_anim.valueChanged.connect(self._on_panel_slide)
        self._panel_anim.finished.connect(self._gl.update)   # settle HUD inset once, at rest
        # Invisible "near-it" hotzone at the left edge (no button): click anywhere in this
        # strip of empty scene and the panel slides back out. Active only while hidden.
        self._GRIP_W = self._PANEL_W - 20    # reopen hotzone: almost the full panel width
        self._panel_grip = QPushButton(self._central)
        self._panel_grip.setFlat(True)
        self._panel_grip.setStyleSheet("QPushButton{background:transparent;border:none;}")
        self._panel_grip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._panel_grip.clicked.connect(self._show_panel)
        self._panel_grip.hide()
        # idle timer: slides the panel shut after inactivity ON THE TABS (scene-only work
        # lets it run down); reset in eventFilter on any tab interaction.
        # >>> EDIT HERE to change the auto-hide delay <<<
        self.PANEL_HIDE_SECONDS = 10
        self._panel_idle = QTimer(self)
        self._panel_idle.setSingleShot(True)
        self._panel_idle.setInterval(int(self.PANEL_HIDE_SECONDS * 1000))
        self._panel_idle.timeout.connect(self._hide_panel)
        self._panel_idle.start()
        self._tabs.show(); self._tabs.raise_()

        # status bar
        self._sb = _RunStatusBar()
        self.setStatusBar(self._sb)
        self._sb.showMessage("Ready.  Open a project folder to begin.")
        self._mode_lbl = QLabel("VIEW   ·   ? = shortcuts")     # persistent mode badge, bottom-right
        self._mode_lbl.setStyleSheet(f"color:{PAL['accent2']};font-weight:bold;padding:0 8px;")
        self._sb.addPermanentWidget(self._mode_lbl)
        self._gl.modeLabelChanged.connect(
            lambda s: self._mode_lbl.setText(f"{s}   ·   ? = shortcuts"))

        self._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self._undo_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._undo_shortcut.activated.connect(self._undo_point_change)
        QShortcut(QKeySequence.StandardKey.Save, self, activated=self._save_project)   # Ctrl+S
        QShortcut(QKeySequence.StandardKey.Open, self, activated=self._open_project)   # Ctrl+O

    # ── point undo ───────────────────────────────────────────
    def _snapshot_point_state(self):
        return {
            "sources":   [tuple(p) for p in self._src_tbl.get_points()],
            "receivers": [tuple(p) for p in self._rec_tbl.get_points()],
        }

    def _remember_points_for_undo(self):
        snap = self._snapshot_point_state()
        if not self._point_undo_stack or self._point_undo_stack[-1] != snap:
            self._point_undo_stack.append(snap)

    def _undo_point_change(self):
        if not self._point_undo_stack:
            self._sb.showMessage("Nothing to undo for sources/receivers.")
            return
        snap = self._point_undo_stack.pop()
        self._src_tbl.set_points(snap["sources"], emit_changed=False)
        self._rec_tbl.set_points(snap["receivers"], emit_changed=False)
        self._refresh_gl_points()
        self._sb.showMessage("Undid last source/receiver change.")

    # ── placement mode ───────────────────────────────────────
    # ── auto-hide sliding tab panel (translate the fixed-width overlay) ──
    def _layout_panel(self):
        """Keep the overlay panel full-height at its current x, the reopen hotzone docked
        to the left edge, and push the scene's Src/Rec bar + HUD text clear of the panel
        (indent them by the panel's on-screen right edge) so it never covers them."""
        g = self._gl_wrap.geometry()     # scene area, already inset by the root margin
        self._tabs.setGeometry(g.x() + self._panel_x, g.y(), self._PANEL_W, g.height())
        self._tabs.raise_()
        self._panel_grip.setGeometry(g.x(), g.y(), self._GRIP_W, g.height())
        if not self._panel_open:
            self._panel_grip.raise_()
        vis = max(0, self._panel_x + self._PANEL_W)    # panel's right edge on screen (0 hidden)
        m = self._place_bar.contentsMargins()
        self._place_bar.setContentsMargins(vis + 8, m.top(), m.right(), m.bottom())
        self._gl.set_hud_left(vis)
        if hasattr(self, "_float_prog"):               # progress bar rides the panel edge
            self._float_prog.move(vis + 16, self._float_prog.y())
        if hasattr(self, "_mode_bar"):                 # mode selector docks top-right, above all
            self._mode_bar.adjustSize()
            self._mode_bar.move(self._gl_wrap.width() - self._mode_bar.width() - 12, 12)
            self._mode_bar.raise_()

    def _on_panel_slide(self, v):
        self._panel_x = int(v)
        self._layout_panel()

    def _animate_panel(self, target_x):
        self._panel_anim.stop()
        self._panel_anim.setStartValue(self._panel_x)
        self._panel_anim.setEndValue(target_x)
        self._panel_anim.start()

    def _hide_panel(self):
        if not self._panel_open or getattr(self, "_panel_locked", False):
            return                                     # Results mode: panel never shrinks
        self._panel_open = False
        self._panel_grip.show(); self._panel_grip.raise_()   # strip becomes clickable
        self._animate_panel(-self._PANEL_W - 12)             # overshoot so no sliver/border peeks

    def _show_panel(self):
        if self._panel_open:
            return
        self._panel_open = True
        self._panel_grip.hide()
        self._animate_panel(0)
        if not getattr(self, "_panel_locked", False):
            self._panel_idle.start()           # restart the inactivity countdown

    def _find_common_script(self, name):
        """Scripts (allIRs.py etc.) live NEXT TO gui.py, both in common/."""
        cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        return cand if os.path.isfile(cand) else None

    def _set_mode(self, mode):
        """Switch the right-side area: 'sim' = 3D scene, 'results' = per-receiver IR/FR plots,
        'draw' = sketcher (separate script, added later)."""
        self._view_mode = mode
        self._gl.set_props_mode(mode == "draw")        # Draw: P/R/S transform props, src/rec locked
        self._gl.set_results_pick(mode == "results")   # click-to-select receivers only in Results
        if mode != "results":
            self._gl.set_selected_receivers([])        # drop the multi-select highlight
        if mode != "draw" and getattr(self, "_sketch", None) is not None:
            self._save_sketch_json()                   # persist the sketch, then hide (own window)
            self._sketch.hide()
        if mode != "draw" and getattr(self, "_propswin", None) is not None:
            self._save_props_json(); self._propswin.hide()
        if mode == "sim":
            self._panel_locked = False
            self._results.hide()
            self._layout_panel()
            self._panel_idle.start()
        elif mode == "results":
            self._panel_locked = False             # panel shrinks/idles normally; Results floats
            self._results.set_project(self._proj_root)
            self._results.show(); self._results.raise_()
            self._layout_panel()
            self._panel_idle.start()
            self._gl.set_selected_receivers(sorted(self._results_sel))
            self._ensure_results()                 # (re)compute broadband IRs, then plot
        elif mode == "draw":
            self._panel_locked = False
            self._results.hide()
            self._open_props()          # loads props JSON first, so the sketch footprints populate on first entry
            self._open_sketch()

    def _open_sketch(self):
        """Open the 2D domain sketcher (its own top-level window). Created once, reused."""
        if getattr(self, "_sketch", None) is None:
            from sketch2d import SketchWindow      # lives next to gui.py in common/
            self._sketch = SketchWindow(self)
            self._sketch.canvas.objReady.connect(self._on_sketch_obj)
        if self._gl._grid_h > 0:                   # background grid = solver grid spacing h
            self._sketch.canvas.h = self._gl._grid_h
        self._sketch.canvas.set_prop_footprints(self._prop_footprints())   # top-down prop outlines
        if not self._sketch.canvas.shapes:         # empty -> restore this project's saved sketch
            self._load_sketch_json()
        self._sketch.show(); self._sketch.raise_(); self._sketch.activateWindow()

    def _open_props(self):
        """Props placement window (own top-level, reused), opened alongside the sketcher in Draw mode."""
        self._load_props_json()
        if getattr(self, "_propswin", None) is None:
            self._propswin = PropsWindow(self._gl, self)
            self._gl.propsChanged.connect(self._on_props_changed)   # GL transform -> refresh the panel
        else:
            self._propswin._refresh()
        self._propswin.show(); self._propswin.raise_()

    def _on_props_changed(self):
        pw = getattr(self, "_propswin", None)
        if pw is not None:
            pw._refresh(); pw._on_select(self._gl._prop_sel)
        sk = getattr(self, "_sketch", None)
        if sk is not None:                                       # refresh the 2D footprint overlays
            sk.canvas.set_prop_footprints(self._prop_footprints())

    def _prop_footprints(self):
        """Top-down (XY) outline of each prop for the 2D sketch overlay: convex hull of its world verts
        projected to the floor. Returns a list of polygons (each a list of (x,y))."""
        def hull(pts):
            pts = sorted(set((round(x, 4), round(y, 4)) for x, y in pts))
            if len(pts) < 3:
                return pts
            cr = lambda o, a, b: (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            lo = []
            for p in pts:
                while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
                    lo.pop()
                lo.append(p)
            up = []
            for p in reversed(pts):
                while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0:
                    up.pop()
                up.append(p)
            return lo[:-1] + up[:-1]
        out = []
        for p in self._gl._props:
            v = p.world()[0]
            h = hull([(x, y) for x, y, _ in v])
            if len(h) >= 2:
                out.append(h)
        return out

    def _props_json_path(self):
        return (os.path.join(self._proj_root, "input", "scene", "props.json")
                if self._proj_root else None)

    def _load_props_json(self):
        import json, props3d
        p = self._props_json_path()
        if p and os.path.isfile(p) and not self._gl._props:   # don't clobber unsaved in-session props
            try:
                self._gl._props = [props3d.Prop.from_dict(d) for d in json.load(open(p))]
                self._gl._prop_sel = -1; self._gl.update()
            except Exception:
                pass

    def _save_props_json(self):
        import json
        p = self._props_json_path()
        if not (p and self._gl._props):
            return
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            with open(p, "w") as f:
                json.dump([pr.to_dict() for pr in self._gl._props], f)
        except OSError:
            pass

    def _sketch_json_path(self):
        return (os.path.join(self._proj_root, "input", "scene", "sketch.json")
                if self._proj_root else None)

    def _load_sketch_json(self):
        import json
        p = self._sketch_json_path()
        if p and os.path.isfile(p):
            try:
                self._sketch.canvas.load_shapes(json.load(open(p)))
            except Exception:
                pass

    def _reset_scene_editors(self):
        """Project switch -> clear the in-session sketch + props (the loaders are guarded against clobbering
        in-session state, so without this the new project shows the PREVIOUS project's domain), then load
        the new project's saved editors."""
        self._gl._props = []; self._gl._prop_sel = -1; self._gl.update()
        sk = getattr(self, "_sketch", None)
        if sk is not None:
            c = sk.canvas
            c.shapes = []; c._shape_sel = None; c._shape_xform = None
            c._psel = set(); c._esel = set(); c.sel = None; c._undo = []
            c.selectionChanged.emit(); c.update()
            self._load_sketch_json()
        self._load_props_json()

    def _save_sketch_json(self):
        """Persist the editable sketch. Guard against overwriting a saved sketch with an empty one."""
        import json
        p = self._sketch_json_path()
        sk = getattr(self, "_sketch", None)
        if not (p and sk and sk.canvas.shapes):
            return
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            with open(p, "w") as f:
                json.dump(sk.canvas.dump_shapes(), f)
        except OSError:
            pass

    def _on_sketch_obj(self, obj_text):
        """Extruded sketch -> write input/scene/sketch.obj (+ the .json sidecar) and load the OBJ."""
        if not self._proj_root:
            self._sb.showMessage("Open a project first to save the sketched OBJ."); return
        d = os.path.join(self._proj_root, "input", "scene"); os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "sketch.obj")
        if self._gl._props:                        # bake placed props into the scene OBJ, grouped by layer
            import props3d
            nv = sum(1 for ln in obj_text.splitlines() if ln.startswith("v "))
            obj_text = obj_text.rstrip("\n") + "\n" + props3d.export_groups(self._gl._props, start=nv + 1)
        with open(path, "w") as f:
            f.write(obj_text)
        self._save_sketch_json(); self._save_props_json()
        self._load_obj_path(path)
        self._sb.showMessage("Extruded → loaded. Switch to Simulation to view / name surfaces.")

    def _current_receiver_row(self):
        r = self._rec_tbl._tbl.currentRow()            # currentRow works with SelectItems tables
        return r if r >= 0 else (0 if self._rec_tbl._tbl.rowCount() else -1)

    def _ensure_results(self):
        """Run allIRs.py in output/ to build IR_broad_<r>_<s>.npy, then plot the current receiver.
        Skips if a run is already going (reassigning a live QThread aborts the app), and if the
        broadband files already exist (avoids recomputing every time Results is opened)."""
        if not self._proj_root:
            self._sb.showMessage("Open a project folder first."); return
        t = getattr(self, "_allirs_thread", None)
        if t is not None and t.isRunning():
            return                                       # already computing; done handler will plot
        out = os.path.join(self._proj_root, "output")
        script = self._find_common_script("allIRs.py")
        if script is None:
            self._sb.showMessage("allIRs.py not found next to gui.py."); return
        self._allirs_thread = SolverThread([sys.executable, script], out)
        self._allirs_thread.output.connect(self._log.append)
        self._allirs_thread.done.connect(self._on_allirs_done)
        self._allirs_thread.start()
        self._sb.showMessage("Computing broadband IRs…")

    def _on_allirs_done(self, rc):
        self._sb.showMessage("Broadband IRs ready." if rc == 0 else f"allIRs.py failed (rc={rc}).")
        if rc == 0:
            self._run_acouparams()                         # rebuild params.json from the fresh IRs
        if getattr(self, "_view_mode", None) == "results":
            if self._results_sel:                          # honour the current multi-selection
                self._apply_results_selection(mirror_table=True)
            else:
                self._results.show_receiver(self._current_receiver_row())

    def _run_acouparams(self):
        """Compute the ISO-3382 params (acouparams.py -> params.json) in output/, after allIRs."""
        t = getattr(self, "_acou_thread", None)
        if t is not None and t.isRunning():
            return
        script = self._find_common_script("acouparams.py")
        if script is None:
            return
        self._acou_thread = SolverThread([sys.executable, script],
                                         os.path.join(self._proj_root, "output"))
        self._acou_thread.output.connect(self._log.append)
        self._acou_thread.done.connect(self._on_acouparams_done)
        self._acou_thread.start()

    def _on_acouparams_done(self, rc):
        if rc != 0:
            self._sb.showMessage(f"acouparams.py failed (rc={rc}) — parameters unavailable."); return
        if getattr(self, "_view_mode", None) == "results" and self._results.current_metric() != "IR + Spectrum":
            self._apply_results_selection(mirror_table=False)   # re-render now params.json is fresh

    def _toggle_scene_theme(self):
        """Flip the viewport between Dark and Light and repaint."""
        set_scene_mode(not SCENE_LIGHT_MODE)
        self._restyle_scene_theme()
        self._gl.update()

    def _restyle_scene_theme(self):
        """Recolour the placement-hint text so it stays readable in both scene modes.
        (The Dark/Light button was replaced by the RMB scene-menu toggle.)"""
        hint_col = "#202020" if SCENE_LIGHT_MODE else PAL["subtext"]
        self._hint_lbl.setStyleSheet(f"color:{hint_col};font-size:9px;")

    def _toggle_placement(self, kind):
        btns = {"source": self._src_btn, "receiver": self._rec_btn, "slice": self._slice_btn}
        on = btns[kind].isChecked()
        for k, b in btns.items():             # mutual exclusion
            if k != kind:
                b.setChecked(False)
        if not on:
            self._gl.exit_placement_mode()
            self._sb.showMessage("Ready.")
            return
        if kind == "source":
            self._gl.enter_placement_mode(GLViewer.MODE_SRC)
            self._tabs.setCurrentIndex(1)
            self._sb.showMessage("SOURCE placement: drag LMB to position (hold X/Y/Z to lock an axis); release to place. Esc to cancel")
        elif kind == "receiver":
            self._gl.enter_placement_mode(GLViewer.MODE_REC)
            self._tabs.setCurrentIndex(2)
            self._sb.showMessage("RECEIVER placement: drag LMB to position (hold X/Y/Z to lock an axis); release to place. Esc to cancel")
        else:  # slice
            self._gl.enter_slice_mode()
            self._sb.showMessage("SLICE: hold X / Y / Z and drag to move that plane; release to set. Esc to finish")

    def _on_slice_changed(self, x, y, z):
        """Scene pick -> reflect in the Animation spin boxes (without echoing back)."""
        self._sync_slice_fields(x, y, z)
        self._gl.clear_animation()      # slice moved -> particle mask/sources stale; reload on next play
        self._sb.showMessage(f"Slice planes  x={x:.3f}  y={y:.3f}  z={z:.3f} m")

    def _sync_slice_fields(self, x, y, z):
        self._slice_sync = True
        self._slice_sb['x'].setValue(x); self._slice_sb['y'].setValue(y); self._slice_sb['z'].setValue(z)
        self._slice_sync = False

    def _on_slice_field(self, _=None):
        """User typed a slice coordinate -> move the planes in the scene."""
        if self._slice_sync:
            return
        self._gl.set_slices(self._slice_sb['x'].value(),
                            self._slice_sb['y'].value(),
                            self._slice_sb['z'].value())
        self._gl.clear_animation()      # slice moved -> reload on next play

    # ── slice-pressure animation ──
    def _toggle_all_planes(self):
        all_on = all(cb.isChecked() for cb in self._plane_chk.values())
        for cb in self._plane_chk.values():
            cb.setChecked(not all_on)           # drives set_plane_visible via toggled
        self._planes_all_btn.setText("Show all" if all_on else "Hide all")

    def _toggle_all_pml(self):
        all_on = all(c.isChecked() for c in self._pml.values())
        for c in self._pml.values():
            c.setChecked(not all_on)
        self._sync_pml_all_btn()

    def _sync_pml_all_btn(self):
        on = all(c.isChecked() for c in self._pml.values())
        self._pml_all_btn.setText("Deselect all" if on else "Select all")

    def _on_prange(self, _=None):
        self._gl.set_pressure_range(self._pmin.value(), self._pmax.value())

    def _toggle_animation(self):
        t = getattr(self, "_anim_timer", None)
        if t is not None and t.isActive():          # playing -> pause IN PLACE (no reset)
            self._gl._anim_playing = False; self._gl.update()
            t.stop(); self._anim_btn.setText("▶︎")
            return
        if self._gl._anim is None:                   # (re)load only when not loaded yet
            if not self._proj_root:
                QMessageBox.warning(self, "No project", "Open a project first."); return
            frames = self._gl.load_animation(os.path.join(self._proj_root, "output"))
            if frames <= 0:
                QMessageBox.information(self, "No animation",
                    "No pressureX/Y/Z.bin found in output/. Run a full simulation first."); return
            self._gl.set_pressure_range(self._pmin.value(), self._pmax.value())
            self._scrub_guard = True
            self._anim_slider.setRange(0, frames - 1); self._scrub_guard = False
        if getattr(self, "_anim_timer", None) is None:
            self._anim_timer = QTimer(self)
            self._anim_timer.timeout.connect(self._gl.step_frame)
        self._anim_timer.start(max(1, int(1000 / self._fps.value())))
        self._gl._anim_playing = True
        self._anim_btn.setText("⏸")
        self._sb.showMessage(f"Playing at {self._fps.value()} fps")

    def _on_scrub(self, v):
        if self._scrub_guard:                        # set by playback, not the user
            return
        if self._gl._anim is None and self._proj_root:   # first touch before any play -> load
            if self._gl.load_animation(os.path.join(self._proj_root, "output")) > 0:
                self._gl.set_pressure_range(self._pmin.value(), self._pmax.value())
        self._gl.seek(v)

    def _on_anim_frame(self, k):
        self._scrub_guard = True                     # reflect playback without re-seeking
        self._anim_slider.setValue(k)
        self._scrub_guard = False

    def _on_point_placed(self, mode, x, y, z):
        self._remember_points_for_undo()
        if mode == "source":
            self._src_tbl.add_point(x, y, z, 1.0)
        else:
            self._rec_tbl.add_point(x, y, z)
        self._refresh_gl_points()
        self._sb.showMessage(f"Placed {mode} at ({x:.3f}, {y:.3f}, {z:.3f}) m  —  drag again to add another, Esc to finish")

    @staticmethod
    def _axis_vals(s, e, step):
        """Equally-spaced values from s to e by |step|. step<=0 (or s==e) -> single value [s]
        (no multiplication on that axis)."""
        s, e, step = float(s), float(e), float(step)
        if step <= 0 or abs(e - s) < 1e-12:
            return [s]
        lo, hi = (s, e) if e >= s else (e, s)
        return [round(v, 6) for v in np.arange(lo, hi + step * 0.5, step)]

    @staticmethod
    def _sphere_azel(center, r, az_step, el_min, el_max, el_step):
        """Equiangular spherical shell: az 0..360 by az_step, el el_min..el_max by el_step (deg). az/el
        step 0 = single. Covers listening-room rings (el=0) and directivity grids. Poles deduped."""
        cx, cy, cz = center
        azs = MainWindow._axis_vals(0.0, 360.0 - (az_step if az_step > 0 else 0.0), az_step) if az_step > 0 else [0.0]
        els = MainWindow._axis_vals(el_min, el_max, el_step)
        seen, out = set(), []
        for el in els:
            er = math.radians(el)
            for az in azs:
                ar = math.radians(az)
                p = (round(cx + r*math.cos(er)*math.cos(ar), 6),
                     round(cy + r*math.cos(er)*math.sin(ar), 6),
                     round(cz + r*math.sin(er), 6))
                if p not in seen:
                    seen.add(p); out.append(list(p))
        return out

    @staticmethod
    def _sphere_fib(center, r, n):
        """N near-uniform points on a shell via the golden-spiral (Fibonacci) distribution."""
        cx, cy, cz = center; n = max(1, int(n)); ga = math.pi * (3 - math.sqrt(5))
        out = []
        for i in range(n):
            z = 1 - 2*(i + 0.5)/n; rr = math.sqrt(max(0.0, 1 - z*z)); ph = ga * i
            out.append([round(cx + r*rr*math.cos(ph), 6), round(cy + r*rr*math.sin(ph), 6), round(cz + r*z, 6)])
        return out

    def _make_grid(self, kind):
        """Placement-mode 'G' (kind from the active S/R mode): build a box grid or spherical shell of
        points, then group-move it."""
        res = self._grid_dialog(kind)
        if not res:
            return
        shape, pr = res
        if shape == "box":
            import itertools
            pts = list(itertools.product(*[self._axis_vals(*a) for a in pr["ax"]]))
        elif shape == "azel":
            pts = self._sphere_azel(pr["center"], pr["r"], pr["az_step"], pr["el_min"], pr["el_max"], pr["el_step"])
        else:
            pts = self._sphere_fib(pr["center"], pr["r"], pr["n"])
        if not pts:
            return
        if len(pts) > 4000:
            B = QMessageBox.StandardButton
            if QMessageBox.question(self, "Large grid",
                    f"This creates {len(pts)} points. Continue?", B.Yes | B.No, B.No) != B.Yes:
                return
        self._remember_points_for_undo()
        tbl = self._src_tbl if kind == "source" else self._rec_tbl
        start = tbl._tbl.rowCount()
        for x, y, z in pts:
            tbl.add_point(x, y, z)
        self._refresh_gl_points()
        self._gl.start_group_move(kind, list(range(start, start + len(pts))))
        self._sb.showMessage(f"{len(pts)} {kind}s created — move with the mouse, click/Enter to drop, Esc to keep.")

    def _on_group_xf(self, items):
        """A point group move/rotate (or grid drop) committed -> write coords back into the tables.
        Snapshot ALL coords FIRST: each update_point fires changed -> _refresh_gl_points, which rebuilds
        gl._sources from the table, clobbering the not-yet-synced points if we read them mid-loop."""
        self._remember_points_for_undo()
        snap = []
        for kind, i in items:
            lst = self._gl._sources if kind == "source" else self._gl._receivers
            if 0 <= i < len(lst):
                p = lst[i]; snap.append((kind, i, float(p[0]), float(p[1]), float(p[2])))
        for kind, i, x, y, z in snap:
            (self._src_tbl if kind == "source" else self._rec_tbl).update_point(i, x, y, z)
        self._sb.showMessage(f"Updated {len(snap)} point(s).")

    def _on_points_deleted(self, items):
        """Edit-mode multi-delete -> remove the selected rows (descending, per table)."""
        if not items:
            return
        self._remember_points_for_undo()
        for kind, tbl in (("source", self._src_tbl), ("receiver", self._rec_tbl)):
            rows = sorted((i for k, i in items if k == kind), reverse=True)
            for r in rows:
                if 0 <= r < tbl._tbl.rowCount():
                    tbl._tbl.removeRow(r)
                    if tbl._has_pose and r < len(tbl.poses):
                        del tbl.poses[r]
            if rows:
                tbl.changed.emit()
        self._sb.showMessage(f"Deleted {len(items)} point(s).")

    def _grid_dialog(self, kind):
        """Box grid or spherical shell of `kind`. Returns (shape, params) or None. shapes:
        'box' -> {ax}; 'azel' -> {center,r,az_step,el_min,el_max,el_step}; 'fib' -> {center,r,n}."""
        from PySide6.QtWidgets import QComboBox, QStackedWidget, QWidget
        dlg = QDialog(self); dlg.setWindowTitle(f"Create {kind} array")
        dlg.setStyleSheet(f"QDialog{{background:{PAL['bg']};color:{PAL['text']};}}QLabel{{color:{PAL['text']};}}")
        root = QVBoxLayout(dlg)
        shape = QComboBox(); shape.addItems(["Box grid", "Sphere: azimuth × elevation", "Fibonacci Sphere"])
        root.addWidget(shape)
        stack = QStackedWidget(); root.addWidget(stack)

        def sb(v, mn=-1e4, mx=1e4, dec=3, suf=" m"):             # unit shown next to the value
            s = QDoubleSpinBox(); s.setRange(mn, mx); s.setDecimals(dec); s.setValue(v); s.setSuffix(suf); return s

        # page 0 — box grid (start/end/step per axis; step 0 = single)
        p0 = QWidget(); g = QGridLayout(p0)
        for c, h in enumerate(("start", "end", "step (0=single)")):
            g.addWidget(QLabel(h), 0, c + 1)
        box = {}
        for r, a in enumerate("xyz"):
            g.addWidget(QLabel(a), r + 1, 0); box[a] = (sb(0.0), sb(1.0), sb(0.0))
            for c, w in enumerate(box[a]):
                g.addWidget(w, r + 1, c + 1)
        g.setRowStretch(4, 1)                                     # hug top (stack is sized to the taller pages)
        stack.addWidget(p0)

        # page 1 — equiangular az × el shell (defaults: 8-point horizontal ring at head height)
        p1 = QWidget(); f1 = QGridLayout(p1)
        az = dict(cx=sb(0.0), cy=sb(0.0), cz=sb(1.6), r=sb(1.0, 0), azs=sb(45.0, 0, 360, suf=" °"),
                  elmn=sb(0.0, -90, 90, suf=" °"), elmx=sb(0.0, -90, 90, suf=" °"), els=sb(0.0, 0, 180, suf=" °"))
        for i, (lb, w) in enumerate([("center x", az["cx"]), ("center y", az["cy"]), ("center z", az["cz"]),
                                     ("radius", az["r"]), ("az step", az["azs"]),
                                     ("el min", az["elmn"]), ("el max", az["elmx"]), ("el step (0=single)", az["els"])]):
            f1.addWidget(QLabel(lb), i, 0); f1.addWidget(w, i, 1)
        f1.setRowStretch(8, 1)
        stack.addWidget(p1)

        # page 2 — Fibonacci near-uniform shell
        p2 = QWidget(); f2 = QGridLayout(p2)
        fib = dict(cx=sb(0.0), cy=sb(0.0), cz=sb(1.6), r=sb(1.0, 0), n=sb(32, 1, 100000, 0, suf=""))
        for i, (lb, w) in enumerate([("center x", fib["cx"]), ("center y", fib["cy"]), ("center z", fib["cz"]),
                                     ("radius", fib["r"]), ("N points", fib["n"])]):
            f2.addWidget(QLabel(lb), i, 0); f2.addWidget(w, i, 1)
        f2.setRowStretch(5, 1)
        stack.addWidget(p2)

        shape.currentIndexChanged.connect(stack.setCurrentIndex)
        row = QHBoxLayout(); ok = QPushButton("Create"); cancel = QPushButton("Cancel")
        ok.clicked.connect(dlg.accept); cancel.clicked.connect(dlg.reject)
        row.addStretch(); row.addWidget(cancel); row.addWidget(ok); root.addLayout(row)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        i = shape.currentIndex()
        if i == 0:
            return "box", {"ax": tuple((box[a][0].value(), box[a][1].value(), box[a][2].value()) for a in "xyz")}
        if i == 1:
            return "azel", {"center": (az["cx"].value(), az["cy"].value(), az["cz"].value()), "r": az["r"].value(),
                            "az_step": az["azs"].value(), "el_min": az["elmn"].value(),
                            "el_max": az["elmx"].value(), "el_step": az["els"].value()}
        return "fib", {"center": (fib["cx"].value(), fib["cy"].value(), fib["cz"].value()),
                       "r": fib["r"].value(), "n": int(fib["n"].value())}

    def _on_table_row_selected(self, kind):
        """A row was selected in a point table -> highlight it in the 3D scene.
        Guarded so the reverse (G-hover -> select row) does not loop."""
        if getattr(self, "_pt_sync_guard", False):
            return
        tbl = self._src_tbl if kind == "source" else self._rec_tbl
        idx = tbl._tbl.currentRow()                    # SelectItems: selectedRows() is empty
        self._gl.set_highlight(kind if idx >= 0 else None, idx)
        if kind == "receiver" and getattr(self, "_view_mode", None) == "results" \
                and not getattr(self, "_sel_sync", False):
            self._results_sel = {ix.row() for ix in tbl._tbl.selectedIndexes()}
            self._apply_results_selection(mirror_table=False)   # table already reflects it

    def _on_scene_receiver_pick(self, idx):
        """Click a receiver in the scene (Results mode) -> toggle it in the shared selection."""
        if idx < 0 or getattr(self, "_view_mode", None) != "results":
            return
        self._results_sel ^= {idx}
        self._apply_results_selection(mirror_table=True)

    def _apply_results_selection(self, mirror_table=True):
        """Render the current Results selection and keep scene highlight + table in sync.
        0 -> hint, 1 -> full allIRs plot, >1 -> overlay comparison."""
        recs = sorted(r for r in self._results_sel if r >= 0)
        self._gl.set_selected_receivers(recs)
        if mirror_table:
            from PySide6.QtCore import QItemSelection, QItemSelectionModel
            tbl = self._rec_tbl._tbl; sm = tbl.selectionModel()
            self._sel_sync = True
            sm.clearSelection()
            for r in recs:
                sm.select(QItemSelection(tbl.model().index(r, 0),
                                         tbl.model().index(r, tbl.columnCount() - 1)),
                          QItemSelectionModel.SelectionFlag.Select)
            self._sel_sync = False
        metric = self._results.current_metric()
        if metric == "Summary (avg)":                  # room-average, all receivers, cross-sim compare
            self._render_summary(); return
        if not recs:
            self._results.show_png(None); return
        if metric == "IR + Spectrum":
            self._results.set_export(None)             # no tabular export for IR/Spectrum
            self._render_overlay(recs) if len(recs) > 1 else self._results.show_receiver(recs[0])
        else:
            self._render_params(recs, metric)

    def _render_overlay(self, recs):
        """Overlay IR + frequency response of several receivers from the saved IR_broad_*.npy (no
        re-run). One column per source (rows: IR, spectrum), one colour per receiver, GUI-themed."""
        import glob, re
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = os.path.join(self._proj_root, "output")
        fs = 44100                                     # allIRs saves broadband IRs at this rate
        FG, BG = PAL['text'], PAL['bg']
        names = self._rec_tbl.get_names()              # legend follows the table's receiver names
        irs = {}                                       # (rec, src) -> IR
        for rec in recs:
            for f in glob.glob(os.path.join(out, f"IR_broad_{rec}_*.npy")):
                m = re.search(r"IR_broad_(\d+)_(\d+)\.npy$", f)
                if m:
                    irs[(rec, int(m.group(2)))] = np.load(f)
        if not irs:
            self._results.show_png(None); return
        srcs = sorted({src for _, src in irs})
        fig, axes = plt.subplots(2, len(srcs), figsize=(6 + 3 * len(srcs), 7), squeeze=False)
        for c, src in enumerate(srcs):
            ax_t, ax_f = axes[0, c], axes[1, c]
            for i, rec in enumerate(recs):
                ir = irs.get((rec, src))
                if ir is None:
                    continue
                col = _plot_color(i)
                name = names[rec] if rec < len(names) and names[rec] else f"receiver {rec}"
                ax_t.plot(np.arange(ir.size) / fs, ir, lw=0.7, color=col, label=name)
                H = np.abs(np.fft.rfft(ir)); fr = np.fft.rfftfreq(ir.size, 1 / fs)
                ax_f.semilogx(fr[1:], 20 * np.log10(H[1:] / 2e-5 + 1e-12), lw=0.8, color=col, label=name)   # dB re 20 uPa: direct sound at 1 m reads Lp,1m
            tag = f"source {src} — " if len(srcs) > 1 else "Receiver comparison — "
            ax_t.set_title(f"{tag}impulse response")
            ax_t.set_xlabel("Time [s]"); ax_t.set_ylabel("Amplitude")
            ax_f.set_title(f"{tag}frequency response")
            ax_f.set_xlabel("Frequency [Hz]"); ax_f.set_ylabel("SPL [dB re 20 µPa]"); ax_f.set_xlim(20, fs / 2)
        fig.patch.set_facecolor(BG)
        for ax in axes.ravel():                        # GUI dark theme: bg + white axes/text/grid
            ax.set_facecolor(BG)
            for sp in ax.spines.values(): sp.set_color(FG)
            ax.tick_params(colors=FG, which='both')
            ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG); ax.title.set_color(FG)
            ax.grid(True, which='both', color=PAL['border'], alpha=0.6)
            leg = ax.legend(fontsize=8)
            leg.get_frame().set_facecolor(PAL['panel']); leg.get_frame().set_edgecolor(PAL['border'])
            for t in leg.get_texts(): t.set_color(FG)
        fig.tight_layout()
        path = os.path.join(out, "IR_overlay.png")
        fig.savefig(path, dpi=120, facecolor=fig.get_facecolor()); plt.close(fig)
        self._results.show_png(path)

    def _render_params(self, recs, param):
        """Plot one ISO-3382 parameter vs octave band, one line per receiver, from params.json
        (acouparams.py). Nulls (undefined bands) are dropped. GUI-themed."""
        self._results.set_export(None)                 # cleared until a table is successfully built
        import json
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PySide6.QtGui import QPixmap
        out = os.path.join(self._proj_root, "output")
        pj = os.path.join(out, "params.json")
        if not os.path.isfile(pj):
            self._results._label.setPixmap(QPixmap())
            self._results._label.setText("Computing parameters… (acouparams.py)")
            self._run_acouparams(); return
        with open(pj) as f:
            data = json.load(f)
        names = self._rec_tbl.get_names()
        FG, BG = PAL['text'], PAL['bg']
        units = {"T20": "s", "T30": "s", "EDT": "s", "C50": "dB", "C80": "dB", "D50": ""}
        # One panel per source when acouparams split them ("sources": {src: {rec: params}}),
        # else the summed entry. Lines, not bars: one connected line per receiver (octave centres
        # are the x ticks), the selected-receiver average dashed.
        srcs = sorted(data.get("sources", {}), key=int)
        panels = [(f"source {sv}", data["sources"][sv]) for sv in srcs] if len(srcs) > 1 else [(None, data)]
        fig, axes = plt.subplots(len(panels), 1, figsize=(8, 4.2 * len(panels)), sharex=True, squeeze=False)
        export_cols, export_rows, ref_bands, any_drawn = [], None, None, False
        for ax, (ptitle, pdata) in zip(axes[:, 0], panels):
            drawn = [(rec, pdata.get(str(rec))) for rec in recs
                     if pdata.get(str(rec)) and param in pdata.get(str(rec))]
            ax.set_facecolor(BG)
            for sp in ax.spines.values(): sp.set_color(FG)
            ax.tick_params(colors=FG, which='both')
            ax.yaxis.label.set_color(FG); ax.title.set_color(FG)
            if not drawn:
                ax.set_title(f"{ptitle + ' — ' if ptitle else ''}no {param} values for the selected receiver(s)")
                continue
            any_drawn = True
            ref_bands = drawn[0][1]["bands"]
            x = np.arange(len(ref_bands))
            mat = np.array([[v if v is not None else np.nan for v in d[param]] for _, d in drawn])
            for i, (rec, d) in enumerate(drawn):
                col = _plot_color(recs.index(rec))
                name = names[rec] if rec < len(names) and names[rec] else f"receiver {rec}"
                ax.plot(x, mat[i], color=col, lw=2.2, label=name)
            # spatial average of the SELECTED receivers: ISO energy-mean for the dB params (C50/C80),
            # plain arithmetic mean for times (T20/T30/EDT) and the D50 ratio. NaN bands are ignored.
            with np.errstate(invalid='ignore', divide='ignore'):
                if param in ("C50", "C80"):
                    avg = 10 * np.log10(np.nanmean(10 ** (mat / 10.0), axis=0))
                else:
                    avg = np.nanmean(mat, axis=0)
            ax.plot(x, avg, color="#d7dde3", lw=2.0, ls="--", label="avg (selected)")
            ax.set_xticks(x); ax.set_xticklabels([str(bnd) for bnd in ref_bands])
            ax.set_ylabel(f"{param} [{units[param]}]" if units[param] else param)
            ax.set_title(f"{ptitle + ' — ' if ptitle else 'Receiver comparison — '}{param} per octave band")
            ax.grid(True, color=PAL['border'], lw=0.5, alpha=0.6)
            leg = ax.legend(fontsize=8)
            leg.get_frame().set_facecolor(PAL['panel']); leg.get_frame().set_edgecolor(PAL['border'])
            for t in leg.get_texts(): t.set_color(FG)
            # table behind the plot for Ctrl+C / RMB export (rows = bands, cols = receivers + avg, per source)
            def _cell(v): return None if v is None or not np.isfinite(v) else float(v)
            tag = f" ({ptitle})" if ptitle else ""
            export_cols += [(names[r] if r < len(names) and names[r] else f"receiver {r}") + tag
                            for r, _ in drawn] + ["avg (selected)" + tag]
            cols = [[_cell(d[param][j]) for _, d in drawn] + [_cell(avg[j])] for j in range(len(ref_bands))]
            export_rows = cols if export_rows is None else [r0 + r1 for r0, r1 in zip(export_rows, cols)]
        if not any_drawn:
            plt.close(fig)
            self._results._label.setPixmap(QPixmap())
            self._results._label.setText(f"No {param} values for the selected receiver(s)."); return
        self._results.set_export({
            "title": param, "rowheader": "band [Hz]", "columns": export_cols,
            "rows": [(str(bnd), export_rows[j]) for j, bnd in enumerate(ref_bands)]})
        axes[-1, 0].set_xlabel("Octave band [Hz]"); axes[-1, 0].xaxis.label.set_color(FG)
        fig.patch.set_facecolor(BG)
        fig.tight_layout()
        p = os.path.join(out, "IR_params.png")
        fig.savefig(p, dpi=120, facecolor=fig.get_facecolor()); plt.close(fig)
        self._results.show_png(p)

    # ── Summary: single-number ratings, room-average, cross-simulation compare ──
    _SUMMARY = {"T20": ([500, 1000], "T20 mid", "s"), "T30": ([500, 1000], "T30 mid", "s"),
                "EDT": ([500, 1000], "EDT mid", "s"), "C50": ([500, 1000, 2000], "C50", "dB"),
                "C80": ([500, 1000, 2000], "C80(3)", "dB"), "D50": ([500, 1000, 2000], "D50", "")}

    def _room_avg(self, data, param):
        """(bands, per-band room average over all receivers). Energy-mean for dB params."""
        rows = [[v if v is not None else np.nan for v in d[param]]
                for d in data.values() if isinstance(d, dict) and param in d]
        if not rows:
            return None, None
        bands = next(d["bands"] for d in data.values() if isinstance(d, dict) and "bands" in d)
        mat = np.array(rows)
        with np.errstate(invalid='ignore', divide='ignore'):
            avg = (10 * np.log10(np.nanmean(10 ** (mat / 10.0), axis=0)) if param in ("C50", "C80")
                   else np.nanmean(mat, axis=0))
        return bands, avg

    def _single_number(self, param, bands, band_avg):
        """ISO single-number rating: mid (500-1k) for T/EDT, 500-2k for C/D; energy-mean for dB."""
        idx = [bands.index(b) for b in self._SUMMARY[param][0] if b in bands]
        vals = np.array([band_avg[i] for i in idx], float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        return float(10 * np.log10(np.mean(10 ** (vals / 10.0)))) if param in ("C50", "C80") else float(vals.mean())

    def _add_compare_sim(self, path):
        import json
        try:
            data = json.load(open(path))
        except Exception as ex:
            self._sb.showMessage(f"Could not load {path}: {ex}"); return
        label = os.path.basename(os.path.dirname(os.path.dirname(path)))   # .../PROJECT/output/params.json
        self._compare.append((label or os.path.basename(path), data))
        self._render_summary()

    def _clear_compare_sims(self):
        self._compare = []
        self._render_summary()

    def _render_summary(self):
        """2x3 grid of single-number ratings, one bar per simulation (current + compared), by folder."""
        import json
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from PySide6.QtGui import QPixmap
        out = os.path.join(self._proj_root, "output")
        pj = os.path.join(out, "params.json")
        if not os.path.isfile(pj):
            self._results._label.setPixmap(QPixmap())
            self._results._label.setText("Computing parameters… (acouparams.py)")
            self._run_acouparams(); return
        series = [(os.path.basename(os.path.normpath(self._proj_root)), json.load(open(pj)))] + self._compare
        params = list(self._SUMMARY)
        FG, BG = PAL['text'], PAL['bg']
        table = {}                                     # param -> [value per series]
        for p in params:
            vals = []
            for _, data in series:
                b, a = self._room_avg(data, p)
                vals.append(self._single_number(p, b, a) if b is not None else None)
            table[p] = vals
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        for ax, p in zip(axes.ravel(), params):
            ys = [v if v is not None else np.nan for v in table[p]]
            ax.bar(range(len(series)), ys, width=0.35, color=[_plot_color(i) for i in range(len(series))])
            ax.set_xlim(-0.7, len(series) - 0.3)       # thin bars, not stretched to fill the panel
            lbl, unit = self._SUMMARY[p][1], self._SUMMARY[p][2]
            ax.set_title(lbl, fontsize=10, color=FG)
            ax.set_ylabel(unit, color=FG)
            ax.set_xticks([]); ax.set_facecolor(BG); ax.grid(False)
            for sp in ax.spines.values(): sp.set_color(FG)
            ax.tick_params(colors=FG)
        fig.patch.set_facecolor(BG)
        handles = [Patch(color=_plot_color(i), label=lbl) for i, (lbl, _) in enumerate(series)]
        leg = fig.legend(handles=handles, loc='lower center', ncol=min(4, len(series)), fontsize=8)
        leg.get_frame().set_facecolor(PAL['panel']); leg.get_frame().set_edgecolor(PAL['border'])
        for t in leg.get_texts(): t.set_color(FG)
        fig.suptitle("Single-number ratings — simulation comparison", color=FG)
        fig.tight_layout(rect=[0, 0.06, 1, 0.96])
        p = os.path.join(out, "IR_summary.png")
        fig.savefig(p, dpi=120, facecolor=fig.get_facecolor()); plt.close(fig)
        self._results.show_png(p)
        # export: rows = parameters, columns = simulations
        def _c(v): return None if v is None or not np.isfinite(v) else float(v)
        self._results.set_export({
            "title": "summary", "rowheader": "parameter",
            "columns": [lbl for lbl, _ in series],
            "rows": [(f"{self._SUMMARY[p][1]} [{self._SUMMARY[p][2]}]".strip(" []"),
                      [_c(v) for v in table[p]]) for p in params]})

    def _on_point_hovered(self, kind, idx):
        """G-mode hover over a point -> select its row in the matching table
        so the user sees which one it is. Guarded against feedback loops."""
        self._pt_sync_guard = True
        if kind == "source":
            self._src_tbl._tbl.selectRow(idx) if idx >= 0 else \
                self._src_tbl._tbl.clearSelection()
        elif kind == "receiver":
            self._rec_tbl._tbl.selectRow(idx) if idx >= 0 else \
                self._rec_tbl._tbl.clearSelection()
        else:
            self._src_tbl._tbl.clearSelection()
            self._rec_tbl._tbl.clearSelection()
        self._pt_sync_guard = False

    def _on_point_moved(self, kind, idx, x, y, z):
        # Drag-to-move finished: write the new coordinates into the table row.
        # (The undo snapshot was already taken at drag start, so the move can
        # be reverted with Ctrl+Z.)
        tbl = self._src_tbl if kind == "source" else self._rec_tbl
        tbl.update_point(idx, x, y, z)
        self._refresh_gl_points()
        self._sb.showMessage(f"Moved {kind} #{idx} to ({x:.3f}, {y:.3f}, {z:.3f}) m")

    def _on_point_removed(self, kind, idx):
        self._remember_points_for_undo()
        tbl = self._src_tbl if kind == "source" else self._rec_tbl
        tbl._tbl.blockSignals(True)
        tbl._tbl.removeRow(idx)
        tbl._tbl.blockSignals(False)
        self._refresh_gl_points()
        self._sb.showMessage(f"Removed {kind} #{idx}")

    def keyPressEvent(self, e):
        """Global key bindings. S=source, R=receiver, T=reset view, X/Y/Z=axis lock."""
        if e.matches(QKeySequence.StandardKey.Undo):
            self._undo_point_change()
            e.accept(); return
        key = e.key()
        # Draw mode reserves S/R/L for prop transforms (handled by the GL widget when focused); never
        # place sources/receivers/slices here. Placement is a Simulation-mode action.
        if getattr(self, "_view_mode", "sim") == "draw" and key in (
                Qt.Key.Key_S, Qt.Key.Key_R, Qt.Key.Key_L):
            e.accept(); return
        if key == Qt.Key.Key_S:
            checked = not self._src_btn.isChecked()
            self._src_btn.setChecked(checked)
            self._toggle_placement("source")
            e.accept(); return
        if key == Qt.Key.Key_R:
            checked = not self._rec_btn.isChecked()
            self._rec_btn.setChecked(checked)
            self._toggle_placement("receiver")
            e.accept(); return
        if key == Qt.Key.Key_L:
            self._slice_btn.setChecked(not self._slice_btn.isChecked())
            self._toggle_placement("slice")
            e.accept(); return
        if key == Qt.Key.Key_G and self._gl._mode in (GLViewer.MODE_SRC, GLViewer.MODE_REC):
            self._make_grid("source" if self._gl._mode == GLViewer.MODE_SRC else "receiver")  # S/R + G -> grid
            e.accept(); return
        if key == Qt.Key.Key_T and self._gl._mode == GLViewer.MODE_NONE:
            # reset the 3D view to the exact initial visualization
            self._gl._reset_view()
            self._sb.showMessage("View reset.")
            e.accept(); return
        if key == Qt.Key.Key_Space and self._gl._mode == GLViewer.MODE_NONE:
            on = not self._gl._nav_mode
            self._gl.set_nav_mode(on)
            self._sb.showMessage("Navigation mode: WAXD fly, mouse looks." if on else "Navigation mode off.")
            e.accept(); return
        if key == Qt.Key.Key_Escape and self._gl._nav_mode:
            self._gl.set_nav_mode(False)
            self._sb.showMessage("Navigation mode off.")
            e.accept(); return
        if key == Qt.Key.Key_Escape:
            was_placing = (self._src_btn.isChecked() or self._rec_btn.isChecked()
                           or self._slice_btn.isChecked())
            self._src_btn.setChecked(False)
            self._rec_btn.setChecked(False)
            self._slice_btn.setChecked(False)
            self._gl.exit_placement_mode()
            if not was_placing:
                self._layer_list.setCurrentRow(-1)
                self._gl.clear_selection()
                self._show_no_layer_selected()
                self._sb.showMessage("Layer deselected.")
            else:
                self._sb.showMessage("Ready.")
            e.accept(); return
        # forward R (reset) and X/Y/Z to GL
        self._gl.keyPressEvent(e)

    def keyReleaseEvent(self, e):
        # Forward key releases (X/Y/Z axis-unlock) to the GL viewer.
        self._gl.keyReleaseEvent(e)

    # ── Scene tab ────────────────────────────────────────────
    def _build_scene_tab(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(8)

        grp_proj = QGroupBox("Project")
        pv = QVBoxLayout(grp_proj)
        self._proj_label = QLabel("not set")
        self._proj_label.setStyleSheet(f"color:{PAL['subtext']};font-size:11px;")
        pv.addWidget(self._proj_label)
        pb = QHBoxLayout()
        for lbl, fn in [("\U0001F5C0\uFE0E  New",  self._new_project),
                        ("\U0001F5C1\uFE0E  Open", self._open_project),
                        ("\U0001F5AB\uFE0E  Save as...", self._save_project_as),
                        ("\U0001F5AB\uFE0E  Save", self._save_project)]:
                        
            b = QPushButton(lbl); b.clicked.connect(fn); pb.addWidget(b)
        pv.addLayout(pb)
        lay.addWidget(grp_proj)

        grp_obj = QGroupBox("Scene file (.obj)")
        o_lay = QHBoxLayout(grp_obj)
        self._obj_label = QLabel("none")
        self._obj_label.setStyleSheet(f"color:{PAL['subtext']};font-size:11px;")
        o_lay.addWidget(self._obj_label, 1)
        btn2 = QPushButton("Import OBJ"); btn2.clicked.connect(self._load_obj)
        o_lay.addWidget(btn2)
        lay.addWidget(grp_obj)

        bbox_row = QHBoxLayout()
        self._bbox_chk = QCheckBox("Show bounding box")
        self._bbox_chk.toggled.connect(self._gl.set_bbox_visible)
        self._bbox_chk.setChecked(False)   # off by default; set after connect so GL syncs
        bbox_row.addWidget(self._bbox_chk); bbox_row.addStretch()
        lay.addLayout(bbox_row)

        # Layer list kept alive (still populated + synced from the material table selection,
        # which drives highlight/toggle) but NOT shown -- the material table IS the layer UI.
        self._layer_list = QListWidget()
        self._layer_list.setMaximumHeight(100)
        self._layer_list.currentRowChanged.connect(self._on_layer_row_changed)

        vis_row = QHBoxLayout()
        toggle_btn = QPushButton("Toggle")
        toggle_btn.clicked.connect(self._toggle_selected_vis)
        vis_row.addWidget(toggle_btn)
        self._layers_all_btn = QPushButton("Hide all")   # one button, name flips with state
        self._layers_all_btn.clicked.connect(self._toggle_all_layers)
        vis_row.addWidget(self._layers_all_btn)
        vis_row.addStretch(); lay.addLayout(vis_row)

        lay.addWidget(QLabel("Material - α per octave band"))
        self._mat_panel = MaterialPanel()
        self._mat_panel.status.connect(lambda t: self._sb.showMessage(t))   # shared status bar
        self._mat_panel.rowSelected.connect(self._on_material_row_selected)
        self._mat_panel.fitBegan.connect(self._begin_fit_log)
        self._mat_panel.fitLog.connect(self._append_fit_log)
        self._mat_panel.fitAllDone.connect(self._on_fit_all_finished)
        self._pending_sim = False   # set when user chose "Fit first" before Simulate
        lay.addWidget(self._mat_panel, 1)
        return w

    # ── Config tab ───────────────────────────────────────────
    def _build_config_tab(self):
        inner = QWidget(); root = QVBoxLayout(inner)
        root.setSpacing(10); root.setContentsMargins(12,12,12,12)
        val_box = QGroupBox("Values"); lay = QFormLayout(val_box); lay.setSpacing(8)

        def ds(val, mn, mx, dec=4, sfx=""):
            s = QDoubleSpinBox(); s.setDecimals(dec); s.setRange(mn,mx)
            s.setValue(val); s.setMinimumWidth(130)
            if sfx: s.setSuffix(f"  {sfx}")
            return s

        # Estimate pass: runs the solver with --estimate (parse + V + T60, no FVM)
        # and fills the recommended Cutoff frequency / Number of rays below. Do this
        # first, after setting grid spacing + temperature.
        self._btn_estimate = QPushButton("≈\uFE0E  Estimate")
        self._btn_estimate.clicked.connect(self._estimate_room)
        _eb = QHBoxLayout()                       # keep the button its natural width,
        _eb.addWidget(self._btn_estimate)         # left-aligned (text + padding), not
        _eb.addStretch(1)                         # stretched across the whole row
        # (added to the layout after Temperature + Altitude, below)


        self._cfg = {}
        def _add(key, lbl, val, mn, mx, dec, sfx):
            sp = ds(val, mn, mx, dec, sfx); self._cfg[key] = sp
            lay.addRow(f"{lbl}", sp)
        # Temperature + Altitude first: they set the atmosphere (speed of sound), which the
        # estimate derives h / f_cutoff / rays from. config.txt is written/read by fixed index.
        _add("T_C",      "Temperature 🌡\uFE0E", 20., -30.,   50., 1, "°C")
        _add("altitude", "Altitude ↥\uFE0E",    408., -500., 9000., 1, "m")   # config line 14 (v[13])
        lay.addRow(_eb)                                                # Estimate, under the atmosphere
        _add("h",        "Grid spacing ↔︎",     0.05, 0.001, 2.0,   3, "m")
        _add("t_max",    "Simulation span 🕑\uFE0E",  0.5,  0.001, 60.0,  3, "s")
        _add("f_cutoff", "Cutoff frequency ✁︎", 250., 50.,   10000., 1, "Hz")
        _add("CFL",      "CFL number",       0.40, 0.05,  0.50,  2, "")

        # Number of poles for the ADE (line 6 of config.txt; vectFIT reads v[5])
        self._npoles = QSpinBox(); self._npoles.setRange(1, 64)
        self._npoles.setValue(8); self._npoles.setMinimumWidth(130)
        lay.addRow("ADE poles", self._npoles)

        # Max reflection order for the ISM (line 13 of config.txt; main3D reads v[12])
        self._ism_order = QSpinBox(); self._ism_order.setRange(0, 20)
        self._ism_order.setValue(3); self._ism_order.setMinimumWidth(130)
        lay.addRow("ISM max order", self._ism_order)

        # Number of rays for the ray-traced late tail (line 15 of config.txt; main3D
        # reads v[14]). Seeded from Vol.bin after a solve (Vorlander <k>); user may
        # override. Wide range for large rooms.
        self._nrays = QSpinBox(); self._nrays.setRange(0, 50_000_000)  # 0 = RT off (clean ISM-only test)
        self._nrays.setValue(20000); self._nrays.setMinimumWidth(130)
        lay.addRow("Number of rays", self._nrays)

        # Source type + frequency now live under the source table (Sources tab), built
        # before this tab. Wire the f_cutoff-dependent clamps here, where _cfg exists:
        #   PCS/Ricker global spinbox -> 0.5*f_cutoff (their spectrum is wider than f0),
        #   Sinusoid per-source f0 column -> f_cutoff (a single resolvable line).
        def _apply_src_freq_clamps(fc):
            self._src_freq.setMaximum(max(1, int(0.5 * fc)))
            self._src_tbl.set_freq_max(int(fc))
        self._cfg['f_cutoff'].valueChanged.connect(_apply_src_freq_clamps)
        _apply_src_freq_clamps(self._cfg['f_cutoff'].value())

        root.addWidget(val_box)                 # close the Values box

        pml_box = QGroupBox("PML faces"); grid = QGridLayout(pml_box); self._pml = {}
        # labels: Right=x+, Front=y+, Up=z+ / Left=x-, Back=y-, Down=z- (keys unchanged)
        for idx, (k, lbl) in enumerate([("R_PML","x+"),("F_PML","y+"),("U_PML","z+"),
                                         ("L_PML","x-"),("B_PML","y-"),("D_PML","z-")]):
            chk = QCheckBox(lbl); chk.setChecked(True)
            self._pml[k] = chk; grid.addWidget(chk, idx//3, idx%3)
        self._pml_all_btn = QPushButton("Deselect all")
        self._pml_all_btn.clicked.connect(self._toggle_all_pml)
        grid.addWidget(self._pml_all_btn, 2, 0, 1, 3)
        root.addWidget(pml_box)

        cfg_btns = QHBoxLayout()
        b_load = QPushButton("🗁\uFE0E  Load config.txt"); b_load.clicked.connect(self._load_config); b_load.setFixedHeight(24)
        b_save = QPushButton("🖫\uFE0E  Save config.txt"); b_save.clicked.connect(self._save_config); b_save.setFixedHeight(24)
        cfg_btns.addWidget(b_load); cfg_btns.addWidget(b_save)   # Load left, Save right
        cfg_btns.setContentsMargins(8, 0, 8, 8)     # 8px sides = same width as the src/rec Load/Save + Add/Remove
        root.addStretch(1)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        out = QWidget(); ov = QVBoxLayout(out); ov.setContentsMargins(0, 0, 0, 0)
        ov.addWidget(scroll, 1)
        ov.addLayout(cfg_btns)          # Load/Save pinned to the tab bottom, like the src/rec tabs
        return out

    # ── Sources tab ──────────────────────────────────────────
    def _build_src_tab(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(8,8,8,8)
        info = QLabel("Press S (or the toolbar button) to enter placement mode.\n"
                      "Drag LMB to position; hold X/Y/Z to lock an axis; release to place. Esc to finish.\n"
                      "Or add rows manually below.")
        info.setStyleSheet(f"color:{PAL['subtext']};font-size:11px;")
        info.setWordWrap(True); lay.addWidget(info)
        self._src_tbl = PointTable(has_amplitude=True, has_freq=True)
        self._src_tbl.changed.connect(self._refresh_gl_points)
        self._src_tbl._tbl.itemSelectionChanged.connect(
            lambda: self._on_table_row_selected("source"))

        # Source excitation type (global; config line 16, main3D reads v[15] as src_opt).
        # Placed ABOVE the table -- the Add/Remove buttons live inside the table widget.
        # Exclusive radios with a didactic pulse-shape icon; ids match the solver:
        # 0=PCS, 1=Ricker, 2=Sinusoid.
        self._src_group = QButtonGroup(self)
        type_row = QHBoxLayout(); type_row.setContentsMargins(0, 0, 0, 0)
        type_row.addWidget(QLabel("Source type"))
        for sid, label in [(0, "PCS"), (1, "Ricker"), (2, "Sinusoid")]:
            rb = QRadioButton(label)
            rb.setIcon(make_pulse_icon(label.lower())); rb.setIconSize(QSize(46, 28))
            self._src_group.addButton(rb, sid)
            type_row.addWidget(rb)
        type_row.addStretch(1)
        lay.addLayout(type_row)

        # Global frequency for PCS/Ricker, on its own line under the type row (config line
        # 17). Sinusoid uses the per-source f0 column instead; only one shows at a time.
        freq_row = QHBoxLayout(); freq_row.setContentsMargins(0, 0, 0, 0)
        self._src_freq = QSpinBox(); self._src_freq.setRange(1, 1_000_000)
        self._src_freq.setValue(100); self._src_freq.setSuffix(" Hz")
        self._src_freq.setAccelerated(True); self._src_freq.setMinimumWidth(120)
        self._src_freq_lbl = QLabel("Frequency")
        freq_row.addWidget(self._src_freq_lbl); freq_row.addWidget(self._src_freq); freq_row.addStretch(1)
        lay.addLayout(freq_row)

        # Auralization: run one simulation per source (separable per-source IRs) instead
        # of firing all sources together. Sets config line 18 (src index) per run.
        self._aural = QCheckBox("Auralization mode (one run per source)")
        lay.addWidget(self._aural)

        lay.addWidget(self._src_tbl, 1)          # stretch so Load/Save pins to the bottom

        def _on_src_type(*_):
            is_sin = (self._src_group.checkedId() == 2)
            self._src_tbl.set_freq_column_visible(is_sin)   # f0 column only for Sinusoid
            self._src_freq_lbl.setVisible(not is_sin)       # global freq only for PCS/Ricker
            self._src_freq.setVisible(not is_sin)
        self._src_group.buttonToggled.connect(_on_src_type)
        self._src_group.button(0).setChecked(True)
        _on_src_type()

        br = QHBoxLayout()
        for lbl, fn in [("🗁\uFE0E  Load src.txt", self._load_src),
                        ("🖫\uFE0E  Save src.txt", self._save_src)]:
            b = QPushButton(lbl); b.clicked.connect(fn); b.setFixedHeight(24); br.addWidget(b)
        lay.addLayout(br)
        return w

    def _build_rec_tab(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(8,8,8,8)
        info = QLabel("Press R (or the toolbar button) to enter placement mode.\n"
                      "Drag LMB to position; hold X/Y/Z to lock an axis; release to place. Esc to finish.\n"
                      "Or add rows manually below.")
        info.setStyleSheet(f"color:{PAL['subtext']};font-size:11px;")
        info.setWordWrap(True); lay.addWidget(info)
        self._rec_tbl = PointTable(has_amplitude=False, has_pose=True)
        self._rec_tbl.changed.connect(self._refresh_gl_points)
        self._rec_tbl._tbl.itemSelectionChanged.connect(
            lambda: self._on_table_row_selected("receiver"))
        self._rec_tbl._tbl.itemSelectionChanged.connect(self._load_selected_rec_pose)
        # re-clicking the already-selected row emits no selectionChanged; use clicked so you can
        # always jump the Results plot back to a receiver you viewed before.
        self._rec_tbl._tbl.clicked.connect(
            lambda ix: getattr(self, "_view_mode", None) == "results"
            and self._on_table_row_selected("receiver"))
        lay.addWidget(self._rec_tbl, 1)          # stretch so Load/Save pins to the bottom
        self._aur_btn = QPushButton("🎧\uFE0E  Auralize ▴"); self._aur_btn.setFixedHeight(24)
        self._aur_btn.clicked.connect(self._toggle_auralize_panel)
        self._aur_panel = self._build_auralize_panel()
        ptl = self._rec_tbl.layout()                      # drop Auralize button + panel ABOVE the
        ptl.insertWidget(ptl.count() - 1, self._aur_btn)  # PointTable's own Add/Remove row
        ptl.insertWidget(ptl.count() - 1, self._aur_panel)
        br = QHBoxLayout()
        for lbl, fn in [("🗁\uFE0E  Load rec.txt", self._load_rec),
                        ("🖫\uFE0E  Save rec.txt", self._save_rec)]:
            b = QPushButton(lbl); b.clicked.connect(fn); b.setFixedHeight(24); br.addWidget(b)
        lay.addLayout(br)
        return w

    # ── Binaural auralization panel (Receivers tab) ──────────────
    def _build_auralize_panel(self):
        """Collapsible panel under the receiver table: head pose dials (az/el/roll) for the SELECTED
        receiver, a SOFA + anechoic-audio picker, Convolve, and a blinking play/scrubber transport."""
        from PySide6.QtWidgets import QDial
        box = QGroupBox("Head pose and binaural render"); box.setVisible(False)
        v = QVBoxLayout(box); v.setSpacing(6); v.setContentsMargins(8, 6, 8, 8)

        dr = QHBoxLayout(); self._aur_dials = {}
        for key, lo, hi, cap in [("az", 0, 360, "Azimuth ◠\uFE0E"), ("el", -90, 90, "Elevation ⦠\uFE0E"),
                                 ("roll", -180, 180, "Roll ⭮\uFE0E")]:
            col = QVBoxLayout()
            c = QLabel(cap); c.setAlignment(Qt.AlignmentFlag.AlignCenter)
            c.setStyleSheet(f"color:{PAL['subtext']};font-size:10px;")
            d = QDial(); d.setRange(lo, hi); d.setValue(0); d.setNotchesVisible(True)
            d.setWrapping(key == "az"); d.setFixedSize(88, 88)
            val = QLabel("0°"); val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            d.valueChanged.connect(lambda x, wv=val: wv.setText(f"{x}°"))
            d.valueChanged.connect(self._on_pose_dial_changed)
            col.addWidget(c); col.addWidget(d, 0, Qt.AlignmentFlag.AlignCenter); col.addWidget(val)
            dr.addLayout(col); self._aur_dials[key] = d
        v.addLayout(dr)
        self._pose_loading = False        # guard: suppress store while loading dials programmatically
        self._rendered_pose = {}          # {rec_row: [az,el,roll]} the last Convolve used, for Restore

        sr = QHBoxLayout(); sr.addWidget(QLabel("HRTF"))
        self._aur_sofa = QLineEdit(QSettings("phases", "gui").value("last_sofa", "") or "")
        bs = QPushButton("SOFA…"); bs.setFixedHeight(24); bs.clicked.connect(self._pick_sofa)
        sr.addWidget(self._aur_sofa, 1); sr.addWidget(bs); v.addLayout(sr)
        # Alternative renderer: first-order ambisonics, no HRTF. Head pose is applied LIVE (dials or
        # FPV mouselook stream to ambisonic.py --live); nothing is re-rendered or written per pose.
        self._aur_foa = QCheckBox("Ambisonic live (no HRTF): head pose applied while playing")
        self._aur_foa.setChecked(bool(QSettings("phases", "gui").value("aur_foa", False, type=bool)))
        self._aur_foa.toggled.connect(self._on_foa_toggled); v.addWidget(self._aur_foa)

        ar = QHBoxLayout(); ar.addWidget(QLabel("Audio"))
        self._aur_dry = QLineEdit(QSettings("phases", "gui").value("last_dry", "") or "")
        bd = QPushButton("Anechoic…"); bd.setFixedHeight(24); bd.clicked.connect(self._pick_dry)
        bps = QPushButton("Per-source…"); bps.setFixedHeight(24); bps.clicked.connect(self._edit_src_dry)
        bps.setToolTip("Give each source its own anechoic signal (played simultaneously)")
        ar.addWidget(self._aur_dry, 1); ar.addWidget(bd); ar.addWidget(bps); v.addLayout(ar)
        self._src_dry = {}                # {source_index: anechoic_path}; blank -> uses the default Audio

        self._aur_conv = QPushButton("∗	Convolve"); self._aur_conv.setFixedHeight(26)
        self._aur_conv.clicked.connect(self._run_convolve); v.addWidget(self._aur_conv)

        tr = QHBoxLayout()
        self._aur_play = QPushButton("▶︎"); self._aur_play.setFixedSize(44, 26)
        self._aur_play.setEnabled(False); self._aur_play.clicked.connect(self._toggle_play)
        self._aur_restore = QPushButton("↺"); self._aur_restore.setFixedSize(30, 26)
        self._aur_restore.setToolTip("Restore the head pose the current wav was rendered with")
        self._aur_restore.setEnabled(False); self._aur_restore.clicked.connect(self._restore_rendered_pose)
        self._aur_slider = QSlider(Qt.Horizontal); self._aur_slider.setEnabled(False)
        self._aur_time = QLabel("0:00"); self._aur_time.setStyleSheet(f"color:{PAL['subtext']};")
        tr.addWidget(self._aur_play); tr.addWidget(self._aur_restore)
        tr.addWidget(self._aur_slider, 1); tr.addWidget(self._aur_time)
        v.addLayout(tr)
        self._aur_setup_player()
        return box

    def _aur_setup_player(self):
        """QtMultimedia player wiring (guarded — if unavailable, Convolve still writes the wav)."""
        self._player = None
        try:
            from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QMediaDevices
        except Exception:
            return
        self._audio_out = QAudioOutput()
        self._audio_out.setDevice(QMediaDevices.defaultAudioOutput())   # current default, not just startup's
        self._player = QMediaPlayer(); self._player.setAudioOutput(self._audio_out)
        # QAudioOutput binds the default device once; it does NOT follow later changes. Re-point it at the
        # current default whenever the device list changes -> BT headphones connected after launch get sound.
        self._media_devices = QMediaDevices(self)
        self._media_devices.audioOutputsChanged.connect(self._use_default_audio_out)
        self._player.positionChanged.connect(self._on_media_pos)
        self._player.durationChanged.connect(lambda d: self._aur_slider.setRange(0, d))
        self._player.playbackStateChanged.connect(self._on_media_state)
        self._aur_slider.sliderMoved.connect(self._player.setPosition)
        self._aur_blink = QTimer(self); self._aur_blink.timeout.connect(self._blink_play)
        self._aur_blink_on = False

    def _enter_receiver_view(self):
        """Teleport the scene camera into the selected receiver, facing the az/el/roll dials — the SAME
        world head frame the auralization uses, so what you see matches what you hear."""
        rec = self._current_receiver_row()
        if rec < 0 or rec >= len(self._gl._receivers):
            self._sb.showMessage("Select a receiver row first."); return
        p = self._gl._receivers[rec]
        eye = np.array([float(p[0]), float(p[1]), float(p[2])])
        az, el, roll = (self._aur_dials[k].value() for k in ("az", "el", "roll"))
        self._gl.enter_fpv(eye, az, el, roll)                  # GL builds the head frame from the angles
        self._sb.showMessage(f"Inside receiver {rec} — LMB look · Ctrl+LMB roll · RMB → Go back.")

    # ── per-receiver head pose (dials <-> receiver <-> FPV mouselook) ──
    def _set_dials(self, az, el, roll):
        """Set the three dials without triggering the store handler (used when loading a pose)."""
        self._pose_loading = True
        self._aur_dials["az"].setValue(int(round(az)) % 360)
        self._aur_dials["el"].setValue(int(round(el)))
        self._aur_dials["roll"].setValue(int(round(roll)))
        self._pose_loading = False

    def _load_selected_rec_pose(self):
        """Selecting a receiver shows ITS stored pose in the dials; enable Restore if it has a wav pose."""
        if not hasattr(self, "_aur_dials"):        # selection can fire during project load pre-panel
            return
        row = self._current_receiver_row()
        if row < 0:
            return
        self._set_dials(*self._rec_tbl.get_pose(row))
        if hasattr(self, "_aur_restore"):
            self._aur_restore.setEnabled(row in getattr(self, "_rendered_pose", {}))

    def _on_pose_dial_changed(self, *_):
        """A dial moved: store it on the selected receiver and, if we're inside that receiver, steer
        the frozen camera to match."""
        if getattr(self, "_pose_loading", False):
            return
        pose = [self._aur_dials[k].value() for k in ("az", "el", "roll")]
        row = self._current_receiver_row()
        if row >= 0:
            self._rec_tbl.set_pose(row, pose); self._mark_dirty()
        if self._gl._fpv is not None:
            self._gl.set_fpv_pose(*pose)
        self._foa_send_pose()

    def _on_fpv_pose(self, az, el, roll):
        """FPV mouselook changed the head aim -> reflect it in the dials + store on the receiver."""
        self._set_dials(az, el, roll)
        row = self._current_receiver_row()
        if row >= 0:
            self._rec_tbl.set_pose(row, [self._aur_dials[k].value() for k in ("az", "el", "roll")])
            self._mark_dirty()
        self._foa_send_pose()

    def _restore_rendered_pose(self):
        """Snap dials (and the FPV camera) back to the pose the current wav was rendered with."""
        row = self._current_receiver_row()
        p = getattr(self, "_rendered_pose", {}).get(row)
        if p is None:
            self._sb.showMessage("No rendered pose for this receiver yet — Convolve first."); return
        self._set_dials(*p)
        self._on_pose_dial_changed()          # store + steer the camera if inside
        self._sb.showMessage(f"Restored the rendered pose (az {p[0]}, el {p[1]}, roll {p[2]}).")

    def _toggle_auralize_panel(self):
        vis = not self._aur_panel.isVisible()
        self._aur_panel.setVisible(vis)
        self._aur_btn.setText("🎧\uFE0E  Auralize ▾" if vis else "🎧\uFE0E  Auralize ▴")

    def _pick_sofa(self):
        start = os.path.dirname(self._aur_sofa.text()) or os.path.expanduser("~")
        f, _ = QFileDialog.getOpenFileName(self, "Pick HRTF SOFA", start, "SOFA (*.sofa)")
        if f:
            self._aur_sofa.setText(f); QSettings("phases", "gui").setValue("last_sofa", f)

    def _pick_dry(self):
        start = os.path.dirname(self._aur_dry.text()) or os.path.expanduser("~")
        f, _ = QFileDialog.getOpenFileName(self, "Pick anechoic audio", start,
                                           "Audio (*.wav *.flac *.aiff *.aif *.ogg)")
        if f:
            self._aur_dry.setText(f); QSettings("phases", "gui").setValue("last_dry", f)

    def _edit_src_dry(self):
        """Assign each source its own anechoic signal (all sources play simultaneously into the receiver;
        blank rows fall back to the default Audio file)."""
        from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                                       QFileDialog, QDialogButtonBox)
        pts = self._src_tbl.get_points()
        if not pts:
            self._sb.showMessage("No sources defined yet."); return
        dlg = QDialog(self); dlg.setWindowTitle("Per-source anechoic signals")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Each source plays its own signal. Leave a row blank to use the default Audio."))
        edits = []
        for i in range(len(pts)):
            row = QHBoxLayout(); row.addWidget(QLabel(f"Source {i}"))
            le = QLineEdit(self._src_dry.get(i, "")); row.addWidget(le, 1)
            b = QPushButton("…"); b.setFixedWidth(28)
            b.clicked.connect(lambda _=False, e=le: self._browse_into(e))
            row.addWidget(b); lay.addLayout(row); edits.append(le)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); lay.addWidget(bb)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._src_dry = {i: e.text().strip() for i, e in enumerate(edits) if e.text().strip()}
            self._sb.showMessage(f"Per-source signals: {len(self._src_dry)} set "
                                 f"(rest use the default Audio).")

    def _browse_into(self, line_edit):
        from PySide6.QtWidgets import QFileDialog
        start = os.path.dirname(line_edit.text()) or os.path.dirname(self._aur_dry.text()) or os.path.expanduser("~")
        f, _ = QFileDialog.getOpenFileName(self, "Anechoic signal", start,
                                           "Audio (*.wav *.flac *.aiff *.aif *.ogg)")
        if f:
            line_edit.setText(f)

    def _on_foa_toggled(self, on):
        QSettings("phases", "gui").setValue("aur_foa", bool(on))
        self._aur_sofa.setEnabled(not on)
        self._aur_conv.setText("◉	Live" if on else "∗	Convolve")
        if not on:
            self._stop_foa_live()

    def _foa_alive(self):
        t = getattr(self, "_foa_thread", None)
        return t is not None and t.isRunning()

    def _stop_foa_live(self):
        if self._foa_alive():
            self._foa_thread.send("q"); self._foa_thread.wait(1500)
            if self._foa_thread.isRunning():
                self._foa_thread.stop()
        self._aur_conv.setText("◉	Live" if self._aur_foa.isChecked() else "∗	Convolve")

    def _foa_send_pose(self):
        """Dials/FPV moved: stream the pose to the live renderer (cheap: one short line)."""
        if self._foa_alive():
            self._foa_thread.send(" ".join(str(self._aur_dials[k].value()) for k in ("az", "el", "roll")))

    def _run_foa_live(self, rec, dry, out):
        """Toggle: start ambisonic.py --live for `rec` (plays until stopped), or stop it."""
        if self._foa_alive():
            self._stop_foa_live(); self._sb.showMessage("Live auralization stopped."); return
        script = self._find_common_script("ambisonic.py")
        if script is None:
            self._sb.showMessage("ambisonic.py not found next to gui.py."); return
        pose = [str(self._aur_dials[k].value()) for k in ("az", "el", "roll")]
        cmd = [sys.executable, script, "--rec", str(rec), "--dry", dry, "--live", "--pose", *pose]
        dm = {str(i): p for i, p in getattr(self, "_src_dry", {}).items() if p and os.path.isfile(p)}
        if dm:
            import json
            mp = os.path.join(out, "drymap.json")
            with open(mp, "w") as f:
                json.dump(dm, f)
            cmd += ["--drymap", mp]
        if self._player is not None:
            self._player.stop()
        self._foa_thread = SolverThread(cmd, out, stdin=True)
        self._foa_thread.output.connect(self._log.append)
        self._foa_thread.done.connect(lambda rc: (self._aur_conv.setText("◉	Live"),
                                                  self._sb.showMessage("Live auralization ended."
                                                                       if rc == 0 else f"ambisonic.py failed (rc={rc}) — see log.")))
        self._foa_thread.start()
        self._aur_conv.setText("■	Stop")
        self._sb.showMessage(f"Live auralization of receiver {rec} — move the dials or look around inside it.")

    def _run_convolve(self):
        """Render the selected receiver to a binaural wav via binauralize.py (in output/)."""
        if not self._proj_root:
            self._sb.showMessage("Open a project first."); return
        rec = self._current_receiver_row()
        if rec < 0:
            self._sb.showMessage("Select a receiver row first."); return
        sofa, dry = self._aur_sofa.text().strip(), self._aur_dry.text().strip()
        if not (dry and os.path.isfile(dry)):
            self._sb.showMessage("Pick an anechoic audio file."); return
        if self._aur_foa.isChecked():
            self._run_foa_live(rec, dry, os.path.join(self._proj_root, "output")); return
        if not (sofa and os.path.isfile(sofa)):
            self._sb.showMessage("Pick a valid .sofa file."); return
        script = self._find_common_script("binauralize.py")
        if script is None:
            self._sb.showMessage("binauralize.py not found next to gui.py."); return
        t = getattr(self, "_aur_thread", None)
        if t is not None and t.isRunning():
            return
        out = os.path.join(self._proj_root, "output")
        self._aur_wav = os.path.join(out, f"binaural_rec{rec}.wav")
        pose_i = [self._aur_dials[k].value() for k in ("az", "el", "roll")]
        self._rendered_pose[rec] = pose_i                        # stamp: the pose this wav uses (Restore)
        self._aur_restore.setEnabled(True)
        pose = [str(x) for x in pose_i]
        cmd = [sys.executable, script, sofa, "--rec", str(rec),
               "--dry", dry, "--out", self._aur_wav, "--pose", *pose]
        dm = {str(i): p for i, p in getattr(self, "_src_dry", {}).items() if p and os.path.isfile(p)}
        if dm:                                                   # per-source signals -> write a drymap.json
            import json
            mp = os.path.join(out, "drymap.json")
            with open(mp, "w") as f:
                json.dump(dm, f)
            cmd += ["--drymap", mp]
        if self._player is not None:
            self._player.setSource(QUrl())                       # release the file before overwrite
        self._aur_conv.setEnabled(False); self._aur_play.setEnabled(False)
        self._aur_slider.setEnabled(False)
        self._aur_thread = SolverThread(cmd, out)
        self._aur_thread.output.connect(self._log.append)
        self._aur_thread.done.connect(self._on_convolve_done)
        self._aur_thread.start()
        self._sb.showMessage(f"Convolving receiver {rec}… (pose {'/'.join(pose)})")

    def _on_convolve_done(self, rc):
        self._aur_conv.setEnabled(True)
        if rc != 0 or not os.path.isfile(getattr(self, "_aur_wav", "")):
            self._sb.showMessage(f"Auralization failed (rc={rc}) — see log."); return
        if self._player is None:
            self._sb.showMessage(f"Wrote {os.path.basename(self._aur_wav)} "
                                 "(QtMultimedia missing — open it externally)."); return
        self._player.setSource(QUrl.fromLocalFile(self._aur_wav))
        self._aur_play.setEnabled(True); self._aur_slider.setEnabled(True)
        self._aur_blink.start(500)                                # blink until the user hits play
        self._sb.showMessage("Binaural ready — press play.")

    def _use_default_audio_out(self):
        """Point the player at the CURRENT system default output. QAudioOutput binds the default at
        construction and doesn't follow later changes, so a BT headset paired after launch never gets
        sound otherwise. Called on device-list changes and right before each play."""
        if self._player is None:
            return
        from PySide6.QtMultimedia import QMediaDevices
        self._audio_out.setDevice(QMediaDevices.defaultAudioOutput())

    def _toggle_play(self):
        if self._player is None:
            return
        from PySide6.QtMultimedia import QMediaPlayer
        self._aur_blink.stop(); self._aur_play.setStyleSheet("")
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._use_default_audio_out()                        # follow whatever is default right now
            self._player.play()

    def _blink_play(self):
        self._aur_blink_on = not self._aur_blink_on
        self._aur_play.setStyleSheet(
            f"background:{PAL['accent']};color:{PAL['bg']};" if self._aur_blink_on else "")

    def _on_media_pos(self, p):
        if not self._aur_slider.isSliderDown():
            self._aur_slider.setValue(p)
        self._aur_time.setText(f"{p // 60000}:{(p // 1000) % 60:02d}")

    def _on_media_state(self, st):
        from PySide6.QtMultimedia import QMediaPlayer
        playing = st == QMediaPlayer.PlaybackState.PlayingState
        self._aur_play.setText("⏸" if playing else "▶︎")

    # ── Run tab ──────────────────────────────────────────────
    def _build_run_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(8,18,8,8); lay.setSpacing(8)   # extra top margin off the tab bar

        # Binary path is kept (hidden field) so the launch command still works;
        # default to SOLVER_BINARY (edit at the top of the file) or last browsed.
        _last_bin = QSettings("phases", "gui").value("last_binary", "")
        if not (_last_bin and os.path.isfile(_last_bin)):
            _last_bin = SOLVER_BINARY
        self._bin_le = QLineEdit(_last_bin)
        self._bin_le.setVisible(False)        # path lives in SOLVER_BINARY now

        # ── Ray-Tracing: debug ray overlay (count + order + Trace/Clear) ──
        rt_box = QGroupBox("☈ay-Tracing"); rt = QHBoxLayout(rt_box); rt.setSpacing(8)
        rt.addWidget(QLabel("Track rays ↯"))
        self._dbg_rays = QSpinBox(); self._dbg_rays.setRange(1, 5000)
        self._dbg_rays.setValue(64); self._dbg_rays.setMinimumWidth(70)
        rt.addWidget(self._dbg_rays)
        rt.addWidget(QLabel("Order"))
        self._dbg_order = QSpinBox(); self._dbg_order.setRange(1, 20)
        self._dbg_order.setValue(5); self._dbg_order.setMinimumWidth(55)
        rt.addWidget(self._dbg_order)
        self._btn_rays = QPushButton("Trace rays"); self._btn_rays.clicked.connect(self._toggle_rays)
        self._btn_rays.setFixedHeight(24)
        rt.addWidget(self._btn_rays, 1)
        lay.addWidget(rt_box)

        # ── Animation: fps + slice-plane positions (or pick in scene with Slice [L]) ──
        an_box = QGroupBox("Animation \U0001F39E\uFE0E"); an = QGridLayout(an_box)
        an.setHorizontalSpacing(6); an.setVerticalSpacing(12)   # roomier rows
        self._fps = QSpinBox(); self._fps.setRange(1, 240); self._fps.setValue(24)
        frow = QHBoxLayout(); frow.setSpacing(6)      # keep the box snug to its label
        frow.addWidget(QLabel("FPS")); frow.addWidget(self._fps); frow.addStretch(1)
        an.addLayout(frow, 0, 0, 1, 7)

        an.addWidget(QLabel("Slice planes [m]"), 1, 0, 1, 7)
        self._slice_sync = False        # guard: suppress field->gl while syncing gl->field
        self._slice_sb = {}
        srow = QHBoxLayout(); srow.setSpacing(4)
        for axis in ('x', 'y', 'z'):
            srow.addWidget(QLabel(f"{axis}0"))
            sb = QDoubleSpinBox(); sb.setRange(-1e4, 1e4); sb.setDecimals(3); sb.setSingleStep(0.05)
            sb.setMaximumWidth(90); sb.valueChanged.connect(self._on_slice_field)
            srow.addWidget(sb); self._slice_sb[axis] = sb
        srow.addStretch(1)
        an.addLayout(srow, 2, 0, 1, 7)

        # pressure colour range [Pa] for the animation
        prow = QHBoxLayout(); prow.setSpacing(4)
        prow.addWidget(QLabel("Pressure range [Pa]"))
        self._pmin = QDoubleSpinBox(); self._pmin.setRange(-1e6, 1e6); self._pmin.setDecimals(4)
        self._pmin.setValue(-0.01); self._pmin.setMaximumWidth(90)
        self._pmax = QDoubleSpinBox(); self._pmax.setRange(-1e6, 1e6); self._pmax.setDecimals(4)
        self._pmax.setValue(0.01); self._pmax.setMaximumWidth(90)
        self._pmin.valueChanged.connect(self._on_prange); self._pmax.valueChanged.connect(self._on_prange)
        prow.addWidget(self._pmin); prow.addWidget(QLabel("to")); prow.addWidget(self._pmax)
        prow.addStretch(1)
        an.addLayout(prow, 3, 0, 1, 7)

        # which planes show / animate (whichever are on) + one show/hide-all toggle
        an.addWidget(QLabel("Show:"), 4, 0)
        # own tight HBox so X/Y/Z spacing is even (grid columns are stretched by the fps
        # spinbox and the Particles radio, which was pushing Z out).
        self._plane_chk = {}
        _show = QHBoxLayout(); _show.setSpacing(12)
        for axis in ('x', 'y', 'z'):
            cb = QCheckBox(axis.upper()); cb.setChecked(True)
            cb.toggled.connect(lambda on, a=axis: self._gl.set_plane_visible(a, on))
            _show.addWidget(cb); self._plane_chk[axis] = cb
        self._planes_all_btn = QPushButton("Hide all")
        self._planes_all_btn.clicked.connect(self._toggle_all_planes)
        _show.addSpacing(8); _show.addWidget(self._planes_all_btn); _show.addStretch(1)
        an.addLayout(_show, 4, 1, 1, 6)

        # colour palette for the animation
        an.addWidget(QLabel("Palette:"), 5, 0)
        from PySide6.QtWidgets import QRadioButton, QButtonGroup
        self._pal_group = QButtonGroup(self)
        rb_c = QRadioButton("Colorful"); rb_c.setChecked(True)
        rb_g = QRadioButton("Black and White")
        rb_p = QRadioButton("Particles")
        self._pal_group.addButton(rb_c); self._pal_group.addButton(rb_g); self._pal_group.addButton(rb_p)
        rb_c.toggled.connect(lambda on: on and self._gl.set_anim_palette('color'))
        rb_g.toggled.connect(lambda on: on and self._gl.set_anim_palette('greys'))
        rb_p.toggled.connect(lambda on: on and self._gl.set_anim_palette('particles'))
        an.addWidget(rb_c, 5, 1); an.addWidget(rb_g, 5, 2); an.addWidget(rb_p, 5, 3)

        # transport: play/pause (pauses in place) + frame scrubber (drag = seek/rewind/reset,
        # arrow keys = frame-by-frame). Only a first play / slice / project change (re)loads.
        self._anim_btn = QPushButton("▶︎"); self._anim_btn.setFixedSize(44, 26)
        self._anim_btn.clicked.connect(self._toggle_animation)
        self._anim_slider = QSlider(Qt.Horizontal); self._anim_slider.setRange(0, 0)
        self._scrub_guard = False
        self._anim_slider.valueChanged.connect(self._on_scrub)
        self._gl.frameChanged.connect(self._on_anim_frame)
        an.addWidget(self._anim_btn, 6, 0)
        an.addWidget(self._anim_slider, 6, 1, 1, 6)
        lay.addWidget(an_box)

        # ── Simulation: Fit impedances + Run/Stop ──
        sim_box = QGroupBox("Simulation ∿"); top = QHBoxLayout(sim_box); top.setSpacing(8)
        self._fitimp_btn = QPushButton("⡶\uFE0E⣰\uFE0E  Fit impedances")
        self._fitimp_btn.setStyleSheet(
            f"background:{PAL['btn']};color:{PAL['text']};font-weight:bold;padding:8px;")
        self._fitimp_btn.clicked.connect(self._fit_impedances)
        top.addWidget(self._fitimp_btn, 1)

        # Single Run/Stop toggle: shows ▶ Run normally, ⏹ Stop while running.
        self._run_btn = QPushButton("▷\uFE0E  Simulate")
        self._run_btn.setStyleSheet(
            f"background:{PAL['accent']};color:{PAL['bg']};font-weight:bold;padding:8px;")
        self._run_btn.clicked.connect(self._toggle_run)
        top.addWidget(self._run_btn, 1)
        lay.addWidget(sim_box)

        # groups hug their content (don't inflate to fill the tab)
        from PySide6.QtWidgets import QSizePolicy
        for box in (rt_box, an_box, sim_box):
            box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        # ── output log ── fills all remaining height under the (fixed-height) boxes above
        lay.addWidget(QLabel("Output log:"))
        self._log = QTextEdit(); self._log.setReadOnly(True)
        self._log.setFont(QFont("Monospace", 10))
        self._log.setMinimumHeight(120)
        self._log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._log.setStyleSheet(
            f"background:#000000;color:{PAL['text']};"
            f"border:1px solid {PAL['border']};")
        lay.addWidget(self._log, 1)         # stretch factor -> takes the leftover space

        self._solver_thread = None
        self._aural_queue = []      # remaining source indices to run (auralization mode)
        self._aural_cmd = None
        return w

    # ── fit output -> Run-tab Output log ─────────────────────
    def _begin_fit_log(self, header):
        """A fit run started: clear the Output log, show the command, and bring
        the Output (Run) tab forward so the fitter's progress is visible."""
        if hasattr(self, "_log"):
            self._log.clear()
            self._log.append(header)
            self._log.append("-" * 60)
        if hasattr(self, "_tabs"):
            self._tabs.setCurrentIndex(4)   # ▶ Run tab holds the Output log

    def _append_fit_log(self, line):
        if hasattr(self, "_log"):
            self._log.append(line)

    # ── slots ────────────────────────────────────────────────
    # ── project: manifest helpers ────────────────────────────
    def _manifest_path(self, root=None):
        return os.path.join(root or self._proj_root, MANIFEST_NAME)

    def _read_manifest(self, path):
        """Parse a `key = value` manifest (#-comments ignored) into a dict.
        Missing/garbled files yield an empty dict — callers fall back to the
        canonical file-contract paths, so old projects without a manifest still
        open fine."""
        d = {}
        try:
            with open(path) as f:
                for line in f:
                    line = line.split("#")[0].strip()
                    if not line or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    d[k.strip()] = v.strip()
        except OSError:
            pass
        return d

    def _write_manifest(self):
        """Write/refresh the manifest from current GUI state. Written
        atomically (temp file + os.replace) so a crash mid-save can't leave a
        half-written manifest. `created` is preserved across saves.

        Paths are stored ABSOLUTE — the C solver reads its inputs by absolute
        path, so the manifest records them the same way (no working-directory
        ambiguity). Trade-off: moving the project folder invalidates the stored
        paths; re-open or Save once to refresh them."""
        if not self._proj_root:
            return
        root     = os.path.abspath(self._proj_root)
        path     = self._manifest_path()
        existing = self._read_manifest(path) if os.path.isfile(path) else {}
        created  = existing.get("created") or f"{datetime.now():%Y-%m-%d %H:%M}"
        scene    = os.path.abspath(self._obj_path) if self._obj_path else ""
        # Binary is stored VERBATIM. It is a shared build artifact living
        # outside the project (e.g. .../3D-solver/build/phases3D): absolute
        # paths (from Browse) stay absolute; a relative path means "relative to
        # the project root at run time" (the solver runs with cwd = root) and
        # must NOT be absolutized against the project, where it does not exist.
        binary   = self._bin_le.text() if hasattr(self, "_bin_le") else ""
        layers   = " ".join(g["name"] for g in self._groups
                            if not _is_bbox(g["name"]))
        dmax     = getattr(self._gl, "_domain_max", None)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write("# phases project manifest  —  written by gui.py\n")
            f.write("# All paths are absolute (the solver reads inputs by absolute path).\n")
            f.write(f"name      = {os.path.basename(os.path.normpath(root))}\n")
            f.write(f"root      = {root}\n")
            f.write(f"created   = {created}\n")
            f.write(f"saved     = {datetime.now():%Y-%m-%d %H:%M}\n")
            f.write(f"scene     = {scene}\n")
            f.write(f"binary    = {binary}\n")
            f.write(f"config    = {os.path.join(root, 'config', 'config.txt')}\n")
            f.write(f"sources   = {os.path.join(root, 'input', 'src', 'src.txt')}\n")
            f.write(f"receivers = {os.path.join(root, 'input', 'rec', 'rec.txt')}\n")
            f.write(f"poles     = {os.path.join(root, 'config', 'poles.txt')}\n")
            f.write(f"matdir    = {os.path.join(root, 'input', 'mat')}\n")
            f.write(f"layers    = {layers}\n")
            f.write(f"show_bbox = {int(self._bbox_chk.isChecked())}\n")
            # Display frame for src/rec coordinates. The FILES always hold shifted
            # (solver) coordinates -- this only records which frame to SHOW them in.
            f.write(f"orig_frame = {int(bool(getattr(self._gl, '_orig_frame', False)))}\n")
            f.write(f"shift_positive = {int(bool(getattr(self._gl, '_shift_enabled', True)))}\n")
            # the user's own OBJ, not the _positive/_merged copies derived from it
            f.write(f"scene_src = {getattr(self, '_obj_orig_path', '') or ''}\n")
            # Which frame src.txt/rec.txt are stored in: they hold RAW + this offset.
            # Needed so a later OBJ load translates by the DELTA, never the full shift.
            _os_ = getattr(self, "_obj_shift", (0.0, 0.0, 0.0))
            f.write(f"obj_shift = {_os_[0]:.10g} {_os_[1]:.10g} {_os_[2]:.10g}\n")
            if dmax is not None:
                f.write("# domain extent (Lx Ly Lz) is derived from the scene AABB; informational only:\n")
                f.write(f"# domain  = {float(dmax[0]):.6g} {float(dmax[1]):.6g} {float(dmax[2]):.6g}\n")
        os.replace(tmp, path)

    # ── project: create / open / save ────────────────────────
    def _new_project(self):
        """Scaffold a fresh project directory tree, seed a default config and
        empty src/rec, write the manifest, and make it the active project."""
        parent = QFileDialog.getExistingDirectory(self, "Choose location for new project")
        if not parent:
            return
        name, ok = QInputDialog.getText(self, "New project", "Project name:",
                                        text="new_project")
        if not ok or not name.strip():
            return
        root = os.path.join(parent, name.strip())
        if os.path.isdir(root) and os.listdir(root):
            if QMessageBox.question(
                    self, "Folder not empty",
                    f"{root}\nalready exists and is not empty.\nScaffold into it anyway?"
                ) != QMessageBox.StandardButton.Yes:
                return
        try:
            for sub in PROJECT_SUBDIRS:
                os.makedirs(os.path.join(root, sub), exist_ok=True)
        except OSError as ex:
            QMessageBox.critical(self, "Create failed", str(ex)); return

        self._proj_root = root
        self._proj_label.setText(root)
        self._results.set_project(root)
        self._reset_scene_editors()            # fresh project -> empty domain sketch, not the last one's
        # seed the file contract so the project is immediately runnable-shaped
        self._write_config(os.path.join(root, "config", "config.txt"))
        self._src_tbl.save_file(os.path.join(root, "input", "src", "src.txt"))
        self._rec_tbl.save_file(os.path.join(root, "input", "rec", "rec.txt"))
        self._mat_panel.set_project(root)
        if self._obj_path and os.path.isfile(self._obj_path):
            # A scene was already loaded before the project existed — keep it and
            # re-populate under the new project so its path lands in the manifest.
            self._load_obj_path(self._obj_path)   # also calls _write_manifest()
            self._sb.showMessage(f"Created project: {root}")
        else:
            self._obj_path = None
            self._groups   = []
            self._write_manifest()
            self._sb.showMessage(f"Created project: {root}  —  load an OBJ to add geometry.")
        self._dirty = False        # fresh project is already written to disk

    def _open_project(self):
        """Open a project directory and restore its full state. Reads the
        manifest when present (scene path, binary), then loads config, sources,
        receivers, and the scene OBJ — which in turn repopulates layers and
        their materials. Projects with no manifest fall back to the canonical
        file-contract paths and a lone input/scene/*.obj, so they still open."""
        d = QFileDialog.getExistingDirectory(self, "Open project directory")
        if not d:
            return
        self._proj_root = d
        self._gl.clear_animation()      # new project -> reload animation on next play
        self._proj_label.setText(d)
        self._results.set_project(d)
        self._reset_scene_editors()     # load THIS project's sketch/props, drop the previous project's
        man = self._read_manifest(self._manifest_path())

        def resolve(rel):
            return rel if os.path.isabs(rel) else os.path.join(d, rel)

        # binary: only adopt the manifest's path if it actually exists, so a
        # stale relative entry (e.g. "build/phases3D") never clobbers the field.
        if hasattr(self, "_bin_le"):
            mb = man.get("binary", "")
            if mb and os.path.isfile(mb):
                self._bin_le.setText(mb)
            # else: keep whatever is in the field (SOLVER_BINARY default or a
            # valid last-browsed path) — do not overwrite with a bad path.

        # config / sources / receivers (manifest path if given, else canonical)
        cfg = resolve(man.get("config", "config/config.txt"))
        if os.path.isfile(cfg):
            self._load_config_from(cfg)
        srcp = resolve(man.get("sources", "input/src/src.txt"))
        if os.path.isfile(srcp):
            self._src_tbl.load_file(srcp)
        recp = resolve(man.get("receivers", "input/rec/rec.txt"))
        if os.path.isfile(recp):
            self._rec_tbl.load_file(recp)
        self._refresh_gl_points()

        # scene: manifest 'scene' first (absolute), else any .obj that happens
        # to sit under input/scene/ (kept as a convenience; not auto-created).
        scene = man.get("scene", "")
        scene = resolve(scene) if scene else ""
        if not (scene and os.path.isfile(scene)):
            cand  = sorted(glob.glob(os.path.join(d, "input", "scene", "*.obj")))
            scene = cand[0] if cand else ""   # first by name if several present
        # The frame src.txt/rec.txt were saved in. Restore it BEFORE the OBJ loads so the
        # delta-translate compares against the right baseline; without it the load would
        # see prev=(0,0,0) and shift already-shifted points a second time.
        try:
            _sh = [float(v) for v in man.get("obj_shift", "0 0 0").split()]
            self._obj_shift = tuple(_sh[:3]) if len(_sh) >= 3 else (0.0, 0.0, 0.0)
        except ValueError:
            self._obj_shift = (0.0, 0.0, 0.0)
        self._gl._shift_enabled = (man.get("shift_positive", "1") == "1")
        _src = man.get("scene_src", "")
        if _src and os.path.isfile(_src):
            self._obj_orig_path = _src        # so the frame can still be switched later
        if scene and os.path.isfile(scene):
            self._load_obj_path(scene)          # repopulates layers + materials
        else:
            self._mat_panel.set_project(d)      # wire materials for a later OBJ

        self._bbox_chk.setChecked(man.get("show_bbox", "0") == "1")   # restore bbox toggle
        # Restore the coordinate display frame. AFTER _load_obj_path, which is what
        # computes _obj_shift; with no shift the two frames coincide and this is a no-op.
        self._gl._orig_frame = (man.get("orig_frame", "0") == "1")
        self._apply_orig_frame(self._gl._orig_frame)

        # src/rec were read ABOVE, before _load_obj_path knew the mesh's bmin -- so
        # load_file subtracted a _solver_off of zero. Re-read them now that the offset is
        # known, or save_file would ADD it to values that never had it subtracted and every
        # point would drift by bmin ("[!] Receiver outside domain").
        if any(getattr(self._src_tbl, "_solver_off", (0, 0, 0))):
            if os.path.isfile(srcp):
                self._src_tbl.load_file(srcp)
            if os.path.isfile(recp):
                self._rec_tbl.load_file(recp)
            self._refresh_gl_points()
            print(f"[frame] re-read src/rec with solver offset "
                  f"{tuple(round(v, 4) for v in self._src_tbl._solver_off)}")

        # Persist the manifest now so opening a project always records current
        # state (scene path included) even if no .mat/config was touched.
        self._write_manifest()
        self._dirty = False        # freshly loaded == saved (loading fired valueChanged signals)
        self._sb.showMessage(f"Opened project: {d}")

    def _mark_dirty(self, *a):
        self._dirty = True

    def _wire_dirty(self):
        """Connect the edit sources to _mark_dirty so the close prompt only shows on unsaved work."""
        md = self._mark_dirty
        for sp in self._cfg.values():
            sp.valueChanged.connect(md)
        for w in (self._fps, self._pmin, self._pmax, *self._slice_sb.values()):
            w.valueChanged.connect(md)
        for c in self._pml.values():
            c.toggled.connect(md)
        self._bbox_chk.toggled.connect(md)
        self._src_tbl.changed.connect(md)
        self._rec_tbl.changed.connect(md)
        self._mat_panel._tbl.cellChanged.connect(md)

    def _save_project(self):
        """Explicit, atomic save of the whole project: config, materials,
        sources, receivers, and the manifest."""
        if not self._proj_root:
            QMessageBox.warning(self, "No project",
                                "Create or open a project first."); return
        self._write_config(os.path.join(self._proj_root, "config", "config.txt"))
        self._src_tbl.save_file(os.path.join(self._proj_root, "input", "src", "src.txt"))
        self._rec_tbl.save_file(os.path.join(self._proj_root, "input", "rec", "rec.txt"))
        self._mat_panel.save_all()
        self._mat_panel.prune_orphans()      # drop .mat/.fit/.sctr for materials no longer in the OBJ
        self._write_manifest()
        self._dirty = False
        self._sb.showMessage(f"Saved project → {self._proj_root}")

    def _save_project_as(self):
        """Save the current state to a NEW project folder (a new version) and switch to it."""
        if not self._proj_root:
            QMessageBox.warning(self, "No project", "Create or open a project first."); return
        parent = QFileDialog.getExistingDirectory(self, "Save project as — choose location")
        if not parent:
            return
        base = os.path.basename(os.path.normpath(self._proj_root))
        name, ok = QInputDialog.getText(self, "Save project as", "New project name:", text=base + "_v2")
        if not ok or not name.strip():
            return
        root = os.path.join(parent, name.strip())
        if os.path.isdir(root) and os.listdir(root):
            if QMessageBox.question(self, "Folder not empty",
                    f"{root}\nalready exists and is not empty.\nSave into it anyway?"
                ) != QMessageBox.StandardButton.Yes:
                return
        try:
            for sub in PROJECT_SUBDIRS:
                os.makedirs(os.path.join(root, sub), exist_ok=True)
        except OSError as ex:
            QMessageBox.critical(self, "Create failed", str(ex)); return
        self._proj_root = root
        self._proj_label.setText(root)
        self._results.set_project(root)
        self._mat_panel.set_project(root)   # redirect .mat writes to the new project (keeps the table)
        self._save_project()                # config, src, rec, mat, manifest -> new root (clears dirty)

    # ── suspend hook ────────────────────────────────────────────────────────────────
    # A CUDA context does not survive suspend on this driver (470): a solver left running
    # comes back as CUDA error 999 + an unkillable D-state process until reboot. So hold a
    # logind "delay" inhibitor and, on PrepareForSleep(true), save, stop every child and
    # exit -- exiting closes the inhibitor fd, which is what lets systemd proceed (it waits
    # at most InhibitDelayMaxSec, 5 s by default; stop() escalates to SIGKILL within that).
    def _install_suspend_hook(self):
        try:
            from PySide6.QtDBus import QDBusConnection, QDBusInterface, QDBusReply
        except ImportError:
            return                                     # non-Linux / QtDBus not built
        bus = QDBusConnection.systemBus()
        if not bus.isConnected():
            return
        svc, path, iface = "org.freedesktop.login1", "/org/freedesktop/login1", "org.freedesktop.login1.Manager"
        mgr = QDBusInterface(svc, path, iface, bus)
        reply = QDBusReply(mgr.call("Inhibit", "sleep", "FASES GUI",
                                    "stop the CUDA solver before suspend", "delay"))
        if not reply.isValid():
            return
        self._sleep_inhibit_fd = reply.value()        # QDBusUnixFileDescriptor; keep it alive
        from PySide6.QtCore import SLOT
        bus.connect(svc, path, iface, "PrepareForSleep", self, SLOT("_on_prepare_for_sleep(bool)"))

    @Slot(bool)
    def _on_prepare_for_sleep(self, going_down):
        if not going_down:
            return
        self._suspending = True                        # closeEvent: save without asking
        self.close()

    def closeEvent(self, e):
        # Offer to save before quitting only if there are unsaved edits (_dirty). A save clears it,
        # so clicking Save then closing won't nag. On suspend there is nobody to ask: just save.
        if self._proj_root and getattr(self, "_dirty", False) and getattr(self, "_suspending", False):
            self._save_project()
        elif self._proj_root and getattr(self, "_dirty", False):
            B = QMessageBox.StandardButton
            ans = QMessageBox.question(self, "Save project?",
                    "Save the project before closing?", B.Save | B.Discard | B.Cancel, B.Save)
            if ans == B.Cancel:
                e.ignore(); return
            if ans == B.Save:
                self._save_project()
        # Tear the audio player down FIRST: the QtMultimedia/FFmpeg backend stalls for seconds on
        # destruction if it still holds a decoded source / open audio device. Stop, drop the source,
        # DETACH the output (forces the device closed), then pump the event loop so the backend
        # actually releases before Qt destroys the widgets — otherwise close hangs.
        if getattr(self, "_aur_blink", None) is not None:
            self._aur_blink.stop()
        if getattr(self, "_player", None) is not None:
            self._player.stop(); self._player.setSource(QUrl()); self._player.setAudioOutput(None)
            QApplication.processEvents()
        for tn in ("_solver_thread", "_estimate_thread", "_rays_thread", "_allirs_thread",
                   "_acou_thread", "_aur_thread", "_foa_thread"):
            th = getattr(self, tn, None)               # stop background QThreads so Qt doesn't
            if th is not None and th.isRunning():      # abort with "Destroyed while running"
                th.stop() if hasattr(th, "stop") else None
                th.wait(2000)
        self._results.close()      # top-level floating window won't close itself with the app
        e.accept()
        # QtMultimedia's FFmpeg backend spawns threads that keep the PROCESS alive after the window
        # closes ("won't exit until I kill the terminal"). State is saved and our threads/player are
        # stopped above, so force the process down rather than hang on that backend's teardown.
        os._exit(0)

    def _load_obj(self):
        # input/scene/ is no longer auto-created (the scene can live anywhere and
        # is referenced by absolute path), so start at the project root — or
        # input/scene if it happens to exist.
        start = ""
        if self._proj_root:
            cand = os.path.join(self._proj_root, "input", "scene")
            start = cand if os.path.isdir(cand) else self._proj_root
        path, _ = QFileDialog.getOpenFileName(self, "Import OBJ", start, "OBJ (*.obj)")
        if path:
            self._load_obj_path(path)

    def _load_obj_path(self, path):
        """Parse an OBJ, load it into the viewport, populate layers + materials,
        and (if a project is open) record the scene in the manifest. Shared by
        the Load-OBJ button and project-open restore."""
        try:
            # Never edit the user's original. If the mesh straddles/negatives the
            # axes, write a shifted COPY next to it and load that; else load as-is.
            # The user's OWN file, before any derived rewrite. Switching coordinate frame
            # must reload from THIS -- reloading a derived copy re-reads already-shifted
            # geometry and the frame never changes. Derived names are suffixed on the stem,
            # and they CHAIN (foo_positive_merged.obj), so test the stem for any of them.
            _stem = os.path.splitext(os.path.basename(path))[0]
            _derived = any(_stem.endswith(sfx) or (sfx + "_") in _stem
                           for sfx in ("_positive", "_merged", "_grouped"))
            if not _derived:
                self._obj_orig_path = path
            base, ext = os.path.splitext(path)
            shifted = base + "_positive" + ext
            prev = tuple(getattr(self, "_obj_shift", (0.0, 0.0, 0.0)))
            if getattr(self._gl, "_shift_enabled", True):
                shift = shift_to_positive(path, shifted)   # writes `shifted` only if a shift is needed
                # A derived copy (project reopen: manifest scene = foo_positive_merged.obj) is
                # ALREADY shifted, so shift_to_positive says 0 -- but relative to the user's
                # original it still carries the recorded shift. Reporting 0 here made the
                # delta below -prev and dragged every src/rec back to the original frame.
                if _derived and shift == (0.0, 0.0, 0.0):
                    shift = prev
            else:
                shift = (0.0, 0.0, 0.0)                    # load as-is; solver offset covers it
            # Remember it: the scene menu can re-display src/rec coordinates in the
            # ORIGINAL frame (BRAS publishes centred-origin positions), while the tables
            # keep handing the solver shifted ones.
            self._obj_shift = tuple(float(v) for v in shift)
            self._gl._obj_shift = self._obj_shift
            # src.txt/rec.txt are canonically in the SHIFTED frame, so they already carry
            # `prev`. Re-apply only the difference, otherwise the points double-shift.
            d = tuple(self._obj_shift[i] - prev[i] for i in range(3))
            moved = self._translate_points(d) if any(d) else 0
            print(f"[frame] shift {prev} -> {self._obj_shift}  delta={tuple(round(v,4) for v in d)}"
                  f"  translated={moved} point(s)"
                  + ("  (delta is zero -> points left alone)" if not any(d) else ""))
            self._apply_orig_frame(getattr(self._gl, "_orig_frame", False))
            if shift != (0.0, 0.0, 0.0):
                print(f"[OBJ] off the positive octant; loading shifted copy "
                      f"(by {shift}): {os.path.basename(shifted)}")
                path = shifted
            # The solver offset (-bmin of the mesh main3D will read) must come from the file
            # that is actually LOADED: the positive copy when shifting is on (bmin = 0 on the
            # shifted axes), the original when it is off. Taken from the original in both
            # frames it double-shifted src.txt in the positive frame and the project re-read
            # then pulled every point back by the shift -- geometry moved, points stayed.
            self._set_solver_offset_from(path)
            verts, groups, split_mtl = parse_obj(path, return_split=True)
            raw_n = len(groups)
            # A file WE wrote is already canonical: its groups are exactly the layers the GUI
            # shows. Re-merging it is wrong twice over -- write_obj emits no `usemtl`, so the
            # material split does not re-detect, respect_names goes off, and the coplanar merge
            # fuses different materials back together (8 layers -> 3); the group count then
            # differs from raw_n so ANOTHER _merged file is written, every single open.
            already_merged = False
            try:
                with open(path) as _f:
                    already_merged = _f.readline().startswith("# merged scene")
            except OSError:
                pass
            if not already_merged:
                # material names are meaningful -> never fuse two different ones
                groups = merge_coplanar_groups(verts, groups, respect_names=split_mtl)
            n_faces_before = sum(len(g["tris"]) for g in groups)
            if not already_merged:
                groups = drop_degenerate_faces(verts, groups)   # purge zero-area slivers + the phantom layers they made
            # `split_mtl` forces the rewrite: raw_n is counted AFTER parse_obj, so a
            # material split is already inside it and the two tests below would both pass
            # -- leaving main3D to read the ORIGINAL file, see one group where the GUI shows
            # several, and die looking up a .fit that isn't in its layer list.
            if not already_merged and (split_mtl or len(groups) != raw_n
                    or sum(len(g["tris"]) for g in groups) != n_faces_before):
                # the file's groups no longer match what the GUI shows/materials. Persist the cleaned
                # set to a derived copy (never the user's original) so the solver reads the same layers
                # — else main3D looks up a .fit for a merged-away group and crashes.
                base0, ext0 = os.path.splitext(path)
                if base0.endswith("_merged"):        # belt-and-braces: never chain the suffix
                    base0 = base0[: -len("_merged")]
                merged_path = base0 + "_merged" + ext0
                write_obj(merged_path, verts, groups)
                path = merged_path
        except Exception as ex:
            QMessageBox.critical(self, "OBJ error", str(ex)); return
        self._obj_path = path
        self._obj_label.setText(os.path.basename(path))
        self._groups = groups
        names = [g["name"] for g in groups]
        shown = ", ".join(names[:12]) + (f", … (+{len(names)-12} more)" if len(names) > 12 else "")
        print(f"[OBJ] Parsed {len(groups)} groups: [{shown}]")
        self._gl.load_mesh(verts, groups)
        self._populate_layers(groups)
        if self._proj_root:
            self._write_manifest()          # persist the scene path on open/load
        status = f"Loaded {len(groups)} layer(s)  —  {len(verts)} vertices"
        self._sb.showMessage(status)

    def _populate_layers(self, groups):
        self._layer_list.clear()
        for g in groups:
            name = g["name"]
            item = QListWidgetItem(name)
            if _is_bbox(name):
                item.setForeground(QColor(PAL["subtext"]))
            self._layer_list.addItem(item)
        self._mat_panel.set_project(self._proj_root)
        self._mat_panel.populate([g["name"] for g in groups])
        # populate() selects material row 0 (needed so edits attribute to a row); don't let that
        # auto-highlight a surface in the 3D scene on load.
        self._gl.clear_selection()
        self._layer_list.setCurrentRow(-1)

    def _show_no_layer_selected(self):
        # Leave the material table selection intact: it must always keep one row
        # selected so value edits attribute to a material. Nothing to do here.
        pass

    def _on_layer_picked(self, name):
        """Sync list selection when user clicks in the GL viewport."""
        if not name:
            self._layer_list.setCurrentRow(-1)
            self._show_no_layer_selected()
            self._sb.showMessage("Layer deselected.")
            return
        for i in range(self._layer_list.count()):
            if self._layer_list.item(i).text() == name:
                self._layer_list.setCurrentRow(i)
                break
        self._tabs.setCurrentIndex(0)
        self._sb.showMessage(f"Selected layer: {name}")

    def _on_surfaces_picked(self, names):
        """RMB-selected surfaces -> ask a target layer, then merge those groups into it."""
        names = set(names)
        if not names:
            return
        existing = [g["name"] for g in self._groups if not _is_bbox(g["name"])]
        target, ok = QInputDialog.getItem(
            self, "Group surfaces into layer",
            f"Assign {len(names)} surface(s) to layer:", existing, 0, True)
        if not ok or not target.strip():
            return
        target = target.strip()
        merge_set = names | {target}
        out_groups = ([{"name": target, "tris": g["tris"]} for g in self._gl._groups
                       if g["name"] in merge_set and not g["is_bbox"]] +      # target faces written CONSECUTIVELY
                      [{"name": g["name"], "tris": g["tris"]} for g in self._gl._groups
                       if g["name"] not in merge_set or g["is_bbox"]])        # so parse_obj collapses them to one
        self._rewrite_scene(out_groups, f"Grouped {len(names)} surface(s) into layer '{target}'.")

    def _on_explode_surface(self, idx):
        """RMB 'Explode' -> split ONLY the one clicked group into its coplanar surfaces, named generically
        (surface_N, numbering continued so repeats don't collide) as an unassigned scratch pool. You then
        pull the ones you want into real material layers and sweep the rest with 'Merge all surface_* → …'
        so main3D isn't left with hundreds of materials/.fit files. Acts by INDEX, never by name — several
        groups can share a name (all props are 'props'), and exploding by name would shatter them all."""
        groups = self._gl._groups
        if not (0 <= idx < len(groups)) or not self._obj_path:
            return
        g = groups[idx]
        if g["is_bbox"] or g["name"] in self._gl._prop_layers():
            return
        verts = np.asarray(self._gl._verts)
        comps = split_coplanar_components(verts, g["tris"])
        if len(comps) <= 1:
            self._sb.showMessage(f"'{g['name']}' is a single surface — nothing to explode."); return
        nums = [int(gg["name"].rpartition("_")[2]) for gg in groups if _sibling_base(gg["name"]) == "surface"]
        start = 1 + (max(nums) if nums else 0)               # continue numbering so repeats don't collide
        others = [{"name": gg["name"], "tris": gg["tris"]} for j, gg in enumerate(groups) if j != idx]
        exploded = [{"name": f"surface_{start + i}", "tris": tr} for i, tr in enumerate(comps)]
        self._rewrite_scene(exploded + others,
                            f"Exploded '{g['name']}' into {len(comps)} surfaces (surface_{start}…).")

    def _set_solver_offset_from(self, path):
        """src.txt must hold (world - bmin): parse_obj_3d shifts the mesh by -bmin on
        EVERY axis while main3D reads sources as (int)(val/h)+3. Zero once the OBJ has
        been rewritten to start at the origin; non-zero when it is loaded as-is."""
        mn = [float("inf")] * 3
        try:
            with open(path) as f:
                for ln in f:
                    t = ln.split()
                    if t and t[0] == "v" and len(t) >= 4:
                        for i in range(3):
                            mn[i] = min(mn[i], float(t[i + 1]))
        except (OSError, ValueError):
            return
        if any(m == float("inf") for m in mn):
            return
        off = tuple(-m for m in mn)
        for t in (self._src_tbl, self._rec_tbl):
            t.set_solver_offset(off)
        if any(abs(v) > 1e-9 for v in off):
            print(f"[OBJ] loaded as-is (min {tuple(round(m,4) for m in mn)}); "
                  f"src/rec written to file with offset {tuple(round(v,4) for v in off)}")

    def _on_shift_policy(self, on):
        self._gl._orig_frame = False          # display offset is meaningless without a shift
        """RMB toggle: rewrite the OBJ to the positive octant, or load it as-is.
        Either way src.txt/rec.txt end up in the frame main3D expects, so results
        do not change -- only which coordinates you see and type."""
        self._sb.showMessage("Scene shifted to the positive octant" if on
                             else "Scene loaded as-is (original coordinates)")
        src = getattr(self, "_obj_orig_path", None) or self._obj_path
        if src and os.path.isfile(src):
            self._load_obj_path(src)

    def _translate_points(self, d):
        """Move every source/receiver by `d` (world frame). Uses update_point per row so
        names/amplitude/frequency survive -- set_points would reset the name column."""
        moved = 0
        for t in (self._src_tbl, self._rec_tbl):
            pts = t.get_points()                      # world (shifted) frame
            for i, pt in enumerate(pts):
                t.update_point(i, pt[0] + d[0], pt[1] + d[1], pt[2] + d[2])
            moved += len(pts)
        return moved

    def _apply_orig_frame(self, on):
        """Show src/rec coordinates in the original OBJ frame (BRAS publishes centred-origin
        positions) while the tables keep feeding the solver shifted ones. Purely a display
        change: main3D converts src.txt with "(int)(val/h)+3" and applies no -bmin, so the
        world frame the tables emit must stay the shifted one."""
        off = tuple(self._obj_shift) if on else (0.0, 0.0, 0.0)
        for t in (self._src_tbl, self._rec_tbl):
            t.set_origin_offset(off)
        self._sb.showMessage(
            f"Coordinates now in the ORIGINAL OBJ frame (offset {off[0]:g}, {off[1]:g}, {off[2]:g})"
            if on else "Coordinates now in the shifted (solver) frame")

    def _on_cleanse_layers(self):
        """RMB 'Cleanse phantom layers' -> drop zero/sliver-area groups and rewrite. Also prints every
        layer's area (smallest first) so we can SEE what the phantom rows really are if drop misses them."""
        if not self._obj_path:
            return
        verts = np.asarray(self._gl._verts, np.float64)
        before = [g for g in self._gl._groups if not _is_bbox(g["name"])]
        stats = []
        for g in before:
            t = np.asarray(g["tris"])
            area = float(_tri_area2(verts, t).sum()) / 2.0 if len(t) else 0.0
            stats.append((area, len(t), g["name"]))
        stats.sort()
        print(f"[cleanse] {len(stats)} non-bbox layers, smallest area first:")
        for area, nt, nm in stats[:50]:
            print(f"    {nm:24s} tris={nt:6d}  area={area:.4e}")
        pl = self._gl._prop_layers()                       # never touch prop geometry
        keep_props = [g for g in self._gl._groups if g["name"] in pl]
        cleaned = drop_degenerate_faces(verts, [g for g in self._gl._groups if g["name"] not in pl]) + keep_props
        gone = {g["name"] for g in before} - {g["name"] for g in cleaned}
        if not gone:
            self._sb.showMessage("No zero/sliver-area layers found — see terminal for layer areas.")
            return
        out_groups = [{"name": g["name"], "tris": g["tris"]} for g in cleaned]
        self._rewrite_scene(out_groups, f"Cleansed {len(gone)} phantom layer(s).")

    def _on_merge_siblings(self, base):
        """Sweep the exploded scratch pool: collapse every '<base>_N' (e.g. surface_*) into ONE named
        layer so only the layers you actually use remain."""
        if not base or not self._obj_path:
            return
        pool = [g for g in self._gl._groups if _sibling_base(g["name"]) == base and not g["is_bbox"]]
        if not pool:
            return
        existing = [g["name"] for g in self._groups if not _is_bbox(g["name"])]
        target, ok = QInputDialog.getItem(
            self, "Merge exploded surfaces",
            f"Collapse {len(pool)} '{base}_*' surface(s) into layer:", existing, 0, True)
        if not ok or not target.strip():
            return
        target = target.strip()
        merge_set = {g["name"] for g in pool} | {target}
        out_groups = ([{"name": target, "tris": g["tris"]} for g in self._gl._groups
                       if g["name"] in merge_set and not g["is_bbox"]] +
                      [{"name": g["name"], "tris": g["tris"]} for g in self._gl._groups
                       if g["name"] not in merge_set or g["is_bbox"]])
        self._rewrite_scene(out_groups, f"Merged {len(pool)} '{base}_*' surfaces into '{target}'.")

    def _rewrite_scene(self, out_groups, msg):
        """Persist a new group set next to the scene and reload it. Shared by group/explode. Stable base
        name (strips trailing _grouped/_merged/_positive) so repeats don't chain suffixes."""
        b, e = os.path.splitext(self._obj_path)
        for s in ("_grouped", "_merged", "_positive"):
            while b.endswith(s):
                b = b[:-len(s)]
        out = b + "_grouped" + e
        try:
            write_obj(out, np.asarray(self._gl._verts), out_groups)
        except Exception as ex:
            QMessageBox.critical(self, "Scene error", str(ex)); return
        self._gl.clear_surf_sel()
        self._load_obj_path(out)
        self._sb.showMessage(msg)

    def _on_layer_row_changed(self, row):
        if row < 0 or row >= len(self._groups):
            self._gl.clear_selection()
            self._show_no_layer_selected()
            return
        name = self._groups[row]["name"]
        self._gl.select_layer(name)
        if _is_bbox(name):
            # bbox has no material row; leave the material table selection as-is
            return
        # select the matching row in the material table -> triggers plot
        if not getattr(self, "_sync_guard", False):
            self._sync_guard = True
            tbl = self._mat_panel._tbl
            for r in range(tbl.rowCount()):
                if tbl.item(r, 0) and tbl.item(r, 0).text() == name:
                    tbl.selectRow(r)
                    break
            self._sync_guard = False
        self._sb.showMessage(f"Selected layer: {name}")

    def _on_material_row_selected(self, name):
        """Reverse path: a row picked in the material table highlights the
        layer in the 3D viewport and in the layer list."""
        if getattr(self, "_sync_guard", False) or not name:
            return
        self._sync_guard = True
        # highlight in viewport
        self._gl.select_layer(name)
        # sync the layer list selection
        for i in range(self._layer_list.count()):
            if self._layer_list.item(i).text() == name:
                self._layer_list.setCurrentRow(i)
                break
        self._sync_guard = False
        # Clicking a material row (and the cross-widget sync above) can move
        # keyboard focus off the table — after which Up/Down do nothing. This
        # branch only runs when the user acted on the material table directly
        # (3D-pick / layer-list selections are guarded out above), so returning
        # focus to the table here is what makes arrow-key row navigation work.
        self._mat_panel._tbl.setFocus(Qt.FocusReason.OtherFocusReason)
        self._sb.showMessage(f"Selected layer: {name}")

    def _set_all_vis(self, v):
        for g in self._groups: self._gl.set_layer_visible(g["name"], v)

    def _toggle_all_layers(self):
        all_on = all(self._gl.is_layer_visible(g["name"]) for g in self._groups) if self._groups else True
        self._set_all_vis(not all_on)
        self._layers_all_btn.setText("Show all" if all_on else "Hide all")

    def _toggle_selected_vis(self):
        row = self._layer_list.currentRow()
        if row < 0 or row >= len(self._groups):
            self._sb.showMessage("Select a layer to toggle.")
            return
        name = self._groups[row]["name"]
        new_vis = not self._gl.is_layer_visible(name)
        self._gl.set_layer_visible(name, new_vis)
        self._sb.showMessage(f"{name}: {'visible' if new_vis else 'hidden'}")

    def _save_config(self):
        if not self._proj_root:
            QMessageBox.warning(self,"No project","Open a project folder first."); return
        p = os.path.join(self._proj_root, "config", "config.txt")
        self._write_config(p)
        self._sb.showMessage(f"Saved config → {p}")

    def _write_config(self, path, src_idx=0):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pml = {k: (1.0 if c.isChecked() else 0.0) for k,c in self._pml.items()}
        ln = lambda v, d: f"{str(v):<10}# {d}\n"   # value padded to col 10, then aligned comment
        with open(path, "w") as f:
            f.write(ln(f"{self._cfg['h'].value():.6g}",        "h [m]"))
            f.write(ln(f"{self._cfg['t_max'].value():.6g}",    "t_max [s]"))
            f.write(ln(f"{self._cfg['f_cutoff'].value():.4g}", "f_cutoff [Hz]"))
            f.write(ln(f"{self._cfg['T_C'].value():.4g}",      "temperature [°C]"))
            f.write(ln(f"{self._cfg['CFL'].value():.6g}",      "CFL"))
            f.write(ln(self._npoles.value(),                   "number of poles for the ADE"))
            for k in ["U_PML","D_PML","R_PML","L_PML","F_PML","B_PML"]:
                f.write(ln(f"{pml[k]:.1f}", k))
            f.write(ln(self._ism_order.value(),                "ISM max reflection order"))
            f.write(ln(f"{self._cfg['altitude'].value():.6g}", "altitude [m]"))
            f.write(ln(self._nrays.value(),                    "number of rays (ray-traced tail)"))
            f.write(ln(max(0, self._src_group.checkedId()),    "source type (0=PCS,1=Ricker,2=Sinusoid)"))
            f.write(ln(self._src_freq.value(),                 "global source frequency [Hz] (PCS/Ricker; sinusoid uses per-source f0)"))
            f.write(ln(int(src_idx),                           "source index (auralization: one run per source; 0 otherwise)"))
            # animation slice planes (world coords); main3D converts to i0/j0/k0 in its grid
            s = self._gl._slices if self._gl._slices is not None else (0.0, 0.0, 0.0)
            f.write(ln(f"{float(s[0]):.6g}", "slice x0 [m] (animation)"))
            f.write(ln(f"{float(s[1]):.6g}", "slice y0 [m] (animation)"))
            f.write(ln(f"{float(s[2]):.6g}", "slice z0 [m] (animation)"))
            f.write(ln(self._fps.value(), "animation fps (frames capped to RAM in main3D)"))
            f.write(ln(f"{self._pmin.value():.6g}", "animation pressure min [Pa] (GUI only)"))
            f.write(ln(f"{self._pmax.value():.6g}", "animation pressure max [Pa] (GUI only)"))

    def _load_config(self):
        start = os.path.join(self._proj_root,"config") if self._proj_root else ""
        p, _ = QFileDialog.getOpenFileName(self,"Load config",start,"Text (*.txt)")
        if p: self._load_config_from(p)

    def _load_config_from(self, path):
        vals = []
        with open(path) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if line:
                    try: vals.append(float(line))
                    except ValueError: pass
        for i, k in enumerate(["h","t_max","f_cutoff","T_C","CFL"]):
            if i < len(vals): self._cfg[k].setValue(vals[i])
        if len(vals) > 5:
            self._npoles.setValue(int(vals[5]))   # number of poles for the ADE
        for j, k in enumerate(["U_PML","D_PML","R_PML","L_PML","F_PML","B_PML"]):
            idx = 6+j
            if idx < len(vals): self._pml[k].setChecked(vals[idx] != 0.0)
        self._sync_pml_all_btn()       # label follows the loaded PML state
        if len(vals) > 12:
            self._ism_order.setValue(int(vals[12]))  # ISM max reflection order
        if len(vals) > 13:
            self._cfg["altitude"].setValue(vals[13])  # altitude [m]
        if len(vals) > 14:
            self._nrays.setValue(int(vals[14]))  # number of rays (ray-traced tail)
        if len(vals) > 15:
            btn = self._src_group.button(max(0, min(2, int(vals[15]))))  # 0=PCS,1=Ricker,2=Sinusoid
            if btn: btn.setChecked(True)
        if len(vals) > 16:
            self._src_freq.setValue(int(vals[16]))  # global source frequency [Hz] (PCS/Ricker)
        if len(vals) > 20:
            self._gl.set_slices(vals[18], vals[19], vals[20])  # animation slice planes
            self._sync_slice_fields(vals[18], vals[19], vals[20])
        if len(vals) > 21:
            self._fps.setValue(int(vals[21]))  # animation fps
        if len(vals) > 23:
            self._pmin.setValue(float(vals[22]))   # animation pressure range
            self._pmax.setValue(float(vals[23]))

    def _save_src(self):
        if not self._proj_root: return
        self._src_tbl.save_file(os.path.join(self._proj_root,"input","src","src.txt"))
        self._sb.showMessage("Saved src.txt")

    def _load_src(self):
        start = os.path.join(self._proj_root,"input","src") if self._proj_root else ""
        p, _ = QFileDialog.getOpenFileName(self,"Load src.txt",start,"Text (*.txt)")
        if p:
            self._remember_points_for_undo()
            self._src_tbl.load_file(p)
            self._refresh_gl_points()

    def _save_rec(self):
        if not self._proj_root: return
        self._rec_tbl.save_file(os.path.join(self._proj_root,"input","rec","rec.txt"))
        self._sb.showMessage("Saved rec.txt")

    def _load_rec(self):
        start = os.path.join(self._proj_root,"input","rec") if self._proj_root else ""
        p, _ = QFileDialog.getOpenFileName(self,"Load rec.txt",start,"Text (*.txt)")
        if p:
            self._remember_points_for_undo()
            self._rec_tbl.load_file(p)
            self._refresh_gl_points()

    def _refresh_gl_points(self):
        self._gl.set_sources(  [(p[0],p[1],p[2]) + ((p[3],) if len(p) > 3 else ())
                                for p in self._src_tbl.get_points()])
        self._gl.set_receivers([(p[0],p[1],p[2]) for p in self._rec_tbl.get_points()])
        self._gl.clear_animation()   # source SPL/pos changed -> rebuild the particle sim on next play

    def _browse_binary(self):
        p, _ = QFileDialog.getOpenFileName(self, "Locate solver binary")
        if p:
            QSettings("phases", "gui").setValue("last_binary", p)  # remember
            if self._proj_root and p.startswith(self._proj_root):
                p = os.path.relpath(p, self._proj_root)
            self._bin_le.setText(p)

    # ── GA-only (reuse wave-based) ──────────────────────────────
    def _wave_sig_path(self):
        return os.path.join(self._proj_root, "output", ".wave_sig") if self._proj_root else None

    def _wave_fingerprint(self):
        """{label: sha1} for every input the FVM low-freq IR depends on, hashed on
        NORMALISED content (comments + surrounding whitespace stripped, blank lines
        dropped) so the writers' formatting jitter can't cause a false mismatch. Covers
        config (minus ism_order = index 12 and nrays = index 14, the GA-only knobs),
        poles.txt, sources (src.txt), receivers (rec.txt), the scene OBJ, and every .mat
        (layers/surfaces). Scattering (.sctr) is deliberately left out — GA-only.
        Returns None if config.txt is unreadable."""
        import hashlib, glob as _glob
        if not self._proj_root:
            return None
        root = self._proj_root

        def norm(path, drop=()):
            try:
                out = []
                for i, raw in enumerate(open(path)):
                    s = " ".join(raw.split("#")[0].split())   # drop comment, collapse all whitespace
                    if s and i not in drop:
                        out.append(s)
                return "\n".join(out)
            except Exception:
                return None

        cfg = norm(os.path.join(root, "config", "config.txt"),
                   drop={12, 14, 18, 19, 20, 21})  # ism_order, nrays, slices, fps: not FVM inputs
        if cfg is None:
            return None
        comps = {
            "config": cfg,
            "poles":  norm(os.path.join(root, "config", "poles.txt")) or "",
            "src":    norm(os.path.join(root, "input", "src", "src.txt")) or "",
            "rec":    norm(os.path.join(root, "input", "rec", "rec.txt")) or "",
        }
        obj = self._obj_path if (self._obj_path and os.path.isfile(self._obj_path)) else None
        comps["obj"] = (norm(obj) or "") if obj else ""
        mats = sorted(_glob.glob(os.path.join(root, "input", "mat", "*.mat")))
        comps["mat"] = "\n".join(f"{os.path.basename(m)}={norm(m)}" for m in mats)
        return {k: hashlib.sha1(v.encode()).hexdigest()[:12] for k, v in comps.items()}

    def _ir_bins_exist(self):
        import glob as _glob
        return bool(self._proj_root and
                    _glob.glob(os.path.join(self._proj_root, "output", "IR*_*.bin")))

    def _ga_available(self):
        """True when the current wave fingerprint matches the one stored after the last
        full run AND the IR*.bin are present. Logs which component differs when it isn't,
        so a stray full run is never a mystery."""
        if not self._proj_root:
            return False
        fp = self._wave_fingerprint()
        if fp is None:
            return False
        try:
            stored = dict(l.strip().split("=", 1)
                          for l in open(self._wave_sig_path()) if "=" in l)
        except Exception:
            stored = None
        if not stored:
            print(f"[ga] no/empty stored signature at {self._wave_sig_path()} -> full run")
            return False
        import glob as _g
        patt = os.path.join(self._proj_root, "output", "IR*_*.bin")
        hits = _g.glob(patt)
        if not hits:
            print(f"[ga] no IR*_*.bin (glob '{patt}' found 0) -> full run")
            return False
        changed = sorted({k for k in fp if fp.get(k) != stored.get(k)} |
                         {k for k in stored if k not in fp})
        if changed:
            print("[ga] wave inputs changed:", ", ".join(changed), "-> full run")
            return False
        return True

    def _store_wave_signature(self):
        """Record the current wave fingerprint (one 'label=hash' line each) after a
        completed full run — the baseline the next Simulate compares against."""
        p = self._wave_sig_path()
        fp = self._wave_fingerprint()
        if not (p and fp):
            return
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write("\n".join(f"{k}={v}" for k, v in sorted(fp.items())))
            print(f"[ga] stored wave signature -> {p}")
        except Exception as e:
            print(f"[ga] FAILED to store wave signature at {p}: {e}")

    def _field_binding_h(self, voxels, h):
        """Coarsest-constraint grid spacing that fits the FIELD allocation (36 B/vox host,
        52 B/vox device) -- the BINDING gate, larger than the SDF grid. Returns the suggested
        h (or None if it already fits / free memory is unreadable). Mirrors main3D.cu's budget,
        so one suggestion clears both the SDF and field gates instead of iterating."""
        try:
            free_host = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        except Exception:
            free_host = 0
        free_dev = 0
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            free_dev = float(r.stdout.split()[0]) * 1024 * 1024      # MiB -> bytes
        except Exception:
            free_dev = 0
        over = 0.0
        if free_host > 0: over = max(over, (36.0 * voxels) / (0.7 * free_host))
        if free_dev  > 0: over = max(over, (52.0 * voxels) / (0.9 * free_dev))
        return h * over ** (1.0/3.0) if over > 1.0 else None

    def _memory_preflight(self):
        """Run --estimate and detect a memory blocker for the CURRENT h. Catches BOTH gates:
        the early SDF/grid 'does not fit' abort (fires before [MEM], on stderr) and the parsed
        [MEM] field-allocation budget. Returns a ready-to-show message string, or None if memory
        is fine / the check couldn't run (binary missing, etc.) -> caller fails open."""
        binary = self._bin_le.text()
        bin_cmd = binary if os.path.isabs(binary) else os.path.join(self._proj_root, binary)
        scene = self._obj_path
        if scene.startswith(self._proj_root):
            scene = os.path.relpath(scene, self._proj_root)
        cmd = [bin_cmd, scene, "config/config.txt",
               "input/src/src.txt", "input/rec/rec.txt", "--estimate"]
        try:
            r = subprocess.run(cmd, cwd=self._proj_root, capture_output=True,
                               text=True, timeout=180)
        except Exception:
            return None
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        # (a) early SDF / grid 'does not fit' abort. Its own suggested h only clears the SDF
        # grid; re-derive the BINDING field-allocation h from its voxel count so a single change
        # clears both gates (no iterating up in steps).
        for ln in out.splitlines():
            if ("does not fit" in ln) or ("Increase h" in ln and "voxels" in ln):
                import re
                mv = re.search(r"([0-9.eE+]+)\s*voxels", ln)
                mh = re.search(r"h\s*=\s*([0-9.eE+\-]+)", ln)
                hb = self._field_binding_h(float(mv.group(1)), float(mh.group(1))) if (mv and mh) else None
                if hb:
                    return ("Grid too fine for available memory: increase h to >= {:.4g} m "
                            "(covers the SDF grid AND the field allocation in one step).".format(hb))
                return ln.split("]", 1)[-1].strip()
        # (b) structured field-allocation budget from the [MEM] line (needs the rebuilt binary)
        for ln in out.splitlines():
            if ln.startswith("[MEM]"):
                d = {}
                for tok in ln.split()[1:]:
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        try: d[k] = float(v)
                        except ValueError: pass
                if d.get("over", 0.0) > 1.0:
                    return ("Not enough memory: needs ~{:.2f} GB host / ~{:.2f} GB device, but only "
                            "~{:.2f} / ~{:.2f} GB free. Increase h to >= {:.4g} m.".format(
                                d.get("need_host_gb",0), d.get("need_dev_gb",0),
                                d.get("free_host_gb",0), d.get("free_dev_gb",0), d.get("min_h",0)))
                return None
        return None

    def _preflight_warnings(self):
        """Pre-solve checks. Returns (blockers, advisories):
          blockers   -- would exhaust RAM and freeze/crash the machine (memory): run REFUSED.
          advisories -- accuracy only (resolution, sampling): the run completes, so overridable.
        Resolution/sampling are pure Python (work without a rebuild); memory needs the binary's
        [MEM] line and is skipped silently otherwise (fail-open)."""
        block, warn = [], []
        c  = math.sqrt(1.4 * 287.05 * (self._cfg["T_C"].value() + 273.15))  # speed of sound
        h  = self._cfg["h"].value()
        fc = self._cfg["f_cutoff"].value()
        # SPATIAL RESOLUTION: >= 10 cells per shortest wavelength (dispersion floor). ADVISORY.
        if h > 0.0 and fc > 0.0:
            ppw = c / (fc * h)
            if ppw < 10.0:
                warn.append(
                    "Grid too coarse: {:.1f} cells per wavelength at {:.0f} Hz (need >= 10). "
                    "Reduce h to <= {:.4g} m, or lower the cutoff frequency.".format(ppw, fc, c/(10.0*fc)))
        # OUTPUT SAMPLING: RT/ISM tail is sampled at 44.1 kHz (main3D.cu), so the cutoff must stay
        # under the usable band or high frequencies alias. (The FVM solver's own dt = CFL*h/c gives
        # fs = c/(CFL*h) >= 2*fc automatically once the resolution check holds, so no separate
        # temporal warning is needed -- a coarse grid surfaces as the resolution advisory.) ADVISORY.
        FS_OUT = 44100.0
        if fc > 0.45 * FS_OUT:
            warn.append(
                "Cutoff {:.0f} Hz is above the usable output band (44.1 kHz sample rate, "
                "{:.1f} kHz Nyquist). Lower the cutoff to <= {:.1f} kHz.".format(
                    fc, FS_OUT/2000, 0.45*FS_OUT/1000))
        # MEMORY (both gates: SDF grid + field allocation): exceeding host/device RAM aborts
        # the solver or FREEZES the machine -> HARD BLOCK. _memory_preflight returns the ready
        # message (or None).
        mem_msg = self._memory_preflight()
        if mem_msg:
            block.append(mem_msg)
        return block, warn

    def _run_solver(self):
        if not self._proj_root:
            QMessageBox.warning(self,"No project","Open project first."); return
        if not self._obj_path:
            QMessageBox.warning(self,"No scene","Load an OBJ first."); return
        cfg_path = os.path.join(self._proj_root,"config","config.txt")
        self._write_config(cfg_path)
        self._src_tbl.save_file(os.path.join(self._proj_root,"input","src","src.txt"))
        self._rec_tbl.save_file(os.path.join(self._proj_root,"input","rec","rec.txt"))
        self._mat_panel.save_all()
        # material values changed since last fit -> .fit (and its ADE poles) are stale.
        # Running now would simulate with old impedances. Warn, offer to refit first.
        try:
            stale = self._mat_panel.stale_fits(cfg_path)
        except Exception:
            stale = []              # a stale-check bug must never block Simulate
        if stale:
            B = QMessageBox.StandardButton
            ans = QMessageBox.warning(self, "Impedances out of date",
                    "Materials or resolution changed since the last fit:\n\n  • "
                    + "\n  • ".join(stale)
                    + "\n\nFit impedances first, then simulate?",
                    B.Yes | B.No | B.Cancel, B.Yes)
            if ans == B.Cancel:
                return
            if ans == B.Yes:
                # fit now; _on_fit_all_finished relaunches the run when it completes
                self._pending_sim = True
                self._fit_impedances()
                return
            # B.No -> fall through and simulate with the old fits
        # pre-flight checks. MEMORY over-budget is a HARD STOP -- it would exhaust RAM and
        # freeze/crash the machine, so the run is refused outright (no override). Resolution /
        # sampling only make the result inaccurate (the run still completes), so they warn with
        # an override.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        block, warn = self._preflight_warnings()
        QApplication.restoreOverrideCursor()
        if block:
            QMessageBox.critical(self, "Cannot start simulation",
                                 "\n\n".join("• " + b for b in block) +
                                 "\n\nThe machine would run out of memory — fix this before running.")
            return
        if warn:
            if QMessageBox.warning(self, "Check simulation settings",
                                   "\n\n".join("• " + w for w in warn) + "\n\nStart anyway?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                   QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
        scene = self._obj_path
        if scene.startswith(self._proj_root):
            scene = os.path.relpath(scene, self._proj_root)
        # The solver requires poles.txt (argv[5], written by vectFIT). Catch a
        # missing one here with a clear message instead of a solver error.
        if not os.path.isfile(os.path.join(self._proj_root, "config", "poles.txt")):
            QMessageBox.warning(self, "No poles.txt",
                "config/poles.txt not found — fit the materials first "
                "(Scene tab → Fit all)."); return
        binary = self._bin_le.text()
        # absolute binary paths are used as-is; only bare relative ones get ./
        bin_cmd = binary if os.path.isabs(binary) else f"./{binary}"
        self._aural_cmd = [bin_cmd, scene,
               "config/config.txt","input/src/src.txt","input/rec/rec.txt",
               "config/poles.txt"]
        # DEFAULT to GA-only automatically: if nothing the FVM depends on has changed since
        # the last full run (and its IR*.bin are present), skip the FVM and reuse them.
        # Otherwise run the full pipeline (which then re-stamps the wave signature).
        self._is_ga_only_run = self._ga_available()
        if self._is_ga_only_run:
            self._aural_cmd.append("--ga-only")
        self._log.clear()
        self._log.append("GA-only: wave inputs unchanged — reusing existing IR, skipping the FVM."
                         if self._is_ga_only_run else
                         "Full run: wave inputs changed (or no prior IR) — computing the FVM.")
        self._tabs.setCurrentIndex(4)

        n_src = len(self._src_tbl.get_points())
        if self._aural.isChecked() and n_src > 1:
            # one run per source; _on_solver_done chains the rest
            self._aural_queue = list(range(1, n_src))
            self._log.append(f"Auralization: {n_src} sources, one run each")
            self._run_source(0)
        else:
            self._aural_queue = []
            self._write_config(os.path.join(self._proj_root,"config","config.txt"), 0)
            self._launch_solver(self._aural_cmd)

    def _launch_solver(self, cmd):
        import time
        self._t_start = time.time()                    # for the live ETA (see _on_solver_progress)
        # time-left takes over the status bar for the whole run (released in _on_solver_done)
        self._sb.hold = False
        self._sb.showMessage("Simulating… estimating time left…", priority=True)
        self._sb.hold = True
        self._log.append(f"$ cd {self._proj_root}\n$ " + " ".join(cmd))
        self._log.append("-"*60)
        self._set_running(True)
        self._solver_thread = SolverThread(cmd, self._proj_root)
        self._solver_thread.output.connect(self._log.append)
        self._solver_thread.progress.connect(self._on_solver_progress)
        self._solver_thread.done.connect(self._on_solver_done)
        self._solver_thread.start()

    def _run_source(self, i):
        """Auralization: write config (src index = i) + a 1-source src.txt, then launch. Source 0 does
        the normal full purge (fresh directory); sources 1..N keep the earlier sources' bins so all
        per-source IR/HIR/ARR accumulate (else each run wipes the last)."""
        self._write_config(os.path.join(self._proj_root,"config","config.txt"), i)
        pt = self._src_tbl.get_points()[i]
        with open(os.path.join(self._proj_root,"input","src","src.txt"), "w") as f:
            f.write("# x y z p0 f0\n")
            f.write("  ".join(f"{v:.6g}" for v in pt) + "\n")
        self._log.append(f'<span style="color:{PAL["subtext"]};">— source {i} —</span>')
        cmd = list(self._aural_cmd)
        if i > 0:
            cmd.append("--keep-other-sources")           # don't wipe sources already computed this batch
        self._launch_solver(cmd)

    def _fit_impedances(self):
        """Run vectFIT over ALL materials (was the Scene tab's 'Fit all').
        Defines cutoff frequency + ADE pole count and writes poles.txt / .fit,
        which the solver needs before a run — hence it lives here on the Run tab."""
        if not self._proj_root:
            self._sb.showMessage("Open a project folder first."); return
        # Flush the current GUI state to disk BEFORE fitting, so the fit stamps the same
        # dt/fc/N/alphas the next Simulate checks against. Otherwise the fitter reads a
        # stale config.txt/.mat and the just-made fit still reads as out of date.
        self._write_config(os.path.join(self._proj_root, "config", "config.txt"))
        self._mat_panel.save_all()
        self._mat_panel.run_fit_all()

    def _on_fit_all_finished(self, rc):
        """If the user picked 'Fit first' on Simulate, relaunch the run once the
        fit finishes cleanly. On failure, drop the pending run."""
        if not self._pending_sim:
            return
        self._pending_sim = False
        if rc == 0:
            self._run_solver()   # fits now fresh -> stale check passes, runs
        else:
            QMessageBox.warning(self, "Fit failed",
                                "Impedance fit failed — simulation not started.")

    def _toggle_run(self):
        """The single Run/Stop button: launch if idle, stop if running."""
        if self._solver_thread and self._solver_thread.isRunning():
            self._stop_solver()
        else:
            self._run_solver()

    def _set_running(self, running):
        """Flip the Run/Stop button appearance and progress-bar visibility."""
        if running:
            self._run_btn.setText("⏹  Stop")
            self._run_btn.setStyleSheet(
                f"background:{PAL['err']};color:{PAL['bg']};font-weight:bold;padding:8px;")
            self._float_prog.setValue(0)
            self._float_prog.show()
            self._float_prog.raise_()
            self._fitimp_btn.setEnabled(False)
        else:
            self._run_btn.setText("▷  Simulate")
            self._run_btn.setStyleSheet(
                f"background:{PAL['accent']};color:{PAL['bg']};font-weight:bold;padding:8px;")
            self._fitimp_btn.setEnabled(True)

    @staticmethod
    def _fmt_eta(s):
        s = int(max(0, s))
        if s >= 3600:
            return f"{s // 3600}h {(s % 3600) // 60}m"
        if s >= 60:
            return f"{s // 60}m {s % 60:02d}s"
        return f"{s}s"

    def _on_solver_progress(self, pct):
        self._float_prog.setValue(pct)
        t0 = getattr(self, "_t_start", None)
        if t0 and pct > 0:                             # live ETA: extrapolate from elapsed / percent
            import time
            remaining = (time.time() - t0) * (100 - pct) / pct
            self._sb.showMessage(f"Simulating… {pct}%  ·  ~{self._fmt_eta(remaining)} left",
                                 priority=True)

    def _stop_solver(self):
        self._sb.hold = False       # user aborted: release the ETA hold
        if self._solver_thread and self._solver_thread.isRunning():
            self._solver_thread.stop()
            self._log.append("\n[!] Stopped by user.")
        self._set_running(False)
        self._float_prog.hide()

    def _load_recommended(self):
        """Seed the Setup fields from the single-value binaries the --estimate pass
        writes into config/ (Vol.bin, f_Schroeder.bin). The user can override any of
        them afterwards.

        Derivations
        -----------
        c        : speed of sound from temperature, c = sqrt(gamma * R_spec * T).
        f_cutoff : the Schroeder frequency f_Schroeder = 2000 * sqrt(T60_500 / V).
                   Above it the field is statistically diffuse, so the wave solver
                   only needs to resolve UP TO f_Schroeder (ISM/RT cover higher bands).
        h (grid) : wave-resolution rule. The shortest wavelength is lambda_min =
                   c / f_cutoff; resolving it needs >= PPW cells per wavelength, so
                   h = c / (PPW * f_cutoff). main3D aborts below 10 cells/lambda, so
                   PPW = 12 keeps a margin. Coarser h => fewer voxels => faster.
        t_max    : longest FINITE T60 across octave bands. t_max bounds the whole IR
                   (FVM + ISM + RT), so it must outlast the slowest-decaying band --
                   the last one still ringing sets when the room is silent (60 dB down).
                   estimate_T60s writes 1e9 for near-rigid bands, so those are skipped.
                   (500 Hz, T60[3], is the conventional single-number RT alternative.)
        Nrays    : Vorlander detection-density criterion. Mean ray hits on a detection
                   sphere of radius R per sample bin: <k> = N * c * pi * R^2 / (V * fs);
                   solve for N at a target <k>. Ref: M. Vorlander, "Auralization"
                   (Springer, 2008); H. Kuttruff, "Room Acoustics".

        ISM order: mean-free-path relation. mfp = 4V/S [Kuttruff]; mean time between
                   reflections = mfp/c; so the order reaching a transition time t is
                   order ~ c*t/mfp = c*t*S/(4V). t is the mixing time t_mix ~
                   POLACK_MS*sqrt(V) [ms] -- Polack's rule (J.-D. Polack, "Using Matching
                   Pursuit for estimating mixing time within Room Impulse Responses," 2009).
                   Clamped to [1,8]; the ISM solver also clamps deep trees at runtime,
                   so an over-estimate is safe.
        """
        if not self._proj_root:
            return
        cdir = os.path.join(self._proj_root, "config")
        c = math.sqrt(1.4 * 287.05 * (self._cfg["T_C"].value() + 273.15))  # speed of sound

        # total surface area (for the ISM max-order estimate below); None if absent
        A_tot = None
        a_path = os.path.join(cdir, "A_tot.bin")
        if os.path.isfile(a_path):
            try:
                A_tot = float(np.fromfile(a_path, dtype=np.float32, count=1)[0])
            except Exception:
                A_tot = None

        # f_Schroeder -> recommended cutoff frequency AND grid spacing h
        fs_path = os.path.join(cdir, "f_Schroeder.bin")
        if os.path.isfile(fs_path):
            try:
                fS = float(np.fromfile(fs_path, dtype=np.float32, count=1)[0])
                if fS > 0.0:
                    self._cfg["f_cutoff"].setValue(fS)
                    PPW = 12.0                        # cells per shortest wavelength (>10 floor)
                    self._cfg["h"].setValue(c / (PPW * fS))
            except Exception:
                pass

        # T60 (8 octave bands) -> recommended simulation time = longest finite band
        t60_path = os.path.join(cdir, "T60.bin")
        if os.path.isfile(t60_path):
            try:
                T60 = np.fromfile(t60_path, dtype=np.float32, count=8)
                finite = [float(t) for t in T60 if 0.0 < t < 1.0e6]  # skip rigid-band 1e9 sentinel
                if finite:
                    self._cfg["t_max"].setValue(min(60.0, max(finite)))  # clamp to spin max
            except Exception:
                pass

        # Vol + surface area -> recommended ray count for the NEXT-EVENT diffuse tail.
        # The diffuse detector shoots a shadow ray to the receiver at EVERY bounce, so the
        # echogram is filled by ~Nrays*(bounces/ray) deposits, NOT by rare detection-sphere
        # crossings. With bounces/ray = c*T_IR/mfp and mfp = 4V/A_tot [Kuttruff], the deposits
        # per sample bin come out T_IR-independent:
        #     deposits/bin = Nrays * c * A_tot / (4 V fs)  ->  Nrays = k' * 4 V fs / (c * A_tot)
        # This is R-INDEPENDENT: the receiver radius only drives the minor late-SPECULAR
        # (detection-sphere) channel, so a small auralization receiver (R<0.12 m) does NOT
        # inflate it -- the old sphere formula N=V*fs/(c*pi*R^2) over-counted by ~100x.
        # k' = target diffuse deposits per sample bin; ~100 gives per-bin ~1/sqrt(k')=10%,
        # smoothed to ~1-2% over the ms envelope. Raise it if the LATE tail looks grainy
        # (high-order bounces are the first to thin out in a very absorptive room).
        vol_path = os.path.join(cdir, "Vol.bin")
        if os.path.isfile(vol_path):
            try:
                V = float(np.fromfile(vol_path, dtype=np.float32, count=1)[0])
                if V > 0.0:
                    fs = 44100.0                      # RT/ISM sample rate (main3D.cu)
                    kp = 100.0                        # target diffuse deposits per sample bin
                    S  = A_tot if (A_tot and A_tot > 0.0) else 6.0 * V**(2.0/3.0)  # fallback: cube-ish
                    N  = kp * 4.0 * V * fs / (c * S)
                    self._nrays.setValue(int(max(10_000, min(N, 10_000_000))))

                    # ISM max order from the mean-free-path relation with Polack's
                    # sqrt(V) mixing time: mfp = 4V/A_tot [Kuttruff], order to reach
                    # t_mix is c*t_mix/mfp, t_mix = POLACK_MS*sqrt(V) [ms]. Clamp to the
                    # spin range; the ISM solver also clamps deep trees at runtime.
                    if A_tot and A_tot > 0.0:
                        POLACK_MS = 1.0                    # t_mix[ms] = POLACK_MS*sqrt(V); tune to your ref
                        mfp   = 4.0 * V / A_tot
                        t_mix = POLACK_MS * math.sqrt(V) / 1000.0   # ms -> s
                        order = int(max(1, min(8, round(c * t_mix / mfp))))
                        self._ism_order.setValue(order)

                    self._sb.showMessage(
                        f"Recommended: f_cut={self._cfg['f_cutoff'].value():.0f} Hz, "
                        f"h={self._cfg['h'].value():.3g} m, t_max={self._cfg['t_max'].value():.3g} s, "
                        f"rays={self._nrays.value()}, ISM order={self._ism_order.value()}  (V={V:.1f} m³)")
            except Exception:
                pass

    def _estimate_room(self):
        """Run the solver's --estimate pass (parse + V + T60, no FVM) so the
        recommended Cutoff frequency / Number of rays fill in BEFORE a real run.
        Needs no poles.txt/fit -- T60 uses only the .mat absorption."""
        if not self._proj_root:
            QMessageBox.warning(self, "No project", "Open a project folder first."); return
        if not self._obj_path:
            QMessageBox.warning(self, "No scene", "Load an OBJ first."); return
        # write config + materials so the estimate uses CURRENT values: T60/f_Schroeder depend
        # on the .mat absorptions, which are otherwise only flushed on Run/Fit/Save.
        self._write_config(os.path.join(self._proj_root, "config", "config.txt"))
        self._mat_panel.save_all()
        scene = self._obj_path
        if scene.startswith(self._proj_root):
            scene = os.path.relpath(scene, self._proj_root)
        binary = self._bin_le.text()
        bin_cmd = binary if os.path.isabs(binary) else f"./{binary}"
        # src/rec/poles are never opened in --estimate mode, but the paths must be present
        cmd = [bin_cmd, scene, "config/config.txt",
               "input/src/src.txt", "input/rec/rec.txt", "config/poles.txt",
               "--estimate"]
        self._btn_estimate.setEnabled(False)
        self._sb.showMessage("Estimating V, T60 …")
        self._log.append("$ " + " ".join(cmd))
        self._estimate_thread = SolverThread(cmd, self._proj_root)
        self._estimate_thread.output.connect(self._log.append)   # show its output in the Run log
        self._estimate_thread.done.connect(self._on_estimate_done)
        self._estimate_thread.start()

    def _on_estimate_done(self, rc):
        self._btn_estimate.setEnabled(True)
        if rc == 0:
            self._load_recommended()   # reads Vol.bin / f_Schroeder.bin, fills the fields
            self._sb.showMessage("Estimate done — recommended values filled in.")
        else:
            self._sb.showMessage(f"Estimate failed (rc={rc}).")
            QMessageBox.warning(self, "Estimate failed",
                f"The estimate pass exited with code {rc} and did NOT update the fields.\n\n"
                "The most common cause is that the current grid is too fine to even build "
                "(the estimate builds the SDF at the current h, so a small h can run out of "
                "memory before it can recommend a coarser one).\n\n"
                "Increase h (or lower the cutoff frequency) and retry. The Run-tab log shows "
                "the solver's exact message (e.g. an SDF 'does not fit' line).")

    def _toggle_rays(self):
        """One button: trace if none shown, else clear (text flips Trace <-> Clear rays)."""
        if self._gl._ray_paths:
            self._clear_rays()
        else:
            self._dump_rays()

    def _dump_rays(self):
        """Run the solver's --dump-rays pass (trace 'Debug rays' rays up to 'Order'
        bounces, no FVM) and overlay the paths on the 3D scene (purplish blue)."""
        if not self._proj_root:
            QMessageBox.warning(self, "No project", "Open a project folder first."); return
        if not self._obj_path:
            QMessageBox.warning(self, "No scene", "Load an OBJ first."); return
        self._write_config(os.path.join(self._proj_root, "config", "config.txt"))
        self._src_tbl.save_file(os.path.join(self._proj_root, "input", "src", "src.txt"))
        scene = self._obj_path
        if scene.startswith(self._proj_root):
            scene = os.path.relpath(scene, self._proj_root)
        binary = self._bin_le.text()
        bin_cmd = binary if os.path.isabs(binary) else f"./{binary}"
        cmd = [bin_cmd, scene, "config/config.txt",
               "input/src/src.txt", "input/rec/rec.txt", "config/poles.txt",
               "--dump-rays", str(self._dbg_rays.value()), str(self._dbg_order.value())]
        self._btn_rays.setEnabled(False)
        self._sb.showMessage("Tracing rays …")
        self._rays_thread = SolverThread(cmd, self._proj_root)
        self._rays_thread.done.connect(self._on_dump_rays_done)
        self._rays_thread.start()

    def _on_dump_rays_done(self, rc):
        self._btn_rays.setEnabled(True)
        if rc == 0:
            paths = self._gl.load_ray_paths(
                os.path.join(self._proj_root, "output", "ray_paths.txt"))
            self._populate_ray_table()
            nsrc = len(set(self._gl._ray_src.values()))
            self._sb.showMessage(f"Traced {len(paths)} rays from {nsrc} source(s) — "
                                 f"pick a row to highlight its path")
            if paths:
                self._btn_rays.setText("Clear rays")
        else:
            self._sb.showMessage(f"Ray trace failed (rc={rc})")

    # ── floating ray table ───────────────────────────────────
    def _ensure_ray_table(self):
        if getattr(self, "_ray_dock", None) is not None:
            return
        self._ray_tbl = QTableWidget(0, 3)
        self._ray_tbl.setHorizontalHeaderLabels(["Src", "Ray", "Length (m)"])
        self._ray_tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._ray_tbl.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._ray_tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._ray_tbl.itemSelectionChanged.connect(self._on_ray_row_selected)
        self._ray_dock = QDockWidget("Rays", self)
        self._ray_dock.setWidget(self._ray_tbl)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._ray_dock)
        self._ray_dock.setFloating(True)
        self._ray_dock.resize(320, 400)

    def _populate_ray_table(self):
        self._ensure_ray_table()
        paths = self._gl._ray_paths; srcs = self._gl._ray_src
        names = self._src_tbl.get_names()          # source id -> user name
        t = self._ray_tbl
        t.blockSignals(True)
        t.setRowCount(0)
        for rid in sorted(paths):
            segs = paths[rid]
            length = sum(math.dist(p0, p1) for p0, p1, o in segs if o >= 0)
            sid = srcs.get(rid, 0)
            sname = names[sid] if 0 <= sid < len(names) else str(sid)
            r = t.rowCount(); t.insertRow(r)
            t.setItem(r, 0, QTableWidgetItem(sname))
            t.setItem(r, 1, QTableWidgetItem(str(rid)))
            t.setItem(r, 2, QTableWidgetItem(f"{length:.2f}"))
        t.blockSignals(False)
        t.clearSelection()
        self._ray_dock.show(); self._ray_dock.raise_()

    def _on_ray_row_selected(self):
        rows = self._ray_tbl.selectionModel().selectedRows()
        rid = int(self._ray_tbl.item(rows[0].row(), 1).text()) if rows else None
        self._gl.set_selected_ray(rid)

    def _clear_rays(self):
        self._gl.clear_ray_paths()
        self._btn_rays.setText("Trace rays")
        if getattr(self, "_ray_dock", None) is not None:
            self._ray_tbl.setRowCount(0)
            self._ray_dock.hide()

    def _on_solver_done(self, rc):
        self._sb.hold = False       # run over: the status bar is everyone's again
        # Auralization: if more sources queued and this one succeeded, run the next.
        if rc == 0 and self._aural_queue:
            self._run_source(self._aural_queue.pop(0))
            return
        self._aural_queue = []      # done or aborted on failure
        # Re-enable launching, fill/hide the progress bar, report status.
        self._set_running(False)
        if rc == 0:
            # New run wrote fresh pressure*.bin + config.txt (maybe a different source type). Drop the
            # cached animation so the next play RELOADS it — else playback reuses the previous sim (e.g.
            # a PCS/Ricker anim after switching to Sinusoid), only fixed by restarting the GUI.
            self._gl.clear_animation()
            self._float_prog.setValue(100)
            # A completed FULL run makes the current wave inputs the reusable baseline,
            # so the next Simulate with only GA changes will auto-skip the FVM.
            if not getattr(self, "_is_ga_only_run", False):
                self._store_wave_signature()
        c = PAL["ok"] if rc == 0 else PAL["err"]
        self._log.append(
            f'<span style="color:{c};">{"[OK]" if rc==0 else "[FAIL]"}  rc={rc}</span>')
        self._sb.showMessage(f"Solver finished  (rc={rc})")
        # hide the floating bar shortly after completion
        QTimer.singleShot(1500, self._float_prog.hide)


# ─────────────────────────────────────────────────────────────
class PropsWindow(QWidget):
    """Place primitives/assets as 3D scene props, assign a layer, transform them. Own top-level window
    (opened in Draw mode, mirrors SketchWindow). Edits gl._props in place; gl draws + exports them."""
    def __init__(self, gl, parent=None):
        super().__init__(parent)
        import props3d
        from PySide6.QtWidgets import QComboBox
        self.gl = gl; self.p3 = props3d
        self.setWindowTitle("Props - place 3D objects")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(340, 600)
        root = QVBoxLayout(self)

        root.addWidget(QLabel("Add primitive"))
        grid = QGridLayout()
        for k, name in enumerate(props3d.PRIMITIVES):
            b = QPushButton(name); b.clicked.connect(lambda _=False, n=name: self._add("primitive", n))
            grid.addWidget(b, k // 3, k % 3)
        root.addLayout(grid)

        root.addWidget(QLabel("Placed props (Ctrl/Shift-click for multi-select)"))
        self._list = QListWidget(); self._list.currentRowChanged.connect(self._on_select)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)  # multi-select
        root.addWidget(self._list, 1)
        delb = QPushButton("Delete selected"); delb.clicked.connect(self._delete)
        root.addWidget(delb)

        form = QFormLayout()
        self._pos = [self._spin(step=0.1) for _ in range(3)]
        self._rot = [self._spin(step=5.0) for _ in range(3)]
        self._scl = [self._spin(1.0, step=0.1) for _ in range(3)]
        for s in self._pos: s.setSuffix(" m")                       # units next to each field
        for s in self._rot: s.setSuffix(" °")
        for s in self._scl: s.setSuffix(" ×")
        form.addRow("pos x y z", self._triple(self._pos))
        form.addRow("rot x y z", self._triple(self._rot))
        form.addRow("scale x y z", self._triple(self._scl))
        self._layer = QComboBox(); self._layer.setEditable(True)                # type new OR pick existing
        self._layer.setCurrentText(props3d.DEFAULT_LAYER)
        lrow = QHBoxLayout(); lrow.addWidget(self._layer, 1)
        asg = QPushButton("Assign → selected"); asg.clicked.connect(self._assign_selected)
        lrow.addWidget(asg)
        form.addRow("layer", lrow)
        root.addLayout(form)
        for s in self._pos + self._rot + self._scl:
            s.valueChanged.connect(self._apply)
        self._layer.lineEdit().editingFinished.connect(self._apply)
        self._refresh()

    def _spin(self, val=0.0, step=0.1):
        s = QDoubleSpinBox(); s.setRange(-1e4, 1e4); s.setDecimals(3)
        s.setSingleStep(step); s.setValue(val); s.setFixedWidth(78); return s

    def _triple(self, spins):
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
        for s in spins:
            h.addWidget(s)
        return w

    def _add(self, kind, source):
        if kind == "primitive" and source not in self.p3.PRIMITIVES:
            return
        layer = self._layer.currentText().strip() or self.p3.DEFAULT_LAYER
        self.gl.push_props_undo()
        self.gl.add_prop(self.p3.Prop(kind, source, layer=layer))
        self._refresh(); self._list.setCurrentRow(len(self.gl._props) - 1)

    def _refresh(self):
        self._list.blockSignals(True); self._list.clear()
        for p in self.gl._props:
            self._list.addItem(f"{p.source}  [{p.layer}]")
        self._list.blockSignals(False)
        self._list.setCurrentRow(self.gl._prop_sel)
        # existing layers into the combo (keep the current text so a half-typed new name survives)
        self._layer.blockSignals(True)
        cur = self._layer.currentText()
        self._layer.clear(); self._layer.addItems(sorted({p.layer for p in self.gl._props}))
        self._layer.setCurrentText(cur)
        self._layer.blockSignals(False)

    def _on_select(self, row):
        self.gl.select_prop(row)
        if not (0 <= row < len(self.gl._props)):
            return
        p = self.gl._props[row]
        for grp, vals in ((self._pos, p.pos), (self._rot, p.rot), (self._scl, p.scale)):
            for s, v in zip(grp, vals):
                s.blockSignals(True); s.setValue(float(v)); s.blockSignals(False)
        self._layer.blockSignals(True); self._layer.setCurrentText(p.layer); self._layer.blockSignals(False)

    def _apply(self, *_):
        row = self._list.currentRow()
        if not (0 <= row < len(self.gl._props)):
            return
        p = self.gl._props[row]
        p.pos = [s.value() for s in self._pos]
        p.rot = [s.value() for s in self._rot]
        p.scale = [s.value() for s in self._scl]
        p.layer = self._layer.currentText().strip() or self.p3.DEFAULT_LAYER
        self._list.item(row).setText(f"{p.source}  [{p.layer}]")
        self.gl.update()
        self.gl.propsChanged.emit()          # refresh the 2D sketch footprints live (not just the 3D view)

    def _selected_rows(self):
        return sorted(self._list.row(it) for it in self._list.selectedItems())

    def _assign_selected(self):
        """AutoCAD-style: give every selected prop the same layer (type a new name or pick an existing)."""
        layer = self._layer.currentText().strip() or self.p3.DEFAULT_LAYER
        self.gl.push_props_undo()
        for r in self._selected_rows():
            if 0 <= r < len(self.gl._props):
                self.gl._props[r].layer = layer
        self._refresh(); self.gl.update()

    def _delete(self):
        self.gl.push_props_undo()
        for r in sorted(self._selected_rows(), reverse=True):        # reverse so indices stay valid
            if 0 <= r < len(self.gl._props):
                del self.gl._props[r]
        self.gl._prop_sel = min(self.gl._prop_sel, len(self.gl._props) - 1)
        self.gl.update(); self._refresh()


# ─────────────────────────────────────────────────────────────
def main():
    # Crash/freeze diagnostics. A segfault inside the Qt/GL stack normally kills the
    # process with no Python traceback at all; faulthandler prints the C-level and
    # Python stacks to stderr instead. The SIGUSR1 hook covers the FREEZE case, where
    # there is no crash to catch: `kill -USR1 <pid>` dumps where every thread is stuck
    # without attaching a debugger or disturbing the process.
    import faulthandler, signal
    faulthandler.enable()
    if hasattr(signal, "SIGUSR1"):        # POSIX only
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=True)
        print(f"[gui] pid {os.getpid()} — 'kill -USR1 {os.getpid()}' dumps all thread stacks",
              flush=True)

    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24); fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    app.setApplicationName("FASES GUI")
    app.setFont(QFont("Segoe UI",10) if sys.platform=="win32" else
                QFont("SF Pro Text",10) if sys.platform=="darwin" else
                QFont("Ubuntu",10))
    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
