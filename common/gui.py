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
from pathlib import Path
from datetime import datetime
import subprocess

import numpy as np

from PySide6.QtCore    import Qt, QThread, Signal, QPoint, QPointF, QEvent, QSettings, QTimer, QSize
from PySide6.QtGui     import (QFont, QColor, QPalette, QSurfaceFormat,
                              QPainter, QCursor, QKeySequence,
                              QPixmap, QIcon, QPen, QShortcut)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QDockWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox,
    QListWidget, QListWidgetItem, QTabWidget, QTabBar, QTextEdit,
    QFileDialog, QMessageBox, QDoubleSpinBox, QSpinBox, QAbstractSpinBox,
    QGroupBox, QScrollArea, QAbstractItemView, QInputDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QStatusBar, QSlider, QMenu, QProgressBar, QDialog
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
    "subtext":   "#78909c", "highlight":"#263238","btn":     "#2a3140",
    "btn_hover": "#35404f", "layer_sel":"#251a55",
}

LAYER_COLORS = [
    (45/255,   2/255, 249/255),   # blue-violet
    (100/255, 249/255,  2/255),   # green
    (240/255,  25/255,  2/255),   # red
    (249/255, 225/255,  2/255),   # yellow
    (249/255,   2/255, 237/255),  # magenta
    (2/255,   249/255, 230/255),  # cyan
    (249/255, 130/255,   2/255),  # orange
    (160/255,  2/255,  249/255),  # purple
    (2/255,   130/255, 249/255),  # azure
    (200/255, 249/255,   2/255),  # lime
    (249/255,   2/255, 120/255),  # pink
    (249/255, 160/255, 200/255),  # rose
]
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
SOLVER_BINARY = "/home/vitorotri/Documents/Github/phases/3D-solver/build/phases3D"

# Predefined absorption-material catalogue used by the material table RMB menu.
# Expected line format inside the text file:
#   material_name  a63  a125  a250  a500  a1000  a2000  a4000  a8000
# Example:
#   rough_concrete 0.01 0.01 0.02 0.04 0.06 0.08 0.10 0.10
# The GUI shows material_name as "material name" in the selector, but keeps the
# file format underscore-based and human-editable.
MATERIALS_FILE = "/home/vitorotri/Documents/Github/phases/common/materialsREF.txt"

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
def parse_obj(path: str):
    """
    Parse vertices and grouped faces from an OBJ file.

    Group handling: an object may be introduced by an ``o`` line, a ``g`` line,
    or BOTH (some exporters emit ``o chair1`` immediately followed by
    ``g chair1``). We therefore do NOT start a fresh group on every o/g line —
    that would create a duplicate (often empty) group for each o+g pair. Instead
    we only open a new group when the RESOLVED name actually changes, and we
    drop any group that ends up with no faces. Group names that carry several
    tokens (e.g. ``g Mesh1 ceiling Model``) resolve to the second token, which
    is the human-readable layer name used elsewhere in the GUI.
    """
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
            elif tok[0] == "f":
                face = []
                for t in tok[1:]:
                    vi = int(t.split("/")[0])
                    vi = vi-1 if vi > 0 else len(vertices)+vi
                    face.append(vi)
                if len(face) < 3: continue
                v0 = face[0]
                for i in range(1, len(face)-1):
                    cur_tris.append((v0, face[i], face[i+1]))
    flush()
    verts = (np.array(vertices, np.float32) if vertices
             else np.zeros((0,3), np.float32))
    return verts, groups


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
    pointDragStarted = Signal()                        # a grab/drag just began
    coordsChanged = Signal(float, float, float)        # live HUD coords

    # placement modes
    MODE_NONE = 0
    MODE_SRC  = 1
    MODE_REC  = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 500)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # camera
        self._yaw   =  35.0       # normal orbit: around OBJ/world Z
        self._pitch = -22.0       # normal orbit: tilt
        self._free_x = 0.0        # Alt+LMB free tumble around screen X
        self._free_y = 0.0        # Alt+LMB free tumble around screen Y
        self._roll  =  0.0        # Ctrl+LMB roll around view axis
        self._dist  =   6.0
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
        self._hover_world = None   # world coords under mouse for HUD


        # points
        self._sources   = []  # list of [x,y,z]
        self._receivers = []  # list of [x,y,z]
        # Index of the highlighted source/receiver (-1 = none). Driven either by
        # table selection or by G-mode hover, so the user can tell which point
        # in the scene corresponds to which table row.
        self._hl_src = -1
        self._hl_rec = -1
        self._glyph_queue = []   # per-frame source/receiver glyph overlay list
        self._corner_widget = None   # optional bottom-left overlay (theme button)

        # placement state
        self._mode      = self.MODE_NONE
        self._axis_lock = None          # 'X','Y','Z' or None
        self._place_pos = np.zeros(3, np.float32)  # frozen base position
        self._cursor_world = np.zeros(3, np.float32)

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
        self._groups = []
        color_idx = 0   # counts only non-bbox groups so colors are never skipped
        for g in groups:
            is_bbox = "boundary" in g["name"].lower()
            if is_bbox:
                color = BBOX_COLOR
            else:
                color = LAYER_COLORS[color_idx % len(LAYER_COLORS)]
                color_idx += 1
            self._groups.append({
                "name":    g["name"],
                "tris":    g["tris"],
                "normals": face_normals(verts, g["tris"]) if len(g["tris"]) else None,
                "edges":   feature_edges(verts, g["tris"]) if len(g["tris"]) else None,
                "color":   color,
                "visible": True,
                "is_bbox": is_bbox,
            })
        if len(verts):
            mn, mx = verts.min(0), verts.max(0)
            ctr  = (mn + mx) * 0.5
            span = (mx - mn).max()
            self._pivot      = ctr.copy()
            self._scene_span = span if span > 0 else 1.0
            self._domain_min = mn.copy()
            self._domain_max = mx.copy()
            self._pan        = np.zeros(3, np.float32)
            self._dist       = span * 1.8 if span > 0 else 6.0
            self._yaw   = 35.0
            self._pitch = -22.0
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

    def set_sources(self, pts):
        self._sources = [list(p[:3]) for p in pts]; self.update()

    def set_receivers(self, pts):
        self._receivers = [list(p[:3]) for p in pts]; self.update()

    def set_highlight(self, kind, idx):
        """Highlight a source/receiver by index in the 3D scene. kind is
        'source' / 'receiver' / None (None clears both)."""
        self._hl_src = idx if kind == "source"   else -1
        self._hl_rec = idx if kind == "receiver" else -1
        self.update()

    def enter_placement_mode(self, mode):
        self._mode      = mode
        self._axis_lock = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

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

    def set_corner_widget(self, w):
        """Register a child widget to keep pinned at the bottom-left corner,
        just above the coordinate HUD (10 px in, ~30 px up from the bottom)."""
        self._corner_widget = w
        self._anchor_corner_widget()

    def _anchor_corner_widget(self):
        w = self._corner_widget
        if w is not None:
            w.move(10, self.height() - w.height() - 26)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._anchor_corner_widget()

    def _setup_camera(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        asp = self.width() / max(self.height(), 1)
        near = max(self._dist * 0.001, 0.01)
        far  = self._dist * 500.0
        gluPerspective(45.0, asp, near, far)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Headlight: with the modelview at identity (eye space), position the
        # light just behind the eye looking down -Z. It stays glued to the
        # camera regardless of how the scene is rotated below.
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 1.0, 0.0])
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

    def paintGL(self):
        glClearColor(*SCENE_BG)        # re-applied each frame so Dark/Light toggles take effect
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._setup_camera()
        self._glyph_queue = []            # collected by _draw_sources/_receivers
        self._draw_gizmo()   # replaces grid + world axes
        self._draw_mesh()
        self._draw_rays()
        self._draw_sources()
        self._draw_receivers()
        if self._mode != self.MODE_NONE:
            self._draw_placement_cursor()
        elif self._drag_kind is not None and self._drag_idx >= 0:
            # show the same axis guide + coordinate readout while moving a point
            self._draw_drag_guide()
        self._draw_hud()
        # source/receiver glyphs are drawn last, as a QPainter text overlay
        if self._glyph_queue:
            gp = QPainter(self)
            self._paint_point_glyphs(gp)
            gp.end()

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

    def _draw_mesh(self):
        if len(self._verts) == 0:
            return
        verts = self._verts

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

        # Pass A: FILLS first. Unselected layers get a transparent grey plane;
        # the selected layer gets its own vivid hue at SEL_ALPHA. Drawing every
        # fill BEFORE the wireframe is what keeps the highlighted layer BEHIND
        # the white wireframe instead of painting over it.
        for g in self._groups:
            if g["is_bbox"] or not g["visible"]:
                continue
            if self._selected_layer == g["name"]:
                draw_triangles(g, alpha=SEL_ALPHA)            # selected: own colour
            else:
                draw_triangles(g, alpha=0.22, color=GREY)     # others: grey

        # Pass B: feature-edge wireframe for ALL visible layers (the selected
        # one included), drawn last so the edges always sit on top of every
        # fill. Colour follows the scene scheme (white on dark / black on light).
        glColor4f(*WIRE_COLOR)
        glLineWidth(1.3)
        for g in self._groups:
            if g["is_bbox"] or not g["visible"]:
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
        if self._bbox_visible:
            self._draw_domain_box()
            glDisable(GL_BLEND)

        glEnable(GL_LIGHTING)

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
            hl  = (i == self._hl_src)
            if self._outside_domain(pt[0], pt[1], pt[2]):
                col = OUTSIDE_COLOR
            else:
                col = HL_COLOR if hl else SRC_COLOR
            self._point_glyph(pt[0], pt[1], pt[2], col, "source", hl=hl)

    def _draw_receivers(self):
        for i, pt in enumerate(self._receivers):
            hl  = (i == self._hl_rec)
            if self._outside_domain(pt[0], pt[1], pt[2]):
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

        # preview: draw the SAME glyph as a placed point, at the exact centre
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
                    coord_str = f"  x={p[0]:.3f}  y={p[1]:.3f}  z={p[2]:.3f} m"
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
                txt = f"x={p[0]:.3f}  y={p[1]:.3f}  z={p[2]:.3f} m"
                painter.drawText(10, self.height() - 10, txt)
                painter.end()
            return
        p = self._cursor_world
        mode_str = "SOURCE" if self._mode == self.MODE_SRC else "RECEIVER"
        mode_col = SRC_COLOR if self._mode == self.MODE_SRC else REC_COLOR
        axis_str = f"  locked: {self._axis_lock}" if self._axis_lock else "  (hold X / Y / Z + drag LMB to lock an axis)"
        coord_str = f"  x={p[0]:.3f}  y={p[1]:.3f}  z={p[2]:.3f} m"

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

    # ── mouse / keyboard ────────────────────────────────────
    def leaveEvent(self, e):
        self._hover_world = None
        self.update()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        self._last_pos  = e.position()
        self._mouse_btn = e.button()
        self._drag_moved = False
        # If edit mode (E) is armed and LMB is pressed, grab the nearest
        # placed source/receiver instead of rotating the camera.
        if (self._grab_held and e.button() == Qt.MouseButton.LeftButton
                and self._mode == self.MODE_NONE):
            self._begin_point_drag(e.position())

    def mouseMoveEvent(self, e):
        # ── placement mode: mouse moves the cursor with NO button needed ──
        if self._mode != self.MODE_NONE:
            if self._axis_lock:
                self._slide_axis(e.position())     # slide along locked axis
            else:
                self._update_cursor_world(e.position())
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
                if mods & Qt.KeyboardModifier.AltModifier:
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
            # Place on LMB release (whether the user dragged along a locked
            # axis or just clicked). Releasing the button commits the point.
            if e.button() == Qt.MouseButton.LeftButton:
                p = self._cursor_world.copy()
                mode_name = "source" if self._mode == self.MODE_SRC else "receiver"
                self.pointPlaced.emit(mode_name, float(p[0]), float(p[1]), float(p[2]))
            return

        if not moved and e.button() == Qt.MouseButton.LeftButton:
            self._do_ray_pick(e.position())
        if not moved and e.button() == Qt.MouseButton.RightButton:
            self._do_point_remove(e.position())

    def wheelEvent(self, e):
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
                # leaving edit mode clears the hover highlight
                self.set_highlight(None, -1)
                self.pointHovered.emit("", -1)
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

        # ── axis-constrained movement ──
        if self._drag_axis is not None and self._drag_axis_base is not None:
            ax = {'X': np.array([1,0,0], np.float32),
                  'Y': np.array([0,1,0], np.float32),
                  'Z': np.array([0,0,1], np.float32)}[self._drag_axis]
            base = np.array(self._drag_axis_base, np.float32)
            try:
                # screen-space direction of the axis at the base point
                s0 = np.array(gluProject(*base.astype(float),
                              self._mv, self._proj, viewport)[:2])
                s1 = np.array(gluProject(*(base + ax*0.1).astype(float),
                              self._mv, self._proj, viewport)[:2])
                sdir = s1 - s0
                slen = np.linalg.norm(sdir)
                if slen < 1e-6:
                    return
                sdir /= slen
                anchor = self._drag_anchor_px or QCursor.pos()
                cur = QCursor.pos()
                mdx = cur.x() - anchor.x()
                mdy = -(cur.y() - anchor.y())
                move = float(np.dot(np.array([mdx, mdy], np.float32), sdir))
                scale = self._dist * 0.0015
                newp = base + ax * move * scale
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
        self.makeCurrent()
        W, H = self.width(), self.height()
        ro, rd = ray_from_screen(qpos.x(), qpos.y(), W, H,
                                  self._mv, self._proj)
        best_t, best_name = None, None
        for g in self._groups:
            if not g["visible"]: continue
            if g["is_bbox"] and not self._bbox_visible: continue
            tris = g["tris"]
            if len(tris) == 0: continue
            t = ray_group_hit(ro.astype(np.float64),
                              rd.astype(np.float64),
                              self._verts, tris)
            if t is not None and (best_t is None or t < best_t):
                best_t, best_name = t, g["name"]
        if best_name:
            self._selected_layer = best_name
            self.layerPicked.emit(best_name)
        else:
            # Clicking empty space must genuinely deselect; do not keep the
            # previous layer highlighted or selected in the list/material panel.
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
                   Qt.AlignmentFlag.AlignCenter, "Frequency [Hz]")

        if self._vals is None:
            p.setPen(grey)
            p.setFont(QFont("Sans", 11))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "α per octave band [Hz]")
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mat_dir = None       # input/mat directory
        self._fit_dir = None       # input/fit directory
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
        cols = ["Layer"] + [str(b) for b in OCTAVE_BANDS]
        self._tbl = QTableWidget(0, len(cols))
        self._tbl.setHorizontalHeaderLabels(cols)
        for c, b in enumerate(OCTAVE_BANDS, start=1):   # unit on hover
            it = self._tbl.horizontalHeaderItem(c)
            if it is not None:
                it.setToolTip(f"{b} Hz")
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._tbl.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        hh = self._tbl.horizontalHeader()
        # The Scene-tab panel is a fixed 480 px (~458 px usable), so eight
        # "8000 Hz" labels cannot all fit and would force a horizontal scrollbar.
        # We show the band centre frequency as the number only (unit is in the
        # column-group label above and on each header's hover tooltip); every
        # number then renders in full with NO scrollbar. The Layer column is
        # fixed and narrow — long names elide to "…" with the full name on hover
        # — so the frequency columns share all the remaining width evenly.
        self._tbl.setTextElideMode(Qt.TextElideMode.ElideRight)
        hh.setMinimumSectionSize(44)        # fits the widest band number ("8000")
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        for c in range(1, len(cols)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
        hh.resizeSection(0, 90)
        self._tbl.setStyleSheet(f"""
            QTableWidget {{background:{PAL['bg']};color:{PAL['text']};
                gridline-color:{PAL['border']};border:1px solid {PAL['border']};}}
            QHeaderView::section {{background:{PAL['panel']};color:{PAL['subtext']};
                border:1px solid {PAL['border']};font-size:11px;}}
        """)
        self._tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._tbl.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tbl.itemSelectionChanged.connect(self._on_row_selected)
        self._tbl.itemChanged.connect(lambda *_: self._on_edit())
        # RMB on a layer row: optional predefined-material lookup. Existing
        # manual typing, paste, plotting, save and fit behaviour stays unchanged.
        self._tbl.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tbl.customContextMenuRequested.connect(self._show_material_menu)
        from PySide6.QtWidgets import QSizePolicy
        self._tbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Expanding)
        self._tbl.setMinimumHeight(140)   # table fills remaining space
        lay.addWidget(self._tbl, 1)

        # ── buttons ──
        row = QHBoxLayout()
        self._save_btn = QPushButton("🖫\uFE0E  Save all .mat")
        self._save_btn.clicked.connect(self.save_all)
        row.addWidget(self._save_btn)
        # The fitting buttons moved to the Run tab ("Fit impedances"); the button
        # objects are kept (hidden, not added to any layout) because run_fit_all /
        # run_fit_selected / _on_fit_all_done enable/disable them as busy flags.
        self._fit_btn = QPushButton("⚙  Run vectFIT");  self._fit_btn.setVisible(False)
        self._fit_btn.clicked.connect(self.run_fit_selected)
        self._fit_all_btn = QPushButton("⚙  Fit all");  self._fit_all_btn.setVisible(False)
        self._fit_all_btn.clicked.connect(self.run_fit_all)
        lay.addLayout(row)

        # ── hard/soft toggle for the SELECTED layer (kept off the table so the
        #    column layout stays untouched). 9th value of the .mat: 0/1. ──
        hs = QHBoxLayout()
        hs_lbl = QLabel("Selected material:")
        hs_lbl.setStyleSheet(f"color:{PAL['subtext']};font-size:11px;")
        hs.addWidget(hs_lbl)
        self._soft_btn = ToggleSwitch()          # small on/off slider switch
        self._soft_btn.toggled.connect(self._on_soft_toggled)
        hs.addWidget(self._soft_btn)
        self._soft_lbl = QLabel("Hard (R > 0)")
        self._soft_lbl.setStyleSheet(f"color:{PAL['text']};font-size:11px;")
        hs.addWidget(self._soft_lbl)
        hs.addStretch(1)
        lay.addLayout(hs)

        self._status = QLabel("")
        self._status.setStyleSheet(f"color:{PAL['subtext']};font-size:10px;")
        lay.addWidget(self._status)

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

        chosen = menu.exec(self._tbl.viewport().mapToGlobal(pos))
        if chosen == choose:
            self._select_material_from_ref(row)

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
            self._status.setText("Missing predefined material catalogue.")
            return

        try:
            materials = self._read_materials_ref()
        except Exception:
            QMessageBox.critical(
                self, "Could not read material list",
                "Could not read the predefined material catalogue. Check that it exists and is readable.")
            self._status.setText("Could not read predefined material catalogue.")
            return

        if not materials:
            QMessageBox.information(
                self, "No predefined materials",
                "No valid materials were found. Each non-comment line must be:\n\n"
                "material_name  a63  a125  a250  a500  a1000  a2000  a4000  a8000")
            self._status.setText("No valid predefined materials found.")
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
            self._status.setText("No predefined material selected.")
            return

        file_name, display_name, vals = chosen.data(Qt.ItemDataRole.UserRole)
        self._apply_material_values(row, vals)
        self._status.setText(
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

    def populate(self, layer_names):
        """Fill table rows from layer names, loading existing .mat values."""
        self._tbl.blockSignals(True)
        self._tbl.setRowCount(0)
        for name in layer_names:
            if "boundary" in name.lower():
                continue
            vals, soft = self._load_mat_values(name)
            r = self._tbl.rowCount(); self._tbl.insertRow(r)
            name_item = QTableWidgetItem(name)
            name_item.setToolTip(name)   # full name on hover (cell may be elided)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._tbl.setItem(r, 0, name_item)
            for c, v in enumerate(vals):
                self._tbl.setItem(r, c+1, QTableWidgetItem(f"{v:.2f}"))
            self._set_row_soft(r, soft)   # after the cells exist
        self._tbl.blockSignals(False)
        self._clear_plot()

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
        name = self._replot_current()
        r = self._current_row()
        if r >= 0:
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
            self._status.setText("Select a layer row first."); return
        self._set_row_soft(r, checked)
        self._soft_lbl.setText("Soft (R < 0)" if checked else "Hard (R > 0)")
        name = self._tbl.item(r, 0).text()
        self._status.setText(f"{name}: {'soft' if checked else 'hard'} "
                             "(remember to Save all .mat)")

    def _on_edit(self):
        # a value changed: only replot the curve, no reselection signal
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
        if e.matches(QKeySequence.StandardKey.Paste):
            self._paste_clipboard()
            e.accept(); return
        super().keyPressEvent(e)

    def _paste_clipboard(self):
        text = QApplication.clipboard().text()
        if not text.strip():
            return
        start_row = max(self._tbl.currentRow(), 0)
        start_col = max(self._tbl.currentColumn(), 1)  # never overwrite name col by default
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
        self._on_row_selected()
        self._status.setText("Pasted values from clipboard.")

    # ── save / fit ──
    def save_selected(self):
        r = self._tbl.currentRow()
        if r < 0:
            self._status.setText("Select a layer row first."); return
        if not self._mat_dir:
            self._status.setText("Open a project folder first."); return
        name = self._tbl.item(r, 0).text()
        vals = self._row_values(r)
        os.makedirs(self._mat_dir, exist_ok=True)
        p = os.path.join(self._mat_dir, name + ".mat")
        with open(p, "w") as f:
            f.write(f"# alpha per octave band: {OCTAVE_BANDS}\n")
            f.write("  ".join(f"{v:.4f}" for v in vals) + "\n")
        self._write_all_soft()      # hard/soft registry lives in ALL.mat
        self._status.setText(f"Saved → {name}.mat")

    def save_all(self):
        """Write a .mat file for EVERY layer row in one pass, independent of the
        current selection (does not move the selected row or change the plot)."""
        if not self._mat_dir:
            self._status.setText("Open a project folder first."); return
        if self._tbl.rowCount() == 0:
            self._status.setText("No materials to save."); return
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
            n += 1
        self._write_all_soft()      # one central hard/soft registry
        self._status.setText(f"Saved {n} material file(s) → {self._mat_dir}")

    def run_fit_selected(self):
        r = self._tbl.currentRow()
        if r < 0:
            self._status.setText("Select a layer row first."); return
        if not self._proj_root:
            self._status.setText("Open a project folder first."); return
        self.save_selected()
        name = self._tbl.item(r, 0).text()
        # The fitting scripts live NEXT TO gui.py (both are in common/), so we
        # locate them relative to THIS file — never relative to the project,
        # which can live anywhere on disk.
        script = self._find_fit_script()
        if script is None:
            self._status.setText("vectFIT.py / fminconFIT.py not found next to gui.py"); return
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
        self._status.setText(f"Running {os.path.basename(script)} on {name}…")
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
                self._status.setText(f"Fit done → {name}.fit")
            else:
                msg = out.strip().splitlines()
                tail = msg[-1] if msg else "unknown error"
                self._status.setText(f"Fit failed (rc={proc.returncode}): {tail[:80]}")
        except subprocess.TimeoutExpired:
            self.fitLog.emit("[!] Fit timed out (600 s).")
            self._status.setText("Fit timed out (600 s).")
        except Exception as ex:
            self.fitLog.emit(f"[!] Fit error: {ex}")
            self._status.setText(f"Fit error: {ex}")

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
            self._status.setText("Open a project folder first."); return
        if self._tbl.rowCount() == 0:
            self._status.setText("No materials to fit."); return
        script = self._find_fit_script()
        if script is None:
            self._status.setText("vectFIT.py / fminconFIT.py not found next to gui.py"); return
        # Compatibility probe: multi-material fitting needs the updated script.
        # gui.py and vectFIT.py ship TOGETHER — if the copy next to gui.py is an
        # older one, say so explicitly instead of failing with an argparse error.
        try:
            if "--fitdir" not in open(script, encoding="utf-8", errors="ignore").read():
                self._status.setText(
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
        self._status.setText(f"Fitting {len(mats)} material(s)… close the plot window to finish.")
        self.fitBegan.emit("$ " + " ".join(cmd))
        self._fit_thread = _FitAllThread(cmd, proj)
        self._fit_thread.line.connect(self.fitLog.emit)
        self._fit_thread.done.connect(self._on_fit_all_done)
        self._fit_thread.start()

    def _on_fit_all_done(self, rc, tail):
        self._fit_all_btn.setEnabled(True)
        self._fit_btn.setEnabled(True)
        if rc == 0:
            self._status.setText(f"Fit all done → {self._fit_dir}")
        else:
            self._status.setText(f"Fit all failed (rc={rc}): {tail[:160]}")


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


class PointTable(QWidget):
    changed = Signal()

    def __init__(self, has_amplitude=False, parent=None):
        super().__init__(parent)
        self._has_amp = has_amplitude
        # Column 0 is an editable Name (defaults to the row number). The numeric
        # columns follow; get_points()/save_file deliberately skip the name so
        # the solver file format (x y z [p0]) is unchanged.
        cols = ["Name", "x (m)", "y (m)", "z (m)"] + (["p₀ (Pa)"] if has_amplitude else [])
        self._coff = 1            # numeric coordinates start at this column
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self._tbl = QTableWidget(0, len(cols))
        self._tbl.setHorizontalHeaderLabels(cols)
        hh = self._tbl.horizontalHeader()
        # Name column a bit wider (interactive), coordinate columns share the rest.
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        for c in range(1, len(cols)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
        self._tbl.setColumnWidth(0, 90)
        self._tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # Cells are edited ONLY by a deliberate double-click (or clicking an
        # already-selected cell). Keyboard triggers are excluded so stray
        # keystrokes — e.g. while moving a point in edit mode — can never
        # overwrite coordinates by accident.
        self._tbl.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked |
            QAbstractItemView.EditTrigger.SelectedClicked)
        self._tbl.setStyleSheet(f"""
            QTableWidget {{background:{PAL['bg']};color:{PAL['text']};
                gridline-color:{PAL['border']};border:1px solid {PAL['border']};}}
            QHeaderView::section {{background:{PAL['panel']};color:{PAL['subtext']};
                border:1px solid {PAL['border']};}}
        """)
        self._tbl.itemChanged.connect(lambda: self.changed.emit())
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
            btn_row.addWidget(b)
        lay.addLayout(btn_row)

    def _add(self, x=1.0, y=1.0, z=1.0, amp=1.0, name=None):
        self._tbl.blockSignals(True)
        r = self._tbl.rowCount(); self._tbl.insertRow(r)
        # column 0: editable name, default to the 1-based row number
        self._tbl.setItem(r, 0, QTableWidgetItem(name if name else str(r + 1)))
        vals = [x, y, z] + ([amp] if self._has_amp else [])
        for c, v in enumerate(vals):
            self._tbl.setItem(r, self._coff + c, QTableWidgetItem(f"{v:.4g}"))
        self._tbl.blockSignals(False)
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
        for c, v in enumerate((x, y, z)):
            self._tbl.setItem(idx, self._coff + c, QTableWidgetItem(f"{v:.4g}"))
        self._tbl.blockSignals(False)
        self.changed.emit()

    def _remove(self):
        rows = sorted({i.row() for i in self._tbl.selectedItems()}, reverse=True)
        for r in rows: self._tbl.removeRow(r)
        self.changed.emit()

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
                pts.append(tuple(row))
            except (TypeError, ValueError): pass
        return pts

    def set_points(self, pts, emit_changed=True):
        """Replace all rows from numeric tuples (x, y, z[, p0]). Names reset to
        the row number. Used by Ctrl+Z point undo."""
        ncoord = self._tbl.columnCount() - self._coff   # numeric columns
        self._tbl.blockSignals(True)
        self._tbl.setRowCount(0)
        for pt in pts:
            vals = list(pt[:ncoord])
            if len(vals) < ncoord:
                vals += [1.0 if self._has_amp else 0.0] * (ncoord - len(vals))
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
                    self._tbl.setItem(r, self._coff + c, QTableWidgetItem(v))

    def save_file(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("# " + ("x y z p0" if self._has_amp else "x y z") + "\n")
            for pt in self.get_points():
                f.write("  ".join(f"{v:.6g}" for v in pt) + "\n")


# ─────────────────────────────────────────────────────────────
#  Solver runner thread
# ─────────────────────────────────────────────────────────────
class SolverThread(QThread):
    output   = Signal(str)
    progress = Signal(int)          # parsed NN% from the solver's progress bar
    done     = Signal(int)          # NOT 'finished' (that shadows QThread's own)

    def __init__(self, cmd, cwd):
        super().__init__(); self._cmd = cmd; self._cwd = cwd
        self._proc = None

    def run(self):
        import re
        pct_re = re.compile(r"(\d+)\s*%")
        try:
            self._proc = subprocess.Popen(self._cmd, cwd=self._cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1)
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
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


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
        self.setWindowTitle("PHASES - Parallel Hybrid Acoustic Solver for Enhanced Soundfields")
        self.resize(1440, 860)
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
        return super().eventFilter(obj, e)

    def _apply_theme(self):
        pal = QPalette()
        for role, key in [
            (QPalette.ColorRole.Window,          "bg"),
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
            QTabWidget::pane{{border:1px solid {PAL['border']};background:{PAL['panel']};}}
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
        self.setCentralWidget(central)
        root_lay = QHBoxLayout(central)
        root_lay.setContentsMargins(6,6,6,6); root_lay.setSpacing(6)

        # left — GL + status
        gl_wrap = QWidget()
        gl_vlay = QVBoxLayout(gl_wrap)
        gl_vlay.setContentsMargins(0,0,0,0); gl_vlay.setSpacing(3)

        self._gl = GLViewer()
        self._gl.layerPicked.connect(self._on_layer_picked)
        self._gl.pointPlaced.connect(self._on_point_placed)
        self._gl.pointMoved.connect(self._on_point_moved)
        self._gl.pointDragStarted.connect(self._remember_points_for_undo)
        self._gl.pointHovered.connect(self._on_point_hovered)
        self._gl.pointRemoved.connect(self._on_point_removed)
        gl_vlay.addWidget(self._gl, stretch=1)

        # placement toolbar under the viewer
        place_bar = QHBoxLayout()
        place_bar.setSpacing(4)
        place_bar.setContentsMargins(2, 1, 2, 1)

        self._src_btn = QPushButton("Src [S]")
        self._src_btn.setCheckable(True)
        self._src_btn.setFixedSize(90, 22)
        self._src_btn.clicked.connect(lambda: self._toggle_placement("source"))
        place_bar.addWidget(self._src_btn)

        self._rec_btn = QPushButton("Rec [R]")
        self._rec_btn.setCheckable(True)
        self._rec_btn.setFixedSize(90, 22)
        self._rec_btn.clicked.connect(lambda: self._toggle_placement("receiver"))
        place_bar.addWidget(self._rec_btn)

        self._hint_lbl = QLabel(
                      "Rotate: LMB   Ctrl+LMB roll   Pan: MMB   Scroll: zoom   "
                      "T: reset view   S: src   R: rec   hold X/Y/Z: lock axis (move mouse)   LMB: place   "
                      "E then LMB-drag: edit/move point (hold X/Y/Z to lock)   RMB: del   Ctrl+Z: undo   Esc: cancel")
        place_bar.addWidget(self._hint_lbl)
        place_bar.addStretch()
        bar_wrap = QWidget()
        bar_wrap.setFixedHeight(24)
        bar_wrap.setLayout(place_bar)
        gl_vlay.addWidget(bar_wrap)

        # tabs panel on the LEFT, GL viewport on the RIGHT
        self._tabs = QTabWidget()
        # Custom tab bar that divides the full width across the tabs so the last
        # tab ends flush with the panel edge (set BEFORE adding any tabs).
        self._tabs.setTabBar(FillTabBar())
        self._tabs.setUsesScrollButtons(False)        # no arrows
        self._tabs.setElideMode(Qt.TextElideMode.ElideNone)
        # Fixed-width panel: not user-resizable (no splitter handle to drag) and
        # its width never changes on maximize, so the tabs always fill it exactly.
        self._tabs.setFixedWidth(480)
        # Layout order: panel (fixed) on the left, GL viewport (stretches) right.
        root_lay.addWidget(self._tabs)                 # left, fixed width
        root_lay.addWidget(gl_wrap, stretch=1)         # right, takes the rest

		# append "\uFE0E" in front of symbol to stay monochrome
        self._tabs.addTab(self._build_scene_tab(),   "🗂\uFE0E  Scene")
        self._tabs.addTab(self._build_src_tab(),     "🔊\uFE0E  Sources")
        self._tabs.addTab(self._build_rec_tab(),     "🎙\uFE0E  Receivers")
        self._tabs.addTab(self._build_config_tab(),  "⚙\uFE0E  Setup")
        self._tabs.addTab(self._build_run_tab(),     "▶\uFE0E  Run")

        # Floating, draggable progress bar overlaying the 3D scene. Parented to
        # the GL container so it sits on top of the viewport; starts hidden and
        # is positioned near the top-left, the user can drag it anywhere.
        self._float_prog = DraggableProgress(gl_wrap)
        self._float_prog.move(16, 16)

        # Dark/Light scene toggle, overlaid on the 3D viewport just ABOVE the
        # bottom-left coordinate readout. Parented to the GL widget so it sits
        # over the scene; the GL viewer re-anchors it on resize.
        self._theme_btn = QPushButton("Dark", self._gl)
        self._theme_btn.setFixedSize(60, 22)
        self._theme_btn.clicked.connect(self._toggle_scene_theme)
        self._gl.set_corner_widget(self._theme_btn)   # bottom-left anchoring
        self._theme_btn.show()
        self._restyle_scene_theme()        # initial button + hint styling (both now exist)

        # status bar
        self._sb = QStatusBar()
        self.setStatusBar(self._sb)
        self._sb.showMessage("Ready.  Open a project folder to begin.")

        self._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self._undo_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._undo_shortcut.activated.connect(self._undo_point_change)

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
    def _toggle_scene_theme(self):
        """Flip the viewport between Dark and Light and repaint."""
        set_scene_mode(not SCENE_LIGHT_MODE)
        self._restyle_scene_theme()
        self._gl.update()

    def _restyle_scene_theme(self):
        """Style the Dark/Light button to mirror the scene (background = scene
        bg, border = wireframe colour, text contrasts) and recolour the
        placement-hint text so it stays readable in both modes."""
        def _hex(rgb):
            return "#%02x%02x%02x" % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        bg_hex   = _hex(SCENE_BG)
        wire_hex = _hex(WIRE_COLOR)
        # text contrasts with the background: dark bg -> light text, light bg -> dark
        txt_hex  = wire_hex                     # wireframe colour already contrasts
        self._theme_btn.setText("Light" if SCENE_LIGHT_MODE else "Dark")
        self._theme_btn.setStyleSheet(
            f"QPushButton{{background:{bg_hex};color:{txt_hex};"
            f"border:1px solid {wire_hex};border-radius:3px;font-size:10px;}}")
        # hint text: grey on dark, near-black on light
        hint_col = "#202020" if SCENE_LIGHT_MODE else PAL["subtext"]
        self._hint_lbl.setStyleSheet(f"color:{hint_col};font-size:9px;")

    def _toggle_placement(self, kind):
        if kind == "source":
            if self._src_btn.isChecked():
                self._rec_btn.setChecked(False)
                self._gl.enter_placement_mode(GLViewer.MODE_SRC)
                self._tabs.setCurrentIndex(1)   # Sources tab
                self._sb.showMessage("SOURCE placement: drag LMB to position (hold X/Y/Z to lock an axis); release to place. Esc to cancel")
            else:
                self._gl.exit_placement_mode()
                self._sb.showMessage("Ready.")
        else:
            if self._rec_btn.isChecked():
                self._src_btn.setChecked(False)
                self._gl.enter_placement_mode(GLViewer.MODE_REC)
                self._tabs.setCurrentIndex(2)   # Receivers tab
                self._sb.showMessage("RECEIVER placement: drag LMB to position (hold X/Y/Z to lock an axis); release to place. Esc to cancel")
            else:
                self._gl.exit_placement_mode()
                self._sb.showMessage("Ready.")

    def _on_point_placed(self, mode, x, y, z):
        self._remember_points_for_undo()
        if mode == "source":
            self._src_tbl.add_point(x, y, z, 1.0)
        else:
            self._rec_tbl.add_point(x, y, z)
        self._refresh_gl_points()
        self._sb.showMessage(f"Placed {mode} at ({x:.3f}, {y:.3f}, {z:.3f}) m  —  drag again to add another, Esc to finish")

    def _on_table_row_selected(self, kind):
        """A row was selected in a point table -> highlight it in the 3D scene.
        Guarded so the reverse (G-hover -> select row) does not loop."""
        if getattr(self, "_pt_sync_guard", False):
            return
        tbl = self._src_tbl if kind == "source" else self._rec_tbl
        sel = tbl._tbl.selectionModel().selectedRows()
        idx = sel[0].row() if sel else -1
        self._gl.set_highlight(kind if idx >= 0 else None, idx)

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
        if key == Qt.Key.Key_T and self._gl._mode == GLViewer.MODE_NONE:
            # reset the 3D view to the exact initial visualization
            self._gl._reset_view()
            self._sb.showMessage("View reset.")
            e.accept(); return
        if key == Qt.Key.Key_Escape:
            was_placing = self._src_btn.isChecked() or self._rec_btn.isChecked()
            self._src_btn.setChecked(False)
            self._rec_btn.setChecked(False)
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
        self._bbox_chk.setChecked(False)
        self._bbox_chk.toggled.connect(self._gl.set_bbox_visible)
        bbox_row.addWidget(self._bbox_chk); bbox_row.addStretch()
        lay.addLayout(bbox_row)

        lay.addWidget(QLabel("Layers  — click in viewport or list to select:"))
        self._layer_list = QListWidget()
        self._layer_list.setMaximumHeight(100)
        self._layer_list.currentRowChanged.connect(self._on_layer_row_changed)
        lay.addWidget(self._layer_list)

        vis_row = QHBoxLayout()
        toggle_btn = QPushButton("Toggle")
        toggle_btn.clicked.connect(self._toggle_selected_vis)
        vis_row.addWidget(toggle_btn)
        for lbl, fn in [("Show all", lambda: self._set_all_vis(True)),
                        ("Hide all", lambda: self._set_all_vis(False))]:
            b = QPushButton(lbl); b.clicked.connect(fn); vis_row.addWidget(b)
        vis_row.addStretch(); lay.addLayout(vis_row)

        lay.addWidget(QLabel("Material — α per octave band [Hz] (click a row to plot; Ctrl+V to paste):"))
        self._mat_panel = MaterialPanel()
        self._mat_panel.rowSelected.connect(self._on_material_row_selected)
        self._mat_panel.fitBegan.connect(self._begin_fit_log)
        self._mat_panel.fitLog.connect(self._append_fit_log)
        lay.addWidget(self._mat_panel, 1)
        return w

    # ── Config tab ───────────────────────────────────────────
    def _build_config_tab(self):
        inner = QWidget(); lay = QFormLayout(inner)
        lay.setSpacing(8); lay.setContentsMargins(12,12,12,12)

        def ds(val, mn, mx, dec=4, sfx=""):
            s = QDoubleSpinBox(); s.setDecimals(dec); s.setRange(mn,mx)
            s.setValue(val); s.setMinimumWidth(130)
            if sfx: s.setSuffix(f"  {sfx}")
            return s

        # Estimate pass: runs the solver with --estimate (parse + V + T60, no FDTD)
        # and fills the recommended Cutoff frequency / Number of rays below. Do this
        # first, after setting grid spacing + temperature.
        self._btn_estimate = QPushButton("≈\uFE0E  Estimate values")
        self._btn_estimate.clicked.connect(self._estimate_room)
        lay.addRow(self._btn_estimate)


        self._cfg = {}
        # Temperature + Altitude first: they set the atmosphere (speed of sound),
        # which the estimate derives h / f_cutoff / rays from. All unit-bearing
        # fields stay bundled; the unitless CFL sits last. Display order only --
        # config.txt is still written/read by fixed key/index below.
        for key, lbl, val, mn, mx, dec, sfx in [
            ("T_C",     "Temperature",      20., -30.,   50., 1, "°C"),
            # altitude is written as line 14 of config.txt (v[13])
            ("altitude","Altitude",         408., -500., 9000., 1, "m"),
            ("h",       "Grid spacing",     0.05, 0.001, 2.0, 3, "m"),
            ("t_max",   "Simulation time",  0.5,  0.001, 60.0, 3, "s"),
            ("f_cutoff","Cutoff frequency", 250., 50., 10000., 1, "Hz"),
            ("CFL",     "CFL number",       0.40,  0.05,  0.50, 2, ""),
        ]:
            sp = ds(val, mn, mx, dec, sfx); self._cfg[key] = sp
            lay.addRow(f"{lbl}:", sp)

        # Number of poles for the ADE (line 6 of config.txt; vectFIT reads v[5])
        self._npoles = QSpinBox(); self._npoles.setRange(1, 64)
        self._npoles.setValue(8); self._npoles.setMinimumWidth(130)
        lay.addRow("ADE poles:", self._npoles)

        # Max reflection order for the ISM (line 13 of config.txt; main3D reads v[12])
        self._ism_order = QSpinBox(); self._ism_order.setRange(0, 8)
        self._ism_order.setValue(3); self._ism_order.setMinimumWidth(130)
        lay.addRow("ISM max order:", self._ism_order)

        # Number of rays for the ray-traced late tail (line 15 of config.txt; main3D
        # reads v[14]). Seeded from Vol.bin after a solve (Vorlander <k>); user may
        # override. Wide range for large rooms.
        self._nrays = QSpinBox(); self._nrays.setRange(1000, 50_000_000)
        self._nrays.setValue(20000); self._nrays.setMinimumWidth(130)
        lay.addRow("Number of rays:", self._nrays)

        lay.addRow(QLabel("PML faces  (check = active):"))
        grid = QGridLayout(); self._pml = {}
        for idx, (k, lbl) in enumerate([("U_PML","Up"),("D_PML","Down"),
                                         ("R_PML","Right"),("L_PML","Left"),
                                         ("F_PML","Front"),("B_PML","Back")]):
            chk = QCheckBox(lbl); chk.setChecked(True)
            self._pml[k] = chk; grid.addWidget(chk, idx//3, idx%3)
        lay.addRow(grid)

        for lbl, fn in [("🖫\uFE0E  Save config.txt", self._save_config),
                        ("🗁\uFE0E  Load config.txt", self._load_config)]:
            b = QPushButton(lbl); b.clicked.connect(fn); lay.addRow(b)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        out = QWidget(); QVBoxLayout(out).addWidget(scroll)
        return out

    # ── Sources tab ──────────────────────────────────────────
    def _build_src_tab(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(8,8,8,8)
        info = QLabel("Press S (or the toolbar button) to enter placement mode.\n"
                      "Then drag LMB to position; hold X/Y/Z to lock an axis; release to place.\n"
                      "Or add rows manually below.")
        info.setStyleSheet(f"color:{PAL['subtext']};font-size:11px;")
        info.setWordWrap(True); lay.addWidget(info)
        self._src_tbl = PointTable(has_amplitude=True)
        self._src_tbl.changed.connect(self._refresh_gl_points)
        self._src_tbl._tbl.itemSelectionChanged.connect(
            lambda: self._on_table_row_selected("source"))
        lay.addWidget(self._src_tbl)
        br = QHBoxLayout()
        for lbl, fn in [("🗁\uFE0E  Load src.txt", self._load_src),
                        ("🖫\uFE0E  Save src.txt", self._save_src)]:
            b = QPushButton(lbl); b.clicked.connect(fn); br.addWidget(b)
        lay.addLayout(br)
        return w

    def _build_rec_tab(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(8,8,8,8)
        info = QLabel("Press E (or the toolbar button) to enter placement mode.\n"
                      "Drag LMB to position; hold X/Y/Z to lock an axis; release to place. Esc to finish.")
        info.setStyleSheet(f"color:{PAL['subtext']};font-size:11px;")
        info.setWordWrap(True); lay.addWidget(info)
        self._rec_tbl = PointTable(has_amplitude=False)
        self._rec_tbl.changed.connect(self._refresh_gl_points)
        self._rec_tbl._tbl.itemSelectionChanged.connect(
            lambda: self._on_table_row_selected("receiver"))
        lay.addWidget(self._rec_tbl)
        br = QHBoxLayout()
        for lbl, fn in [("🗁\uFE0E  Load rec.txt", self._load_rec),
                        ("🖫\uFE0E  Save rec.txt", self._save_rec)]:
            b = QPushButton(lbl); b.clicked.connect(fn); br.addWidget(b)
        lay.addLayout(br)
        return w

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

        # ── ray-trace debug overlay: count + order + Trace/Clear ──
        dbg = QHBoxLayout(); dbg.setSpacing(8)
        dbg.addWidget(QLabel("Debug rays:"))
        self._dbg_rays = QSpinBox(); self._dbg_rays.setRange(1, 5000)
        self._dbg_rays.setValue(64); self._dbg_rays.setMinimumWidth(70)
        dbg.addWidget(self._dbg_rays)
        dbg.addWidget(QLabel("Order:"))
        self._dbg_order = QSpinBox(); self._dbg_order.setRange(1, 20)
        self._dbg_order.setValue(5); self._dbg_order.setMinimumWidth(55)
        dbg.addWidget(self._dbg_order)
        self._btn_rays = QPushButton("Trace rays"); self._btn_rays.clicked.connect(self._dump_rays)
        self._btn_clear_rays = QPushButton("Clear rays")
        self._btn_clear_rays.clicked.connect(self._clear_rays)
        dbg.addWidget(self._btn_rays, 1); dbg.addWidget(self._btn_clear_rays, 1)
        lay.addLayout(dbg)
        lay.addSpacing(12)   # breathing room between debug-ray row and Fit/Simulate

        # ── top row: Fit impedances (left)  |  Run/Stop (right) ──
        top = QHBoxLayout(); top.setSpacing(8)
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
        lay.addLayout(top)

        # (progress bar floats over the 3D scene — see DraggableProgress)

        # ── output log ── keep it in the lower ~half so the wall of solver messages
        # (useful for debugging, but noisy) doesn't dominate the tab.
        lay.addStretch(1)
        lay.addWidget(QLabel("Output:"))
        self._log = QTextEdit(); self._log.setReadOnly(True)
        self._log.setFont(QFont("Monospace", 10))
        self._log.setStyleSheet(
            f"background:#000000;color:{PAL['text']};"
            f"border:1px solid {PAL['border']};")
        lay.addWidget(self._log, 1)

        self._solver_thread = None
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
                            if "boundary" not in g["name"].lower())
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
        self._update_cmd_preview()

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
        self._proj_label.setText(d)
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
        if scene and os.path.isfile(scene):
            self._load_obj_path(scene)          # repopulates layers + materials
        else:
            self._mat_panel.set_project(d)      # wire materials for a later OBJ

        # Persist the manifest now so opening a project always records current
        # state (scene path included) even if no .mat/config was touched.
        self._write_manifest()
        self._update_cmd_preview()
        self._sb.showMessage(f"Opened project: {d}")

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
        self._write_manifest()
        self._sb.showMessage(f"Saved project → {self._proj_root}")

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
            verts, groups = parse_obj(path)
        except Exception as ex:
            QMessageBox.critical(self, "OBJ error", str(ex)); return
        self._obj_path = path
        self._obj_label.setText(os.path.basename(path))
        self._groups = groups
        names = [g["name"] for g in groups]
        print(f"[OBJ] Parsed {len(groups)} groups: {names}")
        self._gl.load_mesh(verts, groups)
        self._populate_layers(groups)
        self._update_cmd_preview()
        if self._proj_root:
            self._write_manifest()          # persist the scene path on open/load
        status = f"Loaded {len(groups)} layer(s)  —  {len(verts)} vertices"
        self._sb.showMessage(status)

    def _populate_layers(self, groups):
        self._layer_list.clear()
        for g in groups:
            name = g["name"]
            item = QListWidgetItem(name)
            if "boundary" in name.lower():
                item.setForeground(QColor(PAL["subtext"]))
            self._layer_list.addItem(item)
        self._mat_panel.set_project(self._proj_root)
        self._mat_panel.populate([g["name"] for g in groups])

    def _show_no_layer_selected(self):
        # clear the material table selection -> plot resets to placeholder
        self._mat_panel._tbl.clearSelection()
        self._mat_panel._tbl.setCurrentCell(-1, -1)

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

    def _on_layer_row_changed(self, row):
        if row < 0 or row >= len(self._groups):
            self._gl.clear_selection()
            self._show_no_layer_selected()
            return
        name = self._groups[row]["name"]
        self._gl.select_layer(name)
        if "boundary" in name.lower():
            self._mat_panel._tbl.clearSelection()
            self._mat_panel._tbl.setCurrentCell(-1, -1)
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

    def _write_config(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pml = {k: (1.0 if c.isChecked() else 0.0) for k,c in self._pml.items()}
        with open(path, "w") as f:
            f.write(f"{self._cfg['h'].value():.6g}       # h  [m]\n")
            f.write(f"{self._cfg['t_max'].value():.6g}   # t_max [s]\n")
            f.write(f"{self._cfg['f_cutoff'].value():.4g}  # f_cutoff [Hz]\n")
            f.write(f"{self._cfg['T_C'].value():.4g}      # temperature [°C]\n")
            f.write(f"{self._cfg['CFL'].value():.6g}      # CFL\n")
            f.write(f"{self._npoles.value()}                               # number of poles for the ADE\n")
            for k in ["U_PML","D_PML","R_PML","L_PML","F_PML","B_PML"]:
                f.write(f"{pml[k]:.1f}  # {k}\n")
            f.write(f"{self._ism_order.value()}                               # ISM max reflection order\n")
            f.write(f"{self._cfg['altitude'].value():.6g}      # altitude [m]\n")
            f.write(f"{self._nrays.value()}                               # number of rays (ray-traced tail)\n")

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
        if len(vals) > 12:
            self._ism_order.setValue(int(vals[12]))  # ISM max reflection order
        if len(vals) > 13:
            self._cfg["altitude"].setValue(vals[13])  # altitude [m]
        if len(vals) > 14:
            self._nrays.setValue(int(vals[14]))  # number of rays (ray-traced tail)

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
        self._gl.set_sources(  [(p[0],p[1],p[2]) for p in self._src_tbl.get_points()])
        self._gl.set_receivers([(p[0],p[1],p[2]) for p in self._rec_tbl.get_points()])

    def _browse_binary(self):
        p, _ = QFileDialog.getOpenFileName(self, "Locate solver binary")
        if p:
            QSettings("phases", "gui").setValue("last_binary", p)  # remember
            if self._proj_root and p.startswith(self._proj_root):
                p = os.path.relpath(p, self._proj_root)
            self._bin_le.setText(p)

    def _update_cmd_preview(self):
        if not hasattr(self, '_cmd_preview'): return
        if not self._proj_root:
            self._cmd_preview.setText("(open a project folder first)"); return
        binary = self._bin_le.text() if hasattr(self, "_bin_le") else SOLVER_BINARY
        # absolute binary paths are used as-is; only bare relative ones get ./
        bin_disp = binary if os.path.isabs(binary) else f"./{binary}"
        scene = self._obj_path or "input/scene/scene.obj"
        if self._proj_root and scene.startswith(self._proj_root):
            scene = os.path.relpath(scene, self._proj_root)
        self._cmd_preview.setText(
            f"{bin_disp}  {scene}  config/config.txt  input/src/src.txt  input/rec/rec.txt  config/poles.txt")

    def _run_solver(self):
        if not self._proj_root:
            QMessageBox.warning(self,"No project","Open project first."); return
        if not self._obj_path:
            QMessageBox.warning(self,"No scene","Load an OBJ first."); return
        self._write_config(os.path.join(self._proj_root,"config","config.txt"))
        self._src_tbl.save_file(os.path.join(self._proj_root,"input","src","src.txt"))
        self._rec_tbl.save_file(os.path.join(self._proj_root,"input","rec","rec.txt"))
        self._mat_panel.save_all()
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
        cmd = [bin_cmd, scene,
               "config/config.txt","input/src/src.txt","input/rec/rec.txt",
               "config/poles.txt"]
        self._log.clear()
        self._log.append(f"$ cd {self._proj_root}\n$ " + " ".join(cmd))
        self._log.append("-"*60)
        # switch the button to Stop, reveal + reset the progress bar
        self._set_running(True)
        self._tabs.setCurrentIndex(4)
        self._solver_thread = SolverThread(cmd, self._proj_root)
        self._solver_thread.output.connect(self._log.append)
        self._solver_thread.progress.connect(self._on_solver_progress)
        self._solver_thread.done.connect(self._on_solver_done)
        self._solver_thread.start()

    def _fit_impedances(self):
        """Run vectFIT over ALL materials (was the Scene tab's 'Fit all').
        Defines cutoff frequency + ADE pole count and writes poles.txt / .fit,
        which the solver needs before a run — hence it lives here on the Run tab."""
        if not self._proj_root:
            self._sb.showMessage("Open a project folder first."); return
        self._mat_panel.run_fit_all()

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

    def _on_solver_progress(self, pct):
        self._float_prog.setValue(pct)

    def _stop_solver(self):
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
                   (FDTD + ISM + RT), so it must outlast the slowest-decaying band --
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

        # Vol -> recommended ray count (Vorlander <k>)
        vol_path = os.path.join(cdir, "Vol.bin")
        if os.path.isfile(vol_path):
            try:
                V = float(np.fromfile(vol_path, dtype=np.float32, count=1)[0])
                if V > 0.0:
                    fs = 44100.0                      # RT/ISM sample rate (main3D.cu)
                    R  = 0.15                         # MUST match main3D.cu rt_recv_radius
                    k  = 1.0                          # target detections per sample bin
                    N  = k * V * fs / (c * math.pi * R * R)
                    self._nrays.setValue(int(max(1000, min(N, 50_000_000))))

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
        """Run the solver's --estimate pass (parse + V + T60, no FDTD) so the
        recommended Cutoff frequency / Number of rays fill in BEFORE a real run.
        Needs no poles.txt/fit -- T60 uses only the .mat absorption."""
        if not self._proj_root:
            QMessageBox.warning(self, "No project", "Open a project folder first."); return
        if not self._obj_path:
            QMessageBox.warning(self, "No scene", "Load an OBJ first."); return
        # write config so h / temperature are current for the estimate
        self._write_config(os.path.join(self._proj_root, "config", "config.txt"))
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
        self._estimate_thread = SolverThread(cmd, self._proj_root)
        self._estimate_thread.done.connect(self._on_estimate_done)
        self._estimate_thread.start()

    def _on_estimate_done(self, rc):
        self._btn_estimate.setEnabled(True)
        if rc == 0:
            self._load_recommended()   # reads Vol.bin / f_Schroeder.bin, fills the fields
        else:
            self._sb.showMessage(f"Estimate failed (rc={rc}) — check the scene/materials")

    def _dump_rays(self):
        """Run the solver's --dump-rays pass (trace 'Debug rays' rays up to 'Order'
        bounces, no FDTD) and overlay the paths on the 3D scene (purplish blue)."""
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
        else:
            self._sb.showMessage(f"Ray trace failed (rc={rc})")

    # ── floating ray table ───────────────────────────────────
    def _ensure_ray_table(self):
        if getattr(self, "_ray_dock", None) is not None:
            return
        self._ray_tbl = QTableWidget(0, 5)
        self._ray_tbl.setHorizontalHeaderLabels(["Src", "Ray", "Bounces", "Escaped", "Length (m)"])
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
        t = self._ray_tbl
        t.blockSignals(True)
        t.setRowCount(0)
        for rid in sorted(paths):
            segs = paths[rid]
            bounces = sum(1 for _, _, o in segs if o >= 0)
            escaped = any(o < 0 for _, _, o in segs)
            length = sum(math.dist(p0, p1) for p0, p1, o in segs if o >= 0)
            r = t.rowCount(); t.insertRow(r)
            t.setItem(r, 0, QTableWidgetItem(str(srcs.get(rid, 0))))
            t.setItem(r, 1, QTableWidgetItem(str(rid)))
            t.setItem(r, 2, QTableWidgetItem(str(bounces)))
            t.setItem(r, 3, QTableWidgetItem("yes" if escaped else ""))
            t.setItem(r, 4, QTableWidgetItem(f"{length:.2f}"))
        t.blockSignals(False)
        t.clearSelection()
        self._ray_dock.show(); self._ray_dock.raise_()

    def _on_ray_row_selected(self):
        rows = self._ray_tbl.selectionModel().selectedRows()
        rid = int(self._ray_tbl.item(rows[0].row(), 1).text()) if rows else None
        self._gl.set_selected_ray(rid)

    def _clear_rays(self):
        self._gl.clear_ray_paths()
        if getattr(self, "_ray_dock", None) is not None:
            self._ray_tbl.setRowCount(0)
            self._ray_dock.hide()

    def _on_solver_done(self, rc):
        # Re-enable launching, fill/hide the progress bar, report status.
        self._set_running(False)
        if rc == 0:
            self._float_prog.setValue(100)
        c = PAL["ok"] if rc == 0 else PAL["err"]
        self._log.append(
            f'<span style="color:{c};">{"[OK]" if rc==0 else "[FAIL]"}  rc={rc}</span>')
        self._sb.showMessage(f"Solver finished  (rc={rc})")
        # hide the floating bar shortly after completion
        QTimer.singleShot(1500, self._float_prog.hide)


# ─────────────────────────────────────────────────────────────
def main():
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
