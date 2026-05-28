"""
gui.py  —  PyQt6 GUI for the 3D Acoustic FVM Solver  (v2)
==========================================================
Requirements:
    pip install PyQt6 PyOpenGL PyOpenGL_accelerate numpy

Run from project root:
    python common/gui.py

Key bindings (GL viewport):
    LMB (normal)   : rotate camera
    MMB / drag     : pan
    Scroll         : zoom
    R              : reset camera
    S              : enter "place SOURCE" mode
    E              : enter "place RECEIVER" mode   (R is reset, so E for rEceiver)
    X / Y / Z      : (in placement mode) lock to that axis
    Click          : (in placement mode) confirm placement
    Escape         : cancel placement mode / deselect layer
    LMB click      : (normal mode) ray-pick layer
"""

import sys, os, math
from pathlib import Path
import subprocess

import numpy as np

from PyQt6.QtCore    import Qt, QThread, pyqtSignal, QPoint, QPointF
from PyQt6.QtGui     import (QFont, QColor, QPalette, QSurfaceFormat,
                              QPainter, QCursor)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox,
    QListWidget, QListWidgetItem, QTabWidget, QTextEdit,
    QFileDialog, QMessageBox, QDoubleSpinBox,
    QGroupBox, QScrollArea, QAbstractItemView,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QStatusBar
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL  import *
from OpenGL.GLU import *

# ─────────────────────────────────────────────────────────────
#  Palette
# ─────────────────────────────────────────────────────────────
PAL = {
    "bg":        "#1a1d23", "panel":   "#20242c", "border":  "#2e3340",
    "accent":    "#4fc3f7", "accent2": "#81d4fa", "warn":    "#ffb74d",
    "ok":        "#a5d6a7", "err":     "#ef9a9a", "text":    "#cfd8dc",
    "subtext":   "#78909c", "highlight":"#263238","btn":     "#2a3140",
    "btn_hover": "#35404f", "layer_sel":"#1e3a4a",
}

LAYER_COLORS = [
    (0.90,0.35,0.35),(0.35,0.75,0.55),(0.45,0.65,0.95),
    (0.95,0.75,0.35),(0.75,0.45,0.90),(0.35,0.90,0.90),
    (0.95,0.55,0.75),(0.75,0.90,0.35),(0.60,0.80,0.60),
    (0.90,0.60,0.40),(0.40,0.70,0.90),(0.80,0.40,0.70),
]
BBOX_COLOR = (0.35, 0.65, 1.0)
SRC_COLOR  = (1.00, 0.82, 0.10)
REC_COLOR  = (0.20, 1.00, 0.55)
SEL_OUTLINE= (1.00, 1.00, 1.00)

OCTAVE_BANDS = [125, 250, 500, 1000, 2000, 4000]

# ─────────────────────────────────────────────────────────────
#  OBJ parser  (mirrors parsing3D.h)
# ─────────────────────────────────────────────────────────────
def parse_obj(path: str):
    vertices, groups = [], []
    cur_name, cur_tris = None, []

    def flush():
        if cur_name is not None:
            groups.append({"name": cur_name,
                           "tris": np.array(cur_tris, np.int32) if cur_tris
                                   else np.empty((0,3), np.int32)})

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()
            if tok[0] == "v":
                vertices.append((float(tok[1]), float(tok[2]), float(tok[3])))
            elif tok[0] in ("o", "g"):
                flush(); cur_tris = []
                parts = tok[1:]
                cur_name = (parts[1] if len(parts) >= 2 else
                            parts[0] if parts else "unnamed")
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
    layerPicked   = pyqtSignal(str)    # emitted on ray-pick
    pointPlaced   = pyqtSignal(str, float, float, float)  # mode, x, y, z
    pointRemoved  = pyqtSignal(str, int)                   # kind, index
    coordsChanged = pyqtSignal(float, float, float)        # live HUD coords

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
        self._yaw   =  35.0
        self._pitch = -22.0
        self._dist  =   6.0
        self._pan   = np.zeros(3, np.float32)
        self._last_pos  = None
        self._mouse_btn = None

        # scene
        self._verts  = np.zeros((0,3), np.float32)
        self._groups = []   # {name, tris, color, visible, is_bbox}
        self._bbox_visible  = True
        self._selected_layer = None

        # points
        self._sources   = []  # list of [x,y,z]
        self._receivers = []  # list of [x,y,z]

        # placement state
        self._mode      = self.MODE_NONE
        self._axis_lock = None          # 'X','Y','Z' or None
        self._place_pos = np.zeros(3, np.float32)  # frozen base position
        self._cursor_world = np.zeros(3, np.float32)

        # pivot (orbit centre = mesh centre)
        self._pivot      = np.zeros(3, np.float32)
        self._scene_span = 1.0   # updated by load_mesh
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
            self._pan        = np.zeros(3, np.float32)
            self._dist       = span * 1.8 if span > 0 else 6.0
            self._yaw   = 35.0
            self._pitch = -22.0
            # default placement position at mesh centre
            self._place_pos = ctr.copy()
        self.update()

    def set_layer_visible(self, name, v):
        for g in self._groups:
            if g["name"] == name: g["visible"] = v
        self.update()

    def set_bbox_visible(self, v):
        self._bbox_visible = v; self.update()

    def select_layer(self, name):
        self._selected_layer = name; self.update()

    def set_sources(self, pts):
        self._sources = [list(p[:3]) for p in pts]; self.update()

    def set_receivers(self, pts):
        self._receivers = [list(p[:3]) for p in pts]; self.update()

    def enter_placement_mode(self, mode):
        self._mode      = mode
        self._axis_lock = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def exit_placement_mode(self):
        self._mode      = self.MODE_NONE
        self._axis_lock = None
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
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glDisable(GL_CULL_FACE)
        glLightfv(GL_LIGHT0, GL_POSITION, [ 2.0, 3.0, 4.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [ 0.80, 0.80, 0.80, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [ 0.20, 0.20, 0.22, 1.0])
        glLightfv(GL_LIGHT1, GL_POSITION, [-2.0,-1.0,-2.0, 0.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE,  [ 0.25, 0.25, 0.30, 1.0])
        glLightfv(GL_LIGHT1, GL_AMBIENT,  [ 0.00, 0.00, 0.00, 1.0])
        glClearColor(0.10, 0.12, 0.14, 1.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def _setup_camera(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        asp = self.width() / max(self.height(), 1)
        near = max(self._dist * 0.001, 0.01)
        far  = self._dist * 500.0
        gluPerspective(45.0, asp, near, far)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Standard arcball:
        #   1. pull camera back along Z
        #   2. apply view-space pan  (pan moves the scene left/right/up in screen space)
        #   3. rotate around pivot
        #   4. translate world so pivot sits at origin
        glTranslatef(0.0, 0.0, -self._dist)
        glTranslatef(self._pan[0], self._pan[1], 0.0)   # pan in view/screen space
        glRotatef(self._pitch, 1, 0, 0)
        glRotatef(self._yaw,   0, 1, 0)
        glTranslatef(-self._pivot[0], -self._pivot[1], -self._pivot[2])
        self._mv   = glGetDoublev(GL_MODELVIEW_MATRIX)
        self._proj = glGetDoublev(GL_PROJECTION_MATRIX)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._setup_camera()
        self._draw_gizmo()   # replaces grid + world axes
        self._draw_mesh()
        self._draw_sources()
        self._draw_receivers()
        if self._mode != self.MODE_NONE:
            self._draw_placement_cursor()
        self._draw_hud()

    # ── drawing helpers ──────────────────────────────────────
    def _draw_gizmo(self):
        """
        Fixed top-left axis gizmo.
        Drawn entirely in a small orthographic sub-viewport so it never
        disturbs the scene camera matrices.
        Balls = two crossed screen-aligned filled circles per tip.
        Labels = QPainter pixel text, positions derived from pure trig.
        """
        W, H = self.width(), self.height()
        size  = 80
        pad   = 14
        cx    = pad + size // 2        # pixel centre of gizmo
        cy_gl = pad + size // 2        # in GL coords (origin bottom-left)

        L  = size * 0.36               # stick length in pixels
        br = size * 0.10               # ball radius in pixels

        axes = [
            ((1,0,0), (0.95, 0.30, 0.30), 'X'),
            ((0,1,0), (0.30, 0.88, 0.45), 'Y'),
            ((0,0,1), (0.35, 0.55, 0.95), 'Z'),
        ]

        # ── carve out a sub-viewport, save everything ──────
        glViewport(0, 0, pad*2+size, pad*2+size)
        glScissor (0, 0, pad*2+size, pad*2+size)
        glEnable(GL_SCISSOR_TEST)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix(); glLoadIdentity()
        glOrtho(0, pad*2+size, 0, pad*2+size, -200, 200)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix(); glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)

        # yaw and pitch as radians for tip projection
        yaw_r   = math.radians(self._yaw)
        pitch_r = math.radians(self._pitch)

        # project 3-D axis direction onto the 2-D gizmo plane
        # using the same rotation the scene camera applies
        def project_axis(ax, ay, az):
            # rotate by yaw around Y
            x1 =  ax * math.cos(yaw_r) + az * math.sin(yaw_r)
            y1 =  ay
            z1 = -ax * math.sin(yaw_r) + az * math.cos(yaw_r)
            # rotate by pitch around X
            x2 =  x1
            y2 =  y1 * math.cos(pitch_r) - z1 * math.sin(pitch_r)
            # z2 unused (depth)
            return x2, y2   # screen-space direction

        # ── draw sticks ───────────────────────────────────
        glBegin(GL_LINES)
        for (ax,ay,az), col, _ in axes:
            px, py = project_axis(ax, ay, az)
            glColor3f(*col)
            glVertex2f(cx, cy_gl)
            glVertex2f(cx + px*L, cy_gl + py*L)
        glEnd()

        # ── draw balls as two crossed filled circles ───────
        segs = 20
        for (ax,ay,az), col, _ in axes:
            px, py = project_axis(ax, ay, az)
            tx = cx + px * L
            ty = cy_gl + py * L
            glColor3f(*col)
            # circle 1: in XY plane of screen
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(tx, ty)
            for k in range(segs + 1):
                a = 2 * math.pi * k / segs
                glVertex2f(tx + br * math.cos(a), ty + br * math.sin(a))
            glEnd()
            # circle 2: slightly smaller, tilted (gives depth illusion)
            glColor3f(min(col[0]*1.3,1), min(col[1]*1.3,1), min(col[2]*1.3,1))
            r2 = br * 0.55
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(tx - br*0.2, ty + br*0.2)
            for k in range(segs + 1):
                a = 2 * math.pi * k / segs
                glVertex2f(tx - br*0.2 + r2*math.cos(a),
                           ty + br*0.2 + r2*math.sin(a))
            glEnd()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        # ── restore matrices and full viewport ────────────
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW);  glPopMatrix()
        glDisable(GL_SCISSOR_TEST)
        glViewport(0, 0, W, H)

        # ── labels via QPainter (no gluProject needed) ────
        painter = QPainter(self)
        painter.setFont(QFont("Monospace", 9, QFont.Weight.Bold))
        # QPainter y is flipped vs GL y
        cy_qt = H - cy_gl
        for (ax,ay,az), col, label in axes:
            px, py = project_axis(ax, ay, az)
            sx = int(cx  + px * L) + pad // 2
            sy = int(cy_qt - py * L) - pad // 2
            painter.setPen(QColor(int(col[0]*255),
                                  int(col[1]*255),
                                  int(col[2]*255)))
            painter.drawText(sx + int(br) + 2, sy + 4, label)
        painter.end()

    def _draw_mesh(self):
        if len(self._verts) == 0:
            return
        verts = self._verts

        # ── pass 1: solid geometry (non-bbox) ──────────────
        for g in self._groups:
            if g["is_bbox"] or not g["visible"]:
                continue
            tris = g["tris"]
            if len(tris) == 0:
                continue
            r, gr, b = g["color"]
            is_sel = (self._selected_layer == g["name"])
            norms  = g["normals"]

            glEnable(GL_LIGHTING)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glColor4f(r, gr, b, 0.88)
            glBegin(GL_TRIANGLES)
            for ti, tri in enumerate(tris):
                if norms is not None:
                    glNormal3fv(norms[ti])
                for vi in tri:
                    glVertex3fv(verts[vi])
            glEnd()

            if is_sel:
                glDisable(GL_LIGHTING)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glLineWidth(2.2)
                glColor4f(1.0, 1.0, 1.0, 0.95)
                glBegin(GL_TRIANGLES)
                for tri in tris:
                    for vi in tri: glVertex3fv(verts[vi])
                glEnd()
                glEnable(GL_LIGHTING)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # ── pass 2: bounding box wireframe drawn last ───────
        # Drawn after all solid geometry so its depth writes
        # (even with GL_FALSE) cannot occlude interior objects.
        if self._bbox_visible:
            for g in self._groups:
                if not g["is_bbox"] or not g["visible"]:
                    continue
                tris = g["tris"]
                if len(tris) == 0:
                    continue
                r, gr, b = g["color"]
                is_sel = (self._selected_layer == g["name"])
                glDisable(GL_LIGHTING)
                glDepthMask(GL_FALSE)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glLineWidth(1.0)
                glColor4f(r, gr, b, 0.55 if not is_sel else 0.85)
                glBegin(GL_TRIANGLES)
                for tri in tris:
                    for vi in tri: glVertex3fv(verts[vi])
                glEnd()
                glDepthMask(GL_TRUE)
                glEnable(GL_LIGHTING)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def _speaker_icon(self, x, y, z, scale, color):
        """Draw a stylised loudspeaker at world position (x,y,z)."""
        r, g, b = color
        glDisable(GL_LIGHTING)
        glPushMatrix()
        glTranslatef(x, y, z)
        # billboard — rotate to face camera (undo yaw+pitch)
        glRotatef(-self._yaw,   0, 1, 0)
        glRotatef(-self._pitch, 1, 0, 0)
        s = scale

        # body rectangle
        glColor4f(r, g, b, 1.0)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(-s*0.25, -s*0.35, 0)
        glVertex3f( s*0.25, -s*0.35, 0)
        glVertex3f( s*0.25,  s*0.35, 0)
        glVertex3f(-s*0.25,  s*0.35, 0)
        glEnd()

        # cone shape (trapezoid on right side)
        glBegin(GL_LINE_LOOP)
        glVertex3f( s*0.25, -s*0.35, 0)
        glVertex3f( s*0.65, -s*0.60, 0)
        glVertex3f( s*0.65,  s*0.60, 0)
        glVertex3f( s*0.25,  s*0.35, 0)
        glEnd()

        # sound waves
        glColor4f(r, g, b, 0.55)
        for i in range(1, 3):
            ri = s * (0.75 + i*0.25)
            glBegin(GL_LINE_STRIP)
            for a in range(-5, 6):
                ang = math.radians(a * 10)
                glVertex3f(s*0.65 + ri*math.cos(ang), ri*math.sin(ang), 0)
            glEnd()

        # dot at centre
        glPointSize(5)
        glColor4f(r, g, b, 1.0)
        glBegin(GL_POINTS); glVertex3f(0, 0, 0); glEnd()

        glPopMatrix()
        glEnable(GL_LIGHTING)

    def _mic_icon(self, x, y, z, scale, color):
        """Draw a stylised microphone at world position (x,y,z)."""
        r, g, b = color
        glDisable(GL_LIGHTING)
        glPushMatrix()
        glTranslatef(x, y, z)
        glRotatef(-self._yaw,   0, 1, 0)
        glRotatef(-self._pitch, 1, 0, 0)
        s = scale

        # capsule (head)
        glColor4f(r, g, b, 1.0)
        glLineWidth(2.0)
        segs = 16
        glBegin(GL_LINE_STRIP)
        for i in range(segs+1):
            ang = math.radians(i * 180 / segs)
            glVertex3f(s*0.22*math.cos(ang), s*0.55 + s*0.22*math.sin(ang), 0)
        glEnd()
        glBegin(GL_LINES)
        glVertex3f(-s*0.22,  s*0.55, 0); glVertex3f(-s*0.22, -s*0.10, 0)
        glVertex3f( s*0.22,  s*0.55, 0); glVertex3f( s*0.22, -s*0.10, 0)
        glEnd()

        # stand arc
        glColor4f(r, g, b, 0.65)
        glBegin(GL_LINE_STRIP)
        for i in range(segs+1):
            ang = math.radians(i * 180 / segs)
            glVertex3f(s*0.38*math.cos(ang), -s*0.10 - s*0.28*math.sin(ang), 0)
        glEnd()
        # stem
        glBegin(GL_LINES)
        glVertex3f(0, -s*0.10, 0); glVertex3f(0, -s*0.62, 0)
        # base
        glVertex3f(-s*0.28, -s*0.62, 0); glVertex3f( s*0.28, -s*0.62, 0)
        glEnd()

        glPointSize(5)
        glColor4f(r, g, b, 1.0)
        glBegin(GL_POINTS); glVertex3f(0, 0, 0); glEnd()

        glPopMatrix()
        glEnable(GL_LIGHTING)

    def _draw_sources(self):
        # icon size = 3% of scene span, independent of zoom distance
        s = max(self._scene_span * 0.03, 0.05)
        for pt in self._sources:
            self._speaker_icon(pt[0], pt[1], pt[2], s, SRC_COLOR)

    def _draw_receivers(self):
        s = max(self._scene_span * 0.03, 0.05)
        for pt in self._receivers:
            self._mic_icon(pt[0], pt[1], pt[2], s, REC_COLOR)

    def _draw_placement_cursor(self):
        """Axis arrow + crosshair at _cursor_world."""
        p = self._cursor_world
        glDisable(GL_LIGHTING)

        col = {None: (1.0,1.0,1.0),
               'X': (1.0,0.3,0.3),
               'Y': (0.3,1.0,0.3),
               'Z': (0.4,0.6,1.0)}[self._axis_lock]

        glColor4f(*col, 0.9)
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
            glColor4f(*col, 0.45)
            glBegin(GL_LINES)
            glVertex3f(p[0]-ax[0]*ext, p[1]-ax[1]*ext, p[2]-ax[2]*ext)
            glVertex3f(p[0]+ax[0]*ext, p[1]+ax[1]*ext, p[2]+ax[2]*ext)
            glEnd()

        # preview icon
        s = max(self._scene_span * 0.03, 0.05)
        if self._mode == self.MODE_SRC:
            self._speaker_icon(p[0], p[1], p[2], s, SRC_COLOR)
        else:
            self._mic_icon(p[0], p[1], p[2], s, REC_COLOR)

        glEnable(GL_LIGHTING)

    def _draw_hud(self):
        """Overlay text using QPainter on top of GL."""
        if self._mode == self.MODE_NONE:
            return
        p = self._cursor_world
        mode_str = "SOURCE" if self._mode == self.MODE_SRC else "RECEIVER"
        mode_col = SRC_COLOR if self._mode == self.MODE_SRC else REC_COLOR
        axis_str = f"  locked: {self._axis_lock}" if self._axis_lock else "  (press X / Y / Z to lock axis)"
        coord_str = f"  x={p[0]:.3f}  y={p[1]:.3f}  z={p[2]:.3f} m"

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # semi-transparent banner at top
        from PyQt6.QtGui import QBrush
        from PyQt6.QtCore import QRect
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
    def mousePressEvent(self, e):
        self._last_pos  = e.position()
        self._mouse_btn = e.button()
        self._drag_moved = False

    def mouseMoveEvent(self, e):
        if self._last_pos is None:
            # still update cursor world pos for HUD even without button
            if self._mode != self.MODE_NONE:
                self._update_cursor_world(e.position())
                self.update()
            return

        dx = e.position().x() - self._last_pos.x()
        dy = e.position().y() - self._last_pos.y()
        if abs(dx) > 1 or abs(dy) > 1:
            self._drag_moved = True

        if self._mode != self.MODE_NONE and self._axis_lock:
            # slide along locked axis
            self._slide_axis(e.position(), dx, dy)
        elif self._mode != self.MODE_NONE:
            self._update_cursor_world(e.position())
        else:
            if self._mouse_btn == Qt.MouseButton.LeftButton:
                self._yaw   += dx * 0.4
                self._pitch  = max(-89, min(89, self._pitch + dy*0.4))
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

        if self._mode != self.MODE_NONE:
            if not moved:
                # confirm placement
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

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key.Key_Escape:
            self.exit_placement_mode()
            e.accept(); return
        if key == Qt.Key.Key_R and self._mode == self.MODE_NONE:
            # reset camera
            self._yaw = 35.0; self._pitch = -22.0
            self._pan = np.zeros(3, np.float32)
            self.update()
            e.accept(); return
        if self._mode != self.MODE_NONE:
            if key == Qt.Key.Key_X:
                self._set_axis_lock('X'); e.accept(); return
            elif key == Qt.Key.Key_Y:
                self._set_axis_lock('Y'); e.accept(); return
            elif key == Qt.Key.Key_Z:
                self._set_axis_lock('Z'); e.accept(); return
        e.ignore()  # let MainWindow handle S / T for mode switching

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

    def _set_axis_lock(self, axis):
        self._axis_lock = axis
        # freeze the two orthogonal coords at current cursor
        # (already stored in _cursor_world)
        self._place_pos = self._cursor_world.copy()
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

    def _slide_axis(self, qpos, dx, dy):
        """Move cursor only along the locked axis."""
        ax = {'X': np.array([1,0,0],np.float32),
              'Y': np.array([0,1,0],np.float32),
              'Z': np.array([0,0,1],np.float32)}[self._axis_lock]
        self.makeCurrent()
        W, H = self.width(), self.height()
        try:
            viewport = [int(0), int(0), int(W), int(H)]
            # project two points on the axis to screen to get screen-space direction
            p0 = self._place_pos
            p1 = p0 + ax * 0.1
            s0 = np.array(gluProject(*p0.astype(float), self._mv, self._proj, viewport)[:2])
            s1 = np.array(gluProject(*p1.astype(float), self._mv, self._proj, viewport)[:2])
            screen_dir = s1 - s0
            slen = np.linalg.norm(screen_dir)
            if slen < 1e-6: return
            screen_dir /= slen
            # project mouse delta onto screen axis direction
            mouse_delta = np.array([dx, -dy], np.float32)
            move = np.dot(mouse_delta, screen_dir)
            # scale: how many metres per pixel
            scale = self._dist * 0.004
            self._cursor_world = self._place_pos + ax * move * scale
            # keep sliding base updated
            self._place_pos = self._cursor_world.copy()
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
            self.update()


# ─────────────────────────────────────────────────────────────
#  Material editor
# ─────────────────────────────────────────────────────────────
class MaterialEditor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._path = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(4)

        hdr = QLabel("Absorption  α  per octave band")
        hdr.setStyleSheet(f"color:{PAL['subtext']};font-size:11px;")
        lay.addWidget(hdr)

        self._table = QTableWidget(1, len(OCTAVE_BANDS))
        self._table.setHorizontalHeaderLabels([f"{b} Hz" for b in OCTAVE_BANDS])
        self._table.verticalHeader().setVisible(False)
        self._table.setFixedHeight(56)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setStyleSheet(f"""
            QTableWidget {{background:{PAL['bg']};color:{PAL['text']};
                gridline-color:{PAL['border']};border:1px solid {PAL['border']};}}
            QHeaderView::section {{background:{PAL['panel']};color:{PAL['subtext']};
                border:1px solid {PAL['border']};font-size:11px;}}
        """)
        lay.addWidget(self._table)

        row = QHBoxLayout()
        self._save_btn = QPushButton("💾  Save .mat")
        self._save_btn.clicked.connect(self.save)
        self._save_btn.setEnabled(False)
        row.addWidget(self._save_btn)
        self._fit_btn = QPushButton("⚙  Run vectFIT")
        self._fit_btn.clicked.connect(self.run_fit)
        self._fit_btn.setEnabled(False)
        row.addWidget(self._fit_btn)
        lay.addLayout(row)

        self._status = QLabel("")
        self._status.setStyleSheet(f"color:{PAL['subtext']};font-size:10px;")
        lay.addWidget(self._status)

    def load(self, mat_path):
        self._path = mat_path
        self._save_btn.setEnabled(True); self._fit_btn.setEnabled(True)
        for col in range(len(OCTAVE_BANDS)):
            self._table.setItem(0, col, QTableWidgetItem("0.10"))
        if os.path.isfile(mat_path):
            try:
                vals = []
                with open(mat_path) as f:
                    for line in f:
                        line = line.split("#")[0].strip()
                        if line: vals += [float(x) for x in line.split()]
                for col, v in enumerate(vals[:len(OCTAVE_BANDS)]):
                    self._table.setItem(0, col, QTableWidgetItem(f"{v:.4f}"))
                self._status.setText(f"Loaded: {os.path.basename(mat_path)}")
            except Exception as ex:
                self._status.setText(f"Parse error: {ex}")
        else:
            self._status.setText("New file (defaults loaded)")

    def get_values(self):
        vals = []
        for col in range(len(OCTAVE_BANDS)):
            item = self._table.item(0, col)
            try: vals.append(float(item.text()) if item else 0.0)
            except ValueError: vals.append(0.0)
        return vals

    def save(self):
        if not self._path: return
        vals = self.get_values()
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            f.write(f"# alpha per octave band: {OCTAVE_BANDS}\n")
            f.write("  ".join(f"{v:.4f}" for v in vals) + "\n")
        self._status.setText(f"Saved → {os.path.basename(self._path)}")

    def run_fit(self):
        if not self._path: return
        self.save()
        proj = Path(self._path).parents[3]
        name = Path(self._path).stem
        for script_name in ("vectFIT.py", "fminconFIT.py"):
            script = proj.parent / "common" / script_name
            if script.is_file(): break
        else:
            self._status.setText("vectFIT.py not found"); return
        out = proj / "input" / "fit" / (name + ".fit")
        cmd = [sys.executable, str(script), "--mat", self._path, "--out", str(out)]
        self._status.setText("Running fit…")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out_bytes, _ = proc.communicate()
        self._status.setText("Fit done." if proc.returncode == 0
                             else f"Fit error (rc={proc.returncode})")


# ─────────────────────────────────────────────────────────────
#  Point table  (sources / receivers)
# ─────────────────────────────────────────────────────────────
class PointTable(QWidget):
    changed = pyqtSignal()

    def __init__(self, has_amplitude=False, parent=None):
        super().__init__(parent)
        self._has_amp = has_amplitude
        cols = ["x (m)", "y (m)", "z (m)"] + (["p₀ (Pa)"] if has_amplitude else [])
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self._tbl = QTableWidget(0, len(cols))
        self._tbl.setHorizontalHeaderLabels(cols)
        self._tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._tbl.setStyleSheet(f"""
            QTableWidget {{background:{PAL['bg']};color:{PAL['text']};
                gridline-color:{PAL['border']};border:1px solid {PAL['border']};}}
            QHeaderView::section {{background:{PAL['panel']};color:{PAL['subtext']};
                border:1px solid {PAL['border']};}}
        """)
        self._tbl.itemChanged.connect(lambda: self.changed.emit())
        lay.addWidget(self._tbl)
        btn_row = QHBoxLayout()
        for label, fn in [("＋ Add", self._add), ("－ Remove", self._remove)]:
            b = QPushButton(label); b.clicked.connect(fn); btn_row.addWidget(b)
        lay.addLayout(btn_row)

    def _add(self, x=1.0, y=1.0, z=1.0, amp=1.0):
        self._tbl.blockSignals(True)
        r = self._tbl.rowCount(); self._tbl.insertRow(r)
        vals = [x, y, z] + ([amp] if self._has_amp else [])
        for c, v in enumerate(vals):
            self._tbl.setItem(r, c, QTableWidgetItem(f"{v:.4g}"))
        self._tbl.blockSignals(False)
        self.changed.emit()

    def add_point(self, x, y, z, amp=1.0):
        self._add(x, y, z, amp)

    def _remove(self):
        rows = sorted({i.row() for i in self._tbl.selectedItems()}, reverse=True)
        for r in rows: self._tbl.removeRow(r)
        self.changed.emit()

    def get_points(self):
        pts = []
        for r in range(self._tbl.rowCount()):
            try:
                row = []
                for c in range(self._tbl.columnCount()):
                    item = self._tbl.item(r, c)
                    if item is None: raise ValueError("empty cell")
                    row.append(float(item.text()))
                pts.append(tuple(row))
            except (TypeError, ValueError): pass
        return pts

    def load_file(self, path):
        self._tbl.setRowCount(0)
        if not os.path.isfile(path): return
        with open(path) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if not line: continue
                vals = line.split()
                r = self._tbl.rowCount(); self._tbl.insertRow(r)
                for c, v in enumerate(vals[:self._tbl.columnCount()]):
                    self._tbl.setItem(r, c, QTableWidgetItem(v))

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
    output   = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, cmd, cwd):
        super().__init__(); self._cmd = cmd; self._cwd = cwd

    def run(self):
        try:
            proc = subprocess.Popen(self._cmd, cwd=self._cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            for line in proc.stdout: self.output.emit(line.rstrip())
            proc.wait(); self.finished.emit(proc.returncode)
        except Exception as ex:
            self.output.emit(f"[ERROR] {ex}"); self.finished.emit(-1)


# ─────────────────────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Acoustic FVM Solver  —  Scene Setup")
        self.resize(1440, 860)
        self._proj_root  = None
        self._obj_path   = None
        self._groups     = []
        self._mat_editors= {}
        self._apply_theme()
        self._build_ui()

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
                border:1px solid {PAL['border']};padding:6px 14px;
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
            QSplitter::handle{{background:{PAL['border']};}}
            QStatusBar{{background:{PAL['panel']};color:{PAL['subtext']};font-size:11px;}}
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_lay = QHBoxLayout(central)
        root_lay.setContentsMargins(6,6,6,6); root_lay.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_lay.addWidget(splitter)

        # left — GL + status
        gl_wrap = QWidget()
        gl_vlay = QVBoxLayout(gl_wrap)
        gl_vlay.setContentsMargins(0,0,0,0); gl_vlay.setSpacing(3)

        self._gl = GLViewer()
        self._gl.layerPicked.connect(self._on_layer_picked)
        self._gl.pointPlaced.connect(self._on_point_placed)
        self._gl.pointRemoved.connect(self._on_point_removed)
        gl_vlay.addWidget(self._gl)

        # placement toolbar under the viewer
        place_bar = QHBoxLayout()
        place_bar.setSpacing(4)

        self._src_btn = QPushButton("📡  Place Source  [S]")
        self._src_btn.setCheckable(True)
        self._src_btn.clicked.connect(lambda: self._toggle_placement("source"))
        place_bar.addWidget(self._src_btn)

        self._rec_btn = QPushButton("🎙  Place Receiver  [T]")
        self._rec_btn.setCheckable(True)
        self._rec_btn.clicked.connect(lambda: self._toggle_placement("receiver"))
        place_bar.addWidget(self._rec_btn)

        place_bar.addStretch()
        hint = QLabel("Rotate: LMB   Pan: MMB   Scroll: zoom   R: reset   "
                      "S: source  T: receiver   X/Y/Z: axis lock   RMB: delete   Esc: cancel")
        hint.setStyleSheet(f"color:{PAL['subtext']};font-size:10px;")
        place_bar.addWidget(hint)
        gl_vlay.addLayout(place_bar)

        splitter.addWidget(gl_wrap)

        # right — tabs
        self._tabs = QTabWidget()
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self._tabs.addTab(self._build_scene_tab(),   "🗂  Scene")
        self._tabs.addTab(self._build_config_tab(),  "⚙  Config")
        self._tabs.addTab(self._build_src_tab(),     "📡  Sources")
        self._tabs.addTab(self._build_rec_tab(),     "🎙  Receivers")
        self._tabs.addTab(self._build_run_tab(),     "▶  Run")

        # status bar
        self._sb = QStatusBar()
        self.setStatusBar(self._sb)
        self._sb.showMessage("Ready.  Open a project folder to begin.")

    # ── placement mode ───────────────────────────────────────
    def _toggle_placement(self, kind):
        if kind == "source":
            if self._src_btn.isChecked():
                self._rec_btn.setChecked(False)
                self._gl.enter_placement_mode(GLViewer.MODE_SRC)
                self._tabs.setCurrentIndex(2)
                self._sb.showMessage("SOURCE placement: move mouse, X/Y/Z to lock axis, click to place, Esc to cancel")
            else:
                self._gl.exit_placement_mode()
                self._sb.showMessage("Ready.")
        else:
            if self._rec_btn.isChecked():
                self._src_btn.setChecked(False)
                self._gl.enter_placement_mode(GLViewer.MODE_REC)
                self._tabs.setCurrentIndex(3)
                self._sb.showMessage("RECEIVER placement: move mouse, X/Y/Z to lock axis, click to place, Esc to cancel")
            else:
                self._gl.exit_placement_mode()
                self._sb.showMessage("Ready.")

    def _on_point_placed(self, mode, x, y, z):
        if mode == "source":
            self._src_tbl.add_point(x, y, z, 1.0)
        else:
            self._rec_tbl.add_point(x, y, z)
        self._refresh_gl_points()
        self._sb.showMessage(f"Placed {mode} at ({x:.3f}, {y:.3f}, {z:.3f}) m  —  click to place another, Esc to finish")

    def _on_point_removed(self, kind, idx):
        tbl = self._src_tbl if kind == "source" else self._rec_tbl
        tbl._tbl.blockSignals(True)
        tbl._tbl.removeRow(idx)
        tbl._tbl.blockSignals(False)
        self._refresh_gl_points()
        self._sb.showMessage(f"Removed {kind} #{idx}")

    def keyPressEvent(self, e):
        """Global key bindings. S=source, T=receiver, R=reset, X/Y/Z=axis lock."""
        key = e.key()
        if key == Qt.Key.Key_S:
            checked = not self._src_btn.isChecked()
            self._src_btn.setChecked(checked)
            self._toggle_placement("source")
            e.accept(); return
        if key == Qt.Key.Key_T:
            checked = not self._rec_btn.isChecked()
            self._rec_btn.setChecked(checked)
            self._toggle_placement("receiver")
            e.accept(); return
        if key == Qt.Key.Key_Escape:
            self._src_btn.setChecked(False)
            self._rec_btn.setChecked(False)
            self._gl.exit_placement_mode()
            self._sb.showMessage("Ready.")
            e.accept(); return
        # forward R (reset) and X/Y/Z to GL
        self._gl.keyPressEvent(e)

    # ── Scene tab ────────────────────────────────────────────
    def _build_scene_tab(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(8)

        grp_proj = QGroupBox("Project")
        g_lay = QHBoxLayout(grp_proj)
        self._proj_label = QLabel("not set")
        self._proj_label.setStyleSheet(f"color:{PAL['subtext']};font-size:11px;")
        g_lay.addWidget(self._proj_label, 1)
        btn = QPushButton("Open 3D-solver/"); btn.clicked.connect(self._open_project)
        g_lay.addWidget(btn)
        lay.addWidget(grp_proj)

        grp_obj = QGroupBox("Scene file (.obj)")
        o_lay = QHBoxLayout(grp_obj)
        self._obj_label = QLabel("none")
        self._obj_label.setStyleSheet(f"color:{PAL['subtext']};font-size:11px;")
        o_lay.addWidget(self._obj_label, 1)
        btn2 = QPushButton("Load OBJ"); btn2.clicked.connect(self._load_obj)
        o_lay.addWidget(btn2)
        lay.addWidget(grp_obj)

        bbox_row = QHBoxLayout()
        self._bbox_chk = QCheckBox("Show Boundary (bounding box)")
        self._bbox_chk.setChecked(True)
        self._bbox_chk.toggled.connect(self._gl.set_bbox_visible)
        bbox_row.addWidget(self._bbox_chk); bbox_row.addStretch()
        lay.addLayout(bbox_row)

        lay.addWidget(QLabel("Layers  — click in viewport or list to select:"))
        self._layer_list = QListWidget()
        self._layer_list.setMaximumHeight(170)
        self._layer_list.currentRowChanged.connect(self._on_layer_row_changed)
        lay.addWidget(self._layer_list)

        vis_row = QHBoxLayout()
        for lbl, fn in [("Show all", lambda: self._set_all_vis(True)),
                        ("Hide all", lambda: self._set_all_vis(False))]:
            b = QPushButton(lbl); b.clicked.connect(fn); vis_row.addWidget(b)
        vis_row.addStretch(); lay.addLayout(vis_row)

        lay.addWidget(QLabel("Material — selected layer:"))
        self._mat_stack = QScrollArea()
        self._mat_stack.setWidgetResizable(True)
        ph = QLabel("Select a layer to edit its material.")
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setStyleSheet(f"color:{PAL['subtext']};")
        self._mat_stack.setWidget(ph)
        lay.addWidget(self._mat_stack, 1)
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

        self._cfg = {}
        for key, lbl, val, mn, mx, dec, sfx in [
            ("h",       "Grid spacing",     0.05, 0.001, 2.0,   4, "m"),
            ("t_max",   "Simulation time",  0.5,  0.001,10.0,   3, "s"),
            ("f_cutoff","Cutoff frequency",1000., 50., 8000.,   1, "Hz"),
            ("T_C",     "Temperature",      20., -30.,   60.,   1, "°C"),
            ("CFL",     "CFL number",        0.4,  0.05,  0.49, 3, ""),
        ]:
            sp = ds(val, mn, mx, dec, sfx); self._cfg[key] = sp
            lay.addRow(f"{lbl}:", sp)

        lay.addRow(QLabel("PML faces  (check = active):"))
        grid = QGridLayout(); self._pml = {}
        for idx, (k, lbl) in enumerate([("U_PML","Up"),("D_PML","Down"),
                                         ("R_PML","Right"),("L_PML","Left"),
                                         ("F_PML","Front"),("B_PML","Back")]):
            chk = QCheckBox(lbl); chk.setChecked(True)
            self._pml[k] = chk; grid.addWidget(chk, idx//3, idx%3)
        lay.addRow(grid)

        for lbl, fn in [("💾  Save config.txt", self._save_config),
                        ("📂  Load config.txt", self._load_config)]:
            b = QPushButton(lbl); b.clicked.connect(fn); lay.addRow(b)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        out = QWidget(); QVBoxLayout(out).addWidget(scroll)
        return out

    # ── Sources tab ──────────────────────────────────────────
    def _build_src_tab(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(8,8,8,8)
        info = QLabel("Press S (or the toolbar button) to enter placement mode.\n"
                      "Then optionally lock an axis with X/Y/Z and click to place.\n"
                      "Or add rows manually below.")
        info.setStyleSheet(f"color:{PAL['subtext']};font-size:11px;")
        info.setWordWrap(True); lay.addWidget(info)
        self._src_tbl = PointTable(has_amplitude=True)
        self._src_tbl.changed.connect(self._refresh_gl_points)
        lay.addWidget(self._src_tbl)
        br = QHBoxLayout()
        for lbl, fn in [("📂  Load src.txt", self._load_src),
                        ("💾  Save src.txt", self._save_src)]:
            b = QPushButton(lbl); b.clicked.connect(fn); br.addWidget(b)
        lay.addLayout(br)
        return w

    def _build_rec_tab(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(8,8,8,8)
        info = QLabel("Press E (or the toolbar button) to enter placement mode.\n"
                      "Lock axis with X/Y/Z, click to place.  Esc to finish.")
        info.setStyleSheet(f"color:{PAL['subtext']};font-size:11px;")
        info.setWordWrap(True); lay.addWidget(info)
        self._rec_tbl = PointTable(has_amplitude=False)
        self._rec_tbl.changed.connect(self._refresh_gl_points)
        lay.addWidget(self._rec_tbl)
        br = QHBoxLayout()
        for lbl, fn in [("📂  Load rec.txt", self._load_rec),
                        ("💾  Save rec.txt", self._save_rec)]:
            b = QPushButton(lbl); b.clicked.connect(fn); br.addWidget(b)
        lay.addLayout(br)
        return w

    # ── Run tab ──────────────────────────────────────────────
    def _build_run_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(8,8,8,8); lay.setSpacing(8)

        bin_row = QHBoxLayout()
        bin_row.addWidget(QLabel("Binary:"))
        self._bin_le = QLineEdit("build/phases3D")
        self._bin_le.textChanged.connect(self._update_cmd_preview)
        bin_row.addWidget(self._bin_le, 1)
        bb = QPushButton("Browse"); bb.clicked.connect(self._browse_binary)
        bin_row.addWidget(bb); lay.addLayout(bin_row)

        lay.addWidget(QLabel("Command:"))
        self._cmd_preview = QLabel("(open project first)")
        self._cmd_preview.setStyleSheet(
            f"color:{PAL['accent']};font-family:monospace;font-size:11px;")
        self._cmd_preview.setWordWrap(True); lay.addWidget(self._cmd_preview)

        self._run_btn = QPushButton("▶  Launch solver")
        self._run_btn.setStyleSheet(
            f"background:{PAL['accent']};color:{PAL['bg']};font-weight:bold;padding:8px;")
        self._run_btn.clicked.connect(self._run_solver); lay.addWidget(self._run_btn)

        self._stop_btn = QPushButton("⏹  Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_solver); lay.addWidget(self._stop_btn)

        lay.addWidget(QLabel("Output:"))
        self._log = QTextEdit(); self._log.setReadOnly(True)
        self._log.setFont(QFont("Monospace", 10)); lay.addWidget(self._log, 1)

        self._solver_thread = None
        return w

    # ── slots ────────────────────────────────────────────────
    def _open_project(self):
        d = QFileDialog.getExistingDirectory(self, "Select 3D-solver directory")
        if not d: return
        self._proj_root = d; self._proj_label.setText(d)
        for rel, fn in [("config/config.txt", self._load_config_from),
                        ("input/src/src.txt",  self._src_tbl.load_file),
                        ("input/rec/rec.txt",  self._rec_tbl.load_file)]:
            p = os.path.join(d, rel)
            if os.path.isfile(p): fn(p)
        self._refresh_gl_points(); self._update_cmd_preview()
        self._sb.showMessage(f"Project: {d}")

    def _load_obj(self):
        start = os.path.join(self._proj_root, "input", "scene") if self._proj_root else ""
        path, _ = QFileDialog.getOpenFileName(self, "Load OBJ", start, "OBJ (*.obj)")
        if not path: return
        self._obj_path = path
        self._obj_label.setText(os.path.basename(path))
        try:
            verts, groups = parse_obj(path)
        except Exception as ex:
            QMessageBox.critical(self, "OBJ error", str(ex)); return
        self._groups = groups
        names = [g["name"] for g in groups]
        print(f"[OBJ] Parsed {len(groups)} groups: {names}")
        bbox_found = any("boundary" in n.lower() for n in names)
        if not bbox_found:
            print("[OBJ] WARNING: no group named 'Boundary' found. "
                  "Boundary toggle will have no effect. "
                  f"Group names: {names}")
        self._gl.load_mesh(verts, groups)
        self._populate_layers(groups)
        self._update_cmd_preview()
        status = f"Loaded {len(groups)} layer(s)  —  {len(verts)} vertices"
        if not bbox_found:
            status += "  ⚠ no Boundary group detected"
        self._sb.showMessage(status)

    def _populate_layers(self, groups):
        self._layer_list.clear(); self._mat_editors.clear()
        for g in groups:
            name = g["name"]
            item = QListWidgetItem(name)
            if "boundary" in name.lower():
                item.setForeground(QColor(PAL["subtext"]))
            self._layer_list.addItem(item)
        if self._proj_root:
            for g in groups:
                name = g["name"]
                if "boundary" in name.lower(): continue
                mp = os.path.join(self._proj_root, "input", "mat", name+".mat")
                ed = MaterialEditor(); ed.load(mp)
                self._mat_editors[name] = ed

    def _on_layer_picked(self, name):
        """Sync list selection when user clicks in the GL viewport."""
        for i in range(self._layer_list.count()):
            if self._layer_list.item(i).text() == name:
                self._layer_list.setCurrentRow(i)
                break
        self._tabs.setCurrentIndex(0)

    def _on_layer_row_changed(self, row):
        if row < 0 or row >= len(self._groups): return
        name = self._groups[row]["name"]
        self._gl.select_layer(name)
        if "boundary" in name.lower():
            ph = QLabel("Boundary layer — no material assignment.")
            ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ph.setStyleSheet(f"color:{PAL['subtext']};")
            self._mat_stack.setWidget(ph); return
        if name not in self._mat_editors:
            mp = (os.path.join(self._proj_root, "input","mat",name+".mat")
                  if self._proj_root else "")
            ed = MaterialEditor()
            if mp: ed.load(mp)
            self._mat_editors[name] = ed
        # Rescue widget from Qt ownership before swapping;
        # otherwise Qt deletes it and the dict holds a dangling pointer.
        old = self._mat_stack.takeWidget()
        if old is not None:
            old.setParent(None)
        self._mat_stack.setWidget(self._mat_editors[name])

    def _set_all_vis(self, v):
        for g in self._groups: self._gl.set_layer_visible(g["name"], v)

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
            f.write(f"0                               # reserved\n")
            for k in ["U_PML","D_PML","R_PML","L_PML","F_PML","B_PML"]:
                f.write(f"{pml[k]:.1f}  # {k}\n")

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
        for j, k in enumerate(["U_PML","D_PML","R_PML","L_PML","F_PML","B_PML"]):
            idx = 6+j
            if idx < len(vals): self._pml[k].setChecked(vals[idx] != 0.0)

    def _save_src(self):
        if not self._proj_root: return
        self._src_tbl.save_file(os.path.join(self._proj_root,"input","src","src.txt"))
        self._sb.showMessage("Saved src.txt")

    def _load_src(self):
        start = os.path.join(self._proj_root,"input","src") if self._proj_root else ""
        p, _ = QFileDialog.getOpenFileName(self,"Load src.txt",start,"Text (*.txt)")
        if p: self._src_tbl.load_file(p); self._refresh_gl_points()

    def _save_rec(self):
        if not self._proj_root: return
        self._rec_tbl.save_file(os.path.join(self._proj_root,"input","rec","rec.txt"))
        self._sb.showMessage("Saved rec.txt")

    def _load_rec(self):
        start = os.path.join(self._proj_root,"input","rec") if self._proj_root else ""
        p, _ = QFileDialog.getOpenFileName(self,"Load rec.txt",start,"Text (*.txt)")
        if p: self._rec_tbl.load_file(p); self._refresh_gl_points()

    def _refresh_gl_points(self):
        self._gl.set_sources(  [(p[0],p[1],p[2]) for p in self._src_tbl.get_points()])
        self._gl.set_receivers([(p[0],p[1],p[2]) for p in self._rec_tbl.get_points()])

    def _browse_binary(self):
        p, _ = QFileDialog.getOpenFileName(self, "Locate solver binary")
        if p:
            if self._proj_root and p.startswith(self._proj_root):
                p = os.path.relpath(p, self._proj_root)
            self._bin_le.setText(p)

    def _update_cmd_preview(self):
        if not hasattr(self, '_cmd_preview'): return
        if not self._proj_root:
            self._cmd_preview.setText("(open a project folder first)"); return
        binary = self._bin_le.text() if hasattr(self,'_bin_le') else "build/phases3D"
        scene = self._obj_path or "input/scene/scene.obj"
        if self._proj_root and scene.startswith(self._proj_root):
            scene = os.path.relpath(scene, self._proj_root)
        self._cmd_preview.setText(
            f"./{binary}  {scene}  config/config.txt  input/src/src.txt  input/rec/rec.txt")

    def _run_solver(self):
        if not self._proj_root:
            QMessageBox.warning(self,"No project","Open project first."); return
        if not self._obj_path:
            QMessageBox.warning(self,"No scene","Load an OBJ first."); return
        self._write_config(os.path.join(self._proj_root,"config","config.txt"))
        self._src_tbl.save_file(os.path.join(self._proj_root,"input","src","src.txt"))
        self._rec_tbl.save_file(os.path.join(self._proj_root,"input","rec","rec.txt"))
        for ed in self._mat_editors.values(): ed.save()
        scene = self._obj_path
        if scene.startswith(self._proj_root):
            scene = os.path.relpath(scene, self._proj_root)
        cmd = [f"./{self._bin_le.text()}", scene,
               "config/config.txt","input/src/src.txt","input/rec/rec.txt"]
        self._log.clear()
        self._log.append(f"$ cd {self._proj_root}\n$ " + " ".join(cmd))
        self._log.append("-"*60)
        self._run_btn.setEnabled(False); self._stop_btn.setEnabled(True)
        self._tabs.setCurrentIndex(4)
        self._solver_thread = SolverThread(cmd, self._proj_root)
        self._solver_thread.output.connect(self._log.append)
        self._solver_thread.finished.connect(self._on_solver_done)
        self._solver_thread.start()

    def _stop_solver(self):
        if self._solver_thread and self._solver_thread.isRunning():
            self._solver_thread.terminate()
            self._log.append("\n[!] Stopped by user.")

    def _on_solver_done(self, rc):
        self._run_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        c = PAL["ok"] if rc == 0 else PAL["err"]
        self._log.append(
            f'<span style="color:{c};">{"[OK]" if rc==0 else "[FAIL]"}  rc={rc}</span>')
        self._sb.showMessage(f"Solver finished  (rc={rc})")


# ─────────────────────────────────────────────────────────────
def main():
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24); fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    app.setApplicationName("AcousticFVM GUI")
    app.setFont(QFont("Segoe UI",10) if sys.platform=="win32" else
                QFont("SF Pro Text",10) if sys.platform=="darwin" else
                QFont("Ubuntu",10))
    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
