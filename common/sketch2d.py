#!/usr/bin/env python3
"""2D domain sketcher — increment 1: draw/edit polygon boundaries in the xy-plane.

Opens as its own window from the GUI's Draw mode. LATER increments add: extrude -> OBJ (read back by
gui.py's existing _load_obj), overlapping-shape pruning to the outer boundary, and a 3D surface-naming
mode. No OBJ I/O yet — this step is purely the editable 2D sketch.

Controls (checked against gui.py: Ctrl+K is unused there; the rest are window-scoped so no clash):
  Ctrl+LMB   add a polygon point to the active shape (starts a new shape if none is open)
  LMB click  select the nearest vertex (edit x/y/z in the side panel)
  LMB drag   on a vertex: move it in xy;  inside a closed shape: drag the whole shape in xy
  Ctrl+RMB   remove the nearest vertex
  Ctrl+K     close the active shape (needs > 2 points)
  Ctrl+Z     undo        MMB drag  pan        wheel  zoom
"""
import copy
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtWidgets import (QWidget, QDoubleSpinBox, QLabel, QVBoxLayout, QHBoxLayout, QFormLayout)

HIT_PX = 10          # vertex pick radius, screen pixels
EDGE_HIT_PX = 28     # edge pick radius for Ctrl+LMB insert (looser: no need to be near the edge)
# A polygon point is [x, y, z_bottom, z_top]: one dot in the xy top-view, two stacked heights. Extrude
# builds the bottom ring from z_bottom and the top ring from z_top, so floors AND ceilings can slope.


def _point_in_poly(x, y, pts):
    """Even-odd ray cast; pts = [[x,y,...], ...]. True if (x,y) is inside the polygon."""
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i][0], pts[i][1]
        x2, y2 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def _nearest_point(shapes, wx, wy, tol):
    """(shape_idx, pt_idx) of the closest vertex within tol world units, or None."""
    best, bd = None, tol * tol
    for si, sh in enumerate(shapes):
        for pi, p in enumerate(sh["pts"]):
            d = (p[0] - wx) ** 2 + (p[1] - wy) ** 2
            if d <= bd:
                bd, best = d, (si, pi)
    return best


def _nearest_edge(shapes, wx, wy, tol):
    """(shape_idx, insert_index, z_bottom, z_top) of the closest edge within tol, or None. z's are
    interpolated at the projection; the caller places the new point's xy at the CLICK, so the edge
    bends toward it. Includes the closing edge of closed shapes."""
    best, bd = None, tol * tol
    for si, sh in enumerate(shapes):
        pts, n = sh["pts"], len(sh["pts"])
        m = n if sh["closed"] else n - 1
        for i in range(m):
            a, b = pts[i], pts[(i + 1) % n]
            dx, dy = b[0] - a[0], b[1] - a[1]
            L2 = dx * dx + dy * dy
            if L2 == 0:
                continue
            t = max(0.0, min(1.0, ((wx - a[0]) * dx + (wy - a[1]) * dy) / L2))
            px, py = a[0] + t * dx, a[1] + t * dy
            d = (px - wx) ** 2 + (py - wy) ** 2
            if d <= bd:
                bd = d
                best = (si, i + 1, a[2] + t * (b[2] - a[2]), a[3] + t * (b[3] - a[3]))
    return best


def _signed_area(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i][0], pts[i][1]
        x2, y2 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def _triangulate_xy(pts):
    """Ear-clip the polygon's xy projection -> [(i,j,k), ...] CCW index triples. Handles concave
    (L-shapes) that a fan would tear. ponytail: O(n^2) ear clip; faces have a handful of verts."""
    n = len(pts)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]
    xy = [(p[0], p[1]) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def in_tri(p, a, b, c):
        d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
        return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))

    area = sum(xy[i][0] * xy[(i + 1) % n][1] - xy[(i + 1) % n][0] * xy[i][1] for i in range(n))
    V = list(range(n))
    if area < 0:                                     # orient CCW so ears have positive cross
        V.reverse()
    tris, guard = [], 0
    while len(V) > 3 and guard < 10 * n:
        guard += 1
        m, clipped = len(V), False
        for k in range(m):
            i0, i1, i2 = V[(k - 1) % m], V[k], V[(k + 1) % m]
            a, b, c = xy[i0], xy[i1], xy[i2]
            if cross(a, b, c) <= 0:                  # reflex/degenerate: not an ear
                continue
            if any(j not in (i0, i1, i2) and in_tri(xy[j], a, b, c) for j in V):
                continue                             # another vertex inside: not an ear
            tris.append((i0, i1, i2)); del V[k]; clipped = True; break
        if not clipped:
            break
    if len(V) == 3:
        tris.append((V[0], V[1], V[2]))
    return tris


def extrude_obj(shapes):
    """Closed prisms -> OBJ text (floor + ceiling + one group per wall so 3D-naming can rename them
    later). Bottom ring = each corner's z_bottom, top ring = z_top. Returns None if nothing closed.
    Winding gives INWARD normals assuming a room; ponytail: flip walls if the solver treats them 1-sided."""
    closed = [s for s in shapes if s["closed"] and len(s["pts"]) >= 3]
    if not closed:
        return None
    multi = len(closed) > 1
    vlines, flines = [], []
    base = 1                                             # OBJ vertices are 1-indexed
    for si, sh in enumerate(closed):
        pts = sh["pts"] if _signed_area(sh["pts"]) >= 0 else list(reversed(sh["pts"]))  # force CCW
        n = len(pts)
        for p in pts:
            vlines.append(f"v {p[0]:.6g} {p[1]:.6g} {p[2]:.6g}")  # bottom ring (z_bottom)
        for p in pts:
            vlines.append(f"v {p[0]:.6g} {p[1]:.6g} {p[3]:.6g}")  # top ring (z_top)
        b, t = base, base + n
        sfx = f"_{si}" if multi else ""
        tris = _triangulate_xy(pts)                                       # concave-safe, per-triangle
        for k, (i0, i1, i2) in enumerate(tris):                          # floor: CCW -> normal +z
            flines.append(f"g floor{sfx}_{k + 1}")
            flines.append(f"f {b + i0} {b + i1} {b + i2}")
        for k, (i0, i1, i2) in enumerate(tris):                          # ceiling: reversed -> -z
            flines.append(f"g ceiling{sfx}_{k + 1}")
            flines.append(f"f {t + i0} {t + i2} {t + i1}")
        for i in range(n):
            j = (i + 1) % n
            flines.append(f"g wall{sfx}_{i + 1}")
            flines.append(f"f {b + i} {t + i} {t + j} {b + j}")                      # inward
        base += 2 * n
    return "\n".join(["# generated by sketch2d"] + vlines + flines) + "\n"


class _Canvas(QWidget):
    selectionChanged = Signal()
    objReady = Signal(str)          # extruded OBJ text -> gui.py writes + loads it

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 400)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.shapes = []             # [{"pts": [[x,y,z], ...], "closed": bool}]
        self.sel = None              # (shape_idx, pt_idx) or None
        self._undo = []              # snapshots (shapes, sel), pushed BEFORE each mutation
        self._drag = None            # ("pt", si, pi) | ("shape", si) | ("pan",) | None
        self._last = None            # last mouse pos (screen)
        self.scale = 40.0            # px per metre
        self.off = QPointF(260.0, 360.0)   # world origin on screen

    # world <-> screen (y up)
    def _w2s(self, x, y):
        return QPointF(self.off.x() + x * self.scale, self.off.y() - y * self.scale)

    def _s2w(self, sx, sy):
        return (sx - self.off.x()) / self.scale, (self.off.y() - sy) / self.scale

    def _snap(self):
        # ponytail: snapshot on every mutation/drag-start; a pure click may push a no-op undo. Dedupe
        # against the current state so a redundant one never stacks. Cap the history.
        snap = copy.deepcopy(self.shapes)
        if not (self._undo and self._undo[-1][0] == snap):
            self._undo.append((snap, self.sel))
            self._undo = self._undo[-100:]

    def _active(self):
        """Index of the open (unclosed) shape to append to, else None."""
        for i in range(len(self.shapes) - 1, -1, -1):
            if not self.shapes[i]["closed"]:
                return i
        return None

    # ── input ──────────────────────────────────────────────
    def mousePressEvent(self, e):
        self._last = e.position()
        wx, wy = self._s2w(e.position().x(), e.position().y())
        ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
        tol = HIT_PX / self.scale
        b = e.button()
        if b == Qt.MouseButton.LeftButton and ctrl:                 # insert on nearest edge (loose tol)
            hit = _nearest_edge(self.shapes, wx, wy, EDGE_HIT_PX / self.scale)
            if hit:
                self._snap()
                si, idx, zb, zt = hit                               # xy at the CLICK, z's from the edge
                self.shapes[si]["pts"].insert(idx, [round(wx, 4), round(wy, 4), round(zb, 4), round(zt, 4)])
                self.sel = (si, idx); self.selectionChanged.emit(); self.update()
            return
        if b == Qt.MouseButton.RightButton and ctrl:                # remove nearest point
            hit = _nearest_point(self.shapes, wx, wy, tol)
            if hit:
                self._snap()
                si, pi = hit
                del self.shapes[si]["pts"][pi]
                if len(self.shapes[si]["pts"]) < 2:                 # no lonely point/edge left behind
                    del self.shapes[si]
                self.sel = None; self.selectionChanged.emit(); self.update()
            return
        if b == Qt.MouseButton.LeftButton:                          # vertex drag / add point / shape drag
            hit = _nearest_point(self.shapes, wx, wy, tol)
            if hit:                                                 # on a vertex -> select + drag it
                self.sel = hit; self._snap(); self._drag = ("pt", hit[0], hit[1])
                self.selectionChanged.emit()
            elif self._active() is not None:                        # drawing an open shape -> append
                self._snap(); si = self._active()
                self.shapes[si]["pts"].append([round(wx, 4), round(wy, 4), 0.0, 0.0])
                self.sel = (si, len(self.shapes[si]["pts"]) - 1); self.selectionChanged.emit()
            else:
                for si, sh in enumerate(self.shapes):               # inside a closed shape -> drag it
                    if sh["closed"] and _point_in_poly(wx, wy, sh["pts"]):
                        self._snap(); self._drag = ("shape", si); break
                else:                                               # empty -> start a new shape
                    self._snap()
                    self.shapes.append({"pts": [[round(wx, 4), round(wy, 4), 0.0, 0.0]], "closed": False})
                    self.sel = (len(self.shapes) - 1, 0); self.selectionChanged.emit()
            self.update(); return
        if b == Qt.MouseButton.MiddleButton:                        # pan
            self._drag = ("pan",)

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        p = e.position()
        if self._drag[0] == "pan":
            self.off += p - self._last; self._last = p; self.update(); return
        dx = (p.x() - self._last.x()) / self.scale
        dy = -(p.y() - self._last.y()) / self.scale
        if self._drag[0] == "pt":
            _, si, pi = self._drag
            pt = self.shapes[si]["pts"][pi]
            pt[0] = round(pt[0] + dx, 4); pt[1] = round(pt[1] + dy, 4)
            self.selectionChanged.emit()
        elif self._drag[0] == "shape":
            for pt in self.shapes[self._drag[1]]["pts"]:
                pt[0] = round(pt[0] + dx, 4); pt[1] = round(pt[1] + dy, 4)
        self._last = p; self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def contextMenuEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            return                                              # Ctrl+RMB = delete point (mousePress)
        if not any(s["closed"] and len(s["pts"]) >= 3 for s in self.shapes):
            return
        from PySide6.QtWidgets import QMenu, QInputDialog
        m = QMenu(self); act = m.addAction("Extrude to OBJ…")
        if m.exec(e.globalPos()) is act:
            h, ok = QInputDialog.getDouble(self, "Extrude", "Domain height (m):", 3.0, 0.01, 1e4, 3)
            if ok:
                self._snap()
                for sh in self.shapes:                              # raise only flat corners; keep any
                    for p in sh["pts"]:                             # per-corner z_top the user edited
                        if p[3] <= p[2]:
                            p[3] = round(p[2] + h, 4)
                txt = extrude_obj(self.shapes)
                if txt:
                    self.objReady.emit(txt)
                self.selectionChanged.emit(); self.update()

    def wheelEvent(self, e):
        p = e.position()
        wx, wy = self._s2w(p.x(), p.y())
        self.scale *= 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.off = QPointF(p.x() - wx * self.scale, p.y() + wy * self.scale)  # zoom about the cursor
        self.update()

    def keyPressEvent(self, e):
        ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if ctrl and e.key() == Qt.Key.Key_K:                        # close the active shape
            si = self._active()
            if si is not None and len(self.shapes[si]["pts"]) > 2:
                self._snap(); self.shapes[si]["closed"] = True; self.update()
            return
        if ctrl and e.key() == Qt.Key.Key_Z:                        # undo
            if self._undo:
                self.shapes, self.sel = self._undo.pop()
                self.selectionChanged.emit(); self.update()
            return
        super().keyPressEvent(e)

    # ── paint ──────────────────────────────────────────────
    def paintEvent(self, _):
        qp = QPainter(self); qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.fillRect(self.rect(), QColor("#1a1d23"))
        qp.setPen(QPen(QColor("#2e3340"), 1))                       # x/y axes at the world origin
        o = self.off
        qp.drawLine(0, int(o.y()), self.width(), int(o.y()))
        qp.drawLine(int(o.x()), 0, int(o.x()), self.height())
        for si, sh in enumerate(self.shapes):
            sp = [self._w2s(p[0], p[1]) for p in sh["pts"]]
            qp.setPen(QPen(QColor("#7c5cfc"), 2))
            for i in range(len(sp) - 1):
                qp.drawLine(sp[i], sp[i + 1])
            if sh["closed"] and len(sp) > 1:
                qp.drawLine(sp[-1], sp[0])
            for pi, s in enumerate(sp):
                selp = self.sel == (si, pi)
                qp.setPen(Qt.PenStyle.NoPen)
                qp.setBrush(QBrush(QColor("#0ff091") if selp else QColor("#cfd8dc")))
                r = 5 if selp else 4
                qp.drawEllipse(s, r, r)


class SketchWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Domain sketch — 2D")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(900, 620)
        self.canvas = _Canvas(self)
        self.canvas.selectionChanged.connect(self._sync)
        self._sx, self._sy, self._szb, self._szt = (self._spin() for _ in range(4))
        for s in (self._sx, self._sy, self._szb, self._szt):
            s.valueChanged.connect(self._apply)
        form = QFormLayout()
        form.addRow("x (m)", self._sx); form.addRow("y (m)", self._sy)
        form.addRow("z bottom", self._szb); form.addRow("z top", self._szt)
        panel = QWidget(); panel.setLayout(form); panel.setFixedWidth(160)
        hint = QLabel("LMB add · LMB drag vertex/shape · Ctrl+LMB insert on edge · Ctrl+RMB del · "
                      "Ctrl+K close · Ctrl+Z undo · RMB extrude · MMB pan · wheel zoom")
        hint.setWordWrap(True); hint.setStyleSheet("color:#8a94a6;font-size:11px;")
        side = QVBoxLayout()
        side.addWidget(QLabel("Selected point")); side.addWidget(panel)
        side.addWidget(hint); side.addStretch(1)
        root = QHBoxLayout(self); root.addWidget(self.canvas, 1); root.addLayout(side)
        self._sync()

    def _spin(self):
        s = QDoubleSpinBox(); s.setRange(-1e4, 1e4); s.setDecimals(3); s.setSingleStep(0.1)
        s.setEnabled(False); return s

    def showEvent(self, e):
        super().showEvent(e); self.canvas.setFocus()

    def _sync(self):
        sel = self.canvas.sel
        on = sel is not None
        for s in (self._sx, self._sy, self._szb, self._szt):
            s.blockSignals(True); s.setEnabled(on)
        if on:
            p = self.canvas.shapes[sel[0]]["pts"][sel[1]]
            self._sx.setValue(p[0]); self._sy.setValue(p[1])
            self._szb.setValue(p[2]); self._szt.setValue(p[3])
        for s in (self._sx, self._sy, self._szb, self._szt):
            s.blockSignals(False)

    def _apply(self, *_):        # valueChanged passes a float; absorb it (else TypeError, edit ignored)
        sel = self.canvas.sel
        if sel is None:
            return
        self.canvas.shapes[sel[0]]["pts"][sel[1]] = [self._sx.value(), self._sy.value(),
                                                     self._szb.value(), self._szt.value()]
        self.canvas.update()


def _selfcheck():
    sq = [[0, 0, 0, 3], [2, 0, 0, 3], [2, 2, 0, 3], [0, 2, 0, 3]]   # [x,y,zb,zt]
    assert _point_in_poly(1, 1, sq) and not _point_in_poly(3, 1, sq), "point-in-poly"
    shapes = [{"pts": sq, "closed": True}]
    assert _nearest_point(shapes, 0.05, 0.05, 0.2) == (0, 0), "nearest hit"
    assert _nearest_point(shapes, 5, 5, 0.2) is None, "nearest miss"
    ed = _nearest_edge(shapes, 1.0, 0.03, 0.2)          # near bottom edge -> insert at 1, zb~0 zt~3
    assert ed and ed[:2] == (0, 1) and abs(ed[2]) < 0.05 and abs(ed[3] - 3) < 0.05, "nearest edge"
    assert _nearest_edge(shapes, 5, 5, 0.2) is None, "edge miss"
    obj = extrude_obj(shapes)
    assert sum(l.startswith("v ") for l in obj.splitlines()) == 8, "8 verts (2x4)"
    assert " 3\n" in obj or " 3 " in obj, "top ring uses z_top"
    assert obj.count("g wall") == 4 and obj.count("g floor") == 2 and obj.count("g ceiling") == 2, "groups"
    assert extrude_obj([{"pts": sq, "closed": False}]) is None, "open -> no obj"
    L = [[0, 0], [4, 0], [4, 4], [2, 4], [2, 2], [0, 2]]        # concave L: n-2 = 4 triangles
    assert len(_triangulate_xy(L)) == 4, "ear-clip concave L"
    print("sketch2d self-check OK")


if __name__ == "__main__":
    _selfcheck()
