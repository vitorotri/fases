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
    """(shape_idx, idx) of the closest boundary vertex within tol, or None."""
    best, bd = None, tol * tol
    for si, sh in enumerate(shapes):
        for i, p in enumerate(sh["pts"]):
            d = (p[0] - wx) ** 2 + (p[1] - wy) ** 2
            if d <= bd:
                bd, best = d, (si, i)
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
    xy = [(p[0], p[1]) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def in_tri(p, a, b, c):
        d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
        return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))

    if n < 3:
        tris = []
    elif n == 3:
        tris = [(0, 1, 2)]
    else:
        area = sum(xy[i][0] * xy[(i + 1) % n][1] - xy[(i + 1) % n][0] * xy[i][1] for i in range(n))
        V = list(range(n))
        if area < 0:                                 # orient CCW so ears have positive cross
            V.reverse()
        tris, guard = [], 0
        while len(V) > 3 and guard < 10 * n:
            guard += 1
            m, clipped = len(V), False
            for k in range(m):
                i0, i1, i2 = V[(k - 1) % m], V[k], V[(k + 1) % m]
                a, b, c = xy[i0], xy[i1], xy[i2]
                if cross(a, b, c) <= 0:              # reflex/degenerate: not an ear
                    continue
                if any(j not in (i0, i1, i2) and in_tri(xy[j], a, b, c) for j in V):
                    continue                         # another vertex inside: not an ear
                tris.append((i0, i1, i2)); del V[k]; clipped = True; break
            if not clipped:
                break
        if len(V) == 3:
            tris.append((V[0], V[1], V[2]))
    return tris


def _poly_inside(inner, outer):
    """True if `inner` sits fully inside `outer` (ALL inner vertices inside). Auto-detects nested
    sub-regions without the user tagging parent/child. Centroid alone is wrong: a big outer's centroid
    can land in a small sub that overlaps the middle."""
    return all(_point_in_poly(p[0], p[1], outer) for p in inner)


def _tri_with_holes(verts_xy, ring_ends):
    """Triangulate an outer ring with holes via mapbox_earcut (ISC). verts_xy = [(x,y),...] outer ring
    then each hole ring; ring_ends = cumulative end index per ring (first = outer). -> [(i,j,k),...]
    ponytail: earcut earns the dep here — robust polygon-with-holes triangulation is a hand-roll trap."""
    import numpy as np
    import mapbox_earcut as earcut
    idx = earcut.triangulate_float64(np.asarray(verts_xy, np.float64),
                                     np.asarray(ring_ends, np.uint32))
    return [(int(idx[i]), int(idx[i + 1]), int(idx[i + 2])) for i in range(0, len(idx), 3)]


def _delaunay_region(boundary, interior, zi):
    """Delaunay-triangulate boundary + interior points in xy; z from column `zi` (3=z_top, 2=z_bottom).
    Triangles whose centroid falls OUTSIDE the boundary polygon are dropped (concave outlines). Interior
    lonely points / open lines become Steiner points wired to the border by the Delaunay net -> a shaped
    ceiling (ridge, peak, valley). Returns (verts3, tris). ponytail: unconstrained Delaunay — a drawn
    interior LINE is honoured as its vertices, not a forced edge; add the `triangle` CDT lib if a segment
    must be guaranteed. Needs scipy (already a pipeline dep)."""
    import numpy as np
    from scipy.spatial import Delaunay
    pts = list(boundary) + list(interior)
    xy = np.array([[p[0], p[1]] for p in pts], np.float64)
    verts3 = [(p[0], p[1], p[zi]) for p in pts]
    tris = []
    for a, b, c in Delaunay(xy).simplices:
        cx = (xy[a, 0] + xy[b, 0] + xy[c, 0]) / 3.0
        cy = (xy[a, 1] + xy[b, 1] + xy[c, 1]) / 3.0
        if _point_in_poly(cx, cy, boundary):
            tris.append((int(a), int(b), int(c)))
    return verts3, tris


def extrude_obj(shapes):
    """Closed shapes -> room OBJ. A shape inside another = a raised/lowered SUB-REGION: the ceiling
    ring between outer and inner SLOPES from the outer z_top down/up to the inner z_top (a cellar/hip
    roof), with a flat cap over the inner region — no vertical column. Floor is the full outer at
    z_bottom (shared; sub floor-steps later). Coplanar floor/ceiling = ONE face; only non-planar
    (sloped) surfaces are triangulated (per-triangle groups). None if nothing closed.
    Winding is irrelevant to the solver (verified): FVM signs by +X parity ray (parsing3D.h), ISM
    canonicalizes normals (ism_canon_normal) + plane reflection, RT skips normal orientation
    (raytrace3D.h:58) and flips N to face the ray. So face order here never needs flipping."""
    closed = [s for s in shapes if s["closed"] and len(s["pts"]) >= 3]
    if not closed:
        return None
    outers = [o for o in closed
              if not any(s is not o and _poly_inside(o["pts"], s["pts"]) for s in closed)]
    V, F = [], []                                        # V: "x y z" lines; F: (group_name, [idx...])

    def vert(x, y, z):
        V.append(f"v {x:.6g} {y:.6g} {z:.6g}"); return len(V)   # 1-based OBJ index

    def ccw(pts):
        return pts if _signed_area(pts) >= 0 else list(reversed(pts))

    def emit_poly(name, pts3, reverse):
        """Coplanar polygon -> ONE face; non-planar -> per-triangle. The GUI's merge_coplanar_groups
        fuses coplanar triangles/facets into single layers at load, so sketch2d stays dumb here."""
        zs = [p[2] for p in pts3]
        ring = [vert(p[0], p[1], p[2]) for p in pts3]
        if max(zs) - min(zs) < 1e-6:
            F.append((name, ring[::-1] if reverse else ring))
        else:
            for k, (a, b, c) in enumerate(_triangulate_xy(pts3)):
                tri = [ring[a], ring[c], ring[b]] if reverse else [ring[a], ring[b], ring[c]]
                F.append((f"{name}_{k + 1}", tri))

    multi = len(outers) > 1
    for oi, O in enumerate(outers):
        ob = ccw(O["pts"])
        subs = [ccw(s["pts"]) for s in closed if s is not O and _poly_inside(s["pts"], O["pts"])]
        # open shapes inside this outer = interior ceiling controls (lonely points / open lines)
        interior = [p for s in shapes if not s["closed"] and _poly_inside(s["pts"], O["pts"])
                    for p in s["pts"]]
        osfx = f"_{oi}" if multi else ""

        emit_poly(f"floor{osfx}", [(p[0], p[1], p[2]) for p in ob], reverse=False)   # full floor @ z_b
        fb = [vert(p[0], p[1], p[2]) for p in ob]        # rings for the walls
        ft = [vert(p[0], p[1], p[3]) for p in ob]
        nb = len(ob)
        for i in range(nb):
            j = (i + 1) % nb
            F.append((f"wall{osfx}_{i + 1}", [fb[i], ft[i], ft[j], fb[j]]))

        if subs:
            # ceiling = the ring BETWEEN outer and inner, SLOPED (outer verts @ outer z_top, inner verts
            # @ the sub's z_top) -> a cellar/hip roof, not a vertical column. earcut fills the annulus.
            rxy = [(p[0], p[1]) for p in ob]; rz = [p[3] for p in ob]; ends = [nb]
            for sb in subs:
                for p in sb:
                    rxy.append((p[0], p[1])); rz.append(p[3])    # inner ring at the SUB height -> slope
                ends.append(len(rxy))
            av = [vert(rxy[i][0], rxy[i][1], rz[i]) for i in range(len(rxy))]
            for k, (a, b, c) in enumerate(_tri_with_holes(rxy, ends)):   # gui merges coplanar facets
                F.append((f"ceiling{osfx}_{k + 1}", [av[a], av[c], av[b]]))   # reversed -> faces down
            for sj, sb in enumerate(subs):                       # flat cap over each inner region
                emit_poly(f"ceiling_region{osfx}_r{sj + 1}", [(p[0], p[1], p[3]) for p in sb], reverse=True)
            # ponytail: interior ceiling points ignored when sub-regions present — sub annulus owns the roof
        elif interior:
            cv, ct = _delaunay_region(ob, interior, 3)           # ceiling shaped by interior pts/lines
            cverts = [vert(x, y, z) for x, y, z in cv]
            for k, (a, b, c) in enumerate(ct):
                F.append((f"ceiling{osfx}_{k + 1}", [cverts[a], cverts[c], cverts[b]]))   # faces down
        else:
            emit_poly(f"ceiling{osfx}", [(p[0], p[1], p[3]) for p in ob], reverse=True)

    lines = ["# generated by sketch2d"] + V
    for g, idx in F:
        lines.append(f"g {g}"); lines.append("f " + " ".join(map(str, idx)))
    return "\n".join(lines) + "\n"


def _drop_out_of_range(shapes):
    """Remove vertices flung far past the scene — corrupt saves clamp coords to the spinbox +-1e4 max,
    which then extrudes a wall shooting to 10000 and wrecks the contour. Bound = 50x the median extent
    (min 100 m), so a legitimately large room survives. Returns (cleaned_shapes, n_dropped); shapes left
    with too few points (closed<3, open<1) are removed."""
    ext = sorted(max(abs(p[0]), abs(p[1])) for sh in shapes for p in sh["pts"])
    if not ext:
        return shapes, 0
    bound = max(100.0, 50.0 * ext[len(ext) // 2])                  # median -> robust to <50% junk
    out, dropped = [], 0
    for sh in shapes:
        keep = [p for p in sh["pts"] if abs(p[0]) <= bound and abs(p[1]) <= bound]
        dropped += len(sh["pts"]) - len(keep)
        if len(keep) >= (3 if sh.get("closed") else 1):
            sh = dict(sh); sh["pts"] = keep; out.append(sh)
    return out, dropped


class _Canvas(QWidget):
    selectionChanged = Signal()
    objReady = Signal(str)          # extruded OBJ text -> gui.py writes + loads it

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 400)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)  # hover coords without a button held
        self._hover = None           # world (x,y) under cursor, for the readout
        self._prop_foot = []         # read-only top-down prop outlines (from gui), [[(x,y),...], ...]
        self.shapes = []             # [{"pts": [[x,y,z], ...], "closed": bool}]
        self.sel = None              # (shape_idx, pt_idx) or None
        self._undo = []              # snapshots (shapes, sel), pushed BEFORE each mutation
        self._drag = None            # ("pt", si, idx) | ("shape", si) | ("pan",) | None
        self._last = None            # last mouse pos (screen)
        self.scale = 40.0            # px per metre
        self.off = QPointF(260.0, 360.0)   # world origin on screen
        self.h = 0.5                 # grid spacing [m]; gui sets it from _grid_h. Fine line per h,
        #                              thick per 10*h. Scene is drawn in the POSITIVE quadrant only
        #                              (parser shifts to positive anyway — draw there from the start).

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
            sh = self.shapes[i]
            if not sh["closed"] and not sh.get("done"):      # finished interior lines are skipped
                return i
        return None

    def dump_shapes(self):
        """JSON-able copy of the sketch (for the .json sidecar so it can be re-edited later)."""
        return copy.deepcopy(self.shapes)

    def set_prop_footprints(self, polys):
        """Read-only top-down outlines of the 3D props (from gui), drawn under the sketch."""
        self._prop_foot = polys or []
        self.update()

    def load_shapes(self, data):
        data, dropped = _drop_out_of_range(data or [])            # self-heal a corrupt save (see helper)
        if dropped:
            print(f"[sketch2d] dropped {dropped} out-of-range vertex/vertices from a corrupt sketch — "
                  "contour repaired. Re-save (extrude / leave draw mode) to persist the repair.")
        self.shapes = data
        self.sel = None; self._undo = []
        self.selectionChanged.emit(); self.update()

    # ── input ──────────────────────────────────────────────
    def mousePressEvent(self, e):
        self._last = e.position()
        wx, wy = self._s2w(e.position().x(), e.position().y())
        wx, wy = max(0.0, wx), max(0.0, wy)                         # positive quadrant only
        ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
        alt = bool(e.modifiers() & Qt.KeyboardModifier.AltModifier)
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
                si, idx = hit
                del self.shapes[si]["pts"][idx]
                if len(self.shapes[si]["pts"]) < 2:               # no lonely boundary point left
                    del self.shapes[si]
                self.sel = None; self.selectionChanged.emit(); self.update()
            return
        if b == Qt.MouseButton.LeftButton:                          # vertex drag / add point / shape drag
            hit = _nearest_point(self.shapes, wx, wy, tol)
            if hit:                                                 # on a point -> select + drag it
                self.sel = hit; self._snap(); self._drag = ("pt",) + hit
                self.selectionChanged.emit()
            elif self._active() is not None:                        # drawing an open shape -> append
                self._snap(); si = self._active()
                self.shapes[si]["pts"].append([round(wx, 4), round(wy, 4), 0.0, 0.0])
                self.sel = (si, len(self.shapes[si]["pts"]) - 1); self.selectionChanged.emit()
            else:
                inside = next((si for si, sh in enumerate(self.shapes)
                               if sh["closed"] and _point_in_poly(wx, wy, sh["pts"])), None)
                if inside is not None and alt:                      # Alt+LMB inside -> drag the shape
                    self._snap(); self._drag = ("shape", inside)
                else:                                               # start a new shape; LMB inside a
                    self._snap()                                    # closed one begins a nested sub-region
                    self.shapes.append({"pts": [[round(wx, 4), round(wy, 4), 0.0, 0.0]], "closed": False})
                    self.sel = (len(self.shapes) - 1, 0); self.selectionChanged.emit()
            self.update(); return
        if b == Qt.MouseButton.MiddleButton:                        # pan
            self._drag = ("pan",)

    def mouseMoveEvent(self, e):
        p = e.position()
        self._hover = self._s2w(p.x(), p.y())                       # live cursor coords (raw, may be <0)
        if self._drag is None:
            self.update(); return
        if self._drag[0] == "pan":
            self.off += p - self._last; self._last = p; self.update(); return
        dx = (p.x() - self._last.x()) / self.scale
        dy = -(p.y() - self._last.y()) / self.scale
        if self._drag[0] == "pt":
            _, si, idx = self._drag
            pt = self.shapes[si]["pts"][idx]
            pt[0] = round(max(0.0, pt[0] + dx), 4); pt[1] = round(max(0.0, pt[1] + dy), 4)  # positive only
            self.selectionChanged.emit()
        elif self._drag[0] == "shape":
            sh = self.shapes[self._drag[1]]
            dx = max(dx, -min(p[0] for p in sh["pts"]))             # clamp translation: keep whole shape
            dy = max(dy, -min(p[1] for p in sh["pts"]))             # in the positive quadrant (rigid)
            for pt in sh["pts"]:                                    # move the whole shape in xy
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
        m = QMenu(self)
        act_ex = m.addAction("Extrude to OBJ… (set height)")
        act_up = m.addAction("Update OBJ")                          # rebuild from current z's, no prompt
        chosen = m.exec(e.globalPos())
        if chosen is act_ex:
            h, ok = QInputDialog.getDouble(self, "Extrude", "Domain height (m):", 3.0, 0.01, 1e4, 3)
            if ok:
                self._snap()
                for sh in self.shapes:                              # raise only flat corners; keep any
                    for p in sh["pts"]:                             # per-corner z_top the user edited
                        if p[3] <= p[2]:
                            p[3] = round(p[2] + h, 4)
                self._emit_obj()
        elif chosen is act_up:
            self._emit_obj()

    def _emit_obj(self):
        from PySide6.QtWidgets import QMessageBox
        try:
            txt = extrude_obj(self.shapes)
        except ImportError:                              # sub-regions need earcut
            QMessageBox.warning(self, "Missing dependency",
                                "Nested sub-regions / interior ceilings need mapbox_earcut + scipy:\n\n"
                                "    pip install mapbox_earcut scipy")
            return
        except Exception as ex:                          # surface any geometry error instead of silence
            QMessageBox.warning(self, "Extrude failed", f"{type(ex).__name__}: {ex}")
            return
        if txt:
            self.objReady.emit(txt)
        self.selectionChanged.emit(); self.update()

    def wheelEvent(self, e):
        p = e.position()
        wx, wy = self._s2w(p.x(), p.y())
        self.scale *= 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale = min(max(self.scale, 0.2), 5000.0)             # floor stops a tiny-scale drag/click
        #                                                            from flinging points to ~10000 (source
        #                                                            of the corruption the sanitizer cleans)
        self.off = QPointF(p.x() - wx * self.scale, p.y() + wy * self.scale)  # zoom about the cursor
        self.update()

    def keyPressEvent(self, e):
        ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if ctrl and e.key() == Qt.Key.Key_K:                        # close the active shape
            si = self._active()
            if si is not None and len(self.shapes[si]["pts"]) > 2:
                self._snap(); self.shapes[si]["closed"] = True; self.update()
            return
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):       # finish open line -> interior ceiling ctrl
            si = self._active()
            if si is not None and self.shapes[si]["pts"]:
                self._snap(); self.shapes[si]["done"] = True; self.update()
            return
        if ctrl and e.key() == Qt.Key.Key_Z:                        # undo
            if self._undo:
                self.shapes, self.sel = self._undo.pop()
                self.selectionChanged.emit(); self.update()
            return
        super().keyPressEvent(e)

    def fit_view(self):
        """Center the origin and pick a scale so the drawn scene fits with margin (default scale if
        empty). Called on show so opening the sketcher always frames the scene."""
        W, Hpx = self.width(), self.height()
        if W <= 0 or Hpx <= 0:
            return                                                 # not laid out yet; showEvent can fire early
        self.off = QPointF(W / 2.0, Hpx / 2.0)                     # origin in the center
        ext = sorted(max(abs(p[0]), abs(p[1])) for sh in self.shapes for p in sh["pts"])
        if ext:
            mx = ext[int(0.9 * (len(ext) - 1))]                   # 90th pct: frame the DRAWING, shrug off
            if mx > 0:                                            # a few stray far points (corrupt saves)
                self.scale = 0.42 * min(W, Hpx) / mx             # scene spans ~84% of the half-extent
        self.scale = min(max(self.scale, 0.2), 5000.0)            # never a degenerate zoom
        self.update()

    def showEvent(self, e):
        super().showEvent(e); self.fit_view()

    # ── paint ──────────────────────────────────────────────
    def paintEvent(self, _):
        qp = QPainter(self)                                        # end() in finally: a paint exception must
        try:                                                       # never leave an active painter (that
            self._paint(qp)                                        # hard-closes the whole app)
        finally:
            qp.end()

    def _paint(self, qp):
        o = self.off
        W, Hpx = self.width(), self.height()
        qp.fillRect(self.rect(), QColor("#1a1d23"))
        h = self.h
        # COARSEN the iteration unit so drawn lines stay >=6px apart. A fine solver h (mm), or a tiny
        # scale from a stray far point, would make h*scale sub-pixel -> the loop runs W/step (tens of
        # thousands) times PER paint -> freeze. Cap the coarsening; if it can't reach 6px, SKIP the grid.
        unit = h; steps = 0
        while h > 0 and unit * self.scale < 6.0 and steps < 24:    # 1*h, 10*h, 100*h... until visible
            unit *= 10; steps += 1
        if h > 0 and self.scale > 0 and unit * self.scale >= 6.0:  # only draw when lines are visibly spaced
            ratio = int(round(unit / h))                           # 1 => per-h (thick every 10th); >=10 => all 10h+
            sp = unit * self.scale                                 # px between drawn lines (>=6) -> loop <= W/6
            i0 = int((0 - o.x()) // sp); i1 = int((W - o.x()) // sp) + 1   # // floors (both signs)
            j0 = int((o.y() - Hpx) // sp); j1 = int(o.y() // sp) + 1

            def grid_pass(fine, thick, axis):
                for i in range(i0, i1 + 1):
                    sx = int(o.x() + i * sp)
                    qp.setPen(axis if i == 0 else thick if (i * ratio) % 10 == 0 else fine)
                    qp.drawLine(sx, 0, sx, Hpx)
                for j in range(j0, j1 + 1):
                    sy = int(o.y() - j * sp)
                    qp.setPen(axis if j == 0 else thick if (j * ratio) % 10 == 0 else fine)
                    qp.drawLine(0, sy, W, sy)

            grid_pass(QPen(QColor("#202329"), 1), QPen(QColor("#282d35"), 1),   # whole plane: DIM (negatives)
                      QPen(QColor("#3a424d"), 1))
            xL = int(min(max(0.0, o.x()), W)); yB = int(min(max(0.0, o.y()), Hpx))
            qp.save(); qp.setClipRect(xL, 0, W - xL, yB)           # positive quadrant = drawable -> BRIGHT
            grid_pass(QPen(QColor("#2b313a"), 1), QPen(QColor("#3c4552"), 1),
                      QPen(QColor("#5a6675"), 1))
            qp.restore()
        # origin marker + live cursor readout
        op = self._w2s(0.0, 0.0)
        qp.setPen(QPen(QColor("#6a7684"), 1)); qp.setBrush(Qt.BrushStyle.NoBrush)
        qp.drawEllipse(op, 3, 3)
        qp.drawText(int(op.x()) + 6, int(op.y()) + 14, "0")
        if self._hover is not None:
            qp.setPen(QColor("#cfd8dc"))
            qp.drawText(8, Hpx - 8, f"x {self._hover[0]:.3g} m    y {self._hover[1]:.3g} m")
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)         # shapes/points antialiased
        if self._prop_foot:                                        # read-only 3D-prop footprints, under shapes
            from PySide6.QtGui import QPolygonF
            qp.setPen(QPen(QColor("#3f7d8c"), 1, Qt.PenStyle.DashLine))
            qp.setBrush(QBrush(QColor(63, 125, 140, 45)))
            for poly in self._prop_foot:
                if len(poly) >= 2:
                    qp.drawPolygon(QPolygonF([self._w2s(x, y) for x, y in poly]))
        for si, sh in enumerate(self.shapes):
            sp = [self._w2s(p[0], p[1]) for p in sh["pts"]]
            col = QColor("#7c5cfc") if sh["closed"] else QColor("#e0a54f")  # boundary vs interior line
            qp.setPen(QPen(col, 2))
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
        hint = QLabel("LMB add/drag pt · LMB inside closed = draw interior · Enter finish line (interior "
                      "ceiling: raise z top for a ridge/peak) · Ctrl+K close = sub-region · Alt+LMB drag "
                      "shape · Ctrl+LMB insert on edge · Ctrl+RMB del · Ctrl+Z undo · RMB extrude/update · "
                      "MMB pan · wheel zoom")
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
    assert _poly_inside([[.5, .5], [1.5, .5], [1.5, 1.5], [.5, 1.5]], sq), "nesting: inside"
    assert not _poly_inside([[5, 5], [6, 5], [6, 6]], sq), "nesting: outside"
    ed = _nearest_edge(shapes, 1.0, 0.03, 0.2)          # near bottom edge -> insert at 1, zb~0 zt~3
    assert ed and ed[:2] == (0, 1) and abs(ed[2]) < 0.05 and abs(ed[3] - 3) < 0.05, "nearest edge"
    assert _nearest_edge(shapes, 5, 5, 0.2) is None, "edge miss"
    obj = extrude_obj(shapes)                            # flat room -> floor/ceiling are SINGLE faces
    assert obj.count("g wall") == 4 and obj.count("g floor") == 1 and obj.count("g ceiling") == 1, "flat=1 face"
    assert " 3\n" in obj, "ceiling uses z_top"
    sloped = [[0, 0, 0, 3], [2, 0, 0, 3], [2, 2, 0, 5], [0, 2, 0, 4]]   # non-planar roof -> triangulated
    assert extrude_obj([{"pts": sloped, "closed": True}]).count("g ceiling") > 1, "sloped ceiling triangulated"
    assert extrude_obj([{"pts": sq, "closed": False}]) is None, "open -> no obj"
    assert len(_triangulate_xy([[0, 0], [4, 0], [4, 4], [2, 4], [2, 2], [0, 2]])) == 4, "ear-clip concave L"
    clean, nd = _drop_out_of_range([{"pts": [[0, 0, 0, 3], [2, 0, 0, 3], [2, 2, 0, 3],
                                             [10000, 10000, 0, 3]], "closed": True}])   # 1 stray vertex
    assert nd == 1 and len(clean) == 1 and len(clean[0]["pts"]) == 3, "sanitizer drops far vertex, keeps room"
    assert _drop_out_of_range([{"pts": [[0, 0, 0, 3], [5, 0, 0, 3], [5, 5, 0, 3]], "closed": True}])[1] == 0, \
        "sanitizer keeps a clean room untouched"
    try:                                                # sub-region path needs mapbox_earcut
        import mapbox_earcut  # noqa: F401
        sub = {"pts": [[.5, .5, 0, 5], [1.5, .5, 0, 5], [1.5, 1.5, 0, 5], [.5, 1.5, 0, 5]], "closed": True}
        o2 = extrude_obj([{"pts": sq, "closed": True}, sub])
        assert "g step" not in o2 and "g ceiling_region" in o2 and " 5\n" in o2, "sub = sloped, no column"
    except ImportError:
        print("(mapbox_earcut not installed -> sub-region test skipped)")
    try:                                                # interior ceiling point needs scipy
        from scipy.spatial import Delaunay  # noqa: F401
        peak = {"pts": [[1, 1, 0, 5]], "closed": False, "done": True}   # lonely interior pt, high z_top
        o3 = extrude_obj([{"pts": sq, "closed": True}, peak])
        assert o3.count("g ceiling") > 1 and " 5\n" in o3, "interior point shapes the ceiling"
    except ImportError:
        print("(scipy not installed -> interior-ceiling test skipped)")
    print("sketch2d self-check OK")


if __name__ == "__main__":
    _selfcheck()
