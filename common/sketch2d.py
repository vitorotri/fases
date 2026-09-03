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
import math
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtWidgets import (QWidget, QDoubleSpinBox, QLabel, QVBoxLayout, QHBoxLayout, QFormLayout,
                               QPushButton)

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


def _rect_ellipse_pts(p0, p1, kind, n=48):
    """2D outline (list of (x,y)) of a rectangle or ellipse spanning the drag bbox p0..p1."""
    x0, x1 = sorted((p0[0], p1[0])); y0, y1 = sorted((p0[1], p1[1]))
    if kind == "rect":
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0; rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    return [(cx + rx * math.cos(2 * math.pi * k / n), cy + ry * math.sin(2 * math.pi * k / n)) for k in range(n)]


def _boundary_union(new_xy, others):
    """Union the new polygon with any `others` it CROSSES (overlaps the boundary of) -> the outermost
    exterior ring + the absorbed indices. Fully-contained shapes (sub-regions) are NOT unioned. Needs
    shapely (polygon boolean); returns (None, []) if shapely is absent or nothing crosses."""
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except Exception:
        return None, []
    npoly = Polygon(new_xy)
    if not npoly.is_valid or npoly.area <= 0:
        return None, []
    merged, absorbed = npoly, []
    for i, xy in enumerate(others):
        if len(xy) < 3:
            continue
        poly = Polygon(xy)
        if poly.is_valid and npoly.intersects(poly) and not npoly.within(poly) and not poly.within(npoly):
            merged = unary_union([merged, poly]); absorbed.append(i)
    if not absorbed:
        return None, []
    g = merged
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda p: p.area)
    return [(round(x, 4), round(y, 4)) for x, y in list(g.exterior.coords)[:-1]], absorbed


def _split_polygon(xy, p0, p1):
    """Split polygon `xy` with the segment p0..p1 -> list of exterior rings (>=2 if it actually cut),
    else []. Needs shapely; [] if absent or the line doesn't divide the polygon."""
    try:
        from shapely.geometry import Polygon, LineString
        from shapely.ops import split
    except Exception:
        return []
    poly = Polygon(xy)
    line = LineString([p0, p1])
    if not poly.is_valid or not line.intersects(poly):
        return []
    try:
        parts = split(poly, line)
    except Exception:
        return []
    polys = [g for g in getattr(parts, "geoms", [parts]) if g.geom_type == "Polygon" and g.area > 1e-9]
    if len(polys) <= 1:
        return []
    return [[(round(x, 4), round(y, 4)) for x, y in list(g.exterior.coords)[:-1]] for g in polys]


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
        self._tool = "free"          # free | rect | ellipse | measure | select | cut
        self._draw_drag = None       # [(x0,y0),(x1,y1)] world bbox while dragging a rect/ellipse
        self._tool_drag = None       # active measure/select/cut drag [(x0,y0),(x1,y1)]
        self._measure = None         # last ruler measurement [(x0,y0),(x1,y1)] (persistent)
        self._psel = set()           # multi-selected points {(shape_idx, pt_idx)} for delete / z-apply
        self._shape_sel = None       # selected WHOLE polygon (Shape tool): index or None
        self._shape_xform = None      # modal rotate/scale of the selected shape {mode,center,base,anchor}
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

    def _grid_snap(self, x, y):
        """Snap to the nearest h-grid corner in FREE mode (rect/ellipse keep the exact drag)."""
        h = self.h
        if self._tool == "free" and h and h > 0:
            return round(x / h) * h, round(y / h) * h
        return x, y

    def dump_shapes(self):
        """JSON-able copy of the sketch (for the .json sidecar so it can be re-edited later)."""
        return copy.deepcopy(self.shapes)

    def _union_shape(self, idx):
        """If closed shape `idx` CROSSES other closed shapes, union it to the outermost outline and remove
        the absorbed ones (shapely). Shared by rect/ellipse draw and Free-mode Ctrl+K close."""
        if not (0 <= idx < len(self.shapes)):
            return
        sh = self.shapes[idx]
        if not sh["closed"] or len(sh["pts"]) < 3:
            return
        others_idx = [i for i in range(len(self.shapes)) if i != idx
                      and self.shapes[i]["closed"] and len(self.shapes[i]["pts"]) >= 3]
        others = [[(p[0], p[1]) for p in self.shapes[i]["pts"]] for i in others_idx]
        ext, absorbed = _boundary_union([(p[0], p[1]) for p in sh["pts"]], others)
        if ext:
            zt = max((p[3] for p in sh["pts"]), default=0.0)
            sh["pts"] = [[x, y, 0.0, round(zt, 4)] for x, y in ext]   # sh is held by ref; safe to del others
            for i in sorted((others_idx[a] for a in absorbed), reverse=True):
                del self.shapes[i]

    def _finish_draw_shape(self, p0, p1):
        """Create the dragged rect/ellipse (closed), add it, and union it into any boundary it crosses."""
        xy = [(round(max(0.0, x), 4), round(max(0.0, y), 4)) for x, y in _rect_ellipse_pts(p0, p1, self._tool)]
        self._snap()
        self.shapes.append({"pts": [[x, y, 0.0, 0.0] for x, y in xy], "closed": True})
        self._union_shape(len(self.shapes) - 1)
        self.sel = None; self.selectionChanged.emit(); self.update()

    def _shape_centroid(self, sh):
        n = len(sh["pts"])
        return (sum(p[0] for p in sh["pts"]) / n, sum(p[1] for p in sh["pts"]) / n)

    def _start_shape_xform(self, mode):
        """Begin a modal rotate ('r') / scale ('s') of the selected shape about its centroid."""
        if self._shape_sel is None or self._shape_sel >= len(self.shapes):
            return
        sh = self.shapes[self._shape_sel]
        self._snap()                                               # snapshot for Ctrl+Z
        self._shape_xform = {"mode": mode, "center": self._shape_centroid(sh),
                             "base": [list(p) for p in sh["pts"]], "anchor": self._hover}
        self.update()

    def _apply_shape_xform(self, cw):
        """Drive the rotate/scale from the cursor world pos `cw` (about the shape centroid)."""
        xf = self._shape_xform
        if xf is None or self._shape_sel is None or self._shape_sel >= len(self.shapes):
            return
        sh = self.shapes[self._shape_sel]; cx, cy = xf["center"]; ax, ay = xf["anchor"]
        if xf["mode"] == "r":
            ang = math.atan2(cw[1] - cy, cw[0] - cx) - math.atan2(ay - cy, ax - cx)
            ca, sa = math.cos(ang), math.sin(ang)
            for pt, b in zip(sh["pts"], xf["base"]):
                dx, dy = b[0] - cx, b[1] - cy
                pt[0] = round(max(0.0, cx + dx * ca - dy * sa), 4)
                pt[1] = round(max(0.0, cy + dx * sa + dy * ca), 4)
        else:                                                      # scale
            d0 = math.hypot(ax - cx, ay - cy) or 1e-6
            f = max(0.05, math.hypot(cw[0] - cx, cw[1] - cy) / d0)
            for pt, b in zip(sh["pts"], xf["base"]):
                pt[0] = round(max(0.0, cx + (b[0] - cx) * f), 4)
                pt[1] = round(max(0.0, cy + (b[1] - cy) * f), 4)
        self.selectionChanged.emit()

    def _select_in_box(self, p0, p1):
        """Select all shape points inside the box p0..p1 (for delete / height-apply)."""
        x0, x1 = sorted((p0[0], p1[0])); y0, y1 = sorted((p0[1], p1[1]))
        self._psel = {(si, pi) for si, sh in enumerate(self.shapes)
                      for pi, pt in enumerate(sh["pts"]) if x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1}
        self.sel = None; self.selectionChanged.emit()

    def _delete_selected(self):
        """Delete the box-selected points; drop any shape left with too few points."""
        if not self._psel:
            return
        self._snap()
        bysh = {}
        for si, pi in self._psel:
            bysh.setdefault(si, []).append(pi)
        for si in sorted(bysh, reverse=True):                      # high->low so indices stay valid
            if si >= len(self.shapes):
                continue
            for pi in sorted(bysh[si], reverse=True):
                if pi < len(self.shapes[si]["pts"]):
                    del self.shapes[si]["pts"][pi]
            if len(self.shapes[si]["pts"]) < 2:
                del self.shapes[si]
        self._psel = set(); self.sel = None; self.selectionChanged.emit(); self.update()

    def apply_z_to_selection(self, zb, zt):
        """Set z_bottom/z_top on every box-selected point (equal heights across many corners)."""
        if not self._psel:
            return
        self._snap()
        for si, pi in self._psel:
            if si < len(self.shapes) and pi < len(self.shapes[si]["pts"]):
                self.shapes[si]["pts"][pi][2] = round(zb, 4); self.shapes[si]["pts"][pi][3] = round(zt, 4)
        self.update()

    def _cut_with_line(self, p0, p1):
        """Cut every closed shape the segment crosses into pieces (shapely split). Footprint z carried
        as z_top; z reassigned by extrude/panel afterwards."""
        new, changed = [], False
        for sh in self.shapes:
            if sh["closed"] and len(sh["pts"]) >= 3:
                rings = _split_polygon([(p[0], p[1]) for p in sh["pts"]], p0, p1)
                if rings:
                    zt = max((p[3] for p in sh["pts"]), default=0.0)
                    for r in rings:
                        new.append({"pts": [[x, y, 0.0, round(zt, 4)] for x, y in r], "closed": True})
                    changed = True
                    continue
            new.append(sh)
        if changed:
            self._snap(); self.shapes = new
            self.sel = None; self._psel = set(); self.selectionChanged.emit(); self.update()

    def _draw_measure_label(self, qp, p0, p1):
        d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        a = self._w2s(*p0); b = self._w2s(*p1)
        txt = f"{d:.3g} m"
        if self.h and self.h > 0:
            txt += f"  (Δi {abs(p1[0]-p0[0])/self.h:.1f}, Δj {abs(p1[1]-p0[1])/self.h:.1f})"
        qp.setPen(QColor("#f0c040"))
        qp.drawText(QPointF((a.x() + b.x()) / 2.0, (a.y() + b.y()) / 2.0 - 6), txt)

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
        gx, gy = self._grid_snap(wx, wy)                            # free-mode: placed points snap to grid
        if self._tool in ("rect", "ellipse") and e.button() == Qt.MouseButton.LeftButton:
            self._draw_drag = [(wx, wy), (wx, wy)]                  # rubber-band a bbox; finish on release
            return
        if self._tool in ("measure", "select", "cut") and e.button() == Qt.MouseButton.LeftButton:
            self._tool_drag = [(wx, wy), (wx, wy)]                  # ruler line / select box / cut line
            return
        if self._tool == "shape" and e.button() == Qt.MouseButton.LeftButton:
            if self._shape_xform is not None:                      # confirm an in-progress rotate/scale
                self._shape_xform = None; self.update(); return
            idx = next((si for si, sh in enumerate(self.shapes)    # pick the shape under the cursor
                        if sh["closed"] and _point_in_poly(wx, wy, sh["pts"])), None)
            self._shape_sel = idx
            if idx is not None:
                self._snap(); self._drag = ("shape", idx)          # grab -> drag to move (reuses shape drag)
            self.selectionChanged.emit(); self.update(); return
        ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
        alt = bool(e.modifiers() & Qt.KeyboardModifier.AltModifier)
        tol = HIT_PX / self.scale
        b = e.button()
        if b == Qt.MouseButton.LeftButton and ctrl:                 # insert on nearest edge (loose tol)
            hit = _nearest_edge(self.shapes, wx, wy, EDGE_HIT_PX / self.scale)
            if hit:
                self._snap()
                si, idx, zb, zt = hit                               # xy at the CLICK, z's from the edge
                self.shapes[si]["pts"].insert(idx, [round(gx, 4), round(gy, 4), round(zb, 4), round(zt, 4)])
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
                self.shapes[si]["pts"].append([round(gx, 4), round(gy, 4), 0.0, 0.0])
                self.sel = (si, len(self.shapes[si]["pts"]) - 1); self.selectionChanged.emit()
            else:                                                   # start a new shape; LMB inside a closed
                self._snap()                                        # one begins a nested sub-region
                self.shapes.append({"pts": [[round(gx, 4), round(gy, 4), 0.0, 0.0]], "closed": False})
                self.sel = (len(self.shapes) - 1, 0); self.selectionChanged.emit()
            self.update(); return
        if b == Qt.MouseButton.MiddleButton:                        # pan
            self._drag = ("pan",)

    def mouseMoveEvent(self, e):
        p = e.position()
        self._hover = self._s2w(p.x(), p.y())                       # live cursor coords (raw, may be <0)
        if self._shape_xform is not None:                          # modal rotate/scale of the selected shape
            self._apply_shape_xform(self._hover); self.update(); return
        if self._draw_drag is not None:                            # extend the rect/ellipse bbox
            wx, wy = self._s2w(p.x(), p.y())
            self._draw_drag[1] = (max(0.0, wx), max(0.0, wy)); self.update(); return
        if self._tool_drag is not None:                            # extend the ruler/select/cut drag
            wx, wy = self._s2w(p.x(), p.y())
            self._tool_drag[1] = (max(0.0, wx), max(0.0, wy)); self.update(); return
        if self._drag is None:
            self.update(); return
        if self._drag[0] == "pan":
            self.off += p - self._last; self._last = p; self.update(); return
        dx = (p.x() - self._last.x()) / self.scale
        dy = -(p.y() - self._last.y()) / self.scale
        if self._drag[0] == "pt":
            _, si, idx = self._drag
            pt = self.shapes[si]["pts"][idx]
            pt[0] = round(max(0.0, pt[0] + dx), 4); pt[1] = round(max(0.0, pt[1] + dy), 4)  # free while dragging
            self.selectionChanged.emit()
        elif self._drag[0] == "shape":
            sh = self.shapes[self._drag[1]]
            dx = max(dx, -min(p[0] for p in sh["pts"]))             # clamp translation: keep whole shape
            dy = max(dy, -min(p[1] for p in sh["pts"]))             # in the positive quadrant (rigid)
            for pt in sh["pts"]:                                    # move the whole shape in xy
                pt[0] = round(pt[0] + dx, 4); pt[1] = round(pt[1] + dy, 4)
        self._last = p; self.update()

    def mouseReleaseEvent(self, e):
        if self._draw_drag is not None and e.button() == Qt.MouseButton.LeftButton:
            p0, p1 = self._draw_drag; self._draw_drag = None
            if abs(p1[0] - p0[0]) > 1e-3 and abs(p1[1] - p0[1]) > 1e-3:   # ignore a zero-size click
                self._finish_draw_shape(p0, p1)
            else:
                self.update()
            return
        if self._tool_drag is not None and e.button() == Qt.MouseButton.LeftButton:
            p0, p1 = self._tool_drag; self._tool_drag = None
            if self._tool == "measure":
                self._measure = [p0, p1]
            elif self._tool == "select":
                self._select_in_box(p0, p1)
            elif self._tool == "cut":
                self._cut_with_line(p0, p1)
            self.update(); return
        if self._drag and self._drag[0] == "pt":                  # snap the moved point to grid ON RELEASE
            _, si, idx = self._drag
            pt = self.shapes[si]["pts"][idx]
            nx, ny = self._grid_snap(pt[0], pt[1])
            pt[0] = round(nx, 4); pt[1] = round(ny, 4)
            self.selectionChanged.emit(); self.update()
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
        # Shape tool: R/S start a rotate/scale of the selected shape; Enter confirms, Esc cancels.
        if self._tool == "shape":
            if self._shape_xform is not None:
                if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    self._shape_xform = None; self.update(); return
                if e.key() == Qt.Key.Key_Escape:                   # cancel -> restore base points
                    sh = self.shapes[self._shape_sel]
                    for pt, b in zip(sh["pts"], self._shape_xform["base"]):
                        pt[0], pt[1] = b[0], b[1]
                    self._shape_xform = None; self.selectionChanged.emit(); self.update(); return
            elif self._shape_sel is not None and not ctrl and e.key() in (Qt.Key.Key_R, Qt.Key.Key_S):
                self._start_shape_xform('r' if e.key() == Qt.Key.Key_R else 's'); return
        if ctrl and e.key() == Qt.Key.Key_K:                        # close the active shape
            si = self._active()
            if si is not None and len(self.shapes[si]["pts"]) > 2:
                self._snap(); self.shapes[si]["closed"] = True
                self._union_shape(si)                              # prune to outermost if it crosses a boundary
                self.selectionChanged.emit(); self.update()
            return
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):       # finish open line -> interior ceiling ctrl
            si = self._active()
            if si is not None and self.shapes[si]["pts"]:
                self._snap(); self.shapes[si]["done"] = True; self.update()
            return
        if ctrl and e.key() == Qt.Key.Key_Z:                        # undo
            if self._undo:
                self.shapes, self.sel = self._undo.pop()
                self._psel = set(); self._shape_sel = None; self._shape_xform = None
                self.selectionChanged.emit(); self.update()
            return
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):    # delete selected shape / points
            if self._tool == "shape" and self._shape_sel is not None and self._shape_sel < len(self.shapes):
                self._snap(); del self.shapes[self._shape_sel]
                self._shape_sel = None; self._shape_xform = None; self.selectionChanged.emit(); self.update()
            else:
                self._delete_selected()
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
        if self._draw_drag is not None:                            # live rect/ellipse preview
            a = self._w2s(*self._draw_drag[0]); b = self._w2s(*self._draw_drag[1])
            qp.setPen(QPen(QColor("#7c5cfc"), 2, Qt.PenStyle.DashLine)); qp.setBrush(Qt.BrushStyle.NoBrush)
            r = QRectF(a, b).normalized()
            (qp.drawRect if self._tool == "rect" else qp.drawEllipse)(r)
        for si, sh in enumerate(self.shapes):
            sp = [self._w2s(p[0], p[1]) for p in sh["pts"]]
            sel_shape = (si == self._shape_sel)                    # Shape tool selection -> highlight
            col = (QColor("#0ff091") if sel_shape
                   else QColor("#7c5cfc") if sh["closed"] else QColor("#e0a54f"))  # boundary vs interior
            qp.setPen(QPen(col, 3 if sel_shape else 2))
            for i in range(len(sp) - 1):
                qp.drawLine(sp[i], sp[i + 1])
            if sh["closed"] and len(sp) > 1:
                qp.drawLine(sp[-1], sp[0])
            for pi, s in enumerate(sp):
                msel = (si, pi) in self._psel
                selp = self.sel == (si, pi)
                qp.setPen(Qt.PenStyle.NoPen)
                qp.setBrush(QBrush(QColor("#ff5c7a") if msel else QColor("#0ff091") if selp
                                   else QColor("#cfd8dc")))
                r = 6 if (msel or selp) else 4
                qp.drawEllipse(s, r, r)
        # ruler measurement (persistent) + active measure/select/cut drag preview
        if self._measure is not None and self._tool == "measure":
            a = self._w2s(*self._measure[0]); b = self._w2s(*self._measure[1])
            qp.setPen(QPen(QColor("#f0c040"), 2)); qp.drawLine(a, b)
            self._draw_measure_label(qp, *self._measure)
        td = self._tool_drag
        if td is not None:
            a = self._w2s(*td[0]); b = self._w2s(*td[1])
            if self._tool == "select":
                qp.setPen(QPen(QColor("#0ff091"), 1, Qt.PenStyle.DashLine))
                qp.setBrush(QBrush(QColor(15, 240, 145, 30)))
                qp.drawRect(QRectF(a, b).normalized())
            else:
                qp.setPen(QPen(QColor("#f0c040" if self._tool == "measure" else "#ff5c7a"), 2))
                qp.setBrush(Qt.BrushStyle.NoBrush); qp.drawLine(a, b)
                if self._tool == "measure":
                    self._draw_measure_label(qp, td[0], td[1])
        if self._shape_xform is not None:                          # rotate/scale pivot marker
            c = self._w2s(*self._shape_xform["center"])
            qp.setPen(QPen(QColor("#0ff091"), 1)); qp.setBrush(Qt.BrushStyle.NoBrush)
            qp.drawEllipse(c, 4, 4)


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
        hint = QLabel(
            "<b>Free</b>: LMB add pt (snaps to grid) · drag pt to move · LMB inside closed = sub-region · "
            "Enter finish line = interior ceiling (raise z top for ridge/peak) · Ctrl+K close · "
            "Ctrl+LMB insert on edge · Ctrl+RMB delete pt<br>"
            "<b>Rect / Ellipse</b>: drag a box (unions into a boundary it overlaps)<br>"
            "<b>Measure</b>: drag to read distance (m + Δi/Δj)<br>"
            "<b>Select</b>: box-select points → Delete, or set z + 'Apply z→selected'<br>"
            "<b>Cut</b>: drag a line fully across a shape to split it in two<br>"
            "<b>Shape</b>: click a polygon then drag = move · R = rotate · S = scale · Delete = remove<br>"
            "Global: MMB pan · wheel zoom · Ctrl+Z undo · RMB = extrude / update OBJ")
        hint.setWordWrap(True); hint.setStyleSheet("color:#8a94a6;font-size:11px;")
        # tool selector: Free (click polygon) / Rect / Ellipse (drag a bbox); rect/ellipse union into the
        # boundary they overlap.
        self._tool_btns = {}
        trow = QHBoxLayout()
        for label, tool in (("Free", "free"), ("Rect", "rect"), ("Ellipse", "ellipse"),
                            ("Measure", "measure"), ("Select", "select"), ("Cut", "cut"), ("Shape", "shape")):
            b = QPushButton(label); b.setCheckable(True); b.setChecked(tool == "free")
            b.setStyleSheet("QPushButton:checked{background:#7c5cfc;color:#1a1d23;font-weight:bold;}")
            b.clicked.connect(lambda _=False, t=tool: self._set_tool(t))
            trow.addWidget(b); self._tool_btns[tool] = b
        self._applyz_btn = QPushButton("Apply z → selected pts")   # equalize heights over a Select box
        self._applyz_btn.clicked.connect(self._apply_z_sel)
        side = QVBoxLayout()
        side.addWidget(QLabel("Tool")); side.addLayout(trow)
        side.addWidget(QLabel("Selected point")); side.addWidget(panel)
        side.addWidget(self._applyz_btn)
        side.addWidget(hint); side.addStretch(1)
        root = QHBoxLayout(self); root.addWidget(self.canvas, 1); root.addLayout(side)
        self._sync()

    def _set_tool(self, tool):
        c = self.canvas
        c._tool = tool
        c._shape_xform = None; c._shape_sel = None                 # clear stale shape/point selections
        c._psel = set(); c._tool_drag = None; c._draw_drag = None
        for t, b in self._tool_btns.items():
            b.setChecked(t == tool)
        c.selectionChanged.emit(); c.update(); c.setFocus()

    def _spin(self):
        s = QDoubleSpinBox(); s.setRange(-1e4, 1e4); s.setDecimals(3); s.setSingleStep(0.1)
        s.setEnabled(False); return s

    def showEvent(self, e):
        super().showEvent(e); self.canvas.setFocus()

    def _sync(self):
        sel = self.canvas.sel
        has_psel = bool(self.canvas._psel)
        on = sel is not None or has_psel                          # enable z spins for box-selection too
        for s in (self._sx, self._sy, self._szb, self._szt):
            s.blockSignals(True); s.setEnabled(on)
        if sel is not None:                                       # only a single point fills the fields
            p = self.canvas.shapes[sel[0]]["pts"][sel[1]]
            self._sx.setValue(p[0]); self._sy.setValue(p[1])
            self._szb.setValue(p[2]); self._szt.setValue(p[3])
        for s in (self._sx, self._sy, self._szb, self._szt):
            s.blockSignals(False)
        self._applyz_btn.setEnabled(has_psel)

    def _apply_z_sel(self):
        self.canvas.apply_z_to_selection(self._szb.value(), self._szt.value())

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
    assert _rect_ellipse_pts((0, 0), (2, 1), "rect") == [(0, 0), (2, 0), (2, 1), (0, 1)], "rect from bbox"
    assert len(_rect_ellipse_pts((0, 0), (2, 2), "ellipse")) == 48, "ellipse polygon 48 pts"
    try:                                                # boundary union needs shapely
        import shapely  # noqa: F401
        # two overlapping unit squares -> union exterior spans 0..1.5 in x (outermost boundary)
        ext, absorbed = _boundary_union([(0, 0), (1, 0), (1, 1), (0, 1)],
                                        [[(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)]])
        assert absorbed == [0] and ext and abs(max(x for x, _ in ext) - 1.5) < 1e-6, "overlap unions to outer"
        # a fully-contained square is NOT unioned (stays a sub-region)
        _, ab2 = _boundary_union([(0, 0), (3, 0), (3, 3), (0, 3)], [[(1, 1), (2, 1), (2, 2), (1, 2)]])
        assert ab2 == [], "contained shape not unioned"
        rings = _split_polygon([(0, 0), (2, 0), (2, 2), (0, 2)], (1, -1), (1, 3))   # vertical cut at x=1
        assert len(rings) == 2, "cut line splits the square into two"
    except ImportError:
        print("(shapely not installed -> boundary-union / cut tests skipped)")
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
