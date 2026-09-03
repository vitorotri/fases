#!/usr/bin/env python3
"""Placeable 3D props for the scene: exact procedural primitives + loaded furniture assets.

Two sources, one Prop model:
  * PRIMITIVES  — plane / box / cylinder (+ dodecahedron later): generated in code, math-exact, no
                  visual check needed.
  * assets/*.obj — dummy / sofa / chair / table: SIMPLIFIED low-poly meshes you decimate and eyecheck
                  ONCE, dropped in props3d's assets/ dir. Loaded via load_asset(), then treated identically.

A Prop carries a transform (pos, rot degrees XYZ, scale) and a LAYER name. bake() applies the transform;
export_groups() writes OBJ with faces grouped BY LAYER (AutoCAD-style: many props can share one layer ->
one `g <layer>` -> one material .fit in the solver). Pure Python (math only) so it self-checks headless.

Qt panel + GL drawing live in gui.py; this module is logic only.
"""
import math
import os

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
DEFAULT_LAYER = "props"


# ── procedural primitives: (verts[(x,y,z)], tris[(a,b,c)]) , CCW outward ──────────────────────
def plane(w=1.0, d=1.0):
    """Flat quad in the XY plane, centered, normal +Z. Open surface (2 tris). Rotate for a wall."""
    hw, hd = w / 2.0, d / 2.0
    return [(-hw, -hd, 0.0), (hw, -hd, 0.0), (hw, hd, 0.0), (-hw, hd, 0.0)], [(0, 1, 2), (0, 2, 3)]


def box(w=1.0, d=1.0, h=1.0):
    """Axis-aligned box centered at origin. 8 verts, 12 tris, outward-facing, closed."""
    hw, hd, hh = w / 2.0, d / 2.0, h / 2.0
    v = [(-hw, -hd, -hh), (hw, -hd, -hh), (hw, hd, -hh), (-hw, hd, -hh),
         (-hw, -hd, hh), (hw, -hd, hh), (hw, hd, hh), (-hw, hd, hh)]
    t = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),        # -z, +z
         (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),        # -y, +y
         (1, 2, 6), (1, 6, 5), (3, 0, 4), (3, 4, 7)]        # +x, -x
    return v, t


def cylinder(r=0.5, h=1.0, n=24):
    """Z-axis cylinder centered at origin. 2n rim + 2 center verts, closed (fan caps)."""
    v = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        x, y = r * math.cos(a), r * math.sin(a)
        v.append((x, y, -h / 2.0)); v.append((x, y, h / 2.0))   # 2i bottom, 2i+1 top
    cb = len(v); v.append((0.0, 0.0, -h / 2.0))                 # bottom center
    ct = len(v); v.append((0.0, 0.0, h / 2.0))                  # top center
    t = []
    for i in range(n):
        b0, t0 = 2 * i, 2 * i + 1
        b1, t1 = 2 * ((i + 1) % n), 2 * ((i + 1) % n) + 1
        t.append((b0, b1, t1)); t.append((b0, t1, t0))         # side quad
        t.append((cb, b1, b0))                                 # bottom cap (-z)
        t.append((ct, t0, t1))                                 # top cap (+z)
    return v, t


def _sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def _cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(p, r=1.0):
    l = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) or 1.0
    return (p[0] / l * r, p[1] / l * r, p[2] / l * r)


def _merge(meshes):
    """Concatenate (verts, tris) meshes, offsetting tri indices."""
    V, T, off = [], [], 0
    for v, t in meshes:
        V.extend(v); T.extend((a + off, b + off, c + off) for a, b, c in t); off += len(v)
    return V, T


def _icosahedron():
    """Unit-ish icosahedron: 12 verts, 20 CCW-outward triangular faces (canonical table)."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
         (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
         (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    return v, f


def sphere(r=0.5, subdiv=2):
    """Icosphere: subdivide an icosahedron `subdiv` times, project to radius r. Closed, uniform."""
    v, f = _icosahedron()
    v = [_normalize(p, r) for p in v]
    for _ in range(subdiv):
        mid, nf = {}, []

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in mid:
                m = _normalize(((v[a][0] + v[b][0]) / 2, (v[a][1] + v[b][1]) / 2, (v[a][2] + v[b][2]) / 2), r)
                mid[key] = len(v); v.append(m)
            return mid[key]

        for a, b, c in f:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return v, f


def _rotate_points(pts, a, b):
    """Rotate all points by the rotation that sends unit vector a -> unit vector b (Rodrigues)."""
    a, b = _normalize(a), _normalize(b)
    v = _cross(a, b); s = math.sqrt(_dot(v, v)); c = _dot(a, b)
    if s < 1e-9:                                                     # parallel
        if c > 0:
            return list(pts)
        ax = _cross(a, (1, 0, 0))                                   # antiparallel: 180deg about any perp axis
        if _dot(ax, ax) < 1e-9:
            ax = _cross(a, (0, 1, 0))
        ax = _normalize(ax)
        return [(2 * _dot(ax, p) * ax[0] - p[0], 2 * _dot(ax, p) * ax[1] - p[1],
                 2 * _dot(ax, p) * ax[2] - p[2]) for p in pts]
    k = (1.0 - c) / (s * s)

    def rot(p):
        vp = _cross(v, p); vvp = _cross(v, vp)
        return (p[0] + vp[0] + vvp[0] * k, p[1] + vp[1] + vvp[1] * k, p[2] + vp[2] + vvp[2] * k)
    return [rot(p) for p in pts]


def dodecahedron(r=0.5):
    """Dodecahedron as the DUAL of an icosahedron: dodeca verts = icosa face centroids; each dodeca
    pentagon = the ring of icosa faces around one icosa vertex. Connectivity is correct by construction
    (no hand-typed face table). Oriented so ONE pentagon lies flat on the floor (z=0) — like a real
    dodecahedron loudspeaker standing on a face. Pentagons fan-triangulated, wound outward."""
    iv, if_ = _icosahedron()
    dv = [_normalize(((iv[a][0] + iv[b][0] + iv[c][0]) / 3.0,
                      (iv[a][1] + iv[b][1] + iv[c][1]) / 3.0,
                      (iv[a][2] + iv[b][2] + iv[c][2]) / 3.0), r) for a, b, c in if_]
    faces = []
    for j in range(len(iv)):
        ring = [i for i, face in enumerate(if_) if j in face]        # 5 icosa faces touching vertex j
        order = [ring[0]]
        while len(order) < len(ring):                                # walk the ring by shared-edge
            cur = order[-1]
            order.append(next(i for i in ring if i not in order
                              and len(set(if_[i]) & set(if_[cur])) == 2))
        n = _cross(_sub(dv[order[1]], dv[order[0]]), _sub(dv[order[2]], dv[order[0]]))
        if _dot(n, dv[order[0]]) < 0:                                # face outward (verts ~ outward normal)
            order = order[::-1]
        faces.append(order)
    # stand it on a face: rotate one face's outward normal to -z, then lift so the bottom rests at z=0
    fc = faces[0]
    fn = _normalize((sum(dv[i][0] for i in fc), sum(dv[i][1] for i in fc), sum(dv[i][2] for i in fc)))
    dv = _rotate_points(dv, fn, (0.0, 0.0, -1.0))
    minz = min(p[2] for p in dv)
    dv = [(x, y, z - minz) for x, y, z in dv]
    tris = [(poly[0], poly[k], poly[k + 1]) for poly in faces for k in range(1, len(poly) - 1)]
    return dv, tris


def dummy(head_r=0.12, torso_r=0.16, torso_h=0.7, shoulder=1.3):
    """Simplified human proxy (scatterer stand-in): elliptical cylinder torso (widened in x -> shoulders)
    + round sphere head, merged. Low-poly, no asset file to author. Swap for a decimated .obj asset when
    you have a real one; same Prop pipeline."""
    tv, tt = cylinder(torso_r, torso_h, 16)
    tv = bake(tv, scale=(shoulder, 1.0, 1.0))                        # widen left-right -> shoulders
    hv, ht = sphere(head_r, 1)
    hv = bake(hv, pos=(0.0, 0.0, torso_h / 2.0 + head_r))            # head sits on top of the torso
    return _merge([(tv, tt), (hv, ht)])


def tetrahedron(r=0.4, h=0.7):
    """3-sided pyramid: equilateral triangular base on z=0 + apex up. Closed (4 faces)."""
    base = [(r * math.cos(math.radians(90 + 120 * k)), r * math.sin(math.radians(90 + 120 * k)), 0.0)
            for k in range(3)]
    v = base + [(0.0, 0.0, h)]                                        # 3 base verts + apex(3)
    t = [(0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1)]                 # 3 sides + base (facing -z)
    return v, t


def cone(r=0.4, h=0.8, n=24):
    """Z-axis cone, base on z=0, apex up. Closed. (Internal helper; also the soundbox driver.)"""
    v = [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), 0.0) for i in range(n)]
    apex = len(v); v.append((0.0, 0.0, h))
    cb = len(v); v.append((0.0, 0.0, 0.0))
    t = []
    for i in range(n):
        i1 = (i + 1) % n
        t.append((i, i1, apex)); t.append((cb, i1, i))
    return v, t


def _box_at(w, d, h, pos):
    v, t = box(w, d, h)
    return bake(v, pos=pos), t


def _open_cone(r, h, n):
    """Cone with the wide end OPEN (sides only, no rim cap) — a speaker cone you look INTO. Base ring
    on z=0, apex at z=h."""
    v = [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), 0.0) for i in range(n)]
    apex = len(v); v.append((0.0, 0.0, h))
    return v, [(i, (i + 1) % n, apex) for i in range(n)]


def chair(w=0.45, d=0.45, seat_h=0.45, back_h=0.45, th=0.03):
    """SIMPLIFIED chair for acoustics: seat slab + backrest only (no legs). Sits on z=0."""
    return _merge([_box_at(w, d, th, (0, 0, seat_h - th / 2.0)),                        # seat
                   _box_at(w, th, back_h, (0, -d / 2.0 + th / 2.0, seat_h + back_h / 2.0))])  # back


def sofa(w=1.9, d=0.9, seat_h=0.43, back_h=0.45, back_t=0.15, arm=0.2, arm_h=0.22):
    """Simplified ergonomic 2-3 seat sofa: seat block (top ~0.43 m) + backrest (top ~0.88 m) + 2 arms
    (top ~0.65 m). Sits on z=0."""
    parts = [_box_at(w, d, seat_h, (0, 0, seat_h / 2.0)),                                   # seat base
             _box_at(w, back_t, back_h, (0, -d / 2.0 + back_t / 2.0, seat_h + back_h / 2.0))]  # backrest
    for sx in (-1, 1):                                                                      # arms
        parts.append(_box_at(arm, d, arm_h, (sx * (w / 2.0 - arm / 2.0), 0, seat_h + arm_h / 2.0)))
    return _merge(parts)


def _ring_fill(inner, outer):
    """Triangulate the annulus between two angle-sorted closed loops (lists of (angle,(x,y,z))) by a
    two-pointer merge on angle -> no gaps even when the loops have different vertex counts. inner verts
    first, then outer."""
    verts = [p for _, p in inner] + [p for _, p in outer]
    ni, no = len(inner), len(outer)
    tris, i, j = [], 0, 0
    for _ in range(ni + no):
        a_nxt = inner[(i + 1) % ni][0] + 2 * math.pi * ((i + 1) // ni)
        b_nxt = outer[(j + 1) % no][0] + 2 * math.pi * ((j + 1) // no)
        if i < ni and (j >= no or a_nxt <= b_nxt):
            tris.append((i % ni, ni + j % no, (i + 1) % ni)); i += 1
        else:
            tris.append((i % ni, ni + j % no, ni + (j + 1) % no)); j += 1
    return verts, tris


def _annulus(w, h, r, cz, y, n=24):
    """Front baffle: rectangle (w x h, centered in x,z) MINUS a circle (radius r at (0,cz)), at plane y.
    The 4 rectangle corners are inserted into the outer loop so the face is gap-free at the corners."""
    hw, hh = w / 2.0, h / 2.0
    inner = [(2 * math.pi * i / n, (r * math.cos(2 * math.pi * i / n), y, cz + r * math.sin(2 * math.pi * i / n)))
             for i in range(n)]
    outer = []
    for i in range(n):
        th = 2 * math.pi * i / n; cx, cs = math.cos(th), math.sin(th)
        ts = ([(hw if cx > 0 else -hw) / cx] if abs(cx) > 1e-9 else [])
        ts += ([((hh - cz) if cs > 0 else (-hh - cz)) / cs] if abs(cs) > 1e-9 else [])
        t = min(tt for tt in ts if tt > 0)
        outer.append((th, (t * cx, y, cz + t * cs)))
    for cxr, czr in ((hw, hh), (-hw, hh), (-hw, -hh), (hw, -hh)):   # exact corners -> no corner gaps
        outer.append((math.atan2(czr - cz, cxr) % (2 * math.pi), (cxr, y, czr)))
    outer.sort(key=lambda e: e[0])
    return _ring_fill(inner, outer)


def soundbox(w=0.4, d=0.35, h=0.6, driver_r=0.12, driver_depth=0.14, n=24):
    """Loudspeaker cabinet: box with the front (+y) baffle opened by a circular hole (gap-free annulus,
    corners included), a driver cone recessed INWARD through it (concave cone visible from the front).
    Sits on z=0. Built centered, then lifted onto the floor."""
    hh, yf = h / 2.0, d / 2.0
    dz = h * 0.1                                                    # driver center height (box-centered)
    bv, bt = box(w, d, h)
    open_box = (bv, [tri for tri in bt if tri not in ((2, 3, 7), (2, 7, 6))])   # drop solid +y face
    baffle = _annulus(w, h, driver_r, dz, yf, n)                   # front face minus driver hole
    cv, ct = _open_cone(driver_r, driver_depth, n)                 # OPEN mouth -> you see into the cone
    driver = (bake(cv, pos=(0, yf, dz), rot=(90, 0, 0)), ct)       # apex INWARD (-y): concave, visible
    cap_frac = 0.5                                                 # depth fraction from the mouth
    cap_r = driver_r * (1.0 - cap_frac)                            # = cone radius there -> disc sits FLUSH
    capv, capt = cylinder(cap_r, driver_r * 0.06, n)              # dustcap caps the cone cross-section
    cap = (bake(capv, pos=(0, yf - driver_depth * cap_frac, dz), rot=(90, 0, 0)), capt)
    v, t = _merge([open_box, baffle, driver, cap])
    return bake(v, pos=(0, 0, hh)), t                              # sit on the floor


def skyline(N=7, cell=0.05, step=0.02, base=0.02):
    """2D skyline diffuser from the 1D Schroeder QR sequence with ROWS SHIFTED: qr[k] = (k^2 mod N),
    height(i,j) = qr[(i+j) mod N] — each row is the 1D QRD cyclically shifted by j (diagonal shift),
    NOT the symmetric 2D (i^2+j^2) form. Shorter columns (smaller step). Sits on z=0."""
    qr = [(k * k) % N for k in range(N)]
    W = N * cell
    parts = [_box_at(W, W, base, (0, 0, base / 2.0))]
    for i in range(N):
        for j in range(N):
            res = qr[(i + j) % N]
            if res == 0:
                continue                                           # zero-depth well = just the base plate
            hgt = base + step * res
            x = -W / 2.0 + cell * (i + 0.5); y = -W / 2.0 + cell * (j + 0.5)
            parts.append(_box_at(cell * 0.96, cell * 0.96, hgt, (x, y, hgt / 2.0)))
    v, t = _merge(parts)
    return bake(v, rot=(0, 90, 0)), t                              # start rotated 90 about y


PRIMITIVES = {"plane": plane, "box": box, "cylinder": cylinder, "tetrahedron": tetrahedron,
              "sphere": sphere, "dodecahedron": dodecahedron, "dummy": dummy,
              "chair": chair, "sofa": sofa, "soundbox": soundbox, "skyline": skyline}


# ── furniture / dummy assets (simplified OBJ, eyechecked once) ─────────────────────────────────
def load_asset(name):
    """Load assets/<name>.obj -> (verts, tris). Minimal reader: `v x y z` + `f` (polys fanned,
    1-based, ignores /vt/vn). These are the low-poly dummy/sofa/chair/table you verify visually."""
    path = name if (os.sep in name or os.path.exists(name)) else os.path.join(ASSET_DIR, name + ".obj")
    verts, tris = [], []
    with open(path) as f:
        for ln in f:
            s = ln.split()
            if not s:
                continue
            if s[0] == "v":
                verts.append((float(s[1]), float(s[2]), float(s[3])))
            elif s[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in s[1:]]        # 1-based -> 0-based
                for k in range(1, len(idx) - 1):                       # fan-triangulate the polygon
                    tris.append((idx[0], idx[k], idx[k + 1]))
    return verts, tris


def list_assets():
    """Names of available furniture assets (assets/*.obj), for the panel dropdown."""
    if not os.path.isdir(ASSET_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(ASSET_DIR) if f.endswith(".obj"))


# ── transform ────────────────────────────────────────────────────────────────────────────────
def _rot_cols(rx, ry, rz):
    """Rotation matrix rows for R = Rz @ Ry @ Rx (angles in radians). Returns 3 row tuples."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # Rz @ Ry @ Rx, expanded
    return (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy,     cy * sx,                cy * cx),
    )


def bake(verts, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
    """Apply scale, then rotation (deg, XYZ euler), then translation. Returns new vert list."""
    sx, sy, sz = scale
    (r0, r1, r2) = _rot_cols(math.radians(rot[0]), math.radians(rot[1]), math.radians(rot[2]))
    px, py, pz = pos
    out = []
    for x, y, z in verts:
        x, y, z = x * sx, y * sy, z * sz
        out.append((r0[0] * x + r0[1] * y + r0[2] * z + px,
                    r1[0] * x + r1[1] * y + r1[2] * z + py,
                    r2[0] * x + r2[1] * y + r2[2] * z + pz))
    return out


def feature_edges(verts, tris, angle_thresh_deg=1.0):
    """Crease + boundary edges only (drop within-face triangulation): an edge is kept if it borders one
    face (open boundary, e.g. a plane's rim) or two faces whose normals differ by > thresh. Low thresh
    (1 deg) drops only COPLANAR splits (box quad diagonals = 0 deg) -> a box shows its 12 cube edges,
    while curved facets (sphere ~10 deg, cylinder ~22 deg) stay visible. Same idea as the scene overlay."""
    from collections import defaultdict

    def fn(t):
        a, b, c = verts[t[0]], verts[t[1]], verts[t[2]]
        return _normalize(_cross(_sub(b, a), _sub(c, a)))

    normals = [fn(t) for t in tris]
    emap = defaultdict(list)
    for fi, t in enumerate(tris):
        for u, v in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            emap[(min(u, v), max(u, v))].append(fi)
    cos_t = math.cos(math.radians(angle_thresh_deg))
    out = []
    for (i, j), fs in emap.items():
        if len(fs) == 1 or _dot(normals[fs[0]], normals[fs[1]]) < cos_t:
            out.append((i, j))
    return out


class Prop:
    """A placed object: a source mesh (primitive name or asset name) + transform + layer."""
    def __init__(self, kind, source="box", pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1),
                 layer=DEFAULT_LAYER, params=None):
        self.kind = kind                 # "primitive" | "asset"
        self.source = source             # primitive key or asset name
        self.pos = list(pos); self.rot = list(rot); self.scale = list(scale)
        self.layer = layer
        self.params = params or {}       # e.g. {"w":2,"h":1} for a primitive
        self._base = None                # cached (verts,tris) of the untransformed mesh
        self._edges = None               # cached feature edges (topology -> same for any transform)

    def mesh(self):
        """Untransformed (verts, tris), cached. Rebuilds a primitive from params or loads an asset."""
        if self._base is None:
            self._base = (PRIMITIVES[self.source](**self.params) if self.kind == "primitive"
                          else load_asset(self.source))
        return self._base

    def world(self):
        """(baked verts, tris) at the prop's transform — what gui draws / exports."""
        v, t = self.mesh()
        return bake(v, self.pos, self.rot, self.scale), t

    def edges(self):
        """Feature edges (indices into the vert list) for a clean wireframe — no triangulation shown."""
        if self._edges is None:
            v, t = self.mesh()
            self._edges = feature_edges(v, t)
        return self._edges

    def to_dict(self):
        return {"kind": self.kind, "source": self.source, "pos": self.pos, "rot": self.rot,
                "scale": self.scale, "layer": self.layer, "params": self.params}

    @staticmethod
    def from_dict(d):
        return Prop(d["kind"], d["source"], d["pos"], d["rot"], d["scale"],
                    d.get("layer", DEFAULT_LAYER), d.get("params"))


# ── OBJ export, grouped BY LAYER (AutoCAD model: N props -> 1 `g <layer>` -> 1 .fit) ───────────
def export_groups(props, start=1):
    """OBJ text for all props, faces grouped by `layer`. `start` = first 1-based vertex index (so it
    can be appended after the room's verts). Every prop's faces land under `g <its layer>`; props
    sharing a layer share the group -> the solver reads one material .fit for the whole layer."""
    vlines, by_layer, idx = [], {}, start
    for p in props:
        verts, tris = p.world()
        base = idx
        for x, y, z in verts:
            vlines.append(f"v {x:.6g} {y:.6g} {z:.6g}"); idx += 1
        by_layer.setdefault(p.layer, []).extend((base + a, base + b, base + c) for a, b, c in tris)
    lines = ["# props3d"] + vlines
    for layer, faces in by_layer.items():
        lines.append(f"g {layer}")
        lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    return "\n".join(lines) + "\n"


# ── self-check ─────────────────────────────────────────────────────────────────────────────────
def _euler_closed(verts, tris):
    """True if the mesh is a closed manifold: every edge shared by exactly 2 tris AND V-E+F==2."""
    from collections import Counter
    edges = Counter()
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            edges[(min(u, v), max(u, v))] += 1
    if any(cnt != 2 for cnt in edges.values()):
        return False
    return len(verts) - len(edges) + len(tris) == 2


def _selfcheck():
    # primitives closed + right counts
    v, t = box(2, 3, 4); assert len(v) == 8 and len(t) == 12 and _euler_closed(v, t), "box"
    v, t = cylinder(0.5, 1.0, 12); assert len(v) == 2 * 12 + 2 and _euler_closed(v, t), "cylinder closed"
    v, t = plane(2, 1); assert len(v) == 4 and len(t) == 2, "plane"
    v, t = sphere(0.5, 2); assert _euler_closed(v, t), "icosphere closed"
    v, t = dodecahedron(0.5)
    assert len(v) == 20 and len(t) == 36 and _euler_closed(v, t), "dodecahedron closed (20v, 12 pentagons)"
    v, t = dummy(); assert v and t and all(0 <= i < len(v) for tri in t for i in tri), "dummy valid"

    # feature edges: box shows its 12 cube edges (not quad diagonals), plane shows its 4-edge outline
    assert len(feature_edges(*box())) == 12, "box feature edges = 12 (no triangulation)"
    assert len(feature_edges(*plane())) == 4, "plane feature edges = 4 (outline only)"

    # new props: tetra closed; composites valid (nonempty, in-range indices); cone NOT a public prop
    v, t = tetrahedron(); assert len(v) == 4 and _euler_closed(v, t), "3-sided pyramid closed"
    for name in ("chair", "sofa", "soundbox", "skyline"):
        v, t = PRIMITIVES[name]()
        assert v and t and all(0 <= i < len(v) for tri in t for i in tri), f"{name} valid"
    assert "cone" not in PRIMITIVES, "cone stays an internal helper (soundbox driver), not a prop"

    # soundbox baffle must be GAP-FREE (corners covered): its area == rectangle - driver circle
    bv, bt = _annulus(0.4, 0.6, 0.12, 0.06, 0.0)
    area = sum(abs((bv[b][0] - bv[a][0]) * (bv[c][2] - bv[a][2])
                   - (bv[c][0] - bv[a][0]) * (bv[b][2] - bv[a][2])) * 0.5 for a, b, c in bt)
    assert abs(area - (0.4 * 0.6 - math.pi * 0.12 ** 2)) < 0.01 * 0.4 * 0.6, "baffle has corner gaps"

    # bake: identity is a no-op; 90deg about Z sends +x -> +y; scale scales; translate offsets
    assert bake([(1, 2, 3)])[0] == (1.0, 2.0, 3.0), "identity bake"
    x, y, z = bake([(1, 0, 0)], rot=(0, 0, 90))[0]
    assert abs(x) < 1e-9 and abs(y - 1) < 1e-9 and abs(z) < 1e-9, "yaw 90: +x -> +y"
    x, y, z = bake([(1, 1, 1)], pos=(10, 0, 0), scale=(2, 2, 2))[0]
    assert (x, y, z) == (12.0, 2.0, 2.0), "scale then translate"

    # export groups BY LAYER: two props on 'wood' share one `g wood`; indices continue across props
    p1 = Prop("primitive", "box", layer="wood")
    p2 = Prop("primitive", "box", pos=(5, 0, 0), layer="wood")
    p3 = Prop("primitive", "box", pos=(0, 5, 0), layer="glass")
    obj = export_groups([p1, p2, p3])
    assert obj.count("g wood") == 1 and obj.count("g glass") == 1, "one group per layer, shared"
    assert obj.count("\nv ") == 24, "3 boxes -> 24 verts"                    # 8*3
    assert obj.count("\nf ") == 36, "3 boxes -> 36 faces"                    # 12*3
    # faces reference valid, in-range, 1-based indices
    vmax = obj.count("\nv ")
    for ln in obj.splitlines():
        if ln.startswith("f "):
            assert all(1 <= int(i) <= vmax for i in ln[2:].split()), "face index in range"

    # Prop round-trips through dict (for the .json sidecar)
    d = p2.to_dict(); assert Prop.from_dict(d).pos == [5, 0, 0], "prop dict round-trip"
    print("props3d self-check OK")


if __name__ == "__main__":
    _selfcheck()
