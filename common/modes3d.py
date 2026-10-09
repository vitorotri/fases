#!/usr/bin/env python3
"""Room modes (didactic): eigenmodes of the room's AIR on a voxel grid, sampled on sheets at a height h
above EVERY floor level (raked floors, steps, balconies), for the GUI's Run tab -> Modes.

Not the FVM: a lossless Helmholtz eigenproblem -Lap(p) = k^2 p on the room's own geometry.
  * Rigid walls (every mesh surface, thin ones included): zero normal particle velocity, dp/dn = 0.
  * Domain faces with the PML ON and no wall there: p = 0 (pressure release, a stand-in for "open").
    Domain faces with the PML off: rigid, like the solver's box.
Absorption is ignored (ideal room); the planned FVM Mode view is the one with the real materials.

Run from the GUI, or by hand:
    python modes3d.py --geom geom.npz --fmax 70.8 --h 1.2 --c 343 --pml "x+,z+" --out modes.npz
geom.npz: V (N,3) vertices, T (M,3) triangles (active layers, no bounding box), bmin, bmax (domain box).
Self-check (rectangular room vs the textbook modes):  python modes3d.py
"""
import argparse
import numpy as np

PPW = 10.0            # cells per wavelength at fmax (frequencies corrected for the grid's dispersion)
MAX_CELLS = 60_000    # grid cap: keeps memory (~ few hundred MB) / time small (a room up to ~250 Hz)
MAX_MODES = 200       # lowest modes computed at most


def _grid(bmin, bmax, fmax, c, ppw=PPW, max_cells=MAX_CELLS):
    """Cell-centred grid over the domain box: (lo (3,), h (3,), n (3,)), coarsened if over the cell cap."""
    bmin, bmax = np.asarray(bmin, float), np.asarray(bmax, float)
    L = np.maximum(bmax - bmin, 1e-6)
    h0 = c / (ppw * fmax)
    h0 = max(h0, (np.prod(L) / max_cells) ** (1.0 / 3.0))
    n = np.maximum(np.ceil(L / h0).astype(np.int64), 2)
    return bmin, L / n, n


def _crossings(V, T, axis, grid, max_pairs=4_000_000):
    """Every crossing of the mesh by the rays along `axis` through the cell-centre lines of the grid.
    -> (rid, t): ray id = iu * n[va] + iv (u, v = the other two axes, ascending), t = coordinate along
    `axis`. The rays are offset by an irrational fraction of a cell, so they don't run exactly through
    mesh edges / vertices (the solver's SDF sign rays do the same)."""
    lo, h, n = grid
    ua, va = [a for a in range(3) if a != axis]
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    nrm = np.cross(B - A, C - A)
    keep = np.abs(nrm[:, axis]) > 1e-12 * (np.abs(nrm).sum(1) + 1e-300)   # triangles parallel to the ray: none
    A, B, C, nrm = A[keep], B[keep], C[keep], nrm[keep]
    if not len(A):
        return np.zeros(0, np.int64), np.zeros(0)
    ju, jv = 0.0137 * h[ua], 0.0091 * h[va]

    def rng(a, j):
        mn = np.minimum(np.minimum(A[:, a], B[:, a]), C[:, a])
        mx = np.maximum(np.maximum(A[:, a], B[:, a]), C[:, a])
        i0 = np.clip(np.ceil((mn - lo[a] - j) / h[a] - 0.5), 0, n[a] - 1).astype(np.int64)
        i1 = np.clip(np.floor((mx - lo[a] - j) / h[a] - 0.5), -1, n[a] - 1).astype(np.int64)
        return i0, np.maximum(i1 - i0 + 1, 0)
    i0u, cu = rng(ua, ju)
    i0v, cv = rng(va, jv)
    cnt = cu * cv
    d = (nrm * A).sum(1)
    rids, ts = [], []
    ends = np.cumsum(cnt)
    k0 = 0
    while k0 < len(A):                                   # chunks of <= max_pairs (triangle, ray) pairs
        base = ends[k0 - 1] if k0 else 0
        k1 = max(int(np.searchsorted(ends, base + max_pairs, side="right")), k0 + 1)
        sel = np.arange(k0, k1)
        c = cnt[sel]
        tot = int(c.sum())
        k0 = k1
        if tot == 0:
            continue
        tri = np.repeat(sel, c)
        loc = np.arange(tot) - np.repeat(np.cumsum(c) - c, c)       # pair index within its triangle
        cvr = cv[tri]
        iu = i0u[tri] + loc // cvr
        iv = i0v[tri] + loc % cvr
        pu = lo[ua] + (iu + 0.5) * h[ua] + ju
        pv = lo[va] + (iv + 0.5) * h[va] + jv
        au, av, bu, bv = A[tri, ua], A[tri, va], B[tri, ua], B[tri, va]
        cu_, cv_ = C[tri, ua], C[tri, va]
        e1 = (bu - au) * (pv - av) - (bv - av) * (pu - au)
        e2 = (cu_ - bu) * (pv - bv) - (cv_ - bv) * (pu - bu)
        e3 = (au - cu_) * (pv - cv_) - (av - cv_) * (pu - cu_)
        hit = ((e1 > 0) & (e2 > 0) & (e3 > 0)) | ((e1 < 0) & (e2 < 0) & (e3 < 0))
        tri, iu, iv, pu, pv = tri[hit], iu[hit], iv[hit], pu[hit], pv[hit]
        t = (d[tri] - nrm[tri, ua] * pu - nrm[tri, va] * pv) / nrm[tri, axis]
        rids.append(iu * n[va] + iv); ts.append(t)
    if not rids:
        return np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(rids), np.concatenate(ts)


class _Line:
    """Sorted crossings along one axis, queried as 'how many crossings below s on ray r'."""
    def __init__(self, rid, t, axis, grid):
        lo, h, n = grid
        self.lo, self.span, self.pad = lo[axis], (n[axis] + 4) * h[axis], 2.0 * h[axis]
        self.keys = np.sort(rid * self.span + np.clip(t - lo[axis] + self.pad, 0.0, self.span))

    def below(self, rid, s):
        """crossings of ray rid with t < s (arrays broadcast)."""
        q = rid * self.span + np.clip(s - self.lo + self.pad, 0.0, self.span)
        return np.searchsorted(self.keys, q) - np.searchsorted(self.keys, rid * self.span)

    def nearest(self, rid, s, above):
        """t of the nearest crossing of ray rid below s (above=False) / above s (True); NaN if none."""
        q = rid * self.span + np.clip(s - self.lo + self.pad, 0.0, self.span)
        i = np.searchsorted(self.keys, q)
        j = i if above else i - 1
        ok = (j >= 0) & (j < len(self.keys))
        jj = np.clip(j, 0, max(len(self.keys) - 1, 0))
        k = self.keys[jj] if len(self.keys) else np.zeros_like(q)
        ok &= np.floor(k / self.span) == rid
        return np.where(ok, k - rid * self.span + self.lo - self.pad, np.nan)


def solve(V, T, bmin, bmax, fmax, c=343.0, h_listen=1.2, pml=(), ppw=PPW, max_cells=MAX_CELLS,
          max_modes=MAX_MODES, log=print):
    """-> dict(freqs (M,), pts (P,3) sheet tiles at h above every floor level, level (P,), vals (M,P) each
    mode on the tiles normalised to its 3D maximum, tile (2,) tile size, info str)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import eigsh
    V = np.asarray(V, float); T = np.asarray(T, np.int64)
    grid = _grid(bmin, bmax, fmax, c, ppw, max_cells)
    lo, h, n = grid
    hi = lo + h * n
    ppw_got = c / fmax / float(np.max(h))
    log(f"[modes] grid {n[0]} x {n[1]} x {n[2]} = {int(np.prod(n))} cells, h = "
        f"{' x '.join(f'{x:.3f}' for x in h)} m ({ppw_got:.1f} cells per wavelength at {fmax:.1f} Hz)")
    if ppw_got < 5.0:
        log(f"[modes] !! coarse grid (cell cap {max_cells}): frequencies near {fmax:.0f} Hz are approximate")
    I = np.meshgrid(*[np.arange(k) for k in n], indexing="ij")
    inside = np.zeros(tuple(n), np.int8)
    lines, open_wall = {}, {}
    for a in range(3):                                   # rays along x, y, z: parity + blocking surfaces
        ua, va = [b for b in range(3) if b != a]
        rid_c, t_c = _crossings(V, T, a, grid)
        ln = _Line(rid_c, t_c, a, grid)
        lines[a] = ln
        rid = I[ua] * n[va] + I[va]
        s = lo[a] + (I[a] + 0.5) * h[a]
        cnt = ln.below(rid, s)
        inside += (cnt % 2).astype(np.int8)
        lines[(a, "cnt")] = cnt
        # a wall between a boundary cell and its domain face (then that face is rigid even with a PML)
        r0 = np.take(rid, 0, axis=a); s0 = lo[a] + 0.5 * h[a]
        r1 = np.take(rid, -1, axis=a); s1 = hi[a] - 0.5 * h[a]
        open_wall[(a, 0)] = ln.below(r0, s0) - ln.below(r0, lo[a] - 0.01 * h[a]) == 0
        open_wall[(a, 1)] = ln.below(r1, hi[a] + 0.01 * h[a]) - ln.below(r1, s1) == 0
        log(f"[modes] axis {'xyz'[a]}: {len(t_c)} surface crossings")
    air = inside >= 2                                    # majority of the three axes (like the particles)
    idx = -np.ones(tuple(n), np.int64)
    idx[air] = np.arange(int(air.sum()))
    rows, cols, w = [], [], []
    for a in range(3):                                   # open faces between neighbouring air cells
        cnt = lines[(a, "cnt")]
        sl0 = [slice(None)] * 3; sl1 = [slice(None)] * 3
        sl0[a] = slice(0, -1); sl1[a] = slice(1, None)
        sl0, sl1 = tuple(sl0), tuple(sl1)
        ok = air[sl0] & air[sl1] & (cnt[sl1] == cnt[sl0])            # no surface in between
        i, j = idx[sl0][ok], idx[sl1][ok]
        rows += [i, j]; cols += [j, i]; w += [np.full(len(i), 1.0 / h[a] ** 2)] * 2
    N = int(air.sum())
    if N < 8:
        raise ValueError("no air found inside the scene (is the mesh closed?)")
    Adj = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()
    ncomp, lab = connected_components(Adj, directed=False)
    main = np.argmax(np.bincount(lab))
    keep = lab == main
    log(f"[modes] air: {N} cells, main volume {int(keep.sum())} cells "
        f"({int(keep.sum()) * float(np.prod(h)):.1f} m^3)" + (f", {ncomp - 1} small pockets ignored" if ncomp > 1 else ""))
    sub = np.nonzero(keep)[0]
    Adj = Adj[sub][:, sub]
    diag = np.asarray(Adj.sum(1)).ravel()
    faces = {"x-": (0, 0), "x+": (0, 1), "y-": (1, 0), "y+": (1, 1), "z-": (2, 0), "z+": (2, 1)}
    kmap = -np.ones(N, np.int64); kmap[sub] = np.arange(len(sub))
    nd = 0
    for f in pml:                                        # p = 0 just outside an open PML face
        a, side = faces[f]
        sl = [slice(None)] * 3; sl[a] = 0 if side == 0 else -1
        bnd = air[tuple(sl)] & open_wall[(a, side)]
        k = kmap[idx[tuple(sl)][bnd]]
        k = k[k >= 0]
        diag[k] += 2.0 / h[a] ** 2
        nd += len(k)
    if pml:
        log(f"[modes] PML faces {','.join(pml)}: {nd} boundary cells set to p = 0")
    from scipy.sparse import diags
    Lap = (diags(diag) - Adj).tocsc()
    M = len(sub)
    vol = M * float(np.prod(h))
    area = 2.0 * (vol ** (2.0 / 3.0)) * 3.0              # rough surface for the Weyl count
    weyl = 4 * np.pi * vol * fmax ** 3 / (3 * c ** 3) + np.pi * area * fmax ** 2 / (4 * c ** 2)
    nev = int(min(max_modes, max(10, 1.3 * weyl + 8), M - 2))
    hg = float(np.prod(h)) ** (1.0 / 3.0)
    for attempt in range(2):
        log(f"[modes] solving for the lowest {nev} modes...")
        lam, vec = eigsh(Lap, k=nev, sigma=-1e-4, which="LM")
        order = np.argsort(lam); lam, vec = lam[order], vec[:, order]
        k = 2.0 / hg * np.arcsin(np.clip(hg * np.sqrt(np.maximum(lam, 0.0)) / 2.0, 0.0, 1.0))  # grid dispersion
        f = c * k / (2 * np.pi)
        if f[-1] >= fmax or nev >= min(max_modes, M - 2) or attempt:
            break
        nev = int(min(max_modes, M - 2, nev * 1.6))      # Weyl under-counted: one more, larger solve
    sel = (f > 1.0) & (f <= fmax)                        # drop the constant (0 Hz) mode
    if f[-1] < fmax and nev >= max_modes:
        log(f"[modes] !! {max_modes} modes reach only {f[-1]:.1f} Hz: shown up to there")
        sel = f > 1.0
    f, vec = f[sel], vec[:, sel]
    vec = vec / np.maximum(np.abs(vec).max(0), 1e-30)
    # sheets: every vertical run of air in the main volume starts at a floor -> a tile at floor + h
    full = np.zeros(tuple(n), bool)
    flat_keep = np.zeros(N, bool); flat_keep[sub] = True
    full[air] = flat_keep
    col = kmap[np.where(full, idx, 0)]
    cz = lines[(2, "cnt")]
    link = full[:, :, :-1] & full[:, :, 1:] & (cz[:, :, 1:] == cz[:, :, :-1])   # same run: air, no surface between
    z1 = np.zeros(tuple(n[:2]) + (1,), bool)
    up = np.concatenate([z1, link], axis=2)              # the cell below continues this run
    dn = np.concatenate([link, z1], axis=2)              # the cell above continues this run
    sx, sy, sz = np.nonzero(full & ~up)                  # run starts (bottom cells), ordered per column
    ex, ey, ez = np.nonzero(full & ~dn)                  # run ends (top cells), same order
    zl = lines[2]
    rid = sx * n[1] + sy
    floor = zl.nearest(rid, lo[2] + (sz + 0.5) * h[2], above=False)
    floor = np.where(np.isnan(floor), lo[2], floor)
    ceil = zl.nearest(rid, lo[2] + (ez + 0.5) * h[2], above=True)
    ceil = np.where(np.isnan(ceil), hi[2], ceil)
    zs = floor + h_listen
    ok = zs < ceil - 0.05
    iz = np.clip(np.floor((zs - lo[2]) / h[2]).astype(np.int64), sz, ez)
    first = np.r_[True, (sx[1:] != sx[:-1]) | (sy[1:] != sy[:-1])]
    level = np.arange(len(sx)) - np.maximum.accumulate(np.where(first, np.arange(len(sx)), 0))
    sx, sy, iz, zs, level = sx[ok], sy[ok], iz[ok], zs[ok], level[ok]
    cell = col[sx, sy, iz]
    pts = np.stack([lo[0] + (sx + 0.5) * h[0], lo[1] + (sy + 0.5) * h[1], zs], 1).astype(np.float32)
    vals = vec[cell].T.astype(np.float16)                # (modes, tiles)
    info = (f"{len(f)} modes up to {f[-1] if len(f) else 0:.1f} Hz, {len(pts)} sheet tiles on "
            f"{int(level.max()) + 1 if len(level) else 0} level(s), h = {h_listen:.2f} m")
    log(f"[modes] {info}")
    return dict(freqs=f, pts=pts, level=level.astype(np.int16), vals=vals,
                tile=np.array([h[0], h[1]], np.float32), info=np.array(info))


def _box(L, open_x=False):
    """Triangles of an axis-aligned box [0,L] (without its two x walls if open_x)."""
    Lx, Ly, Lz = L
    V = np.array([[0, 0, 0], [Lx, 0, 0], [Lx, Ly, 0], [0, Ly, 0],
                  [0, 0, Lz], [Lx, 0, Lz], [Lx, Ly, Lz], [0, Ly, Lz]], float)
    F = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],                 # floor, ceiling
         [0, 5, 1], [0, 4, 5], [3, 2, 6], [3, 6, 7]]                 # y walls
    if not open_x:
        F += [[0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]]            # x walls
    return V, np.array(F)


def _selfcheck():
    c, L = 343.0, (5.0, 3.5, 2.8)
    analytic = sorted(c / 2 * np.sqrt((l / L[0]) ** 2 + (m / L[1]) ** 2 + (k / L[2]) ** 2)
                      for l in range(8) for m in range(8) for k in range(8) if l + m + k)
    V, T = _box(L)
    r = solve(V, T, (0, 0, 0), L, 120.0, c, h_listen=1.2, log=lambda s: None)
    got, ref = r["freqs"][:15], np.array(analytic[:15])
    err = np.max(np.abs(got - ref) / ref)
    print(f"rigid box {L}: first 15 modes, max error {100 * err:.2f} %")
    assert err < 0.03, (got, ref)
    assert abs(r["pts"][:, 2] - 1.2).max() < 1e-4, "sheet at 1.2 m above the floor"
    # open x sides with PML -> p = 0 there: sin modes in x (l >= 1), cos in y, z
    V, T = _box(L, open_x=True)
    r = solve(V, T, (0, 0, 0), L, 120.0, c, pml=("x-", "x+"), log=lambda s: None)
    analytic = sorted(c / 2 * np.sqrt((l / L[0]) ** 2 + (m / L[1]) ** 2 + (k / L[2]) ** 2)
                      for l in range(1, 9) for m in range(8) for k in range(8))
    got, ref = r["freqs"][:10], np.array(analytic[:10])
    err = np.max(np.abs(got - ref) / ref)
    print(f"box open in x with PML (p = 0): first 10 modes, max error {100 * err:.2f} %")
    assert err < 0.04, (got, ref)
    # a balcony slab half way up -> two floor levels in the columns under it
    V, T = _box(L)
    S = np.array([[0, 0, 1.4], [2.5, 0, 1.4], [2.5, L[1], 1.4], [0, L[1], 1.4]], float)
    V2 = np.vstack([V, S]); T2 = np.vstack([T, [[8, 9, 10], [8, 10, 11]]])
    r = solve(V2, T2, (0, 0, 0), L, 80.0, c, h_listen=0.5, log=lambda s: None)
    lv = r["level"]
    assert lv.max() == 1 and np.allclose(np.sort(np.unique(np.round(r["pts"][:, 2], 3))), [0.5, 1.9]), \
        "two sheets: 0.5 m above the floor and above the slab"
    print("balcony slab: sheets on 2 levels (0.5 m and 1.9 m)")
    print("modes3d self-check OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Room eigenmodes on sheets at a height above every floor level.")
    ap.add_argument("--geom", help="npz with V, T, bmin, bmax")
    ap.add_argument("--fmax", type=float, help="highest frequency [Hz]")
    ap.add_argument("--h", type=float, default=1.2, help="sheet height above each floor [m]")
    ap.add_argument("--c", type=float, default=343.0, help="speed of sound [m/s]")
    ap.add_argument("--pml", default="", help="domain faces with the PML on, e.g. 'x+,z+' (p = 0 there)")
    ap.add_argument("--ppw", type=float, default=PPW, help="cells per wavelength at fmax")
    ap.add_argument("--out", help="output npz")
    a = ap.parse_args()
    if not a.geom:
        _selfcheck()
    else:
        g = np.load(a.geom)
        pml = tuple(s for s in a.pml.replace(" ", "").split(",") if s)
        res = solve(g["V"], g["T"], g["bmin"], g["bmax"], a.fmax, a.c, a.h, pml, ppw=a.ppw,
                    log=lambda s: print(s, flush=True))           # flushed: the GUI log shows it live
        np.savez(a.out, **res)
        print(f"[modes] wrote {a.out}", flush=True)
