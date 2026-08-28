#!/usr/bin/env python3
"""Live 2D staggered-grid FDTD on a slice plane — a fast, didactic particle animation.

This is NOT the real 3D field: it is an independent 2D acoustic sim seeded by the scene's
sources (projected onto the plane), purely for a pleasant/teaching visual. Normalized units
(c = dx = rho = 1). Walls are the real room cross-section as a rigid mask (v = 0 on any face
touching a non-room cell) — totally reflective, no absorption, so a pulse rings forever.

The GUI builds one per visible slice plane, calls step() each animation tick, and renders
`disp` (H, W, 2) = per-cell (col, row) displacement in cells, through the existing GL particle
path (same as before). Source types: PCS/Ricker -> one zero-mean Ricker pulse (same for both,
fixed amplitude); Sinusoid -> continuous sine injection.
"""
import numpy as np

P_REF = 20e-6            # Pa, 0 dB SPL reference
DEFAULT_SPL = 94.0       # dB, fallback when a source has no level (94 dB SPL = 1 Pa)
# ponytail: calibration knob — maps a source's SPL (Pa) to the 2D-sim injection amplitude. The
# display gain auto-normalises brightness, so this mainly sets RELATIVE loudness between sources and
# the working numeric range. Tune against tests; the user will report values.
SPL_AMP_CALIB = 1.5


def spl_to_amp(spl_db):
    """Source SPL [dB] -> injection amplitude for the sim (pressure at the '1 m' ref, times calib)."""
    return P_REF * 10.0 ** (float(spl_db) / 20.0) * SPL_AMP_CALIB


class Slice2DSim:
    def __init__(self, inside, sources, src_opt=0, f=0.06, cfl=0.9,
                 absorb=(False, False, False, False)):
        """inside: (H,W) bool room mask. sources: iterable of (row, col[, spl_dB]) on the plane; the
        injection amplitude is spl_to_amp(spl) per source (SPL missing -> DEFAULT_SPL).
        src_opt: 0/1 = PCS/Ricker (single-node impulse), 2 = sinusoid. f: normalized freq (cyc/step).
        absorb: (top, bottom, left, right) grid edges to make outgoing (PML sides) vs rigid."""
        self.inside = np.ascontiguousarray(inside, bool)
        self.H, self.W = self.inside.shape
        self.p  = np.zeros((self.H, self.W),     np.float32)
        self.vx = np.zeros((self.H, self.W - 1), np.float32)   # x-faces between cols i,i+1
        self.vy = np.zeros((self.H - 1, self.W), np.float32)   # y-faces between rows j,j+1
        self.rx = np.zeros((self.H, self.W),     np.float32)   # cumulative displacement (cells)
        self.ry = np.zeros((self.H, self.W),     np.float32)
        self.disp = np.zeros((self.H, self.W, 2), np.float32)
        # rigid walls: a face lives only if BOTH adjacent cells are inside the room
        self.vx_ok = self.inside[:, 1:] & self.inside[:, :-1]
        self.vy_ok = self.inside[1:, :] & self.inside[:-1, :]
        self.dt = float(cfl) / np.sqrt(2.0)                    # < 1/sqrt(2) is stable in 2D
        self.absorb = tuple(bool(a) for a in absorb)           # (top, bottom, left, right)
        self.f, self.src_opt = float(f), int(src_opt)
        self.n = 0
        # sources snapped to the nearest in-room cell so a source projected onto a wall still fires;
        # per-source amplitude from its SPL (dB) -> Pa via spl_to_amp
        ys, xs = np.where(self.inside)
        self.sources = []                                      # (row, col, amp_linear)
        for s in sources:
            r = int(np.clip(s[0], 0, self.H - 1)); c = int(np.clip(s[1], 0, self.W - 1))
            if not self.inside[r, c] and len(ys):
                k = int(np.argmin((ys - r) ** 2 + (xs - c) ** 2)); r, c = int(ys[k]), int(xs[k])
            self.sources.append((r, c, spl_to_amp(s[2] if len(s) > 2 else DEFAULT_SPL)))
        self.amp = max([a for _, _, a in self.sources], default=spl_to_amp(DEFAULT_SPL))  # ref for gain/checks
        # absorbing (PML) edges = a damping sponge: p and v ramped down over a band near the edge,
        # each step, so outgoing waves die instead of reflecting. Rigid (non-PML) edges stay at 1.
        top, bot, left, right = self.absorb
        nb = max(4, min(self.H, self.W) // 8)
        ramp = np.linspace(0.82, 1.0, nb, dtype=np.float32)     # 0.82/step at the very edge
        d = np.ones((self.H, self.W), np.float32)
        if top:   d[:nb, :]  = np.minimum(d[:nb, :],  ramp[:, None])
        if bot:   d[-nb:, :] = np.minimum(d[-nb:, :], ramp[::-1, None])
        if left:  d[:, :nb]  = np.minimum(d[:, :nb],  ramp[None, :])
        if right: d[:, -nb:] = np.minimum(d[:, -nb:], ramp[None, ::-1])
        self.damp   = d
        self.damp_x = 0.5 * (d[:, :-1] + d[:, 1:])              # to x-faces
        self.damp_y = 0.5 * (d[:-1, :] + d[1:, :])              # to y-faces
        self._absorbing = any(self.absorb)

    def reset(self):
        """Zero the field back to t=0 (used after the gain warm-up so playback starts at the pulse)."""
        self.p[:] = 0; self.vx[:] = 0; self.vy[:] = 0
        self.rx[:] = 0; self.ry[:] = 0; self.disp[:] = 0; self.n = 0

    def _inject(self):
        """PCS/Ricker: one high-amplitude single-node impulse at n==0 (then rings). Sinusoid:
        continuous sine every step."""
        if self.src_opt == 2:                                  # sinusoid: continuous
            s = np.sin(2.0 * np.pi * self.f * self.n)
            for r, c, a in self.sources:
                self.p[r, c] += a * s
        elif self.n == 0:                                      # PCS/Ricker: one soft node impulse
            for r, c, a in self.sources:
                self.p[r, c] += a

    def step(self, substeps=2):
        """Advance `substeps` FDTD steps and refresh `disp`. A few substeps/tick = livelier waves."""
        for _ in range(substeps):
            # velocity <- -grad(p), then kill faces at/through walls (rigid)
            self.vx -= self.dt * (self.p[:, 1:] - self.p[:, :-1]); self.vx *= self.vx_ok
            self.vy -= self.dt * (self.p[1:, :] - self.p[:-1, :]); self.vy *= self.vy_ok
            # pressure <- -div(v)  (zero-pad interior faces to the cell grid; a zero boundary face
            # is a rigid wall -- correct for non-PML sides)
            vxf = np.zeros((self.H, self.W + 1), np.float32); vxf[:, 1:self.W] = self.vx
            vyf = np.zeros((self.H + 1, self.W), np.float32); vyf[1:self.H, :] = self.vy
            self.p -= self.dt * ((vxf[:, 1:] - vxf[:, :-1]) + (vyf[1:, :] - vyf[:-1, :]))
            self.p *= self.inside                              # exterior cells stay silent
            self._inject()
            # integrate displacement from face velocities interpolated to cell centres
            self.rx += self.dt * 0.5 * (vxf[:, 1:] + vxf[:, :-1])
            self.ry += self.dt * 0.5 * (vyf[1:, :] + vyf[:-1, :])
            # PML sponge: ramp p & v down near absorbing edges so outgoing waves die (not reflect)
            if self._absorbing:
                self.p *= self.damp; self.rx *= self.damp; self.ry *= self.damp
                self.vx *= self.damp_x; self.vy *= self.damp_y
            self.n += 1
        self.disp[..., 0] = self.rx
        self.disp[..., 1] = self.ry
        return self.disp


def _selfcheck():
    # 1) free-field square, centred pulse -> stays finite, stays symmetric (no bias)
    H = W = 81
    inside = np.ones((H, W), bool)
    sim = Slice2DSim(inside, [(H // 2, W // 2)], src_opt=1)
    for _ in range(300):
        sim.step(1)
    assert np.isfinite(sim.p).all(), "blew up / NaN"
    assert np.abs(sim.p).max() < 20 * sim.amp, "unstable growth"
    # symmetry: left/right and up/down mirror of a centred source in a square must match
    assert np.allclose(sim.p, sim.p[:, ::-1], atol=1e-3), "left-right asymmetry"
    assert np.allclose(sim.p, sim.p[::-1, :], atol=1e-3), "up-down asymmetry"

    # 2) rigid box conserves energy (no source after the pulse) -> doesn't decay to zero
    e_mid = float((sim.p ** 2).sum())
    for _ in range(300):
        sim.step(1)
    e_late = float((sim.p ** 2).sum())
    assert e_late > 0.2 * e_mid, "rigid box should not bleed energy away"
    assert np.abs(sim.p).max() < 20 * sim.amp, "late-time instability"

    # 3) displacement stays bounded (no integration runaway)
    assert np.isfinite(sim.disp).all() and np.abs(sim.disp).max() < 50.0 * sim.amp, "displacement drift"

    # 4) rigid mask really blocks: a walled-off half stays exactly silent
    inside2 = np.ones((40, 40), bool); inside2[:, 20:] = False
    s2 = Slice2DSim(inside2, [(20, 5)], src_opt=1)
    for _ in range(120):
        s2.step(1)
    assert np.abs(s2.p[:, 20:]).max() == 0.0, "energy leaked through a rigid wall"

    # 5) absorbing edges bleed energy away vs a rigid box (same pulse, all 4 edges open)
    big = np.ones((81, 81), bool)
    rigid = Slice2DSim(big, [(40, 40)], src_opt=1)
    openb = Slice2DSim(big, [(40, 40)], src_opt=1, absorb=(True, True, True, True))
    for _ in range(500):
        rigid.step(1); openb.step(1)
    e_rigid = float((rigid.p ** 2).sum()); e_open = float((openb.p ** 2).sum())
    assert np.isfinite(e_open) and e_open < 0.5 * e_rigid, "absorbing edges did not remove energy"
    print("slice2d self-check OK")


if __name__ == "__main__":
    _selfcheck()
