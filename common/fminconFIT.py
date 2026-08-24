#!/usr/bin/env python3
"""
Vector fitting of normal-incedance reflection coefficient using scikit-rf (skrf).

This is a Python translation of the provided MATLAB script.
- MATLAB vectfit3(...)  ->  skrf.vectorFitting.VectorFitting.vector_fit(...)
- Saving .mat          ->  scipy.io.savemat(...)

Notes (important and intentional):
- In the MATLAB script, s = j*omega with omega = 2*pi*f (rad/s).
  Therefore poles should also be in rad/s. Here we initialize poles in rad/s.
- opts.asymp = 1 (no constant term) is mapped to fit_constant=False and fit_proportional=False.
"""

import numpy as np
import skrf as rf
import matplotlib.pyplot as plt
import os
import sys
import glob
import shutil
import struct
from scipy.io import savemat
from scipy.interpolate import PchipInterpolator
from scipy.signal import hilbert
from scipy.optimize import minimize, NonlinearConstraint
from skrf.vectorFitting import VectorFitting

def min_phase_octave_hilbert(freq_hz, R_oct,
                             fc=np.array([63,125,250,500,1000,2000,4000,8000], float),
                             n_w=4096,
                             ext_oct=2,
                             taper_frac=0.08,
                             mag_floor=1e-6,
                             mag_ceil=1-1e-6):
    freq_hz = np.asarray(freq_hz, float)
    R_oct = np.clip(np.asarray(R_oct, float), mag_floor, mag_ceil)

    # --- make a dense frequency grid (log is fine for interpolation) ---
    f0, f1 = float(freq_hz[0]), float(freq_hz[-1])
    f_lo = f0 / (2.0**ext_oct)
    f_hi = f1 * (2.0**ext_oct)
    f_dense = np.exp(np.linspace(np.log(f_lo), np.log(f_hi), 4096))

    # --- interpolate ln|R| vs ln f using PCHIP + constant extension ---
    lfc = np.log(fc)
    yfc = np.log(R_oct)
    pchip = PchipInterpolator(lfc, yfc, extrapolate=True)

    lf_dense = np.log(f_dense)
    lnmag_dense = pchip(lf_dense)
    lnmag_dense[lf_dense < lfc[0]]  = yfc[0]
    lnmag_dense[lf_dense > lfc[-1]] = yfc[-1]

    # --- enforce passivity at magnitude level BEFORE Hilbert ---
    mag_dense = np.clip(np.exp(lnmag_dense), mag_floor, mag_ceil)
    lnmag_dense = np.log(mag_dense)

    # --- resample ln|R| onto uniform linear omega grid (this is key) ---
    w0, w1 = 2*np.pi*f_lo, 2*np.pi*f_hi
    w_u = np.linspace(w0, w1, int(n_w))
    lnmag_u = np.interp(w_u, 2*np.pi*f_dense, lnmag_dense)

    # --- mirror + taper to reduce wrap/edge artifacts ---
    ln_ext = np.r_[lnmag_u, lnmag_u[::-1]]
    N = len(ln_ext)
    T = max(16, int(taper_frac*N))
    win = np.ones(N)
    t = np.linspace(0, np.pi/2, T)
    win[:T] = np.sin(t)**2
    win[-T:] = win[:T][::-1]
    ln_ext = ln_ext*win + (1-win)*np.linspace(ln_ext[0], ln_ext[-1], N)

    # --- Hilbert -> minimum-phase ---
    phi_ext = -np.imag(hilbert(ln_ext))
    phi_u = phi_ext[:len(w_u)]

    # --- back to your target grid ---
    w_t = 2*np.pi*freq_hz
    lnmag_t = np.interp(w_t, w_u, lnmag_u)
    phi_t   = np.interp(w_t, w_u, phi_u)

    mag_t = np.clip(np.exp(lnmag_t), mag_floor, mag_ceil)
    phi_t = np.unwrap(phi_t)
    R_t = mag_t * np.exp(1j*phi_t)
    return R_t, mag_t, phi_t

def min_phase_octave_cepstrum(freq_hz, R_oct,
                             fc=np.array([63,125,250,500,1000,2000,4000,8000], float),
                             n_w=2048, ext_oct=2,
                             mag_floor=1e-2, mag_ceil=1-1e-12):
    freq_hz = np.asarray(freq_hz, float)
    R_oct = np.clip(np.asarray(R_oct, float), mag_floor, mag_ceil)

    # Data-driven one-octave extension below the lowest given band (31.5 Hz). The
    # user only supplies 63..8000 Hz, but the FVM solver needs the reflection below
    # 63 Hz, where a flat hold is physically wrong. Extrapolate ln|R| along the slope
    # of the two lowest bands (63,125) -> 31.5 Hz follows the material's OWN trend
    # (absorbers rise toward rigid, resonant ones can fall). Below 31.5 Hz the curve
    # then holds flat, but only after one honest octave of slope.
    fc = np.asarray(fc, float)
    slope = (np.log(R_oct[1]) - np.log(R_oct[0])) / (np.log(fc[1]) - np.log(fc[0]))
    f_ext = fc[0] / 2.0                                    # 31.5 Hz
    R_ext = np.clip(np.exp(np.log(R_oct[0]) + slope*(np.log(f_ext) - np.log(fc[0]))),
                    mag_floor, mag_ceil)
    fc    = np.r_[f_ext, fc]
    R_oct = np.r_[R_ext, R_oct]

    f0, f1 = float(freq_hz[0]), float(freq_hz[-1])
    f_lo = max(1e-6, f0 / (2.0**ext_oct))
    f_hi = f1 * (2.0**ext_oct)

    # ---- interpolate ln|R| vs ln f (PCHIP) on a dense log grid ----
    lfc = np.log(fc)
    yfc = np.log(R_oct)
    pchip = PchipInterpolator(lfc, yfc, extrapolate=True)

    # ---- uniform omega grid (include 0..wmax for cepstrum periodicity) ----
    wmax = 2*np.pi*f_hi
    w_u = np.linspace(0.0, wmax, int(n_w))
    f_u = np.maximum(w_u/(2*np.pi), f_lo)
    lf_u = np.log(f_u)

    lnmag_u = pchip(lf_u)
    lnmag_u[lf_u < lfc[0]]  = yfc[0]
    lnmag_u[lf_u > lfc[-1]] = yfc[-1]

    # enforce passivity in magnitude (important when close to 1)
    mag_u = np.clip(np.exp(lnmag_u), mag_floor, mag_ceil)
    lnmag_u = np.log(mag_u)

    # ---- build even periodic sequence for FFT: [0..wmax] + mirror(excluding endpoints) ----
    x = lnmag_u
    x_ext = np.r_[x, x[-2:0:-1]]  # length 2*(n_w-1)

    # ---- real cepstrum -> minimum phase ----
    c = np.fft.ifft(x_ext).real
    n = len(c)

    c_min = np.zeros_like(c)
    c_min[0] = c[0]
    c_min[1:n//2] = 2.0*c[1:n//2]  # double positive quefrency
    # c_min[n//2] left as 0 for even length; OK

    X_min = np.fft.fft(c_min)
    H_ext = np.exp(X_min)  # complex spectrum of minimum-phase response

    # recover on 0..wmax grid portion
    H_u = H_ext[:len(x)]  # corresponds to w_u

    # ---- interpolate to target freq grid ----
    w_t = 2*np.pi*freq_hz
    # interpolate real/imag separately
    Re = np.interp(w_t, w_u, np.real(H_u))
    Im = np.interp(w_t, w_u, np.imag(H_u))
    R_t = Re + 1j*Im

    # also return mag/phase
    mag_t = np.abs(R_t)
    phi_t = np.unwrap(np.angle(R_t))
    return R_t, mag_t, phi_t

def Z_from_R(R, z0):
    R = np.asarray(R, complex)
    return z0 * (1 + R) / (1 - R)

def R_expected(Rinf, A, B, C, alpha, beta, zeta, w):
    s = 1j*w
    R = (Rinf + 0j) * np.ones_like(s)

    if len(zeta):
        R += (A[:, None] / (zeta[:, None] + s[None, :])).sum(axis=0)

    if len(alpha):
        den = (s[None, :] + alpha[:, None])**2 + beta[:, None]**2
        R += ((B[:, None] * s[None, :] + C[:, None] * beta[:, None] + alpha[:, None] * B[:, None])/ den).sum(axis=0)

    return R


# ============================================================
# OPTION A (Hornikx/Cotte style): constrained continuation fit
# - Minimize real-part mismatch
# - Subject to imag-part mismatch SSE <= epsilon
# - Hard bounds on alpha/beta/zeta via sigmoid mapping
# ============================================================

def _sigmoid(x):
    # numerically stable sigmoid
    x = np.asarray(x, float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def _pack_params(Rinf, A, B, C, uzeta, ualpha, ubeta):
    return np.r_[np.atleast_1d(Rinf).astype(float),
                 np.asarray(A, float).ravel(),
                 np.asarray(B, float).ravel(),
                 np.asarray(C, float).ravel(),
                 np.asarray(uzeta, float).ravel(),
                 np.asarray(ualpha, float).ravel(),
                 np.asarray(ubeta, float).ravel()]

def _unpack_params(x, K, L, zeta_max, alpha_max, beta_max):
    x = np.asarray(x, float).ravel()
    p = 0
    Rinf = float(x[p]); p += 1

    A = x[p:p+K]; p += K
    B = x[p:p+L]; p += L
    C = x[p:p+L]; p += L

    uzeta  = x[p:p+K]; p += K
    ualpha = x[p:p+L]; p += L
    ubeta  = x[p:p+L]; p += L

    # hard-bounded poles
    zeta  = zeta_max  * _sigmoid(uzeta)  if K else np.array([], float)
    alpha = alpha_max * _sigmoid(ualpha) if L else np.array([], float)
    beta  = beta_max  * _sigmoid(ubeta)  if L else np.array([], float)

    return Rinf, A, B, C, zeta, alpha, beta, uzeta, ualpha, ubeta

def _objective_real_sse(x, K, L, zeta_max, alpha_max, beta_max, w, R_tgt):
    Rinf, A, B, C, zeta, alpha, beta, *_ = _unpack_params(x, K, L, zeta_max, alpha_max, beta_max)
    Rm = R_expected(Rinf, A, B, C, alpha, beta, zeta, w)
    e = (Rm - R_tgt)
    return float(np.sum(np.real(e)**2))

def _constraint_imag_sse(x, K, L, zeta_max, alpha_max, beta_max, w, R_tgt):
    Rinf, A, B, C, zeta, alpha, beta, *_ = _unpack_params(x, K, L, zeta_max, alpha_max, beta_max)
    Rm = R_expected(Rinf, A, B, C, alpha, beta, zeta, w)
    e = (Rm - R_tgt)
    return float(np.sum(np.imag(e)**2))

def hornikx_cotte_constrained_fit(
    R_tgt, w,
    Rinf0, A0, B0, C0, zeta0, alpha0, beta0,
    dt, omega_cutoff,
    max_outer=8,
    eps_reduction=0.5,
    eps_floor=1e-12,
    maxiter=4000,
    passivity_margin=1e-3,   # keep |R_model| <= 1 - margin (Re Z >= 0)
    passivity_weight=1e3     # penalty weight; large enough to dominate overshoot
):
    '''
    Bounded refinement of the ADE coefficients.

    The poles are HARD-bounded to the SSP-RK3 discrete-stability limits via a
    sigmoid map (zeta,alpha <= 2.5127/dt ; beta <= min(1.732/dt, omega_cutoff)),
    so the optimiser can never leave the stable region. Starting from the
    vector-fit solution we minimise the total complex least-squares error in
    that bounded parametrisation.

    (Earlier this used a trust-constr continuation with a NonlinearConstraint on
    the imaginary-part error; in practice that exhausted its evaluation budget on
    the first step and fell back to the vector-fit result without refining. A
    plain bounded BFGS in the sigmoid coordinates is robust, converges, and
    respects exactly the same hard bounds.)

    Returns: (Rinf, A, B, C, zeta, alpha, beta)
    '''
    R_tgt = np.asarray(R_tgt, complex).ravel()
    w = np.asarray(w, float).ravel()

    K = int(len(zeta0))
    L = int(len(alpha0))

    zeta_max  = 2.5127 / dt
    alpha_max = 2.5127 / dt
    beta_max  = min(1.732 / dt, omega_cutoff)

    def inv_sigmoid(y):
        y = np.clip(np.asarray(y, float), 1e-12, 1-1e-12)
        return np.log(y/(1-y))

    uzeta0  = inv_sigmoid(zeta0 / zeta_max)   if K else np.array([], float)
    ualpha0 = inv_sigmoid(alpha0 / alpha_max) if L else np.array([], float)
    ubeta0  = inv_sigmoid(beta0 / beta_max)   if L else np.array([], float)

    x0 = _pack_params(Rinf0, A0, B0, C0, uzeta0, ualpha0, ubeta0)

    # Passivity is a GLOBAL property (all omega>=0), but the fit grid `w` starts at
    # 20 Hz. Overshoot lives below that and between grid points, so check on a
    # denser grid from 1 Hz up to the fit band edge. |R|<=1 <=> Re(Z)>=0 for a
    # locally-reacting surface, so penalising |R|>1-margin removes the non-passivity.
    f_hi_chk = float(np.max(w)) / (2*np.pi)
    w_chk = 2*np.pi*np.logspace(np.log10(1.0), np.log10(max(f_hi_chk, 2.0)), 512)

    def total_sse(x):
        Rinf, A, B, C, zeta, alpha, beta, *_ = _unpack_params(
            x, K, L, zeta_max, alpha_max, beta_max)
        e = R_expected(Rinf, A, B, C, alpha, beta, zeta, w) - R_tgt
        data = float(np.sum(e.real**2 + e.imag**2))
        Rc = R_expected(Rinf, A, B, C, alpha, beta, zeta, w_chk)
        ov = np.maximum(np.abs(Rc) - (1.0 - passivity_margin), 0.0)
        return data + passivity_weight * float(np.sum(ov**2))

    sse0 = total_sse(x0)
    if os.environ.get("VFIT_VERBOSE", "0") == "1":
        print(f"[A] bounds: zeta,alpha<= {zeta_max:.6g}, beta<= {beta_max:.6g}")
        print(f"[A] initial total SSE = {sse0:.6e}")

    res = minimize(total_sse, x0, method="BFGS",
                   options=dict(maxiter=int(maxiter)))

    # Use the optimiser's iterate whether or not SciPy flags success — for this
    # flat, well-bounded landscape res.x is the best point reached; only reject
    # it if it is somehow worse than where we started.
    x_best = res.x if total_sse(res.x) <= sse0 else x0
    if os.environ.get("VFIT_VERBOSE", "0") == "1":
        print(f"[A] refined total SSE = {total_sse(x_best):.6e}  "
              f"(success={res.success}, nfev={res.nfev})")

    Rinf, A, B, C, zeta, alpha, beta, *_ = _unpack_params(
        x_best, K, L, zeta_max, alpha_max, beta_max)
    return Rinf, A, B, C, zeta, alpha, beta



def R_from_alpha(alpha):
    # |R| from absorption coefficient alpha (energy): |R| = sqrt(1 - alpha).
    # The GUI material table stores alpha per octave band; this matches vectFIT.
    return np.sqrt(1.0 - np.asarray(alpha, float))


def _read_soft_registry(mat_dir):
    """Parse ALL.mat (one '<name>  <flag>' line per material; 0=hard, 1=soft)
    from the material directory. Returns {name: bool soft}. The hard/soft flag
    is kept OUT of the per-material .mat files (8 alphas only) and lives here,
    shared with the GUI — same convention as vectFIT."""
    reg = {}
    p = os.path.join(mat_dir, "ALL.mat")
    if os.path.isfile(p):
        try:
            with open(p) as f:
                for line in f:
                    line = line.split("#")[0].strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        reg[parts[0]] = (parts[1] != "0")
        except Exception:
            pass
    return reg


def _alpha_key(mat_path, soft):
    """Hashable key from the 8 octave-band alphas + soft flag. Used to detect
    duplicate materials so their .fit/.imp can be copied instead of re-fitted."""
    arr = np.atleast_1d(np.loadtxt(mat_path)).ravel()[:8]
    return (tuple(np.round(arr, 8)), bool(soft))

def is_near_one(arr, atol=0.02):
    return np.all(np.isclose(arr, 1.0, atol=atol))

def R_constant_from_octaves(freq, R_oct, mag_ceil=1-1e-12):
    # choose a conservative constant close to the minimum band value
    Rinf = float(np.clip(np.min(R_oct), 0.0, mag_ceil))
    R = Rinf * np.ones_like(freq, dtype=complex)  # phase = 0
    return R, Rinf


# --------------------
# Block 1: User input (CLI arguments — non-interactive, project-aware)
# --------------------
# Path behavior intentionally matches the project-aware vectFIT.py:
#   python3 fminconFIT.py --mat walls.mat ceiling.mat floor.mat \
#                         --fitdir <dir for .fit files> \
#                         --config <config.txt> --poles <poles.txt> \
#                         [--refit] [--no-plot]
#
#   python3 fminconFIT.py --mat walls.mat --out walls.fit ...   (single material)
#
# Legacy compatibility:
#   python3 fminconFIT.py <material.mat> <sim_option>
# still works, but project-aware paths are preferred. If the material is inside
# <project>/input/mat, the script resolves config/, input/fit/, and poles.txt
# from that project even when launched from another directory.
#
# The fitting/refinement math below is intentionally the same as the previous
# fminconFIT.py. Only path handling, batching, plotting, and poles.txt bookkeeping
# were changed.

import argparse

_p = argparse.ArgumentParser(description="fmincon/Hornikx-Cotte fitting of reflection coefficient (ADE export)")
_p.add_argument("mat_pos", nargs="*", default=[], help="material .mat file(s) (positional, legacy)")
_p.add_argument("--mat",    dest="mat",    nargs="+", default=None, help="material .mat file(s)")
_p.add_argument("--out",    dest="out",    default=None, help="output .fit file (single material only)")
_p.add_argument("--fitdir", dest="fitdir", default=None, help="output directory for .fit files")
_p.add_argument("--config", dest="config", default=None, help="solver config.txt")
_p.add_argument("--poles",  dest="poles",  default=None, help="poles.txt bookkeeping file")
_p.add_argument("--refit",  action="store_true", help="delete poles.txt before fitting")
_p.add_argument("--no-plot", action="store_true", help="do not show matplotlib figures")
_p.add_argument("--sim", dest="sim_opt", choices=["0", "1"], default=None,
                help="legacy solver selector only used if project paths cannot be inferred: 0=2D, 1=3D")
_args = _p.parse_args()

# Accept the old positional form: fminconFIT.py material.mat 1
legacy_pos = list(_args.mat_pos)
if _args.mat is None and legacy_pos and len(legacy_pos) >= 2 and legacy_pos[-1] in ("0", "1"):
    if _args.sim_opt is None:
        _args.sim_opt = legacy_pos[-1]
    legacy_pos = legacy_pos[:-1]

mat_files = [os.path.abspath(m) for m in (_args.mat or legacy_pos)]
if not mat_files:
    sys.exit("\nUsage: python3 fminconFIT.py --mat <m1.mat> [m2.mat ...] "
             "[--out <single.fit> | --fitdir <dir>] "
             "--config <config.txt> --poles <poles.txt> [--refit] [--no-plot]\n")
if _args.out and len(mat_files) > 1:
    sys.exit("\n[!] --out is only valid with a single material; use --fitdir for several.\n")

for m in mat_files:
    if not os.path.isfile(m):
        sys.exit(f"\n[!] .mat file not found: {m}\n")


def _read_project_manifest(project_root):
    """Read project.txt key=value pairs. Comments and blank lines are ignored."""
    manifest = os.path.join(project_root, "project.txt")
    out = {}
    if not os.path.isfile(manifest):
        return out
    try:
        with open(manifest) as f:
            for raw in f:
                line = raw.split("#", 1)[0].strip()
                if not line or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out[k.strip().lower()] = v.strip()
    except OSError:
        return {}
    return out


def _abs_from_root(project_root, value):
    if not value:
        return None
    value = os.path.expanduser(value)
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(project_root, value))


def _project_root_of(mat_path):
    d = os.path.dirname(os.path.abspath(mat_path))
    probe = d
    for _ in range(8):
        if os.path.isfile(os.path.join(probe, "project.txt")):
            return probe
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    # fallback for <project>/input/mat/material.mat
    if os.path.basename(d) == "mat" and os.path.basename(os.path.dirname(d)) == "input":
        return os.path.dirname(os.path.dirname(d))
    return None


project_root = _project_root_of(mat_files[0])
if project_root is None and _args.config:
    cfg_dir = os.path.dirname(os.path.abspath(_args.config))
    if os.path.basename(cfg_dir) == "config":
        project_root = os.path.dirname(cfg_dir)

# Final fallback only for old manual use outside a project tree.
if project_root is None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if _args.sim_opt == "0":
        project_root = os.path.abspath(os.path.join(script_dir, "..", "2D-solver"))
    elif _args.sim_opt == "1":
        project_root = os.path.abspath(os.path.join(script_dir, "..", "3D-solver"))
    else:
        project_root = os.getcwd()
else:
    print(f"[paths] project root: {project_root}")

_manifest = _read_project_manifest(project_root)

# Explicit CLI paths always win. Otherwise use project.txt if available, then
# project layout defaults. For fitdir, matdir is the source of truth: input/mat
# and input/fit are siblings.
config_path = os.path.abspath(_args.config) if _args.config else _abs_from_root(project_root, _manifest.get("config"))
if config_path is None:
    config_path = os.path.abspath(os.path.join(project_root, "config", "config.txt"))

Npoles_path = os.path.abspath(_args.poles) if _args.poles else os.path.abspath(os.path.join(os.path.dirname(config_path), "poles.txt"))

if _args.fitdir:
    fit_dir = os.path.abspath(_args.fitdir)
else:
    mat_dir = _abs_from_root(project_root, _manifest.get("matdir"))
    if mat_dir:
        fit_dir = os.path.abspath(os.path.join(os.path.dirname(mat_dir), "fit"))
    else:
        fit_dir = os.path.abspath(os.path.join(project_root, "input", "fit"))

if not os.path.isfile(config_path):
    sys.exit(f"\n[!] config.txt not found: {config_path}\n")

if _args.refit and os.path.exists(Npoles_path):
    os.remove(Npoles_path)

print(f"[paths] config   : {config_path}")
print(f"[paths] fit dir  : {fit_dir}")
print(f"[paths] poles    : {Npoles_path}")

# --------------------
# Shared setup (config, air constants, frequency grid)
# --------------------

v = np.loadtxt(config_path)

h = float(v[0])
C_CFL = float(v[4])
f_cutoff = v[2] #balance with N_poles for stability and better distribution of poles
Temp = v[3]
R_spec = 287.05
gamma = 1.4

omega_cutoff = 2 * np.pi * f_cutoff

c0 = np.sqrt(gamma * R_spec * Temp)         # speed of sound (m/s)
rho0 = 1.225                                # air density
z0 = c0 * rho0                              # characteristic impedance of air

dt = C_CFL * h / c0

Niter = 30       # number of iterations (MATLAB did outer loop; skrf uses max_iterations)
N = int(v[5])    # "number of poles" in MATLAB script (here: number of REAL starting poles)

# Frequency samples (Hz). The vector fit places up to N+4 poles; if the grid has fewer
# points than that the least-squares system is underdetermined -> singular (cond=inf) ->
# crash. Decouple the count from f_cutoff: keep the old ~5 Hz spacing as a lower bound but
# FLOOR it well above the pole count, so a low f_cutoff (coarse grid / skipping the
# wave-based part) still yields a well-conditioned fit.
f_min = 20.0
f_max = f_cutoff
frequency_step = 5.0
#freq = np.arange(f_min, f_max + 0.5 * frequency_step, frequency_step)  # include f_max robustly
Npts = max(int((f_max - f_min)/frequency_step) + 1, 8*(N + 4) + 1)  # >> pole count, any f_cutoff
freq = np.logspace(np.log10(f_min), np.log10(f_max), Npts) # log-spaced

# --------------------------------------------------------------------------------
# Persistent (cross-run) fit cache: a .fit is stamped with a signature of EXACTLY
# what produced it -- the material's octave alphas + soft flag + every config value
# that changes the result. On the next run, a material whose current signature
# matches its stored one is up to date, so we load its .fit and skip the expensive
# vector-fit + refinement. It is still plotted (R_tgt/R_hat recomputed cheaply).
# --------------------------------------------------------------------------------
import hashlib

def _fit_sig(alpha_oct, soft):
    """Stable content hash. Uses hashlib (NOT hash(), which is per-process salted)."""
    a = np.round(np.atleast_1d(alpha_oct).ravel()[:8], 8)
    # key on dt (= what the pole bounds actually depend on) not h/cfl/T separately, so a
    # resolution change that leaves dt unchanged does not invalidate the fit.
    canon = ("|".join(f"{x:.8g}" for x in a)
             + f"|soft={int(bool(soft))}|dt={dt:.10g}"
             + f"|fc={float(f_cutoff):.10g}|N={int(N)}")
    return hashlib.sha1(canon.encode()).hexdigest()[:16]

def _read_fit_sig(path):
    """Return the '# sig <hex>' stamped in a .fit, or None if absent/unreadable."""
    try:
        with open(path) as f:
            for line in f:
                s = line.strip()
                if s.lower().startswith("# sig"):
                    return s.split()[-1]
    except OSError:
        return None
    return None

def _load_fit_coeffs(path):
    """Parse an existing .fit back into (Rinf, A, zeta, B, C, alpha, beta) so a cached
    material can be plotted without refitting. Mirrors the writer in fit_material()."""
    section = None; Rinf = 0.0
    A = []; zeta = []; B = []; C = []; alpha = []; beta = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                k = line.lower()
                section = ("RINF"  if "r_inf" in k else
                           "REAL"  if ("a_k" in k and "zeta_k" in k) else
                           "CMPLX" if ("b_l" in k and "alpha_l" in k) else None)
                continue
            p = line.split()
            if section == "RINF":
                Rinf = float(p[0])
            elif section == "REAL":
                A.append(float(p[0])); zeta.append(float(p[1]))
            elif section == "CMPLX":
                B.append(float(p[0])); C.append(float(p[1]))
                alpha.append(float(p[2])); beta.append(float(p[3]))
    return (Rinf, np.array(A, float), np.array(zeta, float),
            np.array(B, float), np.array(C, float),
            np.array(alpha, float), np.array(beta, float))

def _target_R(alpha_oct, soft):
    """Recompute the fit TARGET (min-phase only, no VF) so cached materials still plot.
    Same branches as fit_material()."""
    Rmag_oct = R_from_alpha(alpha_oct)
    if is_near_one(Rmag_oct):
        R_tgt, _ = R_constant_from_octaves(freq, Rmag_oct)
    else:
        R_tgt, _, _ = min_phase_octave_cepstrum(freq, Rmag_oct, ext_oct=2, n_w=4096)
    return -R_tgt if soft else R_tgt

# --------------------------------------------------------------------------------
# Block 2+3 per material: vector fit + fmincon/Hornikx-Cotte refinement
# --------------------------------------------------------------------------------
# The numerical pipeline is the original fminconFIT.py pipeline, wrapped so that
# several materials can be fitted in one run and all generated .fit files update
# one authoritative poles.txt.


def fit_material(read_file, coefs_path, soft=False):
    """Fit one material and write its .fit file. `soft` (from the ALL.mat
    registry) flips the sign of R (pressure-release surface, global pi phase).
    Returns (name, R_tgt, R_hat, K, L, rms) for summary plotting."""
    name = os.path.splitext(os.path.basename(read_file))[0]
    savename = name

    alpha_oct = np.atleast_1d(np.loadtxt(read_file)).ravel()[:8]
    Rmag_oct = R_from_alpha(alpha_oct)

    if is_near_one(Rmag_oct):
        R_tgt, Rinf = R_constant_from_octaves(freq, Rmag_oct)
        if soft:
            # global pi phase shift: R = -|R| (also flips the saved constant)
            R_tgt = -R_tgt; Rinf = -Rinf
            print(f"[{name}] soft material: R sign inverted")
        # skip min-phase + skip vector fitting
        K = 0; L = 0
        Acoef = np.array([], float); zeta = np.array([], float)
        Bcoef = np.array([], float); Ccoef = np.array([], float)
        alpha = np.array([], float); beta = np.array([], float)
        rmserr = 0.0
        print(f"\n[{name}] Near rigid material\n")
    else:
        R_tgt, Rmag_interp, phi = min_phase_octave_cepstrum(freq, Rmag_oct, ext_oct=2, n_w=4096)
        #R_tgt, Rmag_interp, phi = min_phase_octave_hilbert(freq, Rmag_oct, ext_oct=2, n_w=8192)
        if soft:
            # soft (pressure-release) surface: R = -|R|·e^{j·phi_min}, a global
            # pi phase shift. Causality/stability of the min-phase completion
            # are untouched; the fit absorbs the sign.
            R_tgt = -R_tgt
            print(f"[{name}] soft material: R sign inverted")

        w = 2.0 * np.pi * freq
        w_chk = 2*np.pi*np.logspace(0.0, np.log10(max(f_cutoff, 2.0)), 512)

        def _vf_refine(n_poles):
            """One VF (scikit-rf) + Option-A refinement at a given real starting-pole
            count. Returns (coeffs, vf_rms, refined_rms, max|R|)."""
            frequency = rf.Frequency.from_f(freq, unit="Hz")
            nw = rf.Network(frequency=frequency)
            nw.s = R_tgt.reshape(-1, 1, 1)
            nw.name = savename
            vf = VectorFitting(nw)
            vf.max_iterations = int(Niter)
            # custom initial poles (rad/s), s = j*2*pi*f
            vf.poles = (-2.0*np.pi*np.linspace(freq[0], freq[-1], n_poles)).astype(complex)
            vf.vector_fit(n_poles_real=0, n_poles_cmplx=0, init_pole_spacing="custom",
                          parameter_type="s", fit_constant=True, fit_proportional=False)
            vf_rms = float(vf.get_rms_error(i=0, j=0))
            poles = np.array(vf.poles, dtype=complex)
            res   = np.array(vf.residues[0], dtype=complex)
            Rc = float(vf.constant_coeff[0]) if vf.constant_coeff is not None else 0.0
            rm = np.isclose(np.imag(poles), 0.0)             # real vs complex split
            zt = -poles[rm].real;  Aa = res[rm].real         # real poles  -> A_k, zeta_k
            al = -poles[~rm].real; be = poles[~rm].imag       # complex pairs -> alpha,beta
            Bb = 2.0*res[~rm].real; Cc = -2.0*res[~rm].imag
            # Option A (Hornikx/Cotte) refinement: bounded, passivity-penalised
            Rc,Aa,Bb,Cc,zt,al,be = hornikx_cotte_constrained_fit(
                R_tgt=R_tgt, w=w, Rinf0=Rc, A0=Aa, B0=Bb, C0=Cc,
                zeta0=zt, alpha0=al, beta0=be, dt=dt, omega_cutoff=omega_cutoff)
            R_hat = R_expected(Rc, Aa, Bb, Cc, al, be, zt, w)
            r_rms = float(np.sqrt(np.mean(np.abs(R_hat - R_tgt)**2)))
            maxR  = float(np.max(np.abs(R_expected(Rc,Aa,Bb,Cc,al,be,zt,w_chk))))
            return (Rc,Aa,Bb,Cc,zt,al,be), vf_rms, r_rms, maxR

        # A steep target (a band at alpha=1 -> |R| floored to 1e-2) can drop VF in a bad
        # basin, and the real/complex pole split is BLAS-noise sensitive, so a fit can come
        # out non-passive at random. Retry with a few more starting poles, keep the best
        # (passive first, then lowest rms). Break on the first passive fit -> good materials
        # pay for exactly one VF+refine, no slowdown.
        best = None
        for i, n_poles in enumerate((N, N+2, N+4)):
            print(f"[{name}] vector fitting (scikit-rf), {n_poles} start poles"
                  + (f"  [retry {i}]" if i else "") + "...")
            coeffs, rmserr, r_rms, maxR = _vf_refine(n_poles)
            score = (maxR > 1.0 + 1e-6, r_rms)   # False (passive) sorts before True
            if best is None or score < best[0]:
                best = (score, coeffs, rmserr, r_rms, maxR)
            if maxR <= 1.0 + 1e-6:
                break
        _, coeffs, rmserr, r_rms, maxR = best
        Rinf, Acoef, Bcoef, Ccoef, zeta, alpha, beta = coeffs
        print(f"[{name}] RMS (VF) {rmserr:.6e}  refined rms {r_rms:.3e}  max|R| {maxR:.4f}")

    # ------------------------------------------------------------
    # Causality and reality test
    # ------------------------------------------------------------
    tst_alpha = np.all(alpha >= 0)
    tst_beta = np.all(beta >= 0)
    tst_zeta = np.all(zeta >= 0)

    if not (tst_alpha and tst_beta and tst_zeta):
        print(f"\n[!] [{name}] alpha's, beta's or zeta's have negative real values: failed causality test\n")
    if (np.any(alpha > 2.5127 / dt) or
        np.any(zeta  > 2.5127 / dt) or
        np.any(beta  > min(1.732 / dt, omega_cutoff))):
        print(f"\n[!] [{name}] alpha/beta/zeta are above expected values, may cause innacuracies / instability\n")

    # passivity check: |R_model| must stay <= 1 for all omega (Re Z >= 0), else the
    # boundary can add energy on reflection -> slow growth / blow-up.
    w_chk = 2*np.pi*np.logspace(0.0, np.log10(max(f_cutoff, 2.0)), 1024)
    Rmax = float(np.max(np.abs(R_expected(Rinf, Acoef, Bcoef, Ccoef, alpha, beta, zeta, w_chk))))
    if Rmax > 1.0 + 1e-6:
        print(f"\n[!] [{name}] max|R_model| = {Rmax:.5f} > 1 : non-passive, may cause instability\n")

    # ------------------------------------------------------------
    # SAVE IN C++ FORMAT
    # ------------------------------------------------------------
    K = len(zeta)
    L = len(alpha)

    os.makedirs(os.path.dirname(coefs_path), exist_ok=True)
    with open(coefs_path, 'w') as f:
        f.write(f"# sig {_fit_sig(alpha_oct, soft)}\n")
        f.write("# R_inf\n")
        f.write(f"{Rinf}\n")

        f.write("# K L\n")
        f.write(f"{K} {L}\n")

        f.write("# A_k zeta_k\n")
        for k in range(K):
            f.write(f"{Acoef[k]} {zeta[k]}\n")

        f.write("# B_l C_l alpha_l beta_l\n")
        for l in range(L):
            f.write(f"{Bcoef[l]} {Ccoef[l]} {alpha[l]} {beta[l]}\n")

    print(f"[{name}] Saved coefficients to {coefs_path}")

    # .imp: normalized octave-band impedance zeta = (1+R)/(1-R) at the 8 band centres,
    # for the ISM's angle-dependent complex reflection (R_oblique reads Z directly).
    # Built from the material's OWN octave |R| (min-phase), NOT the ADE model -- the fit is
    # only valid up to f_cutoff, so evaluating it at higher octaves would extrapolate garbage
    # exactly in the bands the ISM uses. Two rows: Re[8] then Im[8].
    octs = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], float)
    Rmag_oct = np.clip(R_from_alpha(alpha_oct), 1e-6, 1 - 1e-6)   # clip: keeps zeta finite (rigid)
    if is_near_one(Rmag_oct):
        R_oct = Rmag_oct.astype(complex)
    else:
        R_oct, _, _ = min_phase_octave_cepstrum(octs, Rmag_oct)
    if soft:
        R_oct = -R_oct
    zeta_oct = (1.0 + R_oct) / (1.0 - R_oct)
    imp_path = os.path.splitext(coefs_path)[0] + ".imp"
    with open(imp_path, "w") as f:
        f.write(" ".join(f"{v:.8g}" for v in np.real(zeta_oct)) + "\n")
        f.write(" ".join(f"{v:.8g}" for v in np.imag(zeta_oct)) + "\n")
    print(f"[{name}] Saved impedance to {imp_path}")

    # Model response for summary plot / final RMS
    w = 2 * np.pi * freq
    R_hat = R_expected(Rinf, Acoef, Bcoef, Ccoef, alpha, beta, zeta, w)
    rms_final = float(np.sqrt(np.mean(np.abs(R_hat - R_tgt)**2)))
    print(f"[{name}] RMS error (after Option A): {rms_final:.6e}")
    return name, R_tgt, R_hat, K, L, rms_final


# --------------------
# Fit every material
# --------------------

results = []
K_all = 0; L_all = 0
# Central hard/soft registry (ALL.mat) lives in the same dir as the materials.
soft_reg = _read_soft_registry(os.path.dirname(mat_files[0])) if mat_files else {}

# Cache: alpha_key -> (src_fit_path, src_imp_path, K, L, result_tuple)
# Avoids re-fitting materials that share identical octave-band absorptions.
_alpha_cache = {}

for m in mat_files:
    base = os.path.splitext(os.path.basename(m))[0]
    if base == "ALL":
        continue                         # registry file, not a material
    if _args.out:
        coefs_path = os.path.abspath(_args.out)
    else:
        coefs_path = os.path.join(fit_dir, base + ".fit")
    soft = soft_reg.get(base, False)

    # PERSISTENT skip: a .fit already built from these exact inputs is up to date.
    # Load it, skip the VF + refinement, but still recompute R_tgt/R_hat so it PLOTS.
    # --refit forces a full re-fit (it already wipes poles.txt above).
    alpha_oct = np.atleast_1d(np.loadtxt(m)).ravel()[:8]
    sig = _fit_sig(alpha_oct, soft)
    if (not _args.refit) and os.path.isfile(coefs_path) and _read_fit_sig(coefs_path) == sig:
        Rinf, A, zeta, B, Cc, alpha, beta = _load_fit_coeffs(coefs_path)
        w = 2.0 * np.pi * freq
        R_tgt = _target_R(alpha_oct, soft)
        R_hat = R_expected(Rinf, A, B, Cc, alpha, beta, zeta, w)
        rms = float(np.sqrt(np.mean(np.abs(R_hat - R_tgt) ** 2)))
        K = len(zeta); L = len(alpha)
        print(f"[{base}] unchanged — kept existing .fit (skipped refit)")
        results.append((base, R_tgt, R_hat, rms))
        K_all = max(K_all, K); L_all = max(L_all, L)
        _alpha_cache[_alpha_key(m, soft)] = (coefs_path, K, L, (base, R_tgt, R_hat, rms))
        continue

    key = _alpha_key(m, soft)
    if key in _alpha_cache:
        src_fit, K, L, cached_result = _alpha_cache[key]
        src_name = os.path.splitext(os.path.basename(src_fit))[0]
        print(f"[{base}] alphas identical to '{src_name}' — copying .fit")
        os.makedirs(os.path.dirname(coefs_path), exist_ok=True)
        shutil.copy2(src_fit, coefs_path)
        _si = os.path.splitext(src_fit)[0] + ".imp"          # copy the .imp too
        if os.path.exists(_si):
            shutil.copy2(_si, os.path.splitext(coefs_path)[0] + ".imp")
        _cached_name, R_tgt, R_hat, rms = cached_result
        results.append((base, R_tgt, R_hat, rms))
    else:
        name, R_tgt, R_hat, K, L, rms = fit_material(m, coefs_path, soft=soft)
        results.append((name, R_tgt, R_hat, rms))
        K_all = max(K_all, K); L_all = max(L_all, L)
        _alpha_cache[key] = (coefs_path, K, L, (name, R_tgt, R_hat, rms))

    K_all = max(K_all, K); L_all = max(L_all, L)

# --------------------
# SAVE NUMBER OF POLES — authoritative scan of fit_dir
# --------------------
# The solver zero-pads every material's coefficients to (K_m, L_m) read from
# poles.txt. read_fit_file aborts if ANY .fit exceeds those bounds. Therefore
# poles.txt must cover EVERY .fit the solver will read — including ones from
# older runs. Rescan all .fit headers after fitting and write the true maxima.


def _fit_KL(path):
    """Read the 'K L' line (2nd data line, '#' comments skipped) of a .fit."""
    nums = []
    try:
        with open(path) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if not line:
                    continue
                nums.append(line.split())
                if len(nums) >= 2:
                    break
        return int(nums[1][0]), int(nums[1][1])
    except Exception:
        return 0, 0

K_m, L_m = K_all, L_all
_fit_files = glob.glob(os.path.join(fit_dir, "*.fit"))
for p in _fit_files:
    k, l = _fit_KL(p)
    K_m = max(K_m, k); L_m = max(L_m, l)

Npoles = np.array([[int(K_m), int(L_m)]], dtype=int)
os.makedirs(os.path.dirname(Npoles_path), exist_ok=True)
np.savetxt(Npoles_path, Npoles, fmt="%d", delimiter=" ")
print(f"[poles] K_m={int(K_m)} L_m={int(L_m)} (scanned {len(_fit_files)} .fit files) -> {Npoles_path}")

# Write number of distinct materials (unique alpha combinations actually fitted)
# as a single little-endian int32 binary file so C/C++ can read it with:
#   int n; fread(&n, sizeof(int), 1, fp);
nMat_path = os.path.join(os.path.dirname(Npoles_path), "nMat")
n_distinct = len(_alpha_cache)
with open(nMat_path, "wb") as _f:
    _f.write(struct.pack("<I", n_distinct))
print(f"[nMat] {n_distinct} distinct material(s) -> {nMat_path}")

# ============================================================
# OUTPUTS SUMMARY — absolute paths + file counts, so there is never any
# ambiguity about WHERE the .fit files landed. If you don't see this
# banner after a run, you are executing a different script than this one.
# ============================================================
print("\n" + "=" * 60)
print(" OUTPUTS")
print(f"   .fit -> {fit_dir}   ({len(_fit_files)} file(s))")
print("=" * 60 + "\n")

# ============================================================
# SUMMARY PLOTS — three figures (|R|, Re R, Im R), each a grid of
# subplots, one panel per material. The fitting calculations above are unchanged;
# this is only a non-blocking/multi-material presentation layer.
# ============================================================

if _args.no_plot:
    sys.exit(0)

n = len(results)
ncols = int(np.ceil(np.sqrt(n)))
nrows = int(np.ceil(n / ncols))


def _grid(extract, fit_color, ylabel, suptitle, ylim=None):
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.2*nrows),
                             squeeze=False, sharex=True, sharey=True)
    for i, (name, R_tgt, R_hat, rms) in enumerate(results):
        ax = axes[i // ncols][i % ncols]
        ax.semilogx(freq, extract(R_tgt), color='k', linewidth=2,
                    label=f"Target {ylabel}")
        ax.semilogx(freq, extract(R_hat), "--", color=fit_color, linewidth=2,
                    label=f"Fitted {ylabel}")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, which="both", alpha=0.4)
        ax.set_title(f"{name}   (rms {rms:.1e})", fontsize=10)
        if i // ncols == nrows - 1:
            ax.set_xlabel("Frequency [Hz]")
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
    for j in range(n, nrows*ncols):
        axes[j // ncols][j % ncols].axis("off")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

_grid(np.abs,  'r',    "|R|",   "fminconFIT — target vs fitted |R| per material",   (0, 1.05))
_grid(np.real, 'b',    "Re(R)", "fminconFIT — target vs fitted Re(R) per material", (-1.05, 1.05))
_grid(np.imag, 'lime', "Im(R)", "fminconFIT — target vs fitted Im(R) per material", (-1.05, 1.05))
plt.show()
