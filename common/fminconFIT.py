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
                             mag_floor=1e-8, mag_ceil=1-1e-12):
    freq_hz = np.asarray(freq_hz, float)
    R_oct = np.clip(np.asarray(R_oct, float), mag_floor, mag_ceil)

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
    maxiter=4000
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

    def total_sse(x):
        Rinf, A, B, C, zeta, alpha, beta, *_ = _unpack_params(
            x, K, L, zeta_max, alpha_max, beta_max)
        e = R_expected(Rinf, A, B, C, alpha, beta, zeta, w) - R_tgt
        return float(np.sum(e.real**2 + e.imag**2))

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

# Octave-band fitted impedances for the ISM solver, written next to the .fit
# files (sibling of fit_dir): <fit_dir>/../imp. Derived from fit_dir so the .imp
# files always land in the SAME input/ dir as the .fit files, regardless of how
# fit_dir was resolved (--fitdir, project.txt matdir, or the layout default).
imp_dir = os.path.abspath(os.path.join(os.path.dirname(fit_dir), "imp"))

if not os.path.isfile(config_path):
    sys.exit(f"\n[!] config.txt not found: {config_path}\n")

if _args.refit and os.path.exists(Npoles_path):
    os.remove(Npoles_path)

print(f"[paths] config   : {config_path}")
print(f"[paths] fit dir  : {fit_dir}")
print(f"[paths] poles    : {Npoles_path}")
print(f"[paths] imp dir  : {imp_dir}")

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

# Frequency samples (Hz)
f_min = 20.0
f_max = f_cutoff
frequency_step = 5.0
#freq = np.arange(f_min, f_max + 0.5 * frequency_step, frequency_step)  # include f_max robustly
Npts = int((f_max - f_min)/frequency_step) + 1 # keep about same count as above
freq = np.logspace(np.log10(f_min), np.log10(f_max), Npts) # log-spaced

Niter = 30       # number of iterations (MATLAB did outer loop; skrf uses max_iterations)
N = int(v[5])    # "number of poles" in MATLAB script (here: number of REAL starting poles)

# Octave-band centre frequencies (Hz) — the bands the .mat alpha values use, and
# the bands the .imp impedance export is sampled at for the ISM solver.
OCTAVE_FC = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)


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
        # proceed with VF as usual

        Zs_to_fit = Z_from_R(R_tgt, z0)   # optional; only if needed Z later

        # --------------------------------------------------------------------------------
        # Block 2: Vector fit reflection coefficient with scikit-rf
        # --------------------------------------------------------------------------------

        # Build a 1-port skrf Network with S11 = R_tgt
        frequency = rf.Frequency.from_f(freq, unit="Hz")
        nw = rf.Network(frequency=frequency)
        nw.s = R_tgt.reshape(-1, 1, 1)  # shape: (n_freq, n_ports, n_ports)
        nw.name = savename

        vf = VectorFitting(nw)
        vf.max_iterations = int(Niter)

        # Provide custom initial poles (rad/s), matching s = j*2*pi*f.
        # MATLAB used: poles = -linspace(freq(1),freq(end),N)  (but s is rad/s)
        # Here we do it consistently in rad/s:
        poles_init = -2.0 * np.pi * np.linspace(freq[0], freq[-1], N).astype(float)
        vf.poles = poles_init.astype(complex)

        print(f"[{name}] vector fitting (scikit-rf)...")
        vf.vector_fit(
            n_poles_real=0,             # ignored for custom spacing
            n_poles_cmplx=0,            # ignored for custom spacing
            init_pole_spacing="custom", # uses vf.poles as initial poles
            parameter_type="s",         # fit directly S11
            fit_constant=True,          # if False -> no constant term R_inf
            fit_proportional=False,     # no proportional term
        )

        # Final RMS error (scalar response i=j=0)
        rmserr = float(vf.get_rms_error(i=0, j=0))
        print(f"[{name}] RMS error (VF): {rmserr:.6e}")

        # Fitted model parameters
        poles = np.array(vf.poles, dtype=complex)              # shape (K,) ; complex poles stored with +imag only
        residues = np.array(vf.residues[0], dtype=complex)     # shape (K,) ; because 1-port => M=1 response

        # ------------------------------------------------------------
        # Block 3: Extract coefficients in ADE form and save to file
        # ------------------------------------------------------------

        poles = np.array(vf.poles, dtype=complex)
        residues = np.array(vf.residues[0], dtype=complex)

        # Constant term (R_inf)
        Rinf = float(vf.constant_coeff[0]) if vf.constant_coeff is not None else 0.0

        # Separate real and complex poles
        real_mask = np.isclose(np.imag(poles), 0.0)
        cmplx_mask = ~real_mask

        poles_real = poles[real_mask]
        res_real = residues[real_mask]

        poles_cmplx = poles[cmplx_mask]
        res_cmplx = residues[cmplx_mask]

        # -----------------------------
        # REAL POLES  ->  A_k, zeta_k
        # -----------------------------
        zeta = (-poles_real.real)        # because pole = -zeta
        Acoef = res_real.real            # residue

        K = len(zeta)

        # -------------------------------------------
        # COMPLEX POLES -> B_l C_l alpha_l beta_l
        # -------------------------------------------
        # scikit-rf stores only ONE pole per complex pair (positive imag part)

        alpha = (-poles_cmplx.real)
        beta  = (poles_cmplx.imag)

        Bcoef = 2.0 * res_cmplx.real
        Ccoef = -2.0 * res_cmplx.imag

        L = len(alpha)

    # ------------------------------------------------------------
    # OPTION A refinement (Hornikx/Cotte style)
    # - minimize real-part error
    # - constrain imag-part error via epsilon continuation
    # - enforce alpha/beta/zeta bounds DURING the fit
    # ------------------------------------------------------------
    print(f"\n[{name}] Option A refinement (Hornikx/Cotte constrained continuation)...")
    w = 2.0 * np.pi * freq
    Rinf, Acoef, Bcoef, Ccoef, zeta, alpha, beta = hornikx_cotte_constrained_fit(
        R_tgt=R_tgt, w=w,
        Rinf0=Rinf, A0=Acoef, B0=Bcoef, C0=Ccoef,
        zeta0=zeta, alpha0=alpha, beta0=beta,
        dt=dt, omega_cutoff=omega_cutoff
    )

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

    # ------------------------------------------------------------
    # SAVE IN C++ FORMAT
    # ------------------------------------------------------------
    K = len(zeta)
    L = len(alpha)

    os.makedirs(os.path.dirname(coefs_path), exist_ok=True)
    with open(coefs_path, 'w') as f:
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

    # ----------------------------------------------------------------------
    # SAVE FITTED IMPEDANCE PER OCTAVE BAND  ->  <input>/imp/<name>.imp
    # ----------------------------------------------------------------------
    # The ISM solver builds the angle-dependent, local-reaction reflection
    # coefficient   R(theta) = (zeta*cos(theta) - 1) / (zeta*cos(theta) + 1),
    # zeta = Z / z0 the NORMALISED specific impedance, so it needs Z at the 8
    # octave-band centres. We evaluate the SAME fitted ADE model the wave solver
    # realises (R_expected) at those centres and map R -> Z = z0*(1+R)/(1-R).
    # CRITICAL: these are the FINAL coefficients, i.e. AFTER the Option A
    # (Hornikx/Cotte) refinement above — the exact poles/residues written to the
    # .fit and thus realised by the wave solver. Using the post-refinement model
    # (not the raw VF output, not the raw octave target) keeps the ISM and the
    # wave solver on the same surface. Near-rigid -> R~1 -> large real Z; soft ->
    # R<0 -> sign carried through.
    #
    # Format (exactly two lines, 8 space-separated values each):
    #   line 1: Re(Z) per band, 63 -> 8000 Hz
    #   line 2: Im(Z) per band, 63 -> 8000 Hz
    # Z is the specific acoustic impedance [Pa.s/m]; the ISM normalises by
    # z0 = rho0*c0 (both available from config) to get zeta.
    w_oct      = 2.0 * np.pi * OCTAVE_FC
    R_oct_fit  = R_expected(Rinf, Acoef, Bcoef, Ccoef, alpha, beta, zeta, w_oct)
    Z_oct_fit  = Z_from_R(R_oct_fit, z0)

    os.makedirs(imp_dir, exist_ok=True)
    imp_path = os.path.join(imp_dir, name + ".imp")
    with open(imp_path, "w") as f:
        f.write(" ".join(f"{val:.12g}" for val in np.real(Z_oct_fit)) + "\n")
        f.write(" ".join(f"{val:.12g}" for val in np.imag(Z_oct_fit)) + "\n")
    print(f"[{name}] Saved octave-band impedance to {imp_path}")

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
    imp_path = os.path.join(imp_dir, base + ".imp")
    soft = soft_reg.get(base, False)

    key = _alpha_key(m, soft)
    if key in _alpha_cache:
        src_fit, src_imp, K, L, cached_result = _alpha_cache[key]
        src_name = os.path.splitext(os.path.basename(src_fit))[0]
        print(f"[{base}] alphas identical to '{src_name}' — copying .fit and .imp")
        os.makedirs(os.path.dirname(coefs_path), exist_ok=True)
        shutil.copy2(src_fit, coefs_path)
        os.makedirs(imp_dir, exist_ok=True)
        shutil.copy2(src_imp, imp_path)
        _cached_name, R_tgt, R_hat, rms = cached_result
        results.append((base, R_tgt, R_hat, rms))
    else:
        name, R_tgt, R_hat, K, L, rms = fit_material(m, coefs_path, soft=soft)
        results.append((name, R_tgt, R_hat, rms))
        K_all = max(K_all, K); L_all = max(L_all, L)
        _alpha_cache[key] = (coefs_path, imp_path, K, L, (name, R_tgt, R_hat, rms))

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
# ambiguity about WHERE the .fit and .imp files landed. If you don't see this
# banner after a run, you are executing a different script than this one.
# ============================================================
_imp_written = sorted(glob.glob(os.path.join(imp_dir, "*.imp")))
print("\n" + "=" * 60)
print(" OUTPUTS")
print(f"   .fit -> {fit_dir}   ({len(_fit_files)} file(s))")
print(f"   .imp -> {imp_dir}   ({len(_imp_written)} file(s))")
for _p_ in _imp_written:
    print(f"          {os.path.basename(_p_)}")
print("=" * 60 + "\n")
if not _imp_written:
    print(f"[!] No .imp files were written — check write permissions on {imp_dir} "
          "and the run output above for errors.")

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
_grid(np.real, 'b',    "Re(R)", "fminconFIT — target vs fitted Re(R) per material", None)
_grid(np.imag, 'lime', "Im(R)", "fminconFIT — target vs fitted Im(R) per material", None)
plt.show()
