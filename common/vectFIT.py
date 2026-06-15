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
from scipy.io import savemat
from scipy.interpolate import PchipInterpolator
from scipy.signal import hilbert
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


def is_near_one(arr, atol=0.02):
    return np.all(np.isclose(arr, 1.0, atol=atol))

def R_constant_from_octaves(freq, R_oct, mag_ceil=1-1e-12):
    # choose a conservative constant close to the minimum band value
    Rinf = float(np.clip(np.min(R_oct), 0.0, mag_ceil))
    R = Rinf * np.ones_like(freq, dtype=complex)  # phase = 0
    return R, Rinf

# alpha here are the coefficients of acoustic absorption
def R_from_alpha(alpha):
	return np.sqrt(1 - alpha)

def read_soft_flag(mat_path):
    """Hard/soft flag lives in ALL.mat (the central registry the GUI writes),
    NOT as a trailing column in the per-material .mat (those stay 8 alpha values
    only, so every alpha-only reader keeps working). ALL.mat sits next to the
    .mat files; one line per material: '<name>  <flag>'  (0 = hard / R>0,
    1 = soft / pressure-release R<0). Missing file or entry -> hard."""
    name = os.path.splitext(os.path.basename(mat_path))[0]
    allmat = os.path.join(os.path.dirname(os.path.abspath(mat_path)), "ALL.mat")
    if not os.path.isfile(allmat):
        return False
    try:
        with open(allmat) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[0] == name:
                    return parts[1] != "0"
    except Exception:
        pass
    return False

# --------------------
# Block 1: User input (CLI arguments — non-interactive, project-aware)
# --------------------
# All paths are passed explicitly; nothing is derived from the script location.
# One or SEVERAL materials can be fitted in a single run:
#
#   python3 vectFIT.py --mat walls.mat ceiling.mat floor.mat
#                      --fitdir <dir for .fit files>
#                      --config <config.txt> --poles <poles.txt>
#                      [--refit] [--no-plot]
#
#   python3 vectFIT.py --mat walls.mat --out walls.fit ...   (single material)
#
# --fitdir   : output directory for .fit files (one per material, named after
#              the .mat). --out is only valid with a single material.
# --refit    : delete poles.txt before fitting (fresh pole bookkeeping)
# --no-plot  : skip the matplotlib figure (required when run from the GUI,
#              where output is captured and plt.show() would block)
#
# Legacy: bare positional .mat file(s) also work, with config/, input/fit/
# resolved against the CURRENT WORKING DIRECTORY (run from a project root).
#
# With several materials, all fits are summarised in ONE figure: a grid of
# subplots (target |R| vs fitted |R| per material) so the quality of every fit
# can be checked at a glance.

import argparse

_p = argparse.ArgumentParser(description="Vector fitting of reflection coefficient (ADE export)")
_p.add_argument("mat_pos", nargs="*", default=[], help="material .mat file(s) (positional, legacy)")
_p.add_argument("--mat",    dest="mat",    nargs="+", default=None, help="material .mat file(s)")
_p.add_argument("--out",    dest="out",    default=None, help="output .fit file (single material only)")
_p.add_argument("--fitdir", dest="fitdir", default=None, help="output directory for .fit files")
_p.add_argument("--config", dest="config", default=None, help="solver config.txt")
_p.add_argument("--poles",  dest="poles",  default=None, help="poles.txt bookkeeping file")
_p.add_argument("--refit",  action="store_true", help="delete poles.txt before fitting")
_p.add_argument("--no-plot", action="store_true", help="do not show the matplotlib figure")
_args = _p.parse_args()

mat_files = [os.path.abspath(m) for m in (_args.mat or _args.mat_pos)]
if not mat_files:
    sys.exit("\nUsage: python3 vectFIT.py --mat <m1.mat> [m2.mat ...] "
             "[--out <single.fit> | --fitdir <dir>] "
             "--config <config.txt> --poles <poles.txt> [--refit] [--no-plot]\n")
if _args.out and len(mat_files) > 1:
    sys.exit("\n[!] --out is only valid with a single material; use --fitdir for several.\n")

for m in mat_files:
    if not os.path.isfile(m):
        sys.exit(f"\n[!] .mat file not found: {m}\n")

# Defaults for --config / --poles / --fitdir, in priority order:
#   1. explicit arguments (always win);
#   2. the PROJECT the .mat belongs to — found by walking up from the .mat
#      file's directory to the first dir containing project.txt, or via the
#      <root>/input/mat/ layout. This makes a bare
#          python3 vectFIT.py <proj>/input/mat/walls.mat
#      do the right thing from ANY working directory. (cwd-relative defaults
#      were a footgun: poles.txt silently landed wherever you happened to
#      stand, leaving the project's copy stale -> solver 'incompatible poles');
#   3. the directory of an explicit --config (if it is <root>/config/...);
#   4. the current working directory (legacy: run from a project root).
def _project_root_of(mat_path):
    d = os.path.dirname(os.path.abspath(mat_path))
    probe = d
    for _ in range(8):                                    # walk up
        if os.path.isfile(os.path.join(probe, "project.txt")):
            return probe
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    # fall back to the .../input/mat layout
    if os.path.basename(d) == "mat" and os.path.basename(os.path.dirname(d)) == "input":
        return os.path.dirname(os.path.dirname(d))
    return None

project_root = _project_root_of(mat_files[0])
if project_root is None and _args.config:
    cfg_dir = os.path.dirname(os.path.abspath(_args.config))
    if os.path.basename(cfg_dir) == "config":
        project_root = os.path.dirname(cfg_dir)
if project_root is None:
    project_root = os.getcwd()
elif not (_args.config and _args.poles and _args.fitdir):
    print(f"[paths] project root: {project_root}")

config_path = os.path.abspath(_args.config) if _args.config else os.path.abspath(os.path.join(project_root, "config", "config.txt"))
Npoles_path = os.path.abspath(_args.poles)  if _args.poles  else os.path.abspath(os.path.join(project_root, "config", "poles.txt"))
fit_dir     = os.path.abspath(_args.fitdir) if _args.fitdir else os.path.abspath(os.path.join(project_root, "input", "fit"))

if not os.path.isfile(config_path):
    sys.exit(f"\n[!] config.txt not found: {config_path}\n")

if _args.refit and os.path.exists(Npoles_path):
    os.remove(Npoles_path)

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


# --------------------------------------------------------------------------------
# Block 2+3 per material: vector fit (scikit-rf) and ADE coefficient extraction
# --------------------------------------------------------------------------------
# Identical pipeline to the original single-material script, wrapped in a
# function so several materials can be fitted in one run.

def fit_material(read_file, coefs_path):
    """Fit one material's .mat (octave-band alphas) and write its .fit file.
    Returns (name, R_tgt, R_hat, K, L, rms) for the summary plot."""
    name = os.path.splitext(os.path.basename(read_file))[0]
    savename = name

    data = np.atleast_1d(np.loadtxt(read_file)).ravel()
    alpha_oct = data[:8]
    Rmag_oct = R_from_alpha(alpha_oct)
    # Hard/soft flag is NOT in the per-material .mat (8 alpha values only); it
    # lives in the ALL.mat registry the GUI writes, next to the .mat files.
    soft = read_soft_flag(read_file)
    #Rmag_oct = np.loadtxt(read_file)

    rmserr = 0.0
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
        print(f"\n[{name}] Near rigid material\n")
    else:
        R_tgt, Rmag_interp, phi = min_phase_octave_cepstrum(freq, Rmag_oct, ext_oct=2, n_w=4096)
        #R_tgt, Rmag_interp, phi = min_phase_octave_hilbert(freq, Rmag_oct, ext_oct=2, n_w=8192)
        if soft:
            # soft (pressure-release) surface: R = -|R|·e^{j·phi_min}, i.e. a
            # global pi phase shift. Causality/stability of the min-phase
            # completion are untouched; the fit absorbs the sign.
            R_tgt = -R_tgt
            print(f"[{name}] soft material: R sign inverted")

        Zs_to_fit = Z_from_R(R_tgt, z0)   # optional; only if needed Z later

        # Build a 1-port skrf Network with S11 = R_tgt
        frequency = rf.Frequency.from_f(freq, unit="Hz")
        nw = rf.Network(frequency=frequency)
        nw.s = R_tgt.reshape(-1, 1, 1)  # shape: (n_freq, n_ports, n_ports)
        nw.name = savename

        vf = VectorFitting(nw)
        vf.max_iterations = int(Niter)

        # Provide custom initial poles (rad/s), matching s = j*2*pi*f.
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

        rmserr = float(vf.get_rms_error(i=0, j=0))
        print(f"[{name}] RMS error (final): {rmserr:.6e}")

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

        # REAL POLES  ->  A_k, zeta_k
        zeta = (-poles_real.real)        # because pole = -zeta
        Acoef = res_real.real            # residue
        K = len(zeta)

        # COMPLEX POLES -> B_l C_l alpha_l beta_l
        # scikit-rf stores only ONE pole per complex pair (positive imag part)
        alpha = (-poles_cmplx.real)
        beta  = (poles_cmplx.imag)
        Bcoef = 2.0 * res_cmplx.real
        Ccoef = -2.0 * res_cmplx.imag
        L = len(alpha)

    # Causality and reality test
    tst_alpha = np.all(alpha >= 0)
    tst_beta = np.all(beta >= 0)
    tst_zeta = np.all(zeta >= 0)
    if not (tst_alpha and tst_beta and tst_zeta):
        print(f"\n[!] [{name}] alpha's, beta's or zeta's have negative real values: failed causality test\n")
    '''
    if (np.any(alpha > 2.5127 / dt) or
        np.any(zeta  > 2.5127 / dt) or
        np.any(beta  > min(1.732 / dt, omega_cutoff))):
        print("\n[!] alpha/beta/zeta are above expected values, may cause innacuracies / instability\n")
    '''

    # SAVE IN C++ FORMAT
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

    # Model response for the summary plot
    w = 2 * np.pi * freq
    R_hat = R_expected(Rinf, Acoef, Bcoef, Ccoef, alpha, beta, zeta, w)
    return name, R_tgt, R_hat, K, L, rmserr


# --------------------
# Fit every material
# --------------------

results = []
K_all = 0; L_all = 0
for m in mat_files:
    if _args.out:
        coefs_path = os.path.abspath(_args.out)
    else:
        coefs_path = os.path.join(fit_dir, os.path.splitext(os.path.basename(m))[0] + ".fit")
    name, R_tgt, R_hat, K, L, rms = fit_material(m, coefs_path)
    results.append((name, R_tgt, R_hat, rms))
    K_all = max(K_all, K); L_all = max(L_all, L)

# --------------------
# SAVE NUMBER OF POLES — monotonic running maximum (original behaviour)
# --------------------
# The solver reads (K_m, L_m) from poles.txt, allocates K_m*nMat / L_m*nMat
# arrays ZERO-INITIALISED, and read_fit_file fills only each material's own
# K_local/L_local rows — the remaining slots stay zero. The ADE kernel is
# zero-safe: a zero slot has A=B=C=0 (no contribution) and there is no division
# by zeta/alpha/beta anywhere, so trailing zeros are inert. Each .fit therefore
# keeps EXACTLY its own pole count (no padding, no rewriting), and poles.txt
# only needs to be large enough to cover the biggest one. We keep a monotonic
# running max: read the existing poles.txt (if any), take the max with this
# run's (K_all, L_all), write it back. --refit deletes poles.txt first to start
# the count fresh.
K_m, L_m = K_all, L_all
if os.path.exists(Npoles_path):
    try:
        prev = np.atleast_1d(np.loadtxt(Npoles_path)).ravel()
        if prev.size >= 2:
            K_m = max(K_m, int(prev[0]))
            L_m = max(L_m, int(prev[1]))
    except Exception:
        pass

Npoles = np.array([[int(K_m), int(L_m)]], dtype=int)
os.makedirs(os.path.dirname(Npoles_path), exist_ok=True)
np.savetxt(Npoles_path, Npoles, fmt="%d", delimiter=" ")
print(f"[poles] K_m={int(K_m)} L_m={int(L_m)} -> {Npoles_path}")

# ============================================================
# SUMMARY PLOTS — three figures (|R|, Re R, Im R), each a grid of
# subplots, one panel per material. Target always black; fitted curve
# red for |R|, blue for Re, green for Im (matching the original script).
# ============================================================

if _args.no_plot:
    sys.exit(0)

n = len(results)
ncols = int(np.ceil(np.sqrt(n)))
nrows = int(np.ceil(n / ncols))

def _grid(extract, fit_color, ylabel, suptitle, ylim):
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.2*nrows),
                             squeeze=False, sharex=True, sharey=True)
    for i, (name, R_tgt, R_hat, rms) in enumerate(results):
        ax = axes[i // ncols][i % ncols]
        ax.semilogx(freq, extract(R_tgt), color='k', linewidth=2,
                    label=f"Target {ylabel}")
        ax.semilogx(freq, extract(R_hat), "--", color=fit_color, linewidth=2,
                    label=f"Fitted {ylabel}")
        ax.set_ylim(*ylim)
        ax.grid(True, which="both", alpha=0.4)
        ttl = name if rms == 0.0 else f"{name}   (rms {rms:.1e})"
        ax.set_title(ttl, fontsize=10)
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

_grid(np.abs,  'r',    "|R|",   "Vector fit — target vs fitted |R| per material",   (0, 1.05))
_grid(np.real, 'b',    "Re(R)", "Vector fit — target vs fitted Re(R) per material", (-1.05, 1.05))
_grid(np.imag, 'lime', "Im(R)", "Vector fit — target vs fitted Im(R) per material", (-1.05, 1.05))
plt.show()
