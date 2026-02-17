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
        den = (s[None, :] + alpha[:, None])**2 + (beta[:, None]**2)
        R += ((B[:, None] * s[None, :] + C[:, None] * beta[:, None] + alpha[:, None] * B[:, None])/ den).sum(axis=0)

    return R


def is_near_one(arr, atol=0.02):
    return np.all(np.isclose(arr, 1.0, atol=atol))

def R_constant_from_octaves(freq, R_oct, mag_ceil=1-1e-12):
    # choose a conservative constant close to the minimum band value
    Rinf = float(np.clip(np.min(R_oct), 0.0, mag_ceil))
    R = Rinf * np.ones_like(freq, dtype=complex)  # phase = 0
    return R, Rinf


# --------------------
# Block 1: User input
# --------------------

read_file = sys.argv[1] if len(sys.argv) > 1 else sys.exit("\nUsage: python3 vectFIT.py <material_name.mat>\n")

s_opt = int(input("\nEnter with an option:\n\n0) 2D-solver\n1) 3D-solver\n\n"))
if not (s_opt == 0 or s_opt == 1):
    sys.exit("\n[!] No possible option\n")

script_dir = os.path.dirname(os.path.abspath(__file__))
if s_opt == 0:
    solver_path = "2D-solver"
if s_opt == 1:
    solver_path = "3D-solver"

s_delete = input("\nReffiting for new simulation? Y/N\n\n")
if not (s_delete == "Y" or s_delete == "N"):
    sys.exit("\n[!] No possible option\n")

Npoles_path = os.path.join(script_dir, "..", solver_path, "config", "poles.txt")
if s_delete == "Y":
	delete_str = "rm " + Npoles_path
	os.system(delete_str)

Rmag_oct = np.loadtxt(read_file)

config_path = os.path.join(script_dir, "..", solver_path, "config", "config.txt")

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

savename = np.copy(read_file[:-4]) # output .mat filename (without extension)

# Frequency samples (Hz)
f_min = 20.0
f_max = f_cutoff
frequency_step = 5.0
#freq = np.arange(f_min, f_max + 0.5 * frequency_step, frequency_step)  # include f_max robustly
Npts = int((f_max - f_min)/frequency_step) + 1 # keep about same count as above
freq = np.logspace(np.log10(f_min), np.log10(f_max), Npts) # log-spaced

Niter = 25       # number of iterations (MATLAB did outer loop; skrf uses max_iterations)
N = int(v[5])    # "number of poles" in MATLAB script (here: number of REAL starting poles)

# Example impedance (Miki model) to be approximated (not reading from .mat file)
'''
sigma = 100000.0
x = freq / sigma
Z_Miki = z0 * (1.0 + 5.50 * (x * 1000.0) ** (-0.632) - 1j * 8.43 * (x * 1000.0) ** (-0.632))

Zs_to_fit = Z_Miki
R_tgt = (Zs_to_fit / z0 - 1.0) / (Zs_to_fit / z0 + 1.0)
'''
if is_near_one(Rmag_oct):
    R_tgt, Rinf = R_constant_from_octaves(freq, Rmag_oct)
    # skip min-phase + skip vector fitting
    K = 0; L = 0
    Acoef = np.array([], float); zeta = np.array([], float)
    Bcoef = np.array([], float); Ccoef = np.array([], float)
    alpha = np.array([], float); beta = np.array([], float)
    print("\nNear rigid material\n")
else:
    R_tgt, Rmag_interp, phi = min_phase_octave_cepstrum(freq, Rmag_oct, ext_oct=2, n_w=4096)
    #R_tgt, Rmag_interp, phi = min_phase_octave_hilbert(freq, Rmag_oct, ext_oct=2, n_w=8192)
    # proceed with VF as usual

    Zs_to_fit = Z_from_R(R_tgt, z0)   # optional; only if needed Z later

    # --------------------------------------------------------------------------------
    # Block 2: Vector fit reflection coefficient with scikit-rf
    # --------------------------------------------------------------------------------

    # Build a 1-port skrf Network with S11 = R_tgt
    frequency = rf.Frequency.from_f(freq, unit="hz")
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
    
    print("vector fitting (scikit-rf)...")
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
    print(f"RMS error (final): {rmserr:.6e}")
    
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
# Causality and reality test
# ------------------------------------------------------------
tst_alpha = np.all(alpha >= 0)
tst_beta = np.all(beta >= 0)
tst_zeta = np.all(zeta >= 0)

if not (tst_alpha and tst_beta and tst_zeta):
    print("\n[!] alpha's, beta's or zeta's have negative real values: failed causality test\n")
if (np.any(alpha > 2.5127 / dt) or
    np.any(beta  > 2.5127 / dt) or
    np.any(zeta  > min(1.732 / dt, omega_cutoff))):
    print("\n[!] alpha/beta/zeta are above expected values, may cause instability\n")

# ------------------------------------------------------------
# SAVE IN C++ FORMAT
# ------------------------------------------------------------

save_file = os.path.splitext(os.path.basename(read_file))[0] + ".fit"
coefs_path = os.path.join(script_dir, "..", solver_path, "input", "fit", save_file)

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

print(f"Saved coefficients to {coefs_path}")

# ------------------------------------------------------------
# SAVE MAX NUMBER OF POLES IF SUPERIOR
# ------------------------------------------------------------

if os.path.exists(Npoles_path):
    Npoles = np.loadtxt(Npoles_path)
    K_m, L_m = Npoles
else:
    K_m, L_m = K, L

if K > K_m:
    K_m = K
if L > L_m:
    L_m = L

Npoles = np.array([[int(K_m), int(L_m)]], dtype=int)
os.makedirs(os.path.dirname(Npoles_path), exist_ok=True)
np.savetxt(Npoles_path, Npoles, fmt="%d", delimiter=" ")

# ============================================================
# PLOTS (|R| + Re/Im)
# ============================================================

w = 2 * np.pi * freq
R_hat = R_expected(Rinf, Acoef, Bcoef, Ccoef, alpha, beta, zeta, w)

plt.figure(1)
plt.semilogx(freq, np.abs(R_tgt), label="Target |R|", color='k', linewidth=2)
plt.semilogx(freq, np.abs(R_hat), "--", label="Fitted |R|", color='r', linewidth=2)
plt.ylim(0, 1.05)
plt.grid(True, which="both")
plt.xlabel("Frequency [Hz]")
plt.ylabel("|R|")
plt.legend()

plt.figure(2)
plt.semilogx(freq, np.real(R_tgt), label="Target Re(R)", color='k', linewidth=2)
plt.semilogx(freq, np.real(R_hat), "--", label="Fitted Re(R)", color='b', linewidth=2)
plt.grid(True, which="both")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Re(R)")
plt.legend()

plt.figure(3)
plt.semilogx(freq, np.imag(R_tgt), label="Target Im(R)", color='k', linewidth=2)
plt.semilogx(freq, np.imag(R_hat), "--", label="Fitted Im(R)", color='lime', linewidth=2)
plt.grid(True, which="both")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Im(R)")
plt.legend()

plt.show()
