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
from scipy.io import savemat
import os
import skrf as rf
from skrf.vectorFitting import VectorFitting

def octave_centers():
    return np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)

def interp_loglog(f, fc, y_fc):
    # interpolate y in log-f domain
    lf = np.log(f)
    lfc = np.log(fc)
    return np.interp(lf, lfc, y_fc, left=y_fc[0], right=y_fc[-1])

def min_phase_from_mag(f, Rmag):
    """
    f: Hz, 1D increasing
    Rmag: |R(f)|, same shape
    returns phase phi(f) (radians) using PV integral in omega
    """
    Rmag = np.clip(Rmag, 1e-12, 0.999999999)
    omega = 2*np.pi*f
    logmag = np.log(Rmag)

    phi = np.zeros_like(f, dtype=float)
    # PV integral discretization
    for i in range(len(f)):
        w = omega[i]
        denom = omega - w
        denom[i] = np.inf  # remove singular term
        integrand = logmag / denom
        # trapezoidal integral over omega
        phi[i] = -(1/np.pi) * np.trapz(integrand, omega)

    return phi

def Z_from_octave_Rmag(freq, R_oct, z0):
    """
    freq : Hz array (your target grid)
    R_oct: 8 values at octave centers 63..8000 Hz (magnitudes)
    z0   : characteristic impedance
    returns: complex Z(freq)
    """
    fc = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], float)
    R_oct = np.clip(np.asarray(R_oct, float), 1e-6, 0.999999)

    # interpolate ln|R| on ln f (log-log)
    logR = np.interp(np.log(freq), np.log(fc), np.log(R_oct),
                     left=np.log(R_oct[0]), right=np.log(R_oct[-1]))
    Rmag = np.exp(logR)

    # minimum-phase phase via PV integral (simple, O(N^2), OK for ~1e4 points)
    omega = 2*np.pi*freq
    logmag = np.log(np.clip(Rmag, 1e-12, 1.0))
    phi = np.zeros_like(freq, float)
    for i in range(freq.size):
        denom = omega - omega[i]
        denom[i] = np.inf
        phi[i] = -(1/np.pi) * np.trapz(logmag / denom, omega)

    R = Rmag * np.exp(1j*phi)
    Z = z0 * (1 + R) / (1 - R)
    return Z

# --------------------
# Block 1: User input
# --------------------

read_file = "Material1.mat"

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "input", "mat", read_file)
Rmag_oct = np.loadtxt(data_path)

config_path = os.path.join(script_dir, "..", "config", "config.txt")

v = np.loadtxt(config_path)

f_cutoff = v[2]
f_tol = 0.2*f_cutoff
f_cutoff += f_tol
Temp = v[3]
R_spec = 287.05
gamma = 1.4

c0 = np.sqrt(gamma * R_spec * Temp)         # speed of sound (m/s)
rho0 = 1.225                                # air density
z0 = c0 * rho0                              # characteristic impedance of air

savename = np.copy(read_file[:-4]) # output .mat filename (without extension)

# Frequency samples (Hz)
f_min = 20.0
f_max = f_cutoff
frequency_step = 5.0
freq = np.arange(f_min, f_max + 0.5 * frequency_step, frequency_step)  # include f_max robustly

Niter = 20       # number of iterations (MATLAB did outer loop; skrf uses max_iterations)
N = int(v[5])    # "number of poles" in MATLAB script (here: number of REAL starting poles)

# Example impedance (Miki model) to be approximated (not reading from .mat file)
'''
sigma = 100000.0
x = freq / sigma
Z_Miki = z0 * (1.0 + 5.50 * (x * 1000.0) ** (-0.632) - 1j * 8.43 * (x * 1000.0) ** (-0.632))

Zs_to_fit = Z_Miki
'''
Zs_to_fit = Z_from_octave_Rmag(freq, Rmag_oct, z0)

# --------------------------------------------------------------------------------
# Block 2: Vector fit reflection coefficient with scikit-rf
# --------------------------------------------------------------------------------
ReflectionCoefficient = (Zs_to_fit / z0 - 1.0) / (Zs_to_fit / z0 + 1.0)

# Build a 1-port skrf Network with S11 = ReflectionCoefficient
frequency = rf.Frequency.from_f(freq, unit="hz")
nw = rf.Network(frequency=frequency)
nw.s = ReflectionCoefficient.reshape(-1, 1, 1)  # shape: (n_freq, n_ports, n_ports)
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
beta  = (-poles_cmplx.imag)

Bcoef = 2.0 * res_cmplx.real
Ccoef = -2.0 * res_cmplx.imag

L = len(alpha)

# ------------------------------------------------------------
# SAVE IN C++ FORMAT
# ------------------------------------------------------------

save_file = read_file[:-4] + ".fit"
coefs_path = os.path.join(script_dir, "..", "input", "fit", save_file)

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

Npoles_path = os.path.join(script_dir, "..", "config/poles.txt")
Npoles = np.loadtxt(Npoles_path)
K_m = Npoles[0]
L_m = Npoles[1]
if K > K_m:
	K_m = K
if L > L_m:
	L_m = L
Npoles = np.array([[int(K_m), int(L_m)]], dtype=int)
np.savetxt(Npoles_path, Npoles, fmt='%d', delimiter=' ')
