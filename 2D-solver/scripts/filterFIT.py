import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import hilbert

# ============================================================
# DO NOT TOUCH YOUR I/O CONVENTIONS (kept)
# ============================================================
f_oct = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)

read_file = "Material0.mat"

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "input", "mat", read_file)
Rmag_oct = np.loadtxt(data_path)

config_path = os.path.join(script_dir, "..", "config", "config.txt")

v = np.loadtxt(config_path)

K = int(v[5])  # real poles
L = int(v[6])  # complex conjugate pairs

# simulation / medium (kept)
h = v[0]
f_cutoff = v[2]
Temp = v[3]
C_CFL = v[4]
R_spec = 287.05
gamma = 1.4
rho = 1.225
f_tol = 0.2*f_cutoff
f_cutoff += f_tol
c = np.sqrt(gamma * R_spec * Temp)
dt = C_CFL * h / c


# ============================================================
# SETTINGS (assertive defaults)
# ============================================================
USE_MIN_PHASE = True          # you only have |R| => must synthesize a causal phase
LF_ANCHOR_TO_RIGID = True     # |R| -> 1 at LF (recommended)
MAG_FLOOR = 1e-6

VF_ITERS = 13
VF_TOL_IMAG_REAL = 1e-8
VF_RELAX = 0.1               # pole update relaxation (stability)

# ============================================================
# helpers
# ============================================================
def mag_target_from_octaves(f_meas, mag_meas,
                            f_low=20.0, f_high=None,
                            ngrid=512,
                            mag_low=None, mag_high=None):
    f_meas = np.asarray(f_meas, float)
    mag_meas = np.clip(np.asarray(mag_meas, float), MAG_FLOOR, 1.0)
    if f_high is None:
        f_high = float(f_meas.max())
    if mag_low is None:
        mag_low = float(mag_meas[0])
    if mag_high is None:
        mag_high = float(mag_meas[-1])

    f_anchor = np.r_[f_low, f_meas, f_high]
    mag_anchor = np.r_[mag_low, mag_meas, mag_high]
    mag_anchor = np.clip(mag_anchor, MAG_FLOOR, 1.0)

    flog_u = np.linspace(np.log(f_anchor.min()), np.log(f_anchor.max()), ngrid)
    f_u = np.exp(flog_u)

    lnmag_u = np.interp(flog_u, np.log(f_anchor), np.log(mag_anchor))
    mag_u = np.exp(lnmag_u)
    mag_u = np.clip(mag_u, MAG_FLOOR, 1.0)
    return f_u, mag_u

def min_phase_from_mag(f_grid, mag_grid, w_uniform):
    w_src = 2*np.pi*np.asarray(f_grid, float)
    mag_src = np.clip(np.asarray(mag_grid, float), MAG_FLOOR, 1.0)
    lnmag_src = np.log(mag_src)

    lnmag_u = np.interp(w_uniform, w_src, lnmag_src)
    analytic = hilbert(lnmag_u)     # uniform grid required
    phi_u = -np.imag(analytic)
    mag_u = np.exp(lnmag_u)
    mag_u = np.clip(mag_u, MAG_FLOOR, 1.0)
    return mag_u, phi_u

def stack_real_imag(Ac, bc):
    A = np.vstack([np.real(Ac), np.imag(Ac)])
    b = np.hstack([np.real(bc), np.imag(bc)])
    return A, b

def enforce_stable(a):
    # reflect unstable poles into LHP
    ar = np.real(a)
    ai = np.imag(a)
    ar = -np.abs(ar)
    return ar + 1j*ai

def pack_poles_from_zeta_alpha_beta(zeta, alpha, beta):
    a = []
    for k in range(len(zeta)):
        a.append(-zeta[k] + 0j)
    for l in range(len(alpha)):
        a.append(-alpha[l] + 1j*beta[l])
        a.append(-alpha[l] - 1j*beta[l])
    return np.array(a, dtype=complex)

def unpack_to_zeta_alpha_beta(a, K, L):
    # Choose K "most real" poles as real; rest as L conjugate pairs (positive imag)
    a = np.array(a, dtype=complex)

    # snap near-real
    a2 = a.copy()
    a2[np.abs(np.imag(a2)) < VF_TOL_IMAG_REAL] = np.real(a2[np.abs(np.imag(a2)) < VF_TOL_IMAG_REAL]) + 0j

    # pick real poles
    idx_sorted = np.argsort(np.abs(np.imag(a2)))
    real_idx = []
    for idx in idx_sorted:
        if len(real_idx) >= K:
            break
        if np.abs(np.imag(a2[idx])) < 1e-3:
            real_idx.append(idx)
    # if not enough "real", just take smallest imag
    if len(real_idx) < K:
        real_idx = list(idx_sorted[:K])

    real_poles = a2[real_idx]
    rest_mask = np.ones(len(a2), dtype=bool)
    rest_mask[real_idx] = False
    rest = a2[rest_mask]

    # build conjugate pairs by taking positive imag poles
    pos = rest[np.imag(rest) > 0]
    if len(pos) < L:
        # fallback: pick poles with largest |imag| as pairs
        rest_sorted = rest[np.argsort(-np.abs(np.imag(rest)))]
        pos = rest_sorted[np.imag(rest_sorted) >= 0][:L]
    pos = pos[:L]

    zeta = -np.real(real_poles)
    alpha = -np.real(pos)
    beta = np.abs(np.imag(pos))

    # ensure strictly positive
    zeta = np.clip(zeta, 1e-6, None)
    alpha = np.clip(alpha, 1e-6, None)
    beta = np.clip(beta, 1e-6, None)
    return zeta, alpha, beta

# ============================================================
# Hornikx–Wang constrained model (C = -alpha B / beta)
# This simplifies the pair term to:  B * s / ((s+alpha)^2 + beta^2)
# ============================================================
def R_model_constrained(w, Rinf, A, zeta, B, alpha, beta):
    s = 1j*w
    R = (Rinf + 0j) * np.ones_like(s)

    if len(zeta):
        R += (A[:, None] / (zeta[:, None] + s[None, :])).sum(axis=0)

    if len(alpha):
        den = (s[None, :] + alpha[:, None])**2 + (beta[:, None]**2)
        R += (B[:, None] * s[None, :] / den).sum(axis=0)

    return R

# ============================================================
# Vector Fitting: pole relocation (Gustavsen-style)
# ============================================================
def vector_fit_poles(s, F, a_init, n_iters=10, relax=0.6):
    a = np.array(a_init, dtype=complex)

    for _ in range(n_iters):
        # Build LS system for: F*(1 + sum d/(s-a)) ≈ c0 + sum c/(s-a)
        Phi = 1.0 / (s[:, None] - a[None, :])             # (Ns, Np)
        # Unknowns: [c0, c(1..N), d(1..N)]
        Ac = np.hstack([
            np.ones((len(s), 1), dtype=complex),
            Phi,
            -F[:, None] * Phi
        ])
        bc = F

        A, b = stack_real_imag(Ac, bc)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)

        # Extract d
        d = x[1+len(a):1+2*len(a)]
        d = d.astype(float)  # from real-stacked system

        # Relocation: roots of sigma(s)=1+sum d/(s-a)
        # Eigenvalues of: diag(a) - 1*d^T
        Arel = np.diag(a) - np.ones((len(a), 1)) @ d.reshape(1, -1)
        a_new = np.linalg.eigvals(Arel)

        # enforce stability + relax
        a_new = enforce_stable(a_new)
        a = relax*a_new + (1.0-relax)*a
        a = enforce_stable(a)

    return a

# ============================================================
# Final constrained LS solve for (Rinf, A_k, B_l) with fixed poles
# ============================================================
def solve_constrained_coeffs(s, F, zeta, alpha, beta):
    # Basis: 1, 1/(s+zeta_k), s/((s+alpha_l)^2+beta_l^2)
    cols = [np.ones_like(s)]
    for zk in zeta:
        cols.append(1.0 / (s + zk))
    for al, be in zip(alpha, beta):
        cols.append(s / ((s + al)**2 + be**2))

    Ac = np.column_stack(cols)  # complex
    A, b = stack_real_imag(Ac, F)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    x = x.astype(float)

    Rinf = x[0]
    Acoef = x[1:1+len(zeta)]
    Bcoef = x[1+len(zeta):]
    return Rinf, Acoef, Bcoef

# ============================================================
# CUT-OFF MASKING (kept)
# ============================================================
mask = f_oct <= f_cutoff
f_oct = f_oct[mask]
Rmag_oct = Rmag_oct[mask]

# ============================================================
# Build |R| target (kept style) + complex target via min-phase
# ============================================================
f_low = 20.0
f_high = float(f_cutoff)

mag_low = 0.999 if LF_ANCHOR_TO_RIGID else float(Rmag_oct[0])
mag_high = float(Rmag_oct[-1])

f_log, mag_log = mag_target_from_octaves(f_oct, Rmag_oct,
                                         f_low=f_low, f_high=f_high,
                                         ngrid=512,
                                         mag_low=mag_low, mag_high=mag_high)

w_min = 2*np.pi*f_low
w_max = 2*np.pi*f_high
Nw = 2048
w_fit = np.linspace(w_min, w_max, Nw)
f_fit = w_fit / (2*np.pi)
s_fit = 1j*w_fit

mag_fit = np.interp(f_fit, f_log, mag_log)
mag_fit = np.clip(mag_fit, MAG_FLOOR, 1.0)

if USE_MIN_PHASE:
    mag_u, phi_u = min_phase_from_mag(f_log, mag_log, w_fit)
    R_tgt = mag_u * np.exp(1j*phi_u)
else:
    R_tgt = mag_fit.astype(complex)

# ============================================================
# Initial poles (fixed count) with your stability bounds logic
# ============================================================
eps = 1e-6
nu_SSP_RE = 2.5127
nu_SSP_IM = 1.732
omega_max = 2*np.pi*float(f_cutoff)
kappa_band = 5.0

zeta_max  = min(nu_SSP_RE / dt, kappa_band * omega_max)
alpha_max = min(nu_SSP_RE / dt, kappa_band * omega_max)
beta_max  = min(0.85 * omega_max, nu_SSP_IM / dt)
beta_min  = 2*np.pi*max(0.8*float(f_oct.min()), f_low)

zeta0  = np.clip(np.logspace(np.log10(w_fit.min()), np.log10(w_fit.max()), K), eps, zeta_max)
alpha0 = np.clip(np.logspace(np.log10(w_fit.min()), np.log10(w_fit.max()), L), eps, alpha_max)
beta0  = np.clip(np.logspace(np.log10(beta_min), np.log10(w_fit.max()), L), beta_min, beta_max)

a0 = pack_poles_from_zeta_alpha_beta(zeta0, alpha0, beta0)

# ============================================================
# VECTOR FITTING (pole relocation)
# ============================================================
a_vf = vector_fit_poles(s_fit, R_tgt, a0, n_iters=VF_ITERS, relax=VF_RELAX)
a_vf = enforce_stable(a_vf)

# re-quantize into exactly K real + L pairs
zeta, alpha, beta = unpack_to_zeta_alpha_beta(a_vf, K, L)
# clip within your stability limits
zeta  = np.clip(zeta,  eps, zeta_max)
alpha = np.clip(alpha, eps, alpha_max)
beta  = np.clip(beta,  beta_min, beta_max)

# ============================================================
# FINAL constrained coefficients solve (enforces C = -alpha B / beta)
# ============================================================
Rinf, Acoef, Bcoef = solve_constrained_coeffs(s_fit, R_tgt, zeta, alpha, beta)
Ccoef = -(alpha / beta) * Bcoef  # enforced output

# ============================================================
# SAVE (SAME FORMAT, SAME PATHS)
# ============================================================
save_file = read_file[:-4] + ".fit"
coefs_path = os.path.join(script_dir, "..", "input", "fit", save_file)
#np.savetxt(poles_path, np.array([[K, L]]), fmt="%d")

with open(coefs_path, 'w') as f:
    f.write("# R_inf\n")
    f.write(f"{Rinf}\n")

    f.write("# A_k zeta_k\n")
    for k in range(K):
        f.write(f"{Acoef[k]} {zeta[k]}\n")

    f.write("# B_l C_l alpha_l beta_l\n")
    for l in range(L):
        f.write(f"{Bcoef[l]} {Ccoef[l]} {alpha[l]} {beta[l]}\n")
    if L == 0:
        f.write("0.0 0.0 0.0 0.0\n")

# ============================================================
# PLOTS (|R| + Re/Im, you asked for it)
# ============================================================
R_hat = R_model_constrained(w_fit, Rinf, Acoef, zeta, Bcoef, alpha, beta)

plt.figure(1)
plt.semilogx(f_fit, np.abs(R_tgt), label="Target |R|", linewidth=2)
plt.semilogx(f_fit, np.abs(R_hat), "--", label="Fitted |R|", linewidth=2)
plt.ylim(0, 1.05)
plt.grid(True, which="both")
plt.xlim(f_low, f_high)
plt.xlabel("Frequency [Hz]")
plt.ylabel("|R|")
plt.legend()

plt.figure(2)
plt.semilogx(f_fit, np.real(R_tgt), label="Target Re(R)", linewidth=2)
plt.semilogx(f_fit, np.real(R_hat), "--", label="Fitted Re(R)", linewidth=2)
plt.grid(True, which="both")
plt.xlim(f_low, f_high)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Re(R)")
plt.legend()

plt.figure(3)
plt.semilogx(f_fit, np.imag(R_tgt), label="Target Im(R)", linewidth=2)
plt.semilogx(f_fit, np.imag(R_hat), "--", label="Fitted Im(R)", linewidth=2)
plt.grid(True, which="both")
plt.xlim(f_low, f_high)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Im(R)")
plt.legend()

plt.show()
