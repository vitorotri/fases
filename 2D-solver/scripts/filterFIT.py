import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import hilbert
import os

# ============================================================
# CONFIG (assertive defaults)
# ============================================================
USE_MIN_PHASE = True          # If you only have |R|, this should be True.
LF_ANCHOR_TO_RIGID = True     # Enforce |R| -> ~1 as f->0 (recommended).
MAG_FLOOR = 1e-5              # Avoid log(0) in ln|R|.

# ============================================================
# 1) Build smooth |R|(f) target from octave band values
# ============================================================
def mag_target_from_octaves(
    f_meas, mag_meas,
    f_low=None, f_high=None,
    ngrid=512,
    mag_low=None, mag_high=None
):
    """
    Build a smooth magnitude target |R|(f) on a log-frequency grid by
    interpolating log(|R|) vs log(f).
    """
    f_meas = np.asarray(f_meas, float)
    mag_meas = np.clip(np.asarray(mag_meas, float), MAG_FLOOR, 1.0)

    if f_low is None:
        f_low = float(f_meas.min())
    if f_high is None:
        f_high = float(f_meas.max())

    if mag_low is None:
        mag_low = float(mag_meas[0])
    if mag_high is None:
        mag_high = float(mag_meas[-1])

    # Anchors
    f_anchor = np.r_[f_low, f_meas, f_high]
    mag_anchor = np.r_[mag_low, mag_meas, mag_high]
    mag_anchor = np.clip(mag_anchor, MAG_FLOOR, 1.0)

    # Log-spaced frequency grid
    flog = np.log(f_anchor)
    flog_u = np.linspace(flog.min(), flog.max(), ngrid)
    f_u = np.exp(flog_u)

    # Interpolate log-magnitude in log-frequency
    logmag_u = np.interp(flog_u, np.log(f_anchor), np.log(mag_anchor))
    mag_u = np.exp(logmag_u)
    mag_u = np.clip(mag_u, MAG_FLOOR, 1.0)
    return f_u, mag_u


# ============================================================
# 2) Minimum-phase reconstruction from |R|(f)
#    phase = -Hilbert( ln|R| ) on a UNIFORM angular-frequency grid
# ============================================================
def min_phase_from_mag(f_log_grid, mag_log_grid, w_uniform):
    """
    Given mag sampled on some grid (typically log-spaced in f),
    resample ln|R| onto a uniform omega grid, do Hilbert transform,
    and return minimum-phase phi(w) and |R|(w) on that uniform grid.

    This is an assumption. But it's the *correct* assumption if your goal is:
    "invent a causal phase consistent with the given magnitude".
    """
    w_src = 2.0 * np.pi * np.asarray(f_log_grid, float)
    mag_src = np.clip(np.asarray(mag_log_grid, float), MAG_FLOOR, 1.0)

    # Interpolate ln|R| onto uniform omega grid
    lnmag_src = np.log(mag_src)
    lnmag_u = np.interp(w_uniform, w_src, lnmag_src)

    # Hilbert transform of ln|R| on uniform grid
    analytic = hilbert(lnmag_u)
    hilb_lnmag = np.imag(analytic)

    phi_u = -hilb_lnmag  # minimum-phase convention
    mag_u = np.exp(lnmag_u)
    mag_u = np.clip(mag_u, MAG_FLOOR, 1.0)
    return mag_u, phi_u


# ============================================================
# 3) Hornikx–Wang frequency-domain reflection model R(omega)
# ============================================================
def R_model(w, Rinf, A, zeta, B, alpha, beta):
    s = 1j * w
    R = Rinf + 0j

    # real poles: sum_k A_k / (zeta_k + s)
    if A.size:
        R += (A[:, None] / (zeta[:, None] + s[None, :])).sum(axis=0)

    # complex conjugate pairs with constrained numerator
    if B.size:
        s2 = s[None, :]
        al = alpha[:, None]
        bl = beta[:, None]
        C = -(alpha / beta) * B  # Hornikx constraint

        num = B[:, None] * s2 + (C[:, None] * bl) + al * B[:, None]
        den = (al + s2) ** 2 + bl ** 2
        R += (num / den).sum(axis=0)

    return R


def unpack(x, K, L):
    i = 0
    Rinf = x[i]; i += 1
    A    = x[i:i+K]; i += K
    zeta = x[i:i+K]; i += K
    B    = x[i:i+L]; i += L
    alpha= x[i:i+L]; i += L
    beta = x[i:i+L]; i += L
    return Rinf, A, zeta, B, alpha, beta


# ============================================================
# 4) Residual: fit Re and Im simultaneously (THIS IS THE FIX)
# ============================================================
def resid_complex_fit(
    x, K, L, w_fit, R_tgt,
    w_pass=None, lam_pass=500.0,
    aux_param=0.1, lam_damp=500.0,
    r_max=2.0, lam_rmax=500.0
):
    Rinf, A, zeta, B, alpha, beta = unpack(x, K, L)
    Rhat = R_model(w_fit, Rinf, A, zeta, B, alpha, beta)

    # complex fit residual: stack real and imag
    e = (Rhat - R_tgt).ravel()
    r = np.hstack([np.real(e), np.imag(e)])

    # passivity penalty: |R| <= 1
    if w_pass is not None and lam_pass > 0:
        Rp = R_model(w_pass, Rinf, A, zeta, B, alpha, beta)
        viol = np.maximum(np.abs(Rp) - 1.0, 0.0)
        r = np.hstack([r, np.sqrt(lam_pass) * viol])

    # optional safeguard: alpha >= aux_param * beta
    if L > 0 and lam_damp > 0 and aux_param > 0:
        viol_d = np.maximum(aux_param * beta - alpha, 0.0)
        r = np.hstack([r, np.sqrt(lam_damp) * viol_d])

    # optional safeguard: alpha <= r_max * beta
    if L > 0 and lam_rmax > 0 and r_max > 0:
        viol_hi = np.maximum(alpha - r_max * beta, 0.0)
        r = np.hstack([r, np.sqrt(lam_rmax) * viol_hi])

    return r


# ============================================================
# 5) INPUT (your existing structure kept)
# ============================================================
f_oct = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)

# user inputs
read_file = "Material1.mat"
K = 3  # real poles
L = 2  # complex pairs

# simulation / medium
h = 0.005
C_CFL = 0.3
R_spec = 287.05
gamma = 1.4
rho = 1.225
Temp = 288.15
f_cutoff = 1000.0
f_tol = 200

f_cutoff += f_tol
c = np.sqrt(gamma * R_spec * Temp)
dt = C_CFL * h / c

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "input", "mat", read_file)
Rmag_oct = np.loadtxt(data_path)

# Path to config/poles.txt from scripts/
poles_path = os.path.join(script_dir, "..", "config", "poles.txt")
np.savetxt(poles_path, np.array([[K, L]]), fmt="%d")

# Effective cutoff masking
mask = f_oct <= f_cutoff
f_oct = f_oct[mask]
Rmag_oct = Rmag_oct[mask]

# ============================================================
# 6) Build magnitude target with PHYSICAL LF anchoring
# ============================================================
f_low = 20.0
f_high = float(f_cutoff)

if LF_ANCHOR_TO_RIGID:
    mag_low = 0.999  # assertive: low-f tends to rigid reflection for many liners
else:
    mag_low = float(Rmag_oct[0])

mag_high = float(Rmag_oct[-1])

f_log, mag_log = mag_target_from_octaves(
    f_oct, Rmag_oct,
    f_low=f_low,
    f_high=f_high,
    ngrid=512,
    mag_low=mag_low,
    mag_high=mag_high
)

# Build a UNIFORM omega grid for min-phase Hilbert
w_min = 2.0 * np.pi * f_low
w_max = 2.0 * np.pi * f_high
Nw = 2048
w_fit = np.linspace(w_min, w_max, Nw)
f_fit = w_fit / (2.0 * np.pi)

# Resample |R| onto w_fit
mag_fit = np.interp(f_fit, f_log, mag_log)
mag_fit = np.clip(mag_fit, MAG_FLOOR, 1.0)

# ============================================================
# 7) Construct complex target R_tgt(w)
# ============================================================
if USE_MIN_PHASE:
    mag_u, phi_u = min_phase_from_mag(f_log, mag_log, w_fit)
    R_tgt = mag_u * np.exp(1j * phi_u)
else:
    # If you set this False, you are back to an underdetermined problem.
    # You can still fit, but expect failures on resonant/notched materials.
    R_tgt = mag_fit.astype(complex)

# ============================================================
# 8) Initial guess + bounds (kept similar to yours)
# ============================================================
eps = 1e-6

zeta0  = np.logspace(np.log10(w_fit.min()), np.log10(w_fit.max()), K)
alpha0 = np.logspace(np.log10(w_fit.min()), np.log10(w_fit.max()), L)

# seed betas around band of interest
beta0 = np.logspace(np.log10(2*np.pi*max(0.8*float(f_oct.min()), f_low)),
                    np.log10(2*np.pi*float(f_high)),
                    L)

# SSP RK3 stability region factors you used
nu_SSP_RE = 2.5127
nu_SSP_IM = 1.732

omega_max = 2 * np.pi * float(f_high)
kappa_band = 5.0

zeta_max  = min(nu_SSP_RE / dt, kappa_band * omega_max)
alpha_max = min(nu_SSP_RE / dt, kappa_band * omega_max)
beta_max  = min(0.85 * omega_max, nu_SSP_IM / dt)
beta_min  = 2 * np.pi * max(0.8 * float(f_oct.min()), f_low)

zeta0  = np.clip(zeta0,  eps, zeta_max)
alpha0 = np.clip(alpha0, eps, alpha_max)
beta0  = np.clip(beta0,  beta_min, beta_max)

# better Rinf seed: use high-frequency complex target (real part) as start
Rinf0 = float(np.clip(np.real(R_tgt[-1]), -0.99, 0.99))

A0 = np.zeros(K)
B0 = np.zeros(L)

x0 = np.hstack([Rinf0, A0, zeta0, B0, alpha0, beta0])

lb = np.hstack([
    -1.0,
    -np.inf * np.ones(K),
    eps * np.ones(K),
    -np.inf * np.ones(L),
    eps * np.ones(L),
    beta_min * np.ones(L)
])

ub = np.hstack([
     1.0,
     np.inf * np.ones(K),
     zeta_max * np.ones(K),
     np.inf * np.ones(L),
     alpha_max * np.ones(L),
     beta_max * np.ones(L)
])

x0 = np.maximum(x0, lb)
x0 = np.minimum(x0, ub)

# Passivity grid (dense)
w_pass = np.linspace(w_min, w_max, 4096)

# ============================================================
# 9) Solve (stage 1 then stage 2 with passivity)
# ============================================================
res1 = least_squares(
    resid_complex_fit, x0, bounds=(lb, ub),
    args=(K, L, w_fit, R_tgt, None, 0.0),
    method="trf",
    max_nfev=2000, ftol=1e-8, xtol=1e-8, gtol=1e-8,
    verbose=0
)

res2 = least_squares(
    resid_complex_fit, res1.x, bounds=(lb, ub),
    args=(K, L, w_fit, R_tgt, w_pass, 500.0),
    method="trf",
    max_nfev=2500, ftol=1e-10, xtol=1e-10, gtol=1e-10,
    verbose=0
)

Rinf, A, zeta, B, alpha, beta = unpack(res2.x, K, L)
C = -(alpha / beta) * B

# Diagnostics
if L > 0:
    f_res = beta / (2*np.pi)
    print("\n--- Resonance diagnostics ---")
    for l in range(L):
        print(f"Pair {l}: f_res={f_res[l]:.1f} Hz, alpha/beta={alpha[l]/beta[l]:.3f}, "
              f"alpha={alpha[l]:.3e}, beta={beta[l]:.3e}, B={B[l]:.3e}, C={C[l]:.3e}")

# ============================================================
# 10) Save to file (same format as before)
# ============================================================
save_file = read_file[:-4] + ".fit"
coefs_path = os.path.join(script_dir, "..", "input", "fit", save_file)
np.savetxt(poles_path, np.array([[K, L]]), fmt="%d")

with open(coefs_path, 'w') as f:
    f.write("# R_inf\n")
    f.write(f"{Rinf}\n")

    f.write("# A_k zeta_k\n")
    for k in range(K):
        f.write(f"{A[k]} {zeta[k]}\n")

    f.write("# B_l C_l alpha_l beta_l\n")
    for l in range(L):
        f.write(f"{B[l]} {C[l]} {alpha[l]} {beta[l]}\n")
    if L == 0:
        f.write("0.0 0.0 0.0 0.0\n")

# ============================================================
# 11) Evaluate + plots
# ============================================================
R_hat = R_model(w_fit, Rinf, A, zeta, B, alpha, beta)

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
plt.semilogx(f_fit, np.unwrap(np.angle(R_tgt)), label="Target phase (min-phase)", linewidth=2)
plt.semilogx(f_fit, np.unwrap(np.angle(R_hat)), "--", label="Fitted phase", linewidth=2)
plt.grid(True, which="both")
plt.xlim(f_low, f_high)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase [rad]")
plt.legend()

plt.figure(3)
Rp = R_model(w_pass, Rinf, A, zeta, B, alpha, beta)
plt.plot(w_pass/(2*np.pi), np.abs(Rp))
plt.axhline(1.0, linewidth=1, color='k')
plt.ylim(0, 1.2)
plt.grid(True, which="both")
plt.xlim(f_low, f_high)
plt.xlabel("Frequency [Hz]")
plt.ylabel("|R_fit| (passivity check)")

# ============================================================
# EXTRA PLOTS: Re(R) and Im(R) fit (what you asked for)
# ============================================================
plt.figure(4)
plt.semilogx(f_fit, np.real(R_tgt), label="Target Re(R)", linewidth=2)
plt.semilogx(f_fit, np.real(R_hat), "--", label="Fitted Re(R)", linewidth=2)
plt.grid(True, which="both")
plt.xlim(f_low, f_high)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Re(R)")
plt.legend()

plt.figure(5)
plt.semilogx(f_fit, np.imag(R_tgt), label="Target Im(R)", linewidth=2)
plt.semilogx(f_fit, np.imag(R_hat), "--", label="Fitted Im(R)", linewidth=2)
plt.grid(True, which="both")
plt.xlim(f_low, f_high)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Im(R)")
plt.legend()


plt.show()
