import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os

# ----------------------------
# Magnitude-only target from octave-band |R|
# (no minimum-phase, no complex target)
# ----------------------------
def mag_target_from_octaves(f_meas, mag_meas,
                            f_low=None, f_high=None,
                            ngrid=256,
                            mag_low=None,
                            mag_high=None):
    """
    Build a smooth magnitude target |R|(f) on a log-frequency grid by
    interpolating log(|R|) vs log(f). This avoids minimum-phase estimation.

    f_low / f_high:
      - If provided, extend the target with anchors for nicer behavior near edges.
      - If omitted, we fit only between min/max measured.

    mag_low / mag_high:
      - If provided, used for the extensions.
      - If omitted and f_low/f_high are used, defaults to:
          mag_low = mag_meas[0], mag_high = mag_meas[-1]
    """
    f_meas = np.asarray(f_meas, float)
    mag_meas = np.clip(np.asarray(mag_meas, float), 1e-6, 1.0)

    if f_low is None:
        f_low = float(f_meas.min())
    if f_high is None:
        f_high = float(f_meas.max())

    if mag_low is None:
        mag_low = float(mag_meas[0])
    if mag_high is None:
        mag_high = float(mag_meas[-1])

    # Build anchors (optionally extended)
    f_anchor = np.r_[f_low, f_meas, f_high]
    mag_anchor = np.r_[mag_low, mag_meas, mag_high]

    # Log-spaced grid
    flog = np.log(f_anchor)
    flog_u = np.linspace(flog.min(), flog.max(), ngrid)
    f_u = np.exp(flog_u)

    # Interpolate log-magnitude in log-frequency
    logmag_u = np.interp(flog_u, np.log(f_anchor), np.log(mag_anchor))
    mag_u = np.exp(logmag_u)
    mag_u = np.clip(mag_u, 0.0, 1.0)

    return f_u, mag_u


# ----------------------------
# Hornikx–Wang frequency-domain model
#   Enforce C = -(alpha/beta) * B (Hornikx mass-spring-damper constraint)
# ----------------------------
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

        # Hornikx constraint (note the minus sign)
        C = -(alpha / beta) * B          # (L,)
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


def resid_mag_only(x, K, L, w, mag_tgt,
                   w_pass=None, lam_pass=500.0,
                   aux_param=0.1, lam_damp=500.0,
                   r_max=2.0, lam_rmax=500.0):
    """
    Magnitude-only residual:
      r = |R_model(w)| - |R|_target(w)
    plus optional penalties (passivity + alpha/beta safeguards).
    """
    Rinf, A, zeta, B, alpha, beta = unpack(x, K, L)
    Rhat = R_model(w, Rinf, A, zeta, B, alpha, beta)

    # magnitude fit residual
    r = (np.abs(Rhat) - mag_tgt).ravel()

    # passivity penalty: |R| <= 1
    if w_pass is not None and lam_pass > 0:
        Rp = R_model(w_pass, Rinf, A, zeta, B, alpha, beta)
        viol = np.maximum(np.abs(Rp) - 1.0, 0.0)
        r = np.hstack([r, np.sqrt(lam_pass) * viol])

    # optional safeguard: alpha >= aux_param * beta
    if L > 0 and lam_damp > 0 and aux_param > 0:
        viol_d = np.maximum(aux_param * beta - alpha, 0.0)
        r = np.hstack([r, np.sqrt(lam_damp) * viol_d])

    # optional safeguard: alpha <= r_max * beta (keeps pairs meaningfully oscillatory)
    if L > 0 and lam_rmax > 0 and r_max > 0:
        viol_hi = np.maximum(alpha - r_max * beta, 0.0)
        r = np.hstack([r, np.sqrt(lam_rmax) * viol_hi])

    return r


# ----------------------------
# Input: octave-band centers, |R| (0..1) from file and cutoff frequency
# ----------------------------
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

# ----------------------------
# Effective cutoff masking
# ----------------------------
mask = f_oct <= f_cutoff
f_oct = f_oct[mask]
Rmag_oct = Rmag_oct[mask]

# ----------------------------
# Build magnitude-only target (no minimum-phase)
# ----------------------------
# If you want *no* artificial extension, set f_low=None, f_high=None.
# If you want the same kind of gentle edge conditioning as before, keep extensions.
f_fit, mag_target = mag_target_from_octaves(
    f_oct, Rmag_oct,
    f_low=20.0,                    # low anchor (optional)
    f_high=float(f_cutoff),         # high anchor (optional)
    ngrid=256,
    mag_low=float(Rmag_oct[0]),     # or 1.0 if you think it tends to rigid at LF
    mag_high=float(Rmag_oct[-1])    # flat high extension
)

w_fit = 2 * np.pi * f_fit

# ----------------------------
# Initial guess
# ----------------------------
eps = 1e-6

f_seed_min = max(0.8 * float(f_oct.min()), 20.0)  # Hz
w_seed_min = 2 * np.pi * f_seed_min
w_seed_max = 2 * np.pi * float(f_cutoff)

zeta0  = np.logspace(np.log10(w_fit.min()), np.log10(w_fit.max()), K)
alpha0 = np.logspace(np.log10(w_fit.min()), np.log10(w_fit.max()), L)
beta0  = np.logspace(np.log10(w_seed_min), np.log10(w_seed_max), L)

# ----------------------------
# Bounds: stability/causality + |Rinf|<=1
# ----------------------------
nu_SSP_RE = 2.5127
nu_SSP_IM = 1.732

omega_max = 2 * np.pi * float(f_cutoff)
kappa_band = 5.0

zeta_max  = min(nu_SSP_RE / dt, kappa_band * omega_max)
alpha_max = min(nu_SSP_RE / dt, kappa_band * omega_max)
beta_max  = min(0.85 * omega_max, nu_SSP_IM / dt)

beta_min = 2 * np.pi * max(0.8 * float(f_oct.min()), 20.0)

zeta0  = np.clip(zeta0,  eps, zeta_max)
alpha0 = np.clip(alpha0, eps, alpha_max)
beta0  = np.clip(beta0,  beta_min, beta_max)

# With magnitude-only data, sign/phase of Rinf is unknown.
# Using a positive seed near the HF magnitude is a reasonable neutral start.
Rinf0 = float(np.clip(mag_target[-1], -0.99, 0.99))

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

# Passivity grid
f_pass = np.logspace(np.log10(f_fit.min()), np.log10(f_fit.max()), 512)
w_pass = 2 * np.pi * f_pass

# ----------------------------
# Stage 1: fit without passivity penalty
# ----------------------------
res1 = least_squares(
    resid_mag_only, x0, bounds=(lb, ub),
    args=(K, L, w_fit, mag_target, None, 0.0),
    method="trf",
    max_nfev=1200, ftol=1e-7, xtol=1e-7, gtol=1e-7,
    verbose=0
)

# ----------------------------
# Stage 2: add passivity penalty
# ----------------------------
res2 = least_squares(
    resid_mag_only, res1.x, bounds=(lb, ub),
    args=(K, L, w_fit, mag_target, w_pass),
    method="trf",
    max_nfev=1200, ftol=1e-8, xtol=1e-8, gtol=1e-8,
    verbose=0
)

Rinf, A, zeta, B, alpha, beta = unpack(res2.x, K, L)
C = -(alpha / beta) * B  # reported/output for compatibility

# resonance diagnostics
if L > 0:
    f_res = beta / (2 * np.pi)
    damp_ratio = alpha / beta
    print("\n--- Resonance diagnostics ---")
    for l in range(L):
        print(
            f"Pair {l}: "
            f"f_res = {f_res[l]:.1f} Hz, "
            f"alpha = {alpha[l]:.3e}, "
            f"beta = {beta[l]:.3e}, "
            f"alpha/beta = {damp_ratio[l]:.3f}, "
            f"C = {C[l]:.3e}"
        )

# ----------------------------
# Save to file (same format as before: include C even though it's computed)
# ----------------------------
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

# ----------------------------
# Evaluate and plot
# ----------------------------
R_hat = R_model(w_fit, Rinf, A, zeta, B, alpha, beta)

plt.figure(1)
plt.semilogx(f_fit, mag_target, label="Target |R| (interp from octaves)", c='b')
plt.semilogx(f_fit, np.abs(R_hat), "--", label="Fitted |R|", c='r')
plt.ylim(0, 1.05)
plt.grid(True, which="both")
plt.xlim(20, f_cutoff - f_tol)
plt.xlabel("Frequency [Hz]")
plt.ylabel("|R|")
plt.legend()

plt.figure(2)
plt.semilogx(f_pass, np.abs(R_model(w_pass, Rinf, A, zeta, B, alpha, beta)), color='lime')
plt.axhline(1.0, linewidth=1, c='k')
plt.ylim(0, 1.2)
plt.grid(True, which="both")
plt.xlabel("Frequency [Hz]")
plt.ylabel("|R_fit| (passivity check)")

plt.show()

