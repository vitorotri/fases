import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import hilbert
import os

# ----------------------------
# Minimum-phase complex target from |R| only
# ----------------------------
def minimum_phase_from_mag(f_meas, mag_meas,
                           f_low=5.0, f_high=20000.0,
                           ngrid=512, pad_frac=0.5,
                           mag0=1.0):
    f_meas = np.asarray(f_meas, float)
    mag_meas = np.clip(np.asarray(mag_meas, float), 1e-6, 1.0)

    # --- build extended anchors (low -> meas -> high) ---
    f_anchor = np.r_[f_low, f_meas, f_high]
    mag_anchor = np.r_[mag0, mag_meas, mag_meas[-1]]  # flat high extension

    # --- uniform in log-frequency grid ---
    flog = np.log(f_anchor)
    flog_u = np.linspace(flog.min(), flog.max(), ngrid)
    f_u = np.exp(flog_u)

    logmag_u = np.interp(flog_u, np.log(f_anchor), np.log(mag_anchor))

    # --- mirror padding to reduce Hilbert edge artifacts ---
    n_pad = int(pad_frac * ngrid)
    left  = logmag_u[1:n_pad+1][::-1]
    right = logmag_u[-n_pad-1:-1][::-1]
    logmag_pad = np.r_[left, logmag_u, right]

    phase_pad = -np.imag(hilbert(logmag_pad))
    phase_u = phase_pad[n_pad:n_pad+ngrid]

    Rmin = np.exp(logmag_u + 1j * phase_u)
    return f_u, Rmin


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


def resid(x, K, L, w, R_tgt, w_pass=None, lam_pass=500.0,
          aux_param=0.1, lam_damp=500.0,
          r_max=2.0, lam_rmax=500.0):

    Rinf, A, zeta, B, alpha, beta = unpack(x, K, L)
    Rhat = R_model(w, Rinf, A, zeta, B, alpha, beta)
    r = np.hstack([np.real(Rhat - R_tgt), np.imag(Rhat - R_tgt)])

    # passivity penalty: |R| <= 1
    if w_pass is not None and lam_pass > 0:
        Rp = R_model(w_pass, Rinf, A, zeta, B, alpha, beta)
        viol = np.maximum(np.abs(Rp) - 1.0, 0.0)
        r = np.hstack([r, np.sqrt(lam_pass) * viol])

    # optional damping ratio lower bound: alpha >= aux_param * beta (your safeguard)
    if L > 0 and lam_damp > 0 and aux_param > 0:
        viol_d = np.maximum(aux_param * beta - alpha, 0.0)
        r = np.hstack([r, np.sqrt(lam_damp) * viol_d])

    # NEW: prevent overdamped "fake complex pairs": enforce alpha <= r_max * beta
    # (soft constraint via penalty; keeps complex pairs actually oscillatory)
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
L = 3  # complex pairs

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

f_fit, R_target = minimum_phase_from_mag(f_oct, Rmag_oct)
w_fit = 2 * np.pi * f_fit

# ----------------------------
# Initial guess
# ----------------------------
eps = 1e-6

# IMPORTANT: do NOT seed beta0 from the 5 Hz extension; seed from meaningful band instead
f_seed_min = max(0.8 * float(f_oct.min()), 20.0)  # Hz
w_seed_min = 2 * np.pi * f_seed_min
w_seed_max = 2 * np.pi * min(float(f_cutoff), float(f_fit.max()))

zeta0  = np.logspace(np.log10(w_fit.min()), np.log10(w_fit.max()), K)
alpha0 = np.logspace(np.log10(w_fit.min()), np.log10(w_fit.max()), L)
beta0  = np.logspace(np.log10(w_seed_min), np.log10(w_seed_max), L)

# ----------------------------
# Bounds: stability/causality + |Rinf|<=1
# ----------------------------
# stability limits from SSP-RK3 applied to a regular ODE (your choice)
nu_SSP_RE = 2.5127
nu_SSP_IM = 1.732

omega_max = 2 * np.pi * f_cutoff
kappa_band = 5.0  # cap decay rates to a few times the modeled band (prevents "instantaneous" hacks)

zeta_max  = min(nu_SSP_RE / dt, kappa_band * omega_max)
alpha_max = min(nu_SSP_RE / dt, kappa_band * omega_max)
beta_max  = min(0.85 * omega_max, nu_SSP_IM / dt)

# NEW: meaningful beta floor to avoid beta->0 degeneracy with C=-(alpha/beta)B
beta_min = 2 * np.pi * max(0.8 * float(f_oct.min()), 20.0)  # Hz->rad/s; ties to lowest measured band

zeta0  = np.clip(zeta0,  eps, zeta_max)
alpha0 = np.clip(alpha0, eps, alpha_max)
beta0  = np.clip(beta0,  beta_min, beta_max)

Rinf0 = float(np.clip(np.real(R_target[-1]), -0.99, 0.99))
A0 = np.zeros(K)
B0 = np.zeros(L)

# x = [Rinf, A( K ), zeta( K ), B( L ), alpha( L ), beta( L )]
x0 = np.hstack([Rinf0, A0, zeta0, B0, alpha0, beta0])

lb = np.hstack([
    -1.0,                    # Rinf lower
    -np.inf * np.ones(K),    # A_k
    eps * np.ones(K),        # zeta_k >= eps
    -np.inf * np.ones(L),    # B_l
    eps * np.ones(L),        # alpha_l >= eps
    beta_min * np.ones(L)    # beta_l  >= beta_min
])

ub = np.hstack([
     1.0,                    # Rinf upper
     np.inf * np.ones(K),    # A_k
     zeta_max * np.ones(K),  # zeta_k <= zeta_max
     np.inf * np.ones(L),    # B_l
     alpha_max * np.ones(L), # alpha_l <= alpha_max
     beta_max * np.ones(L)   # beta_l  <= beta_max
])

# Make x0 always feasible (prevents "x0 infeasible" if tweaking bounds later)
x0 = np.maximum(x0, lb)
x0 = np.minimum(x0, ub)

# Passivity grid
f_pass = np.logspace(np.log10(f_fit.min()), np.log10(f_fit.max()), 512)
w_pass = 2 * np.pi * f_pass

# ----------------------------
# Stage 1: fit without passivity penalty
# ----------------------------
res1 = least_squares(
    resid, x0, bounds=(lb, ub),
    args=(K, L, w_fit, R_target, None, 0.0),
    method="trf",
    max_nfev=1200, ftol=1e-7, xtol=1e-7, gtol=1e-7,
    verbose=0
)

# ----------------------------
# Stage 2: add passivity penalty
# ----------------------------
res2 = least_squares(
    resid, res1.x, bounds=(lb, ub),
    args=(K, L, w_fit, R_target, w_pass),
    method="trf",
    max_nfev=1200, ftol=1e-8, xtol=1e-8, gtol=1e-8,
    verbose=0
)

Rinf, A, zeta, B, alpha, beta = unpack(res2.x, K, L)
C = -(alpha / beta) * B  # computed for reporting/output (Hornikx)

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
plt.semilogx(f_fit, np.abs(R_target), label="Target |R| (min-phase)", c='b')
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
