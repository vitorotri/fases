import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import hilbert

# ----------------------------
# Input: octave-band centers, |R| (0..1) and cuttof frequency
# ----------------------------
f_oct = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
Rmag_oct = np.array([0.95, 0.90, 0.80, 0.65, 0.50, 0.45, 0.55, 0.70], dtype=float)
f_cutoff = 1200.0 # Hz

# ----------------------------
# Effective cutoff masking
# ----------------------------
mask = f_oct <= f_cutoff
f_oct = f_oct[mask]
Rmag_oct = Rmag_oct[mask]

# ----------------------------
# Model order (keep small to start!)
# ----------------------------
K = 3  # real poles
L = 2  # complex pairs

# ----------------------------
# Minimum-phase complex target from |R| only
# ----------------------------
def minimum_phase_from_mag(f, mag, ngrid=220):
    f = np.asarray(f, float)
    mag = np.clip(np.asarray(mag, float), 1e-6, 1.0)
    flog = np.log(f)
    flog_u = np.linspace(flog.min(), flog.max(), ngrid)
    f_u = np.exp(flog_u)
    logmag_u = np.interp(flog_u, flog, np.log(mag))
    phase_u = -np.imag(hilbert(logmag_u))   # uniform in log-frequency
    return f_u, np.exp(logmag_u + 1j*phase_u)

f_fit, R_target = minimum_phase_from_mag(f_oct, Rmag_oct, ngrid=220)
w_fit = 2*np.pi*f_fit

# ----------------------------
# Hornikx–Wang frequency-domain model
# ----------------------------
def R_model(w, Rinf, A, zeta, B, C, alpha, beta):
    s = 1j*w
    R = Rinf + 0j
    for Ak, zk in zip(A, zeta):
        R += Ak / (zk + s)
    for Bl, Cl, al, bl in zip(B, C, alpha, beta):
        num = (Bl*s + Cl*bl + al*Bl)
        den = (al + s)**2 + bl**2
        R += num / den
    return R

def unpack(x, K, L):
    i=0
    Rinf = x[i]; i+=1
    A = x[i:i+K]; i+=K
    zeta = x[i:i+K]; i+=K
    B = x[i:i+L]; i+=L
    C = x[i:i+L]; i+=L
    alpha = x[i:i+L]; i+=L
    beta  = x[i:i+L]; i+=L
    return Rinf, A, zeta, B, C, alpha, beta

def resid(x, K, L, w, R_tgt, w_pass=None, lam_pass=0.0):
    Rinf, A, zeta, B, C, alpha, beta = unpack(x, K, L)
    Rhat = R_model(w, Rinf, A, zeta, B, C, alpha, beta)
    r = np.hstack([np.real(Rhat - R_tgt), np.imag(Rhat - R_tgt)])

    if w_pass is not None and lam_pass > 0:
        Rp = R_model(w_pass, Rinf, A, zeta, B, C, alpha, beta)
        viol = np.maximum(np.abs(Rp) - 1.0, 0.0)
        r = np.hstack([r, np.sqrt(lam_pass)*viol])
    return r

# ----------------------------
# Initial guess
# ----------------------------
wmin, wmax = w_fit.min(), w_fit.max()
zeta0  = np.logspace(np.log10(wmin), np.log10(wmax), K)
alpha0 = np.logspace(np.log10(wmin), np.log10(wmax), L)
beta0  = np.logspace(np.log10(wmin), np.log10(wmax), L)

Rinf0 = float(np.clip(np.real(R_target[-1]), -0.9, 0.9))
A0 = np.zeros(K)
B0 = np.zeros(L)
C0 = np.zeros(L)
x0 = np.hstack([Rinf0, A0, zeta0, B0, C0, alpha0, beta0])

# Bounds: stability/causality + |Rinf|<=1
eps = 1e-6
lb = np.hstack([-1.0, -np.inf*np.ones(K), eps*np.ones(K), # zeta_k ≥ eps
                -np.inf*np.ones(L), -np.inf*np.ones(L), eps*np.ones(L), eps*np.ones(L)])
ub = np.hstack([ 1.0,  np.inf*np.ones(K), np.inf*np.ones(K),
                 np.inf*np.ones(L),  np.inf*np.ones(L), np.inf*np.ones(L), np.inf*np.ones(L)]) # alpha_l ≥ eps, beta_l ≥ eps

# Passivity grid (coarse but useful)
f_pass = np.logspace(np.log10(f_fit.min()), np.log10(f_fit.max()), 450)
w_pass = 2*np.pi*f_pass

# ----------------------------
# Stage 1: fit without passivity penalty (fast)
# ----------------------------
res1 = least_squares(
    resid, x0, bounds=(lb, ub),
    args=(K, L, w_fit, R_target, None, 0.0),
    method="trf",
    max_nfev=800, ftol=1e-7, xtol=1e-7, gtol=1e-7,
    verbose=0
)

# ----------------------------
# Stage 2: add passivity penalty (still fast)
# ----------------------------
res2 = least_squares(
    resid, res1.x, bounds=(lb, ub),
    args=(K, L, w_fit, R_target, w_pass, 30.0),
    method="trf",
    max_nfev=1200, ftol=1e-7, xtol=1e-7, gtol=1e-7,
    verbose=0
)

Rinf, A, zeta, B, C, alpha, beta = unpack(res2.x, K, L)

#print(f"R_inf = {Rinf:.8g}")
print("# R_inf K L")
print(Rinf, K, L)
# do not mistake those k's with 3rd dimension indexes in 3D scheme
print("# A_k zeta_k")
for k in range(K):
    #print(f"A{k+1}={A[k]:.8g}, zeta{k+1}={zeta[k]:.8g}")
    print(A[k], zeta[k])
print("# B_l C_l alpha_l beta_l")
for l in range(L):
    #print(f"B{l+1}={B[l]:.8g}, C{l+1}={C[l]:.8g}, alpha{l+1}={alpha[l]:.8g}, beta{l+1}={beta[l]:.8g}")
    # do no mistake those alphas and betas with WENO coefficients
    print(B[l], C[l], alpha[l], beta[l])

# Plots
R_hat = R_model(w_fit, Rinf, A, zeta, B, C, alpha, beta)
plt.figure()
plt.semilogx(f_fit, np.abs(R_target), label="Target |R| (min-phase)",c='b')
plt.semilogx(f_fit, np.abs(R_hat), "--", label="Fitted |R|",c='r')
plt.ylim(0, 1.05); plt.grid(True, which="both")
plt.xlabel("Frequency [Hz]"); plt.ylabel("|R|"); plt.legend()

plt.figure()
plt.semilogx(f_pass, np.abs(R_model(w_pass, Rinf, A, zeta, B, C, alpha, beta)),color='lime')
plt.axhline(1.0, linewidth=1, c='k')
plt.ylim(0, 1.2); plt.grid(True, which="both")
plt.xlabel("Frequency [Hz]"); plt.ylabel("|R_fit| (passivity check)")
plt.show()
