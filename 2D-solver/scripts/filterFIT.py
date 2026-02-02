import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import hilbert
import os

# ----------------------------
# Minimum-phase complex target from |R| only
# ----------------------------
'''
def minimum_phase_from_mag(f, mag, ngrid=220):
    f = np.asarray(f, float)
    mag = np.clip(np.asarray(mag, float), 1e-6, 1.0)
    flog = np.log(f)
    flog_u = np.linspace(flog.min(), flog.max(), ngrid)
    f_u = np.exp(flog_u)
    logmag_u = np.interp(flog_u, flog, np.log(mag))
    phase_u = -np.imag(hilbert(logmag_u))   # uniform in log-frequency
    return f_u, np.exp(logmag_u + 1j*phase_u)
'''

def minimum_phase_from_mag(f_meas, mag_meas,
                           f_low=5.0, f_high=20000.0,
                           ngrid=512, pad_frac=0.5,
                           mag0=1.0):
    f_meas = np.asarray(f_meas, float)
    mag_meas = np.clip(np.asarray(mag_meas, float), 1e-6, 1.0)

    # --- build extended anchors (low -> meas -> high) ---
    f_anchor = np.r_[f_low, f_meas, f_high]
    mag_anchor = np.r_[mag0,  mag_meas, mag_meas[-1]]  # flat high extension

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

    Rmin = np.exp(logmag_u + 1j*phase_u)
    return f_u, Rmin

# ----------------------------
# Hornikx–Wang frequency-domain model
# ----------------------------
def R_model(w, Rinf, A, zeta, B, C, alpha, beta):
    s = 1j*w  # (N,)
    R = Rinf + 0j

    # real poles: sum_k A_k / (zeta_k + s)
    if A.size:
        R += (A[:, None] / (zeta[:, None] + s[None, :])).sum(axis=0)

    # complex pairs:
    if B.size:
        s2 = s[None, :]
        al = alpha[:, None]
        bl = beta[:, None]
        num = B[:, None]*s2 + C[:, None]*bl + al*B[:, None]
        den = (al + s2)**2 + bl**2
        R += (num/den).sum(axis=0)

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
# Input: octave-band centers, |R| (0..1) from file and cuttof frequency
# ----------------------------
f_oct = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)

# user inputs

read_file = "Material1.mat"
K = 3  # real poles
L = 3  # complex pairs

# read from file?
h = 0.005 # grid spacing [m]
C_CFL = 0.3 # Courant number: pick values < 0.5 (not equal to)
R_spec = 287.05 # specific gas constant for air [J] [kg]^-1 [K]^-1
gamma = 1.4 # specific heat ratio for air
rho = 1.225 # average sea level density for air [kg/m3]. For dry air, proportional to p0/R_spec*T
Temp = 288.15 # temperature [K]
f_cutoff = 1000.0 # [Hz] Should be the same for each material (simulation-dependent), should read from conf file
f_tol = 200 # tolerance fequency [Hz], should be the same for each material (simulation-dependent)

f_cutoff += f_tol
c = np.sqrt(gamma*R_spec*Temp) # speed of sound [m/s]
dt = C_CFL*h/c # time-step size [s]

script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
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
w_fit = 2*np.pi*f_fit

# ----------------------------
# Initial guess
# ----------------------------
eps = 1e-6
wmin, wmax = w_fit.min(), w_fit.max()
zeta0  = np.logspace(np.log10(wmin), np.log10(wmax), K)
alpha0 = np.logspace(np.log10(wmin), np.log10(wmax), L)
beta0  = np.logspace(np.log10(wmin), np.log10(wmax), L)

# Bounds: stability/causality + |Rinf|<=1
zeta_max  = 0.5 / dt
alpha_max = 0.5 / dt
beta_max  = 0.5 / dt

zeta0  = np.clip(zeta0,  eps, zeta_max)
alpha0 = np.clip(alpha0, eps, alpha_max)
beta0  = np.clip(beta0,  eps, beta_max)


Rinf0 = float(np.clip(np.real(R_target[-1]), -0.99, 0.99))
A0 = np.zeros(K)
B0 = np.zeros(L)
C0 = np.zeros(L)
x0 = np.hstack([Rinf0, A0, zeta0, B0, C0, alpha0, beta0])

lb = np.hstack([
    -1.0,                    # Rinf lower
    -np.inf*np.ones(K),      # A_k
    eps*np.ones(K),          # zeta_k >= eps
    -np.inf*np.ones(L),      # B_l
    -np.inf*np.ones(L),      # C_l
    eps*np.ones(L),          # alpha_l >= eps
    eps*np.ones(L)           # beta_l  >= eps
])

ub = np.hstack([
     1.0,                    # Rinf upper
     np.inf*np.ones(K),      # A_k
     zeta_max*np.ones(K),    # zeta_k <= zeta_max
     np.inf*np.ones(L),      # B_l
     np.inf*np.ones(L),      # C_l
     alpha_max*np.ones(L),   # alpha_l <= alpha_max
     beta_max*np.ones(L)     # beta_l  <= beta_max
])

# Passivity grid (coarse but useful)
f_pass = np.logspace(np.log10(f_fit.min()), np.log10(f_fit.max()), 500) # from ~ 45 Hz to f_cut+tol
w_pass = 2*np.pi*f_pass

# ----------------------------
# Stage 1: fit without passivity penalty (fast)
# ----------------------------
res1 = least_squares(
    resid, x0, bounds=(lb, ub),
    args=(K, L, w_fit, R_target, None, 0.0),
    method="trf",
    max_nfev=1200, ftol=1e-7, xtol=1e-7, gtol=1e-7,
    verbose=0
)

# ----------------------------
# Stage 2: add passivity penalty (still fast)
# ----------------------------
res2 = least_squares(
    resid, res1.x, bounds=(lb, ub),
    args=(K, L, w_fit, R_target, w_pass, 1000.0),
    method="trf",
    max_nfev=1200, ftol=1e-8, xtol=1e-8, gtol=1e-8,
    verbose=0
)

Rinf, A, zeta, B, C, alpha, beta = unpack(res2.x, K, L)

# save to file

save_file = read_file[:-4] + ".fit"
coefs_path = os.path.join(script_dir, "..", "input", "fit", save_file)
np.savetxt(poles_path, np.array([[K, L]]), fmt="%d")

'''
#print(f"R_inf = {Rinf:.8g}")
print("# R_inf")
print(Rinf)
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
'''
  
with open(coefs_path, 'w') as f:
    f.write("# R_inf\n")
    f.write(f"{Rinf}\n")  # Matches print(Rinf)
    
    f.write("# A_k zeta_k\n")
    for k in range(K):
        f.write(f"{A[k]} {zeta[k]}\n")  # Space-separated, one per line
        
    f.write("# B_l C_l alpha_l beta_l\n")
    for l in range(L):
        f.write(f"{B[l]} {C[l]} {alpha[l]} {beta[l]}\n")  # 4 values per line
    if L == 0:
        f.write("0.0 0.0 0.0 0.0\n")


R_hat = R_model(w_fit, Rinf, A, zeta, B, C, alpha, beta)

# Plots
plt.figure()
plt.semilogx(f_fit, np.abs(R_target), label="Target |R| (min-phase)",c='b')
plt.semilogx(f_fit, np.abs(R_hat), "--", label="Fitted |R|",c='r')
plt.ylim(0, 1.05); plt.grid(True, which="both")
plt.xlim(0, f_cutoff-f_tol)
plt.xlabel("Frequency [Hz]"); plt.ylabel("|R|"); plt.legend()

plt.figure()
plt.semilogx(f_pass, np.abs(R_model(w_pass, Rinf, A, zeta, B, C, alpha, beta)),color='lime')
plt.axhline(1.0, linewidth=1, c='k')
plt.ylim(0, 1.2); plt.grid(True, which="both")
plt.xlabel("Frequency [Hz]"); plt.ylabel("|R_fit| (passivity check)")
plt.show()
