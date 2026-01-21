import numpy as np
from scipy.optimize import curve_fit

# ------------------------------------------------------------
# 1) Your target data: frequencies and ISO 9613 attenuation
# ------------------------------------------------------------
# f_hz: 1D numpy array of frequencies (Hz), e.g. log-spaced
# alpha_iso_db: 1D numpy array of alpha(f) from ISO 9613-1 in dB/m

# Example dummy grid (replace with your real f and alpha):
f_hz = np.logspace(np.log10(50.0), np.log10(5000.0), 80)  # 50 Hz ... 5 kHz
alpha_iso_db = 0.0005 * (f_hz / 1000.0)**1.7  # <-- REPLACE with real ISO 9613

# Convert dB/m (for pressure level) -> Np/m (amplitude)
# ISO 9613 alpha is usually in dB per metre for SPL (20*log10).
alpha_iso_np = alpha_iso_db * np.log(10.0) / 20.0

# ------------------------------------------------------------
# 2) Multiple-relaxation model: alpha_model(f; tau_k, A_k)
# ------------------------------------------------------------
def alpha_model_np(f_hz, *theta):
    """
    Multiple-relaxation attenuation model:
        alpha(f) = sum_k A_k * (omega^2 * tau_k) / (1 + (omega * tau_k)^2)

    theta = [log10(tau_1), ..., log10(tau_M),
             log10(A_1),   ..., log10(A_M)]
    """
    f_hz = np.asarray(f_hz)
    omega = 2.0 * np.pi * f_hz

    M = len(theta) // 2
    log10_tau = np.array(theta[:M])
    log10_A   = np.array(theta[M:])

    tau = 10.0**log10_tau  # enforce tau > 0
    A   = 10.0**log10_A    # enforce A > 0

    alpha = np.zeros_like(f_hz, dtype=float)
    for k in range(M):
        wt = omega * tau[k]
        alpha += A[k] * (omega**2 * tau[k]) / (1.0 + wt**2)

    return alpha

# ------------------------------------------------------------
# 3) Fit routine
# ------------------------------------------------------------
def fit_relaxation_to_alpha(f_hz, alpha_target_np, M=3):
    """
    Fit M relaxation processes to target alpha(f) in Np/m.

    Returns:
        tau (s), A (dimensionless) arrays of length M,
        and the fitted alpha(f).
    """
    f_hz = np.asarray(f_hz)
    alpha_target_np = np.asarray(alpha_target_np)

    # Initial guesses: tau spread over time scales of interest,
    # A small positive starting values.
    # You may tweak these ranges depending on your band.
    fmin, fmax = f_hz.min(), f_hz.max()
    tmin = 1.0 / (2.0 * np.pi * fmax)  # shortest period ~ highest freq
    tmax = 1.0 / (2.0 * np.pi * fmin)  # longest period ~ lowest freq

    init_log10_tau = np.linspace(np.log10(tmin), np.log10(tmax), M)
    init_log10_A   = np.log10(alpha_target_np.max() / M + 1e-12) * np.ones(M)

    theta0 = np.concatenate([init_log10_tau, init_log10_A])

    # Fit
    popt, pcov = curve_fit(
        alpha_model_np,
        f_hz,
        alpha_target_np,
        p0=theta0,
        maxfev=20000
    )

    # Decode parameters
    log10_tau_fit = popt[:M]
    log10_A_fit   = popt[M:]

    tau = 10.0**log10_tau_fit
    A   = 10.0**log10_A_fit

    alpha_fit = alpha_model_np(f_hz, *popt)

    # Compute simple relative RMS error
    rel_rms = np.sqrt(np.mean(((alpha_fit - alpha_target_np) / alpha_target_np)**2))

    print("=== Fit results ===")
    print(f"M (number of relaxations) = {M}")
    for k in range(M):
        print(f"k={k}: tau = {tau[k]:.3e} s,  A = {A[k]:.3e}")
    print(f"Relative RMS error: {rel_rms:.3e}")

    return tau, A, alpha_fit

# ------------------------------------------------------------
# 4) Run the fit
# ------------------------------------------------------------
M = 3  # or 2–4; usually enough
tau, A, alpha_fit = fit_relaxation_to_alpha(f_hz, alpha_iso_np, M=M)

# Now tau and A are your fitted relaxation parameters.
# You can plot to visually check:
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.loglog(f_hz, alpha_iso_np, label="target ISO (Np/m)")
    plt.loglog(f_hz, alpha_fit, "--", label="M-relaxation fit (Np/m)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Attenuation α [Np/m]")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()
