# Script to assess TDIBC (Time-Domain Impedance Boundary Conditions) for acoustics using the associated ADE (Auxiliary Differential Equations) - normal incidence test only.

import numpy as np
import math
import matplotlib.pyplot as plt
import os
from scipy.special import hankel1
from scipy.signal.windows import tukey

def dB_SPL(p):
	p0 = 0.00002
	return 20*np.log10(p/p0)

def R_model(w, Rinf, A, zeta, B, C, alpha, beta):
    s = 1j*w
    R = Rinf + 0j
    #eps = 1e-7
    for Ak, zk in zip(A, zeta):
        R += Ak / (zk + s)
    for Bl, Cl, al, bl in zip(B, C, alpha, beta):
        num = (Bl*s + Cl*bl + al*Bl)
        den = (al + s)**2 + bl**2 # + eps
        R += num / den
    return R

def load_R(filepath):

    section = None
    R_inf = None
    A, zeta = [], []
    B, C, alpha, beta = [], [], [], []

    with open(filepath, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("#"):
                key = line.lower()
                if "r_inf" in key:
                    section = "RINF"
                elif "a_k" in key and "zeta_k" in key:
                    section = "REAL"
                elif "b_l" in key and "c_l" in key and "alpha_l" in key and "beta_l" in key:
                    section = "CMPLX"
                else:
                    section = None
                continue

            parts = line.split()

            if section == "RINF":
                R_inf = float(parts[0])

            elif section == "REAL":
                if len(parts) < 2:
                    raise ValueError(f"Bad REAL pole line: {line}")
                A.append(float(parts[0]))
                zeta.append(float(parts[1]))

            elif section == "CMPLX":
                if len(parts) < 4:
                    raise ValueError(f"Bad CMPLX pole line: {line}")
                B.append(float(parts[0]))
                C.append(float(parts[1]))
                alpha.append(float(parts[2]))
                beta.append(float(parts[3]))

    if R_inf is None:
        raise ValueError("R_inf not found in coefficients file.")

    return (R_inf,
            np.array(A, dtype=float), np.array(zeta, dtype=float),
            np.array(B, dtype=float), np.array(C, dtype=float),
            np.array(alpha, dtype=float), np.array(beta, dtype=float))

# 2D outgoing Helmholtz Green's function
def G2D(k, r):
    return 0.25j * hankel1(0, k*r)

def next_pow2(n):
    return 1 << (int(n - 1).bit_length())

def extract_window(x, center, half_width, alpha=0.20):
    i0 = max(0, center - half_width)
    i1 = min(len(x), center + half_width)
    seg = np.array(x[i0:i1], dtype=float)
    if len(seg) < 16:
        raise RuntimeError("Gate too small or too close to boundary.")
    win = tukey(len(seg), alpha) if alpha > 0 else np.ones(len(seg))
    return seg * win

def lowpass_mask_rfft(freq, fc, frac=0.90):
    """
    Raised-cosine lowpass mask for rfft frequencies.
    frac=0.90 means roll-off starts at 0.9*fc and reaches 0 at fc.
    """
    fp = frac * fc
    fs = fc

    M = np.ones_like(freq, dtype=float)

    # stopband
    M[freq >= fs] = 0.0

    # transition band
    trans = (freq > fp) & (freq < fs)
    # map freq in (fp, fs) to theta in (0, pi)
    theta = np.pi * (freq[trans] - fp) / (fs - fp)
    M[trans] = 0.5 * (1.0 + np.cos(theta))   # 1 -> 0 smoothly

    return M

def lowpass_filter_fft(x, dt, fc, Nfft=None, frac=0.90):
    """
    Zero-phase lowpass by FFT masking. Returns filtered time signal (length = len(x)).
    """
    x = np.asarray(x)
    N = len(x)
    if Nfft is None:
        Nfft = 1 << int(np.ceil(np.log2(N)))  # next pow2

    X = np.fft.rfft(x, n=Nfft)
    freq = np.fft.rfftfreq(Nfft, d=dt)

    M = lowpass_mask_rfft(freq, fc, frac=frac)
    Xf = X * M

    xf = np.fft.irfft(Xf, n=Nfft)[:N]
    return xf

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "../..", "config", "config.txt")

v = np.loadtxt(config_path)

h = float(v[0])
C_CFL = float(v[4])
f_cutoff = v[2] #balance with N_poles for stability and better distribution of poles
Temp = v[3] + 273.15
R_spec = 287.05
gamma = 1.4

omega_cutoff = 2 * np.pi * f_cutoff

c = np.sqrt(gamma * R_spec * Temp)         # speed of sound (m/s)
rho0 = 1.225                                # air density
z0 = c * rho0                             # characteristic impedance of air

dt = C_CFL * h / c
print(C_CFL)
print(h)
print(c)
print(dt)
fs = 1/dt # temporal sampling frequency [Hz]

script_dir = os.path.dirname(os.path.abspath(__file__)) # tests/
print("\nRename [outputs].bin if necessary so it can be used right below this line...\n")
data_path_noise = os.path.join(script_dir, "IRnoise.bin") # noise signal
data_path_0 = os.path.join(script_dir, "IR0.bin") # full reflection signal
data_path_1 = os.path.join(script_dir, "IR1.bin") # absorbed signal

p_d = np.fromfile(data_path_noise, dtype=np.float32) # direct pulse
p_f = np.fromfile(data_path_0, dtype=np.float32) # in-phase signal with full reflection
p_r = np.fromfile(data_path_1, dtype=np.float32) # signal with partial reflection

'''
# filtering (?)
p_d = lowpass_filter_fft(np.copy(p_d), dt, f_cutoff, frac=0.90)
p_r = lowpass_filter_fft(np.copy(p_r), dt, f_cutoff, frac=0.90)
p_f = lowpass_filter_fft(np.copy(p_f), dt, f_cutoff, frac=0.90)
'''

p_f = np.copy(p_f) - np.copy(p_d)
p_r = np.copy(p_r) - np.copy(p_d)

#p_d = np.roll(p_d,1862-603)

'''
# gating ?
g_start = 1000
g_end = 3500
p_f = np.copy(p_f[g_start:g_end])
p_r = np.copy(p_r[g_start:g_end])
p_d = np.copy(p_d[g_start:g_end])
'''
'''
# remove DC offsets ? Better to remove with PCS in simulation
p_f -= np.mean(p_f)
p_r -= np.mean(p_r)
'''

N = len(p_r)

# one-sided FFT

Pr = np.fft.rfft(p_r)
Pf = np.fft.rfft(p_f)
Pd = np.fft.rfft(p_d)

freq = np.fft.rfftfreq(N, d=dt)

# masking for spectrrum plot

mask = (
    (freq > 0.0) &
    (freq <= f_cutoff) &
    (np.abs(Pr) > 1e-6*np.max(np.abs(Pr))) # for better division
)


# Loads expected reflection coefficients
script_dir = os.path.dirname(os.path.abspath(__file__)) # tests/
coef_path = os.path.join(script_dir, "../..", "input", "fit", "Material1.fit")

R_inf, A, zeta, B, Cc, alpha, beta = load_R(coef_path)

# angular frequencies
w = 2*np.pi*freq  # freq in Hz

# compute analytical R
R_ana = R_model(w, R_inf, A, zeta, B, Cc, alpha, beta)

# angular frequencies
w = 2*np.pi*freq  # freq in Hz

# wavenumbers
k = w / c

R1 = 1.0 # d source -> receiver [m]
R2 = 3.0 # d receiver -> image-source [m]

eps = 1e-12
R_num = Pr / (Pf + eps)
#R_num = Pr / (Pd + eps)
#R_num = (Pr * np.exp(1j * k * R1)) / (Pd * np.exp(1j * k * R2) + eps)
Rmag_num = np.abs(R_num)

# compute Fourier of p_expected
P_expected = R_ana * Pf

# Bring back to time domain (match original length N)
p_expected = np.fft.irfft(P_expected, n=N)

# single period for given frequency f is just T = 1/f
# in order to properly capture a given frequency in the FFT analysis, it is necessary to simulate at least
# some cycles of that period (minimum 3, ideally 5 ~ 10 cycles)

'''
T_gate = (slice_end - slice_start) * dt
df = 1.0 / T_gate
print("Gate length T_gate =", T_gate, "s")
print("Frequency resolution df =", df, "Hz")
'''
err = 20*np.log10(np.abs(R_ana)/np.abs(R_num))

plt.figure(1)
plt.plot(p_d, 'b', label='Direct signal')
plt.plot(p_f, 'r', linestyle='--', label='Full reflection')
plt.plot(p_r, color='lime', linestyle='--', label='Absorbed reflection')
plt.xlabel('Samples (n)')
plt.ylabel('Sound Pressure (Pa)')
plt.grid(True)
plt.legend()
plt.title("Impulse Response")

plt.figure(2)
plt.semilogx(freq[mask], Rmag_num[mask], 'r', label='|R_num(f)|')
plt.semilogx(freq[mask], np.abs(R_ana)[mask], 'k--', label='|R_ana(f)|')
plt.xlabel("Frequency [Hz]")
plt.ylabel("|R(f)|")
plt.ylim([0, 1])
plt.grid(True, which="both")
plt.legend()
plt.title("Reflection magnitude: numerical vs analytical")

plt.figure(3)
plt.semilogx(freq[mask], err[mask], 'r', label='Error(f)')
plt.xlabel("Frequency [Hz]")
plt.ylabel("Error(f) [dB]")
plt.axhline(y=0, linestyle='--', color='black')
plt.grid(True, which="both")
plt.legend()
plt.title("Error |R_ana / R_num| in dB")

plt.figure(4)
plt.semilogx(freq, 20*np.log10(np.abs(Pd) + 1e-30), 'k')
plt.xlabel("Frequency [Hz]")
plt.ylabel("|X0(f)| [dB]")
plt.axvline(x=f_cutoff, linestyle='--', color='red')
plt.grid(True, which="both")
plt.title(r"Sanity test: is spectrum 'flat'below f_cutoff?")

plt.figure(5)
plt.semilogx(freq[mask], np.real(R_num[mask]), 'r', label='re(R_num(f))')
plt.semilogx(freq[mask], np.real(R_ana[mask]), 'k--', label='re(R_ana(f))')
plt.xlabel("Frequency [Hz]")
plt.ylabel("R(f)")
plt.grid(True, which="both")
plt.legend()

plt.figure(6)
plt.semilogx(freq[mask], np.imag(R_num[mask]), 'r', label='im(R_num(f))')
plt.semilogx(freq[mask], np.imag(R_ana[mask]), 'k--', label='im(R_ana(f))')
plt.xlabel("Frequency [Hz]")
plt.ylabel("R(f)")
plt.grid(True, which="both")
plt.legend()

plt.figure(7)
plt.plot(p_expected, 'k', label='Predicted reflection (p_expected)')
plt.plot(p_r, color='r', linestyle='--', label='Measured absorbed reflection (p_r)')
plt.xlabel('Samples (n)')
plt.ylabel('Sound Pressure (Pa)')
plt.grid(True)
plt.legend()
plt.title("Time-domain check: measured vs predicted reflected waveform")

phi = np.unwrap(np.angle(R_num[mask]))
phi_ana = np.unwrap(np.angle(R_ana[mask]))
dphi = phi - phi_ana

plt.figure(8)
plt.semilogx(freq[mask], dphi)
plt.grid(True, which="both")
plt.title("Phase difference: angle(R_num) - angle(R_ana)")
plt.xlabel("Hz")
plt.ylabel("rad")

plt.show()

