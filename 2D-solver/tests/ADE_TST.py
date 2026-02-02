# Script to assess TDIBC (Time-Domain Impedance Boundary Conditions) for acoustics using the associated ADE (Auxiliary Differential Equations) - normal incidence test only.

import numpy as np
import math
import matplotlib.pyplot as plt
import os

def dB_SPL(p):
	p0 = 0.00002
	return 20*np.log10(p/p0)

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


print("\nRename [outputs].bin if necessary so it can be used right below this line...\n")

# simulation parameters (should automate this with config file logic)
f_cutoff = 1000
C_CFL = 0.3
h = 0.005
t_max = 0.03

R_spec = 287.05
gamma = 1.4
rho = 1.225
Temp = 288.15

c = np.sqrt(gamma*R_spec*Temp) # speed of sound [m/s]
dt = C_CFL*h/c # time-step size [s]
fs = 1/dt # temporal sampling frequency [Hz]

script_dir = os.path.dirname(os.path.abspath(__file__)) # tests/
data_path_0 = os.path.join(script_dir, "..", "output", "IR0.bin") # full signal
data_path_1 = os.path.join(script_dir, "..", "output", "IR1.bin") # absorbed signal

x0 = np.fromfile(data_path_0, dtype=np.float32) # in-phase signal with full reflection
x1 = np.fromfile(data_path_1, dtype=np.float32) # signal with partial reflections

# slice for computing second peak
print("\nPlease check slice values if necessary\n")

slice_start = 1500
slice_end = 7115

x0_s = np.copy(x0[slice_start:slice_end])
x1_s = np.copy(x1[slice_start:slice_end])

# remove DC offsets ? Better to remove with PCS in simulation
#x0_s -= np.mean(x0_s)
#x1_s -= np.mean(x1_s)

N = len(x0_s)

# one-sided FFT
#n_pad = 50000 # zero-padding via FFT
X0 = np.fft.rfft(x0_s)
X1 = np.fft.rfft(x1_s)
freq = np.fft.rfftfreq(N, dt)

# Power spectrums
power0 = np.abs(X0)**2
power1 = np.abs(X1)**2

# masking for spectrrum plot
mask = (
    (freq > 0.0) &
    (freq <= f_cutoff) &
    (np.abs(X0) > 1e-6*np.max(np.abs(X0)))
)


# Loads expected reflection coefficients
script_dir = os.path.dirname(os.path.abspath(__file__)) # tests/
coef_path = os.path.join(script_dir, "..", "input", "fit", "Material1.fit")

R_inf, A, zeta, B, Cc, alpha, beta = load_R(coef_path)

# angular frequencies
w = 2*np.pi*freq  # freq in Hz

# compute analytical R
R_ana = R_model(w, R_inf, A, zeta, B, Cc, alpha, beta)

# compute numerical reflection
eps = 1e-6
R_num = X1 / (X0 + eps)
Rmag_num = np.abs(R_num)

# single period for given frequency f is just T = 1/f
# in order to properly capture a given frequency in the FFT analysis, it is necessary to simulate at least
# some cycles of that period (minimum 3, ideally 5 ~ 10 cycles)
T_gate = (slice_end - slice_start) * dt
df = 1.0 / T_gate
print("Gate length T_gate =", T_gate, "s")
print("Frequency resolution df =", df, "Hz")

err = 20*np.log10(np.abs(R_ana/R_num))

plt.figure(1)
plt.plot(x0, 'b', label='Full reflection')
plt.plot(x1, 'r', linestyle='--', label='Partial reflection')
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
plt.semilogx(freq, 20*np.log10(np.abs(X0) + 1e-30), 'k')
plt.xlabel("Frequency [Hz]")
plt.ylabel("|X0(f)| [dB]")
plt.axvline(x=f_cutoff, linestyle='--', color='red')
plt.grid(True, which="both")
plt.title(r"Sanity test: is spectrum 'flat'below f_cutoff?")

plt.show()
