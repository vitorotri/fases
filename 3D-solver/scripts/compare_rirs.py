# To properly compare IRs and validate ISM vs FVM, remove fractional delay from direct ISM part and simulate
# inside an anechoic booth measuring at 1m, applying the negative delay to FVM later if simulated with the PCS
# (PCS is naturally delayed). Also, keep theta's = 0.0 (look for "amplitude validation" in main3D.cu).

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

# Load impulse responses.

dtype = np.dtype("<f4")

v1 = np.fromfile("IR0_0.bin", dtype=dtype)
v2 = np.fromfile("HIR0_0_500.bin", dtype=dtype)

# Set the common sample rate [Hz].
fs = 44100
dt = 1.0 / fs
dt_half = 0.5*dt

if len(v1) != len(v2):
    v1 = resample(v1, len(v2))

f_in = 200 # Hz

f_x = next(c * 2**0.5 for c in (63, 125, 250, 500, 1000, 2000, 4000, 8000) if f_in <= c * 2**0.5)
delay = int(round(1.806 / f_x * fs))

v1 = np.roll(v1, -delay)

N = len(v2)

# Time axis
t = np.arange(N) / fs

# Time-domain comparison
plt.figure(figsize=(10, 4))
plt.plot(t, v1, "b", lw=1, label="IR0_0 (resampled)")
plt.plot(t, v2, "r", lw=1, alpha=0.8, label="HIR0_0_250")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
