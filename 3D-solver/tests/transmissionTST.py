# Script to assess acoustics boundary conditions on domain according to chapter 5.6.1 of Kowalczyk. The different
# txt files have to be created under this framework.

import numpy as np
import math
import matplotlib.pyplot as plt

def dB_SPL(p):
	p0 = 0.00002
	return 20*np.log10(p/p0)

def compute_T_num(x1_s, x0_s):
	if np.abs(np.max(x1_s)) > np.abs(np.min(x1_s)):
		aux = np.max(x1_s) / np.max(x0_s)
	else:
		aux = np.min(x1_s) / np.max(x0_s)
	return aux

# no negative R's for this test?
# also, for very small values of R_theta, it is expected that R_num may not be that small, and a tolerance must
# be considered, so the difference in dB is not that huge

x0 = np.fromfile('IR0full.bin', dtype=np.float32) # in-phase signal with full reflection
x1 = np.fromfile('IR0.bin', dtype=np.float32) # signal with partial numerical reflection

# slice for computing second peak numerical R from second peak (see Kowalczyk equation 2.21)
# changes depending on angle of incidence
print("\nPlease check slice_start and IR**.bin file if necessary\n")

T_num = compute_T_num(x1, x0) # numerical T
T = float(input("Enter the transmission coefficient for the barrier at normal incidence: "))

ahead = 80 # number of samples undelayed for the transmitted signal (faster transmission in solids)
# also, in 2D, the free field signal has an attenuation proportional to sqrt(R + r/2), so the transmitted signal
# is corrected for the portion attenuated by the barrier's thickness

expected = T*x0 # expected peak slice
thickness = 0.09 # barrier's thickness
R = 0.96 # distance from source to barrier
k1 = 1/np.sqrt(R + thickness) # spatial decay from source to barrier + thickness
k2 = 1/np.sqrt(R) # spatial decay from source to barrier
k3 = np.abs(k1 - k2) # difference correction
portion = 1 + k3 # portion correction for inverse distance law according to 2D profile decay
expected *= portion
expected = np.roll(expected, -ahead)

print('T_num = ',T_num)
print()
print(f'dB diff: {np.abs(dB_SPL(max(x1)) - dB_SPL(max(expected))): .7f}')
print()

plt.figure(1)
plt.plot(x0, 'b', label='Fully transmitted signal')
plt.plot(x1, 'g', linestyle='--', label='Partially transmitted signal')
plt.xlabel('Samples (n)')
plt.ylabel('Sound Pressure (Pa)')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(2)
plt.plot(expected, 'b', label='Expected')
plt.plot(x1, 'r', linestyle='--', label='Numerical')
plt.xlabel('Samples (n)')
plt.ylabel('Sound Pressure (Pa)')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
