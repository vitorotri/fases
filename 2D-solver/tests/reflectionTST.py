# Script to assess acoustics boundary conditions on domain according to chapter 5.6.1 of Kowalczyk. The different
# txt files have to be created under this framework.

import numpy as np
import math
import matplotlib.pyplot as plt

def dB_SPL(p):
	p0 = 0.00002
	return 20*np.log10(p/p0)

def compute_R_num(x1_s, x0_s):
	if np.abs(np.max(x1_s)) > np.abs(np.min(x1_s)):
		aux = np.max(x1_s) / np.max(x0_s)
	else:
		aux = np.min(x1_s) / np.max(x0_s)
	return aux

# no negative R's for this test?
# also, for very small values of R_theta, it is expected that R_num may not be that small, and a tolerance must
# be considered, so the difference in dB is not that huge

x0 = np.fromfile('IR45full.bin', dtype=np.float32) # in-phase signal with full reflection
x1 = np.fromfile('IR45.bin', dtype=np.float32) # signal with partial numerical reflection

# slice for computing second peak numerical R from second peak (see Kowalczyk equation 2.21)
# changes depending on angle of incidence
print("\nPlease check slice_start and IR**.bin file if necessary\n")
slice_start = 2500 # 0 -> 1500, 20 -> 1500, 45 -> 2500, 70 -> 5750
x0_s = x0[slice_start:]
x1_s = x1[slice_start:]

R_num = compute_R_num(x1_s, x0_s) # numerical R
R = float(input("Enter the reflection coefficient for the wall at normal incidence: "))
theta = float(input("Enter the angle of reflection for the receiver: "))

rho = 1.225# density of air
R_spec = 287.05; # specific gas constant for air [J] [kg]^-1 [K]^-1
gamma = 1.4; # specific heat ratio for air
T = 288.15 # temperature [K]
c = math.sqrt(gamma*R_spec*T) # speed of sound [m]/[s]
Z0 = rho * c # impedance of air
cos_theta = np.cos(math.radians(theta))
Zb = Z0 * (1 + R) / (1 - R) # fixed wall impedance
zeta = Zb / Z0 # specific wall impedance
R_theta = (zeta * cos_theta - 1) / (zeta * cos_theta + 1)
print('R_theta = ', R_theta)

err = np.abs(R_theta - R_num)

# compute expected R from angle / impedance ? compute angles from lcoations as well etc

expected_s = R_theta*x0_s # expected peak slice
expected = np.concatenate(( x0[:slice_start] , expected_s )) # expected signal

print('R_num = ',R_num)
print()
print('Error = ', np.abs(R_theta - R_num))
print()
print(f'dB diff: {np.abs(dB_SPL(max(x1_s)) - dB_SPL(max(expected_s))): .7f}')
print()

plt.figure(1)
plt.plot(x0, 'b', label='Full reflection signal')
plt.plot(x1, 'g', linestyle='--', label='Partial reflection signal')
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
