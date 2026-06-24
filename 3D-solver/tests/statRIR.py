# Statistical Impulse Response, as shown in the 2021 lecture notes of ETHZ's Acoustics I, by Kurt Heutschi, as well
# in Svensson (1998).
# Similar technique is found on FRA-RIR (Tencent AI Lab).

import numpy as np
import matplotlib.pyplot as plt

def step(x):
	return np.where(x >= 1, 1, 0)

r = 2.3 # distance between real source and receiver [m]
fc = 20000.0 # sampling frequency [Hz]
c = 340.3 # speed of sound [m/s]
N = 50000 # number of samples of the RIR
V = 30 # room's volume [m^3]
T = 1.2 # room's reverberation time [s]

i = np.arange(0,N,1)
eps = np.random.normal(loc=0.0, scale=1.0, size=i.size)
dell = 3*np.log(10)
aux = np.sqrt(4*np.pi*c/(V*fc)) * np.exp(-i * dell / fc) * eps
val = np.insert(aux, 0, 1/r)

plt.plot(val,'b')
plt.show()
