import matplotlib.pyplot as plt
import numpy as np

Nx = 200
d_x = 0.1*200
R0 = 0.01
m = 4.0
c = 344.0

i = np.linspace(0, Nx-1, Nx)

#sx = -0.5 * ((i / d_x)**m) * c * (m + 1.0) * np.log(R0) / d_x
sx = -0.5 * (((Nx - 1 - i) / d_x)**m) * c * (m + 1.0) * np.log(R0) / d_x # mirrored

plt.plot(i,sx,'b')
plt.xlabel('i')
plt.ylabel(r'$\sigma_x(i)$')
plt.title('Conductivity profile in function of i')
plt.show()
