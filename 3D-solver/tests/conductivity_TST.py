import matplotlib.pyplot as plt
import numpy as np

Nx = 200
h = 0.025

perc_x = 0.1
d_x_cells = perc_x * Nx
d_x = d_x_cells * h

R0 = 0.01
m = 4.0
c = 344.0

i = np.linspace(0, Nx-1, Nx)

# local normalized coordinates (branchless)
xm = (d_x_cells - i) / d_x_cells
xp = (i - (Nx - d_x_cells)) / d_x_cells

# clamp to [0,1]
xm = np.clip(xm, 0.0, 1.0)
xp = np.clip(xp, 0.0, 1.0)

# conductivity profiles
pref = -0.5 * c * (m + 1.0) * np.log(R0) / d_x

sx_minus = pref * (xm**m)   # left PML
sx_plus  = pref * (xp**m)   # right PML

sx = sx_minus + sx_plus

plt.plot(i, sx_minus, 'r', label='sx_minus (x-)')
plt.plot(i, sx_plus,  'b', label='sx_plus (x+)')
plt.plot(i, sx, 'k--', label='total')

plt.xlabel('i')
plt.ylabel(r'$\sigma_x(i)$')
plt.title('Conductivity profile in function of i')
plt.legend()
plt.show()

