# The simplest idea is to model a barrier where inside it the values are multiplied by an alpha < 1.0.
# We have below a similar idea with a barrier with some thickness where the values are absorder in time-domain
# by a multipole filter uwing ADE. Could be just a low-pass or a high-pass, then we would have a single line, not
# a for loop with M poles.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

Lx, Ly = 1.0, 1.0  # domain size [m]
c = 340  # speed of sound [m/s]
h = 0.005  # grid spacing [m]
t_max = 0.05  # maximum simulation time [s]

n = 0  # initial time-step
CFL = 1/np.sqrt(2.0)  # Courant-Friedrich-Lewy number for 2D leapfrog FDTD
dt = CFL*h/c  # time-step [s]
n_max = int(t_max/dt)  # number of samples

Nx, Ny = int(Lx/h), int(Ly/h)

p0 = np.zeros((Nx, Ny), dtype=np.float64)
p1 = np.zeros((Nx, Ny), dtype=np.float64)
p2 = np.zeros((Nx, Ny), dtype=np.float64)

p0[int(Nx/2), int(Ny/4)] = 1.0
p1[int(Nx/2), int(Ny/4)] = 1.0

my_cmap = mcolors.LinearSegmentedColormap.from_list('custom', ['crimson', 'springgreen', 'white', 'lawngreen', 'blueviolet'])
im = plt.imshow(p1, cmap=my_cmap, interpolation='bilinear', origin='lower', extent=[0, Lx, 0, Ly], vmin=-0.01, vmax=0.01)

x_start = int(0.10*Nx)
x_end = int(0.90*Nx)
y_start = int(0.50*Ny)
y_end = int(0.52*Ny)

M = 3  # number of poles from your fit, example
a = np.array([0.15, 0.5, 0.8])      # pole locations (|a_m|<1)
w = np.array([0.5, 0.3, 0.2])       # weights for each section
d = 0.0                             # direct term (if any)

# (1)
# parameter for simplistic barrier
alpha = 0.8

# (2)
# ADE states: one 2D field per pole
barrier_state = np.zeros((M, x_end - x_start, y_end - y_start))

while (n < n_max):
    # ... your FDTD update ...
    p2[1:-1,1:-1] = 0.5*(p1[2:,1:-1] + p1[:-2,1:-1] + p1[1:-1,2:] + p1[1:-1,:-2]) - p0[1:-1,1:-1]
	
    
    # --- ADE barrier ---
    incoming = p2[x_start:x_end, y_start:y_end]

    # update all ADE states
    for m in range(M):
        barrier_state[m] = a[m]*barrier_state[m] + (1.0 - a[m])*incoming

    # combine states to get transmitted field
    out = d*incoming
    for m in range(M):
        out += w[m]*barrier_state[m]
	
    p2[x_start:x_end, y_start:y_end] = out
    # -------------------
    
    
    # simple barrier
    #p2[x_start:x_end, y_start:y_end] *= alpha
    
    # absorbing BCs...
    p2[-1, 1:-1]   = p1[-2, 1:-1]
    p2[0, 1:-1]    = p1[1, 1:-1]
    p2[1:-1, 0]    = p1[1:-1, 1]
    p2[1:-1, -1]   = p1[1:-1, -2]

    p0, p1 = np.copy(p1), np.copy(p2)
    
    im.set_data(p1)
    plt.pause(0.001)
    n += 1

plt.show()

