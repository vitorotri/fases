import numpy as np
import matplotlib.pyplot as plt

'''
n_theta = 50
n_phi = 50
r = 3
theta = np.linspace(0, 2*np.pi, n_theta)
phi   = np.linspace(0, np.pi,  n_phi)

T, P = np.meshgrid(theta, phi)        # shape (n_phi, n_theta)

x = r*np.sin(T)*np.sin(P)
y = r*np.cos(T)*np.sin(P)
z = r*np.cos(P)
'''

N = 10000          # number of phonons
c = 343.0          # speed of sound (use your local c)

i  = np.arange(N) + 0.5
cz = 1 - 2*i/N                      # cos(phi), uniform in (-1, 1)
sz = np.sqrt(1 - cz**2)            # sin(phi) = projection onto xy-plane
theta = np.pi * (1 + 5**0.5) * i   # golden-angle azimuth

vx = c*sz*np.cos(theta)
vy = c*sz*np.sin(theta)
vz = c*cz                          # = c*cos(phi)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(vx.ravel(), vy.ravel(), vz.ravel(), c='r', marker='o')
plt.show()
