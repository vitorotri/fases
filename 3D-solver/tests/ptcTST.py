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

N = 2000
r = 3
# Fibonacci sphere — uniform by construction, arbitrary N
i = np.arange(N) + 0.5
z = 1 - 2*i/N                          # cos(phi) uniform in [-1, 1]
phi = np.arccos(z)
theta = np.pi * (1 + 5**0.5) * i       # golden-angle azimuth

x = r*np.sin(phi)*np.cos(theta)
y = r*np.sin(phi)*np.sin(theta)
z = r*z

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(x.ravel(), y.ravel(), z.ravel(), c='r', marker='o')
plt.show()
