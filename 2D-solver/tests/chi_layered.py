import numpy as np
import matplotlib.pyplot as plt

def chi_layered(s, tol, h):
    s = np.asarray(s)

    # Start with all zeros
    chi = np.zeros_like(s, dtype=float)

    # Region s <= h  -> 1
    mask1 = s <= h
    chi[mask1] = 1.0

    # Region h < s < tol  -> smooth transition
    mask2 = (s > h) & (s < tol)
    r = (s[mask2] - h) / (tol - h)
    smooth = r * r * (3 - 2 * r)
    chi[mask2] = 1 - smooth

    return chi

h = 0.005
tol = 3 * h
s = np.linspace(0, 5 * h, 100)

plt.plot(s, chi_layered(s, tol, h), 'b')
plt.show()

