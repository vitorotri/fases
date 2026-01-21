import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Load data
# --------------------
with open('header.txt', 'r') as file:
    header_line = file.readline().strip()
    values = header_line.split(' ')

Nx, Ny, h, tol = int(values[0]), int(values[1]), float(values[9]), int(values[10])

S = np.fromfile('S.bin', dtype=np.float32).reshape(Ny, Nx)
oid = np.fromfile('oid.bin', dtype=np.uint8).reshape(Ny, Nx)

x = np.arange(Nx) * h
y = np.arange(Ny) * h

# --------------------
# Compute normals from S (mimic C code with clamped neighbors)
# --------------------
# pad with edge values so that im1/ip1, jm1/jp1 are clamped at borders
Spad = np.pad(S, pad_width=1, mode='edge')  # shape (Ny+2, Nx+2)

# central differences, same as:
# gx = (S[i+1,j] - S[i-1,j]) * 0.5/h with clamping at borders
gx = (Spad[1:-1, 2:] - Spad[1:-1, 0:-2]) * (0.5 / h)
gy = (Spad[2:, 1:-1] - Spad[0:-2, 1:-1]) * (0.5 / h)

gn = np.sqrt(gx * gx + gy * gy)

eps = 1e-12
nx = np.zeros_like(S, dtype=np.float32)
ny = np.zeros_like(S, dtype=np.float32)

valid = gn > eps
nx[valid] = gx[valid] / gn[valid]
ny[valid] = gy[valid] / gn[valid]

# --------------------
# Mask: only near obstacles, in a narrow band of S
# --------------------
# In C you had (roughly): if (!(0 < |S| < h)) return; and "only immersed cells"
near_obstacle = (oid != 0xFF)             # obstacle cells
narrow_band   = (np.abs(S) < h) & (np.abs(S) > 0.0)
mask = near_obstacle & narrow_band & valid

X, Y = np.meshgrid(x, y)  # shape (Ny, Nx), consistent with S

# (optional) subsample to avoid too many arrows (you can comment this out)
# idx_i, idx_j = np.indices(S.shape)
# mask = mask & ((idx_i % 2 == 0) & (idx_j % 2 == 0))

# --------------------
# Plot S
# --------------------
plt.figure(figsize=(6, 5))
plt.title("Signed Distance Function S")
plt.xlabel("x [m]")
plt.ylabel("y [m]")

im = plt.imshow(
    S,
    extent=[x[0], x[-1], y[0], y[-1]],
    origin='lower',
    cmap='seismic',
    vmin=-h,
    vmax=h
)
plt.colorbar(im, label="S (m)")

cs = plt.contour(x, y, S, levels=[0.0], colors='k', linewidths=1.0)
plt.clabel(cs, fmt='S=0', colors='k')

plt.tight_layout()

# --------------------
# Plot oid
# --------------------
tol_m = tol * h   # if your tol is in "cells"; if it is already in meters, just tol_m = tol
band_mask = (S >= 0.0) & (np.abs(S) <= tol_m) # since parser do it on fluid side only

# Overlay the band cells as a transparent layer
plt.figure(figsize=(6, 5))
im = plt.pcolormesh(X, Y, oid, shading='nearest', cmap='Blues')
plt.colorbar(im, label="oid values")

# Band overlay: alpha = 0.3 so you see oid below
plt.pcolormesh(X, Y, band_mask, shading='nearest', alpha=0.5, cmap='Reds')

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title(f"oid with ±{tol} cells tolerance band")
plt.tight_layout()
plt.show()

# --------------------
# Plot normals as quiver near obstacles
# --------------------
plt.figure(figsize=(6, 5))
plt.title("Normals near obstacles")
plt.xlabel("x [m]")
plt.ylabel("y [m]")

# background: S field with zero level
im = plt.imshow(
    S,
    extent=[x[0], x[-1], y[0], y[-1]],
    origin='lower',
    cmap='seismic',
    vmin=-h,
    vmax=h
)
plt.colorbar(im, label="S (m)")

cs = plt.contour(x, y, S, levels=[0.0], colors='k', linewidths=1.0)
plt.clabel(cs, fmt='S=0', colors='k')

# quiver of normals only where mask is True
plt.quiver(
    X[mask],
    Y[mask],
    nx[mask],
    ny[mask],
    color='k',
    scale=30.0,   # adjust scale for arrow length as needed
    width=0.003
)

plt.tight_layout()
plt.show()

