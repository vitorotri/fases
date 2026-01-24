import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Load data
# --------------------
with open('../output/pre_header.txt', 'r') as file:
    header_line = file.readline().strip()
    values = header_line.split(' ')

Nx, Ny, h, tol = int(values[0]), int(values[1]), float(values[2]), int(values[3])

S = np.fromfile('../output/S.bin', dtype=np.float32).reshape(Ny, Nx)
oid = np.fromfile('../output/oid.bin', dtype=np.uint8).reshape(Ny, Nx)

x = np.arange(Nx) * h
y = np.arange(Ny) * h

# --------------------
# Compute normals from S (mimic C code with clamped nwweighbors)
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

# Print oid stats in narrow band near obstacles
near_obstacle = (oid != 0xFF)
narrow_band = (np.abs(S) < h) & (np.abs(S) > 0.0)
mask = near_obstacle & narrow_band

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
# --------------------
# Plot oid with better colormap and transparency
# --------------------
tol_m = tol * h
band_mask = (S >= 0.0) & (np.abs(S) <= tol_m)

plt.figure(figsize=(8, 6))

# Mask out 255 to make background clean
oid_display = np.where(oid == 255, np.nan, oid)
im = plt.pcolormesh(X, Y, oid_display, shading='nearest', cmap='viridis', 
                    vmin=0, vmax=10, alpha=0.9)
plt.colorbar(im, label="oid values (255 transparent)")

# Red band overlay only where oid != 255
band_display = band_mask & (oid != 255)
plt.pcolormesh(X, Y, band_display.astype(float), shading='nearest', 
               alpha=0.4, cmap='Reds')

# Add text labels for oid values
for i in range(Ny):
    for j in range(Nx):
        if mask[i,j] and oid[i,j] != 255:
            plt.text(X[i,j], Y[i,j], f'{oid[i,j]}', 
                    ha='center', va='center', fontsize=8, color='white',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.7))

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title(f"oid objects with numeric labels & ±{tol} cells band")
plt.tight_layout()
plt.show()
