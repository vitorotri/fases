# receives an input XYZ file and outputs an equivalent voxelized file

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

v = np.loadtxt(sys.argv[1])
h = float(sys.argv[2])
scale = 1

# Vectorized calculations done once for entire arrays
x1 = np.floor(v[:,0]/(scale*h))
x2 = np.ceil(v[:,0]/(scale*h))
y1 = np.floor(v[:,1]/(scale*h))
y2 = np.ceil(v[:,1]/(scale*h))
z1 = np.floor(v[:,2]/(scale*h))
z2 = np.ceil(v[:,2]/(scale*h))

# Stack coordinates and find unique pairs
S = np.stack((x1, y1), axis=1)  # 2D
# S = np.stack((x1, y1, z1), axis=1)  # 3D, uncomment if needed
S = np.unique(S, axis=0)

plt.scatter(S[:,0], S[:,1], c='black')
plt.show()

original_filename = sys.argv[1]
base_name = os.path.splitext(original_filename)[0]
new_filename = base_name + ".vox"

np.savetxt(new_filename, S, fmt='%d') # if values are sorted, just read latest line from C to get Nx and Ny
