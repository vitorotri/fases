import numpy as np
import matplotlib.pyplot as plt
import os
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "output")

# Find all IR*.bin files
files = sorted(glob.glob(os.path.join(output_dir, "IR*.bin")))

# Custom colors
colors = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#17becf",  # cyan
]

for file, color in zip(files, colors):
    p = np.fromfile(file, dtype=np.float32)

    # Use filename (without path) as legend label
    label = os.path.basename(file)

    plt.plot(p, color=color, label=label)

plt.xlabel('Samples [n]')
plt.ylabel('Sound Pressure [Pa]')
plt.legend()
plt.show()
