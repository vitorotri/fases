import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "..", "output", "IR1.bin")

p = np.fromfile(path, dtype=np.float32)

plt.plot(p,'b')
plt.xlabel('Samples [n]')
plt.ylabel('Sound Pressure [Pa]')
plt.show()
