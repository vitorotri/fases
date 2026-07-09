import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
# Each line: first value = legend label, remaining 8 values = octave band data
data_lines = """
0.01	0.010 0.010 0.010 0.015 0.020 0.035 0.040 0.040
0.02	0.010 0.010 0.010 0.015 0.035 0.080 0.090 0.100
0.05	0.010 0.010 0.010 0.025 0.080 0.200 0.250 0.300
0.10	0.010 0.010 0.020 0.050 0.150 0.260 0.310 0.350
0.20	0.010 0.010 0.030 0.110 0.290 0.370 0.420 0.460
0.30	0.010 0.020 0.040 0.190 0.420 0.490 0.530 0.570	
0.50	0.010 0.030 0.100 0.380 0.620 0.740 0.770 0.800
0.80	0.020 0.050 0.240 0.710 0.860 0.880 0.890 0.900
0.90	0.040 0.110 0.680 0.860 0.920 0.930 0.940 0.950 
1.00	0.100 0.370 0.900 1.000 1.000 1.000 1.000 1.000
"""

# Parse data
lines = data_lines.strip().split("\n")
legend_labels = []
values_by_row = []

for line in lines:
    parts = line.split()
    legend_labels.append(parts[0])
    values_by_row.append([float(v) for v in parts[1:]])

# Octave frequency bands (x-axis labels)
octave_bands = ["63", "125", "250", "500", "1k", "2k", "4k", "8k"]

# Colors from top to bottom in the legend
colors = [
    "cyan",
    "gold",
    "grey",
    "pink",
    "magenta",
    "purple",
    "red",
    "green",
    "orange",
    "blue",
]

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for i, (label, values) in enumerate(zip(legend_labels, values_by_row)):
    ax.plot(
        octave_bands,
        values,
        label=label,
        color=colors[i],
        linewidth=2,
        marker=".",
        markersize=5,
    )

# Y-axis from 0 to 1
ax.set_ylim(0, 1)

ax.set_xlabel("Octave Frequency Band [Hz]")
ax.set_ylabel("Value")
ax.set_title("Data vs Octave Frequency Bands")

# Legend on the left side of the figure
ax.legend(
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    title="Legend",
    frameon=False,
)

ax.grid(True, alpha=0.3)

# Make room for the legend on the right side of the plot area
plt.tight_layout()
plt.subplots_adjust(right=0.75)

plt.show()
