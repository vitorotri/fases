import numpy as np
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

ROOM_DIM  = [3.4, 4.0, 3.0]  		# [Lx, Ly, Lz] metres
FS        = 44100.0           		# Hz  — must match the binary reference
C         = 343.23					# m/s
MAX_ORDER = 3                 		# total ISM reflection order

# Energy absorption coefficients per surface  (α ∈ [0,1], R = sqrt(1−α))
# Convention matches pyroomacoustics ShoeBox:
#   west  / east    →  x = 0   / x = Lx
#   south / north   →  y = 0   / y = Ly
#   floor / ceiling →  z = 0   / z = Lz
ALPHA = {
    "west":    0.51,
    "east":    0.90,
    "south":   0.31,
    "north":   0.10,
    "floor":   0.10,
    "ceiling": 0.04,
}

SOURCES = [
    {"pos": [1.75, 1.0, 1.5], "pressure": 1.0}#,
    #{"pos": [2.5, 1.0, 1.5], "pressure": 0.7},
]

RECEIVER = [1.75, 3.0, 1.5]

# One binary (float32) file per source — edit paths as needed
BINARY_FILES = [
    "RIR0_500.bin"#,
    #"RIR1_1000.bin",
]
BINARY_DTYPE = np.float32

# ── Setup ─────────────────────────────────────────────────────────────────────

Lx, Ly, Lz = ROOM_DIM
R = {k: np.sqrt(max(0.0, 1.0 - v)) for k, v in ALPHA.items()}


# ── ISM ───────────────────────────────────────────────────────────────────────

def ism_rir(src_pos, src_pressure, rec_pos):
    """
    Shoebox ISM for a single frequency band.

    Image sources indexed by (l,m,n) ∈ Z^3 and parity bits (p,q,r) ∈ {0,1}^3:
      coord_img = 2·n·L + (−1)^p · coord_src

    Reflections per axis (e.g. x):  |l−p| from the x=0 wall, |l| from the x=Lx wall.
    """
    xs, ys, zs = src_pos
    xr, yr, zr = rec_pos
    lim = MAX_ORDER + 1

    t_list, a_list = [], []

    for l in range(-lim, lim + 1):
        for m in range(-lim, lim + 1):
            for n in range(-lim, lim + 1):
                for p in range(2):
                    for q in range(2):
                        for r in range(2):
                            order = (abs(l-p) + abs(l)
                                   + abs(m-q) + abs(m)
                                   + abs(n-r) + abs(n))
                            if order > MAX_ORDER:
                                continue

                            xi = 2*l*Lx + (-1)**p * xs
                            yi = 2*m*Ly + (-1)**q * ys
                            zi = 2*n*Lz + (-1)**r * zs

                            d = np.sqrt((xi-xr)**2 + (yi-yr)**2 + (zi-zr)**2)
                            if d < 1e-9:
                                continue

                            # product of reflection coefficients along each axis
                            amp = (R["west"]   ** abs(l-p) * R["east"]    ** abs(l)
                                 * R["south"]  ** abs(m-q) * R["north"]   ** abs(m)
                                 * R["floor"]  ** abs(n-r) * R["ceiling"] ** abs(n))
                            amp *= src_pressure / (4.0 * np.pi * d)

                            t_list.append(int(d / C * FS))
                            a_list.append(amp)

    N = max(t_list) + 1
    rir = np.zeros(N, dtype=np.float64)
    for t, a in zip(t_list, a_list):
        rir[t] += a
    return rir


# ── Assert image source count ─────────────────────────────────────────────────
# Virtual room order |I|+|J|+|K| ≤ N  ↔  |l−p|+|l|+|m−q|+|m|+|n−r|+|n| ≤ N
# Analytic cumulative counts (generating fn [(1+z)/(1-z)]^3):
#   N=0:1  N=1:7  N=2:25  N=3:63  N=4:129  N=5:231
_EXPECTED = {0: 1, 1: 7, 2: 25, 3: 63, 4: 129, 5: 231}
_count = sum(
    1
    for l in range(-(MAX_ORDER+1), MAX_ORDER+2)
    for m in range(-(MAX_ORDER+1), MAX_ORDER+2)
    for n in range(-(MAX_ORDER+1), MAX_ORDER+2)
    for p in range(2) for q in range(2) for r in range(2)
    if abs(l-p)+abs(l) + abs(m-q)+abs(m) + abs(n-r)+abs(n) <= MAX_ORDER
)
assert _count == _EXPECTED[MAX_ORDER], f"image source count {_count} ≠ {_EXPECTED[MAX_ORDER]}"
print(f"[check] {_count} candidate image sources ✓  (order={MAX_ORDER}, before path validity filter)")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(len(SOURCES), 1, figsize=(11, 4 * len(SOURCES)), squeeze=False)

for i, src in enumerate(SOURCES):
    rir = ism_rir(src["pos"], src["pressure"], RECEIVER)
    t   = np.arange(len(rir)) / FS * 1e3  # ms

    ax = axes[i][0]
    ax.plot(t, rir, label=f"ISM analytical (src {i+1})", linewidth=0.8)

    d_direct = np.linalg.norm(np.array(src["pos"]) - np.array(RECEIVER))
    t_direct_py = int(d_direct / C * FS)
    print(f"[src {i+1}] direct distance={d_direct:.6f} m  →  Python sample={t_direct_py}  ({t_direct_py/FS*1e3:.4f} ms)")

    try:
        ref   = np.fromfile(BINARY_FILES[i], dtype=BINARY_DTYPE)
        t_ref = np.arange(len(ref)) / FS * 1e3
        ax.plot(t_ref, ref, label="Reference (binary)", linestyle="--",
                linewidth=0.8, alpha=0.75)
        t_peak_ref = int(np.argmax(np.abs(ref)))
        print(f"[src {i+1}] reference peak sample={t_peak_ref}  ({t_peak_ref/FS*1e3:.4f} ms)  amp={ref[t_peak_ref]:.6f}")
        print(f"[src {i+1}] Python    direct  amp ={rir[t_direct_py]:.6f}")
    except FileNotFoundError:
        print(f"[info] '{BINARY_FILES[i]}' not found — reference skipped.")

    ax.set_title(f"Source {i+1}  pos={src['pos']}  →  receiver={RECEIVER}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Pressure (Pa)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

plt.suptitle(
    f"RIR — single band, order={MAX_ORDER}  |  "
    f"room {Lx}×{Ly}×{Lz} m  |  fs={FS:.0f} Hz",
    fontsize=11,
)
plt.tight_layout()
plt.savefig("rir_comparison.png", dpi=150)
plt.show()
