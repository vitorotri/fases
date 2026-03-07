import numpy as np
import matplotlib.colors as mcolors
from pathlib import Path

# 3D viz
import pyvista as pv


# -------------------------- config --------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (BASE_DIR / ".." / "output").resolve()
SRC_TXT = (BASE_DIR / ".." / "input" / "src" / "src.txt").resolve()

# filenames (adjust if yours differ)
HEADER = OUTPUT_DIR / "header.txt"
OBJ_DIR = OUTPUT_DIR  # if header stores only a name, we try here too

def find_bin(name):
    candidates = list(OUTPUT_DIR.glob(f"{name}*.bin"))
    if not candidates:
        raise FileNotFoundError(f"Could not find {name}*.bin inside {OUTPUT_DIR}")
    return candidates[0]

BIN_PX = find_bin("pressureX")
BIN_PY = find_bin("pressureY")
BIN_PZ = find_bin("pressureZ")

OUT_MP4 = OUTPUT_DIR / "anim3d_slices.mp4"

# your colormap
cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom", ["crimson", "springgreen", "white", "lawngreen", "blueviolet"]
)

# color limits (set None to auto each frame)
V_LIM = 0.001  # Pa


# -------------------------- helpers --------------------------

def read_header(header_path: Path):
    with open(header_path, "r") as f:
        line = f.readline().strip()
    vals = line.split()

    # Expect:
    # Nx Ny Nz Nt Lx Ly Lz pbin_size anim_step dt obj_name h tol_over_h
    if len(vals) < 13:
        raise RuntimeError(f"Header has {len(vals)} tokens, expected 13+. Line: {line}")

    Nx, Ny, Nz, Nt = map(int, vals[0:4])
    Lx, Ly, Lz = map(float, vals[4:7])

    anim_counter = int(vals[7])      # pbin_size
    anim_step    = int(vals[8])      # anim_step
    dt           = float(vals[9])
    obj_name     = str(vals[10])

    h            = float(vals[11])
    tol_over_h   = int(vals[12])

    return Nx, Ny, Nz, Nt, Lx, Ly, Lz, anim_counter, anim_step, dt, obj_name, h, tol_over_h


def load_slices(Nx, Ny, Nz, anim_counter):
    # Each frame:
    # pX: Ny*Nz (your dst_id = j + Ny*k => reshape (Nz, Ny))
    # pY: Nx*Nz (dst = i + Nx*k => reshape (Nz, Nx))
    # pZ: Nx*Ny contiguous block => reshape (Ny, Nx) if you want y rows, x cols
    pX = np.fromfile(BIN_PX, dtype=np.float32)
    pY = np.fromfile(BIN_PY, dtype=np.float32)
    pZ = np.fromfile(BIN_PZ, dtype=np.float32)

    needX = anim_counter * Ny * Nz
    needY = anim_counter * Nx * Nz
    needZ = anim_counter * Nx * Ny
    if pX.size != needX:
        raise RuntimeError(f"pX size mismatch: got {pX.size}, expected {needX}")
    if pY.size != needY:
        raise RuntimeError(f"pY size mismatch: got {pY.size}, expected {needY}")
    if pZ.size != needZ:
        raise RuntimeError(f"pZ size mismatch: got {pZ.size}, expected {needZ}")

    pX = pX.reshape(anim_counter, Nz, Ny)  # (k, j)
    pY = pY.reshape(anim_counter, Nz, Nx)  # (k, i)
    pZ = pZ.reshape(anim_counter, Ny, Nx)  # (j, i)

    return pX, pY, pZ


def find_obj(obj_name: str):
    # header might store "scene.obj" or just "scene"
    name = obj_name
    if name.endswith(".dxf"):
        name = name[:-4] + ".obj"
    if not name.endswith(".obj"):
        name = name + ".obj"

    candidates = [
        Path(name),
        OBJ_DIR / name,
        (OUTPUT_DIR / name),
        (BASE_DIR / name),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find OBJ '{name}' in: {candidates}")


def set_clim(actor, vlim):
    if vlim is None:
        return
    actor.mapper.scalar_range = (-vlim, vlim)

def read_source_pos(src_path: Path):
    with open(src_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            x, y, z, value = map(float, line.split()[:4])
            return x, y, z, value
    raise RuntimeError(f"No source line found in {src_path}")

# -------------------------- main --------------------------

def main():
    Nx, Ny, Nz, Nt, Lx, Ly, Lz, anim_counter, anim_step, dt, obj_name, h, tol_over_h = read_header(HEADER)
    obj_path = find_obj(obj_name)

    pX, pY, pZ = load_slices(Nx, Ny, Nz, anim_counter)

    # grid spacing
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    
    src_x, src_y, src_z, src_val = read_source_pos(SRC_TXT)

    # cell-center coordinates
    xc = (np.arange(Nx) + 0.5) * dx
    yc = (np.arange(Ny) + 0.5) * dy
    zc = (np.arange(Nz) + 0.5) * dz
    
    # nearest cell-center indices to the source position
    i0 = int(np.argmin(np.abs(xc - src_x)))
    j0 = int(np.argmin(np.abs(yc - src_y)))
    k0 = int(np.argmin(np.abs(zc - src_z)))
    
    # fixed slice positions (exactly at the chosen cell centers)
    x0 = xc[i0]
    y0 = yc[j0]
    z0 = zc[k0]
    
    print(f"Source: ({src_x:.3f}, {src_y:.3f}, {src_z:.3f})")
    print(f"Slices at cell centers: x0={x0:.3f} (i0={i0}), y0={y0:.3f} (j0={j0}), z0={z0:.3f} (k0={k0})")

    # --- create plotter ---
    pv.global_theme.background = "white"
    pv.global_theme.font.color = "black"

    pl = pv.Plotter(off_screen=True, window_size=(1600, 896))
    pl.add_axes(line_width=2)

    # --- OBJ wireframe ---
    mesh = pv.read(str(obj_path))
    pl.add_mesh(mesh, style="wireframe", color="black", line_width=1.0, opacity=0.9)

    # --- three slice planes ---
    # --- cell-center coordinates (match how you computed x0,y0,z0) ---
    xc = (np.arange(Nx) + 0.5) * dx
    yc = (np.arange(Ny) + 0.5) * dy
    zc = (np.arange(Nz) + 0.5) * dz
    
    # fixed slice positions (cell centers)
    x0 = xc[i0]
    y0 = yc[j0]
    z0 = zc[k0]
    
    # ---------- YZ slice at x = x0 ----------
    Y, Z = np.meshgrid(yc, zc, indexing="ij")         # (Ny, Nz)
    X = np.full_like(Y, x0)
    plane_X = pv.StructuredGrid(X, Y, Z)              # deterministic ordering
    
    # pX[frame] is (Nz, Ny) with (k, j). We want (j, k) => (Ny, Nz)
    plane_X.point_data["p"] = pX[0].T.astype(np.float32).ravel(order="F")
    
    # ---------- XZ slice at y = y0 ----------
    X, Z = np.meshgrid(xc, zc, indexing="ij")         # (Nx, Nz)
    Y = np.full_like(X, y0)
    plane_Y = pv.StructuredGrid(X, Y, Z)
    
    # pY[frame] is (Nz, Nx) with (k, i). Want (i, k) => (Nx, Nz)
    plane_Y.point_data["p"] = pY[0].T.astype(np.float32).ravel(order="F")
    
    # ---------- XY slice at z = z0 ----------
    X, Y = np.meshgrid(xc, yc, indexing="ij")         # (Nx, Ny)
    Z = np.full_like(X, z0)
    plane_Z = pv.StructuredGrid(X, Y, Z)
    
    # pZ[frame] is (Ny, Nx) with (j, i). Want (i, j) => (Nx, Ny)
    plane_Z.point_data["p"] = pZ[0].T.astype(np.float32).ravel(order="F")

    # Add meshes (semi-transparent so you can see overlap)
    aX = pl.add_mesh(plane_X, scalars="p", cmap=cmap, opacity=0.95, lighting=False, show_scalar_bar=False)
    aY = pl.add_mesh(plane_Y, scalars="p", cmap=cmap, opacity=0.95, lighting=False, show_scalar_bar=False)
    aZ = pl.add_mesh(plane_Z, scalars="p", cmap=cmap, opacity=0.95, lighting=False, show_scalar_bar=False)

    # fixed color limits if desired
    set_clim(aX, V_LIM)
    set_clim(aY, V_LIM)
    set_clim(aZ, V_LIM)

    pl.add_scalar_bar(title="Sound Pressure (Pa)", n_labels=5)
    
    pl.add_axes(line_width=2, color="black")

    # Nice isometric-ish camera
    pl.camera_position = "iso"
    pl.camera.azimuth = 5    # rotate left/right
    #pl.camera.elevation = 5     # tilt up/down
    pl.camera.zoom(1.2)

    # --- write animation ---
    import imageio.v2 as imageio

    FPS = 10
    
    # Force the FFMPEG backend explicitly (this is the key)
    writer = imageio.get_writer(
        str(OUT_MP4),
        format="FFMPEG",
        fps=FPS,
        codec="libx264",
        pixelformat="yuv420p",
    )
    
    for frame in range(anim_counter):
        plane_X.point_data["p"] = pX[frame].T.astype(np.float32).ravel(order="F")
        plane_Y.point_data["p"] = pY[frame].T.astype(np.float32).ravel(order="F")
        plane_Z.point_data["p"] = pZ[frame].T.astype(np.float32).ravel(order="F")
    
        t = frame * dt * anim_step
        pl.add_text(f"t = {t:.6f} s", name="time_text", font_size=14)
    
        if V_LIM is not None:
            aX.mapper.scalar_range = (-V_LIM, V_LIM)
            aY.mapper.scalar_range = (-V_LIM, V_LIM)
            aZ.mapper.scalar_range = (-V_LIM, V_LIM)
    
        pl.render()
        img = pl.screenshot(return_img=True)   # (H, W, 3) uint8
        writer.append_data(img)
    
    writer.close()
    pl.close()
    print(f"[OK] Wrote: {OUT_MP4}")


if __name__ == "__main__":
    main()
