import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from pathlib import Path
import ezdxf
import math

# -------------------------------------------- for DXF wireframe ------------------------------------------------

def _arc_points(cx, cy, r, a0, a1, step_deg=5.0):
    a0r, a1r = math.radians(a0), math.radians(a1)
    if a1r < a0r: a1r += 2*math.pi
    n = max(4, int(math.ceil((a1r - a0r) * 180.0 / (math.pi*step_deg))))
    return [(cx + r*math.cos(a0r + t*(a1r - a0r)/n),
             cy + r*math.sin(a0r + t*(a1r - a0r)/n)) for t in range(n+1)]

def _ellipse_points(cx, cy, majx, majy, ratio, t0, t1, step_deg=5.0):
    a = math.hypot(majx, majy);  b = a * ratio
    phi = math.atan2(majy, majx)
    n = max(8, int(abs(t1 - t0) / (step_deg*math.pi/180.0)))
    pts = []
    for i in range(n+1):
        t = t0 + (t1 - t0)*i/n
        x, y = a*math.cos(t), b*math.sin(t)
        X = cx + x*math.cos(phi) - y*math.sin(phi)
        Y = cy + x*math.sin(phi) + y*math.cos(phi)
        pts.append((X, Y))
    return pts

def draw_dxf_wire(ax, dxf_path, color_boundary='w', color_obst='k',
                  lw=0.8, step_deg=5.0, align_origin=False, zorder=10):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Optionally shift all DXF coords so min x,y = 0 (to match your SDF origin)
    xmin = ymin = float('inf'); xmax = ymax = -float('inf')
    if align_origin:
        for e in msp:
            t = e.dxftype()
            if t == "LWPOLYLINE":
                for (x,y,*_) in e.get_points():
                    xmin=min(xmin,x); ymin=min(ymin,y)
            elif t == "LINE":
                for p in (e.dxf.start, e.dxf.end):
                    xmin=min(xmin,p.x); ymin=min(ymin,p.y)
            elif t in ("ARC","CIRCLE","ELLIPSE"):
                # rough bbox using centers and radii/axes
                if t == "CIRCLE":
                    cx,cy,r = e.dxf.center.x, e.dxf.center.y, e.dxf.radius
                    xmin=min(xmin,cx-r); ymin=min(ymin,cy-r)
                elif t == "ARC":
                    cx,cy,r = e.dxf.center.x, e.dxf.center.y, e.dxf.radius
                    xmin=min(xmin,cx-r); ymin=min(ymin,cy-r)
                else:
                    cx,cy = e.dxf.center.x, e.dxf.center.y
                    a = math.hypot(e.dxf.major_axis.x, e.dxf.major_axis.y)
                    b = a*e.dxf.ratio
                    xmin=min(xmin,cx-a-b); ymin=min(ymin,cy-a-b)
            elif t == "SPLINE":
                for (x,y,_) in e.approximate(segments=64):
                    xmin=min(xmin,x); ymin=min(ymin,y)
        if xmin==float('inf'): xmin=ymin=0.0
    else:
        xmin=ymin=0.0

    for e in msp:
        typ = e.dxftype()
        layer = e.dxf.layer
        col = color_boundary if layer == "Boundary" else color_obst

        pts = None
        if typ == "LWPOLYLINE":
            pts = [(x - xmin, y - ymin) for (x,y,*_) in e.get_points()]
            if e.closed and pts[0] != pts[-1]: pts.append(pts[0])
        elif typ == "LINE":
            pts = [(e.dxf.start.x - xmin, e.dxf.start.y - ymin),
                   (e.dxf.end.x   - xmin, e.dxf.end.y   - ymin)]
        elif typ == "CIRCLE":
            c = e.dxf.center; r = e.dxf.radius
            pts = _arc_points(c.x, c.y, r, 0.0, 360.0, step_deg)
            pts = [(x - xmin, y - ymin) for (x,y) in pts]
        elif typ == "ARC":
            c = e.dxf.center; r = e.dxf.radius
            pts = _arc_points(c.x, c.y, r, e.dxf.start_angle, e.dxf.end_angle, step_deg)
            pts = [(x - xmin, y - ymin) for (x,y) in pts]
        elif typ == "ELLIPSE":
            c = e.dxf.center; maj = e.dxf.major_axis
            pts = _ellipse_points(c.x, c.y, maj.x, maj.y, e.dxf.ratio,
                                  e.dxf.start_param, e.dxf.end_param, step_deg)
            pts = [(x - xmin, y - ymin) for (x,y) in pts]
        elif typ == "SPLINE":
            pts = [(x - xmin, y - ymin) for (x,y,_) in e.approximate(segments=128)]
        else:
            continue

        if pts and len(pts) >= 2:
            x, y = zip(*pts)
            ax.plot(x, y, col, linewidth=lw, zorder=zorder, solid_capstyle='round')

    # Make sure wireframe stays above the image:
    for ln in ax.lines:
        ln.set_animated(False)  # important when blit=True

# ---------------------------------------------------------------------------------------------------------------

# dir paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / '..' / 'output'

# Read header information
with open(OUTPUT_DIR / 'header.txt', 'r') as file:
#with open('headerZuri.txt', 'r') as file:
    header_line = file.readline().strip()
    values = header_line.split(' ')
Nx, Ny, Nt, Lx, Ly, anim_counter, step, dt, dxf_name = int(values[0]), int(values[1]), int(values[2]), float(values[3]), float(values[4]), int(values[5]), int(values[6]), float(values[7]), str(values[8])

# Load data
data = np.fromfile(OUTPUT_DIR / 'pressure.bin', dtype=np.float32).reshape((anim_counter, Ny, Nx))

#print(data.max())
#print(data.min())

# Set up the figure and initial plot
fig, ax = plt.subplots(figsize=(16, 10))

# my personal colormap (not colorblind friendly)
cmap = mcolors.LinearSegmentedColormap.from_list('custom', ['crimson', 'springgreen', 'white', 'lawngreen', 'blueviolet'])
#plt.style.use('dark_background')
#cmap = mcolors.LinearSegmentedColormap.from_list('custom', ['crimson', 'springgreen', 'black', 'lawngreen', 'blueviolet'])

v_lim = 0.01
im = ax.imshow(data[0], cmap=cmap, origin='lower', extent=[0, Lx, 0, Ly], vmin = -v_lim, vmax = v_lim)
#im = ax.imshow(data[0], cmap='seismic', origin='lower', extent=[0, Lx, 0, Ly], vmin = -0.01, vmax = 0.01)
#im = ax.imshow(data[0], cmap='magma', origin='lower', extent=[0, Lx, 0, Ly], vmin = -0.01, vmax = 0.01)
#im = ax.imshow(data[0], cmap='prism', origin='lower', extent=[0, Lx, 0, Ly], vmin = -0.01, vmax = 0.01)
#im = ax.imshow(data[0], cmap='inferno', origin='lower', extent=[0, Lx, 0, Ly], vmin = -0.01, vmax = 0.01)
#im = ax.imshow(data[0], cmap='Greys', origin='lower', extent=[0, Lx, 0, Ly], vmin = -0.01, vmax = 0.01)

dxf_name = dxf_name.removesuffix(".obj")
dxf_name = dxf_name + ".dxf"


# Comment if using PNG
# Draw wireframe on top (shift to origin if SDF/OBJ parser did that)
draw_dxf_wire(ax, dxf_name, color_boundary='w', color_obst='k', lw=0.8, step_deg=5.0, align_origin=True, zorder=10) # argc[1] must be a .dxf


plt.gca().set_aspect('equal')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
title = ax.set_title(f't = 0.000 s')

# Add colorbar outside the axes
cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.5, pad=0.05) # Move it outside
cbar.set_label('Sound Pressure (Pa)')
#cbar.set_label('Normalized Sound Pressure (Pa)')

# Animation update function
def update(frame):
    im.set_array(data[frame])
    #im.set_clim(data[frame].min(), data[frame].max()) # Adjust color limits dynamically
    title.set_text(f't = {frame*dt*step:.5f} s')
    return [im, title]

# Create the animation
anim = animation.FuncAnimation(fig, update, frames=anim_counter, interval=30, blit=True)
# Set up the writer and save the animation
writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
anim.save(OUTPUT_DIR / 'anim.mp4', writer=writer)
plt.close(fig)


plt.figure(figsize=(10,10))
plt.imshow(data[5], cmap=cmap, origin='lower', extent=[0, Lx, 0, Ly], vmin = -0.001, vmax = 0.001, interpolation='spline16')
#plt.scatter(x, y, s=0.2, c='black')
#plt.xticks([])  # Remove x-axis ticks
#plt.yticks([])  # Remove y-axis ticks
cbar = plt.colorbar(im, orientation='vertical', shrink=0.5, pad=0.05)
cbar.set_label('Sound Pressure (Pa)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()
