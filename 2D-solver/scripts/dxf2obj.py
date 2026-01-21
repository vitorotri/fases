#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads 2D entities from a DXF and prints OBJ-style output with per-edge
line equations (vn a b c). Arcs, circles, ellipses, and splines are
discretized into polylines.

Usage:
    python3 dxf2obj.py myfile.dxf > myfile.obj
"""

import ezdxf
import math
import sys

# ---- parameters ----
DXF_PATH = sys.argv[1] if len(sys.argv) > 1 else "TST2.dxf"
ARC_RES_DEG = 5.0  # segment every 5 degrees for arcs/circles

# --------------------

doc = ezdxf.readfile(DXF_PATH)
msp = doc.modelspace()
vertex_counter = 1

def signed_area(pts):
    A = 0.0
    n = len(pts)
    for k in range(n):
        x0, y0 = pts[k]
        x1, y1 = pts[(k + 1) % n]
        A += x0 * y1 - x1 * y0
    return 0.5 * A  # >0: CCW

def arc_to_poly(cx, cy, r, start, end, is_ccw=True):
    """Discretize an arc or circle to points."""
    if is_ccw:
        if end <= start: end += 2 * math.pi
    else:
        if start <= end: start += 2 * math.pi
    total_angle = abs(end - start)
    nseg = max(4, int(math.ceil(total_angle * 180.0 / (math.pi * ARC_RES_DEG))))
    pts = []
    for i in range(nseg + 1):
        t = i / nseg
        ang = start + (end - start) * t
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts

for e in msp:
    typ = e.dxftype()
    name = e.dxf.layer

    # ---- Convert entity to list of (x,y) ----
    points = []
    closed = False

    if typ == "LWPOLYLINE":
        points = [(x, y) for (x, y, *_r) in e.get_points()]
        closed = bool(e.closed) or (points[0] == points[-1])
        if closed and points[0] != points[-1]:
            points.append(points[0])

    elif typ == "LINE":
        points = [(e.dxf.start.x, e.dxf.start.y),
                  (e.dxf.end.x,   e.dxf.end.y)]
        closed = False

    elif typ == "ARC":
        cx, cy = e.dxf.center.x, e.dxf.center.y
        r = e.dxf.radius
        start = math.radians(e.dxf.start_angle)
        end   = math.radians(e.dxf.end_angle)
        pts = arc_to_poly(cx, cy, r, start, end, is_ccw=True)
        points = pts
        closed = False

    elif typ == "CIRCLE":
        cx, cy = e.dxf.center.x, e.dxf.center.y
        r = e.dxf.radius
        points = arc_to_poly(cx, cy, r, 0.0, 2 * math.pi, is_ccw=True)
        closed = True

    elif typ == "ELLIPSE":
        cx, cy = e.dxf.center.x, e.dxf.center.y
        major = e.dxf.major_axis
        ratio = e.dxf.ratio
        start = e.dxf.start_param
        end   = e.dxf.end_param
        a = math.hypot(major.x, major.y)
        b = a * ratio
        phi = math.atan2(major.y, major.x)
        nseg = max(8, int(abs(end - start) / (ARC_RES_DEG * math.pi / 180)))
        pts = []
        for i in range(nseg + 1):
            t = start + (end - start) * i / nseg
            x = a * math.cos(t)
            y = b * math.sin(t)
            X = cx + x * math.cos(phi) - y * math.sin(phi)
            Y = cy + x * math.sin(phi) + y * math.cos(phi)
            pts.append((X, Y))
        points = pts
        closed = abs(end - start) >= 2 * math.pi * 0.999

    elif typ == "SPLINE":
        # approximate spline using ezdxf's native tessellation
        pts = [(p[0], p[1]) for p in e.approximate(segments=64)]
        points = pts
        closed = e.closed

    else:
        continue  # skip unsupported types

    if len(points) < 2:
        continue

    print(f"o {name}")

    # Emit vertices
    indices = []
    for (x, y) in points[:-1] if closed else points:
        print(f"v {x:.6f} {y:.6f} 0.0")
        indices.append(vertex_counter)
        vertex_counter += 1

    # Emit face if closed
    if closed:
        print("f", " ".join(str(i) for i in indices))

    # Orientation (for closed shapes)
    A = signed_area(points[:-1] if closed else points)

    for k in range(len(points) - 1):
        x0, y0 = points[k]
        x1, y1 = points[k + 1]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L == 0.0: continue
        nLx, nLy = (dy / L, -dx / L)
        nRx, nRy = (-dy / L, dx / L)

        if closed:
            interior_is_left = (A > 0.0)
            if name == "Boundary":
                nx, ny = (nLx, nLy) if interior_is_left else (nRx, nRy)
            else:
                nx, ny = (nRx, nRy) if interior_is_left else (nLx, nLy)
        else:
            nx, ny = nLx, nLy

        a, b, c = nx, ny, -(nx * x0 + ny * y0)
        print(f"vn {a:.9f} {b:.9f} {c:.9f}")


