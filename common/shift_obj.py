#!/usr/bin/env python3
"""Shift an OBJ so all geometry lies on the positive side of the axes.

The FVM/GA pipeline expects the scene on the positive octant (the parser then
pads it with the ghost band). A mesh grabbed off the internet often straddles
the origin or sits fully negative, so before it is parsed we translate every
vertex by -min per axis: the mesh's minimum corner lands exactly on the origin
and nothing that was already positive moves further than it must.

Only `v` (vertex position) lines are touched. `vn` (normals) and `vt`
(texture coords) are directions/parameters, NOT positions, so they are left
verbatim. Any extra tokens on a `v` line (w, or per-vertex r g b colour) are
preserved. Idempotent: a mesh already at the origin shifts by 0 -> no rewrite.

Usage (CLI):   python3 shift_obj.py scene.obj            # edits in place
               python3 shift_obj.py scene.obj out.obj    # writes a copy
Importable:    from shift_obj import shift_to_positive
"""
import sys


def shift_to_positive(path, out=None):
    """Translate the mesh in `path` so its min corner is the origin.

    Returns the (dx, dy, dz) shift applied. If the mesh is already fully on the
    positive side (all mins >= 0) nothing is written and (0,0,0) is returned.
    `out` defaults to `path` (in-place edit).
    """
    with open(path) as f:
        lines = f.readlines()

    mins = [float("inf")] * 3
    have_vert = False
    for ln in lines:
        if ln.startswith("v ") or ln.startswith("v\t"):
            parts = ln.split()
            try:
                xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
            except (IndexError, ValueError):
                continue                       # malformed vertex line, skip
            have_vert = True
            for i in range(3):
                if xyz[i] < mins[i]:
                    mins[i] = xyz[i]

    if not have_vert:
        return (0.0, 0.0, 0.0)                 # no geometry to move

    # push each axis up only if it dips negative; leave positive axes untouched
    shift = [(-m if m < 0 else 0.0) for m in mins]
    if shift == [0.0, 0.0, 0.0]:
        return (0.0, 0.0, 0.0)                 # already positive -> don't rewrite

    outlines = []
    for ln in lines:
        if ln.startswith("v ") or ln.startswith("v\t"):
            parts = ln.split()
            try:
                x = float(parts[1]) + shift[0]
                y = float(parts[2]) + shift[1]
                z = float(parts[3]) + shift[2]
            except (IndexError, ValueError):
                outlines.append(ln); continue
            rest = parts[4:]                    # w / colour tokens, kept as-is
            tail = (" " + " ".join(rest)) if rest else ""
            eol = "\n" if ln.endswith("\n") else ""
            outlines.append(f"v {x:.10g} {y:.10g} {z:.10g}{tail}{eol}")
        else:
            outlines.append(ln)

    with open(out or path, "w") as f:
        f.writelines(outlines)
    return tuple(shift)


def _demo():
    import os, tempfile
    obj = "v -2 1 -5\nvn 0 1 0\nv 0 3 -1\nf 1 2 1\n"
    d = tempfile.mkdtemp()
    p = os.path.join(d, "t.obj")
    with open(p, "w") as f:
        f.write(obj)
    s = shift_to_positive(p)
    assert s == (2.0, 0.0, 5.0), s             # min x=-2 -> +2, y ok, min z=-5 -> +5
    txt = open(p).read().splitlines()
    assert txt[0] == "v 0 1 0", txt[0]         # -2+2, 1+0, -5+5
    assert txt[1] == "vn 0 1 0", txt[1]        # normal untouched
    assert txt[2] == "v 2 3 4", txt[2]         # 0+2, 3+0, -1+5
    assert shift_to_positive(p) == (0.0, 0.0, 0.0)   # idempotent
    print("shift_obj demo OK")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _demo()
    else:
        inp = sys.argv[1]
        outp = sys.argv[2] if len(sys.argv) > 2 else None
        s = shift_to_positive(inp, outp)
        print(f"shifted by {s} -> {outp or inp}")
