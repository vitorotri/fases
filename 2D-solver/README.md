# F.A.S.E.S 2D

2D toy solver for enclosed and open spaces.

## Compilation

```bash
# make
```

## Usage

Convert DXF file to OBJ with script dxf2obj.py in the "scripts" directory.\
Fit materials coefficients per octave band with filterFIT.py in the "common" directory. Then:

```bash
# ./run <path_to_scene.obj> <path_to_config.txt> <path_to_sources.txt> <path_to_receivers.txt>
```

## Plotting

```bash
# python3 scripts/plot.py
```

Video animation will be in the "output" directory.
