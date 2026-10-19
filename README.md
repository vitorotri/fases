![image description](FASES_lockup.png)


F.A.S.E.S is the Faithful Acoustic Solver for Extended Spaces.

## Installation

### Requirements

- Linux x86-64 with glibc 2.31 or newer (Debian 11, Ubuntu 20.04 or later)
- An NVIDIA GPU with driver 450.80 or newer (no CUDA toolkit needed)
- Python 3.9 or newer
- An internet connection during installation

### Install

Download the installer of the latest release and run it:

```bash
curl -LO https://github.com/vitorotri/fases/releases/latest/download/install.sh
chmod +x install.sh
./install.sh
```

You can also double-click `install.sh`; it opens a terminal by itself. Run it as a file, not piped into
`bash` (`curl ... | bash` does not work).

The installer asks for your password once (sudo) to install the system packages it needs (Python venv,
PortAudio, OpenGL). It then downloads the program and the prebuilt solver into `~/.local/share/fases`,
sets up a private Python environment, and adds a menu entry, a desktop icon and the `fases` command.
It never installs or changes the NVIDIA driver; it only checks it.

### Start, update, uninstall

- **Start:** from the applications menu, the desktop icon, or the command `fases`.
- **Update:** download and run the newest `install.sh` again. Your projects are not touched.
- **Uninstall:** `~/.local/share/fases/uninstall.sh`. Your projects are not touched.

### Windows

Windows is supported through WSL 2 (Windows 11, or Windows 10 22H2) with the Windows NVIDIA driver.
This route is not tested yet; see the WSL section of the user manual.

### Building the solver from source

Needs the CUDA toolkit (nvcc 11.2 or newer) and a C++17 compiler:

```bash
cd 3D-solver
make            # quick build for the GPU set in the Makefile (ARCH)
make release    # portable build for all GPU generations
```

## Documentation

The user manual is [docs/F_A_S_E_S_manual.pdf](docs/F_A_S_E_S_manual.pdf).

## Ethical use

This software and its derivatives shall not be used for mass surveillance or to harm civilians.
