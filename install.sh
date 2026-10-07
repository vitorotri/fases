#!/usr/bin/env bash
# F.A.S.E.S installer (Linux x86-64). Lives at the top of the shipped folder:
#   install.sh  LICENSE.md  requirements.txt  common/  3D-solver/build/fases3D
# Run: double-click (it opens itself in a terminal) or ./install.sh
# Installs: system packages (sudo), a private Python venv with requirements.txt,
# the program in ~/.local/share/fases, a menu entry, a desktop shortcut and an uninstaller.
# It does NOT install the NVIDIA driver (distribution-specific, Secure Boot, reboot): it checks it.
set -Eeuo pipefail

# double-clicked (no terminal): reopen in one, so sudo can ask for the password
if [ ! -t 0 ] && [ -z "${FASES_IN_TERM:-}" ]; then
    export FASES_IN_TERM=1
    for t in x-terminal-emulator gnome-terminal konsole xfce4-terminal mate-terminal xterm; do
        command -v "$t" >/dev/null 2>&1 || continue
        case "$t" in
            gnome-terminal)               exec gnome-terminal -- bash "$0" ;;
            xfce4-terminal|mate-terminal) exec "$t" -x bash "$0" ;;
            *)                            exec "$t" -e bash "$0" ;;
        esac
    done
fi

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HOME/.local/share/fases"
BIN="$HOME/.local/bin"
APPS="$HOME/.local/share/applications"
ICONS="$HOME/.local/share/icons/hicolor"
SOLVER="3D-solver/build/fases3D"

pause() { if [ -n "${FASES_IN_TERM:-}" ]; then read -rp $'\nPress Enter to close...' _ || true; fi; }
say()   { printf '\n==> %s\n' "$*"; }
warn()  { printf '  [!] %s\n' "$*"; }
die()   { printf '\n!! %s\n' "$*"; pause; exit 1; }
ver_ge() { [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }   # $1 >= $2
trap 'printf "\n!! Installation failed (line %s).\n" "$LINENO"; pause' ERR

# ---------------------------------------------------------------- checks
say "Checking the system"
[ "$(uname -m)" = x86_64 ] || die "F.A.S.E.S needs Linux x86-64 (found $(uname -m))."
glibc="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"
ver_ge "$glibc" 2.31 || die "glibc $glibc is too old; 2.31 or newer is needed (Debian 11 / Ubuntu 20.04 or later)."
[ -f "$SRC/common/gui.py" ] && [ -f "$SRC/$SOLVER" ] || die "Incomplete package: common/gui.py or $SOLVER is missing next to install.sh."
[ "$SRC" -ef "$DEST" ] && die "Run install.sh from the downloaded folder, not from $DEST."

if command -v nvidia-smi >/dev/null 2>&1; then
    drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
    if [ -z "$drv" ]; then
        warn "nvidia-smi is present but reports no GPU (driver not loaded?). Simulations need a working driver."
    elif ver_ge "$drv" 450.80; then
        echo "  NVIDIA driver $drv: OK"
    else
        warn "NVIDIA driver $drv is older than 450.80: update it, or the solver will not start."
    fi
else
    warn "No NVIDIA driver found. The program installs, but simulations need an NVIDIA GPU with"
    warn "driver 450.80 or newer (install it with your distribution's driver manager)."
fi

# ---------------------------------------------------------------- system packages
# REQ: Python + venv, PortAudio (live auralization), OpenGL/GLU (3D view).
# OPT: X11 libraries PySide6's Qt needs on minimal desktops (absent names are skipped).
if   command -v apt-get >/dev/null 2>&1; then PM=apt
    REQ=(python3 python3-venv python3-pip libportaudio2 libgl1 libglu1-mesa)
    OPT=(libegl1 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libxcb-xinerama0)
elif command -v dnf >/dev/null 2>&1; then PM=dnf
    REQ=(python3 python3-pip portaudio mesa-libGL mesa-libGLU)
    OPT=(mesa-libEGL libxkbcommon-x11 xcb-util-cursor xcb-util-wm xcb-util-keysyms)
elif command -v pacman >/dev/null 2>&1; then PM=pacman
    REQ=(python python-pip portaudio mesa glu)
    OPT=(libxkbcommon-x11 xcb-util-cursor xcb-util-wm xcb-util-keysyms)
elif command -v zypper >/dev/null 2>&1; then PM=zypper
    REQ=(python3 python3-pip portaudio libGLU1)
    OPT=(Mesa-libEGL1 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1)
else PM=none
fi

pm_install() {
    case "$PM" in
        apt)    sudo apt-get install -y "$@" ;;
        dnf)    sudo dnf install -y "$@" ;;
        pacman) sudo pacman -S --needed --noconfirm "$@" ;;
        zypper) sudo zypper --non-interactive install "$@" ;;
    esac
}

if [ "$PM" = none ]; then
    warn "Unknown package manager: install Python 3.9+ (with venv), PortAudio and GLU yourself."
else
    say "Installing system packages ($PM; sudo will ask for your password)"
    [ "$PM" = apt ] && sudo apt-get update
    pm_install "${REQ[@]}"
    for p in "${OPT[@]}"; do
        pm_install "$p" >/dev/null 2>&1 || warn "optional package $p not available (only matters if the GUI fails to start)"
    done
fi

command -v python3 >/dev/null 2>&1 || die "python3 not found."
python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' \
    || die "Python 3.9 or newer is needed (found $(python3 --version 2>&1))."

# ---------------------------------------------------------------- program files
say "Copying F.A.S.E.S to $DEST"
mkdir -p "$DEST/3D-solver/build"
rm -rf "$DEST/common"                                   # the venv is kept: a re-install is fast
cp -a "$SRC/common" "$DEST/"
cp -a "$SRC/$SOLVER" "$DEST/$SOLVER"
chmod +x "$DEST/$SOLVER"
cp -a "$SRC/requirements.txt" "$DEST/"
[ -f "$SRC/LICENSE.md" ] && cp -a "$SRC/LICENSE.md" "$DEST/"

# ---------------------------------------------------------------- Python environment
say "Installing Python packages (private environment, this takes a few minutes)"
if ! "$DEST/venv/bin/python" -m pip --version >/dev/null 2>&1; then   # missing, or broken by a failed run
    rm -rf "$DEST/venv"
    python3 -m venv "$DEST/venv"
fi
"$DEST/venv/bin/python" -m pip install --upgrade pip
"$DEST/venv/bin/python" -m pip install -r "$DEST/requirements.txt"

# ---------------------------------------------------------------- launcher, menu, icons, desktop
say "Creating the launcher, menu entry and desktop shortcut"
mkdir -p "$BIN" "$APPS"
cat > "$BIN/fases" <<EOF
#!/usr/bin/env bash
exec "$DEST/venv/bin/python" "$DEST/common/gui.py" "\$@"
EOF
chmod +x "$BIN/fases"

for n in 16 24 32 48 64 128 256 512; do
    f="$DEST/common/icons/fases_$n.png"
    [ -f "$f" ] && install -Dm644 "$f" "$ICONS/${n}x${n}/apps/fases.png"
done
gtk-update-icon-cache -q -t "$ICONS" >/dev/null 2>&1 || true

cat > "$APPS/fases.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=F.A.S.E.S
GenericName=Room acoustics simulator
Comment=Faithful Acoustic Solver for Extended Spaces
Exec="$BIN/fases"
Icon=fases
Terminal=false
Categories=Science;Physics;Education;
EOF
chmod +x "$APPS/fases.desktop"
update-desktop-database "$APPS" >/dev/null 2>&1 || true

DESK="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
if [ -d "$DESK" ]; then
    cp "$APPS/fases.desktop" "$DESK/fases.desktop"
    chmod +x "$DESK/fases.desktop"
    gio set "$DESK/fases.desktop" metadata::trusted true >/dev/null 2>&1 || true   # GNOME/Cinnamon: launchable
fi

cat > "$DEST/uninstall.sh" <<EOF
#!/usr/bin/env bash
# Removes F.A.S.E.S (program, environment, launcher, menu entry, icons). Your projects are not touched.
rm -rf "$DEST" "$BIN/fases" "$APPS/fases.desktop" "$DESK/fases.desktop"
rm -f "$ICONS"/*/apps/fases.png
echo "F.A.S.E.S removed."
EOF
chmod +x "$DEST/uninstall.sh"

trap - ERR
say "Done. Start F.A.S.E.S from the applications menu, the desktop icon, or the command 'fases'."
case ":$PATH:" in *":$BIN:"*) ;; *) warn "$BIN is not on your PATH: the 'fases' command needs it (menu and desktop work anyway)." ;; esac
echo "  Uninstall: $DEST/uninstall.sh"
pause
