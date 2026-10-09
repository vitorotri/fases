#!/usr/bin/env bash
# F.A.S.E.S installer (Linux x86-64). Standalone: download only this file and run it
# (double-click opens it in a terminal, or ./install.sh). It fetches the LATEST GitHub release:
# the source tree of that tag (common/, 3D-solver/, LICENSE.md, requirements.txt) + the release
# asset "fases3D" (prebuilt solver, make release), into ~/.local/share/fases.
# Next to a full checkout that already holds 3D-solver/build/fases3D, it installs that instead (offline/dev).
# Installs: system packages (sudo), a private Python venv with requirements.txt, a menu entry,
# a desktop shortcut and an uninstaller. It does NOT install the NVIDIA driver (distribution-specific,
# Secure Boot, reboot): it checks it.
set -Eeuo pipefail
REPO="vitorotri/fases"

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
[ "$SRC" -ef "$DEST" ] && die "Run a downloaded install.sh, not the copy inside $DEST."
LOCAL=0; [ -f "$SRC/common/gui.py" ] && [ -f "$SRC/$SOLVER" ] && LOCAL=1

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
    REQ=(curl tar python3 python3-venv python3-pip libportaudio2 libgl1 libglu1-mesa)
    OPT=(libegl1 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libxcb-xinerama0)
elif command -v dnf >/dev/null 2>&1; then PM=dnf
    REQ=(curl tar python3 python3-pip portaudio mesa-libGL mesa-libGLU)
    OPT=(mesa-libEGL libxkbcommon-x11 xcb-util-cursor xcb-util-wm xcb-util-keysyms)
elif command -v pacman >/dev/null 2>&1; then PM=pacman
    REQ=(curl tar python python-pip portaudio mesa glu)
    OPT=(libxkbcommon-x11 xcb-util-cursor xcb-util-wm xcb-util-keysyms)
elif command -v zypper >/dev/null 2>&1; then PM=zypper
    REQ=(curl tar python3 python3-pip portaudio libGLU1)
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
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
if [ "$LOCAL" = 1 ]; then
    say "Installing from the folder next to install.sh ($SRC)"
    PKG="$SRC"
else
    command -v curl >/dev/null 2>&1 || die "curl not found (needed to download F.A.S.E.S)."
    # tag of the latest release (github.com/<repo>/releases/latest redirects to .../releases/tag/<tag>)
    TAG="$(curl -fsSIL -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest" | sed 's#.*/tag/##' || true)"
    case "$TAG" in ""|*/*) die "No release found at https://github.com/$REPO/releases (internet connection?)." ;; esac
    say "Downloading F.A.S.E.S $TAG (source + solver)"
    curl -fL --retry 3 -o "$TMP/src.tar.gz" "https://github.com/$REPO/archive/refs/tags/$TAG.tar.gz" \
        || die "Could not download the source of $TAG."
    mkdir -p "$TMP/pkg/3D-solver/build"
    tar -xzf "$TMP/src.tar.gz" -C "$TMP/pkg" --strip-components=1
    curl -fL --retry 3 -o "$TMP/pkg/$SOLVER" "https://github.com/$REPO/releases/download/$TAG/fases3D" \
        || die "Release $TAG has no 'fases3D' asset (the prebuilt solver)."
    PKG="$TMP/pkg"
fi
[ -f "$PKG/common/gui.py" ] && [ -f "$PKG/requirements.txt" ] || die "Incomplete package: common/gui.py or requirements.txt missing."

say "Installing F.A.S.E.S to $DEST"
mkdir -p "$DEST"
find "$DEST" -mindepth 1 -maxdepth 1 ! -name venv -exec rm -rf {} +   # old version out, venv kept (fast re-install)
cp -a "$PKG/." "$DEST/"
rm -rf "$DEST/.git"
chmod +x "$DEST/$SOLVER"

# ---------------------------------------------------------------- Python environment
say "Installing Python packages (private environment, this takes a few minutes)"
if ! "$DEST/venv/bin/python" -m pip --version >/dev/null 2>&1; then   # missing, or broken by a failed run
    rm -rf "$DEST/venv"
    python3 -m venv "$DEST/venv"
fi
"$DEST/venv/bin/python" -m pip install --upgrade pip
# --no-compile: Python 3.9's compileall crashes on PySide6's template file (__init__.tmpl.py) under pip's
# progress bar ("encode() argument 'encoding' must be str, not None"); modules compile on first import anyway
"$DEST/venv/bin/python" -m pip install --no-compile -r "$DEST/requirements.txt"

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
