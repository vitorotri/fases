#!/usr/bin/env python3
# Broadband IR (low-freq FVM + high-freq GA octave bands) per source-receiver pair.
# Discovers every (source, receiver) pair from the IR/HIR bins in the cwd, saves the
# broadband IR array per pair, and writes one figure per receiver (IR + frequency response).
#
# Run from the project's output/ directory (where IR<s>_<r>.bin / HIR<s>_<r>_<band>.bin live).

import glob, re, os, numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless: savefig only, no display needed
import matplotlib.pyplot as plt

# GUI dark theme (mirror of gui.py PAL + PLOT_COLORS): dark bg, white axes/text, purple-first cycle
FG = "#cfd8dc"
plt.rcParams.update({
    "figure.facecolor": "#1a1d23", "savefig.facecolor": "#1a1d23", "axes.facecolor": "#1a1d23",
    "axes.edgecolor": FG, "axes.labelcolor": FG, "axes.titlecolor": FG, "text.color": FG,
    "xtick.color": FG, "ytick.color": FG, "grid.color": "#2e3340",
    "legend.facecolor": "#20242c", "legend.edgecolor": "#2e3340",
    "axes.prop_cycle": plt.cycler(color=["#7c5cfc", "#0ff091", "#ff5c7a"]),
})
from scipy import signal
from fractions import Fraction
from scipy.signal import resample_poly

octave_bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]

fs = 44100                                    # Hz — output sample rate
XOVER_DROP_DB = 6.0                            # ponytail: crossover = pulse's upper -6 dB edge; tune
deconv_reg = 1e-2   # ponytail: Tikhonov floor. Raise if low band gets noisy near f_cutoff (pulse
                    # weak there), lower for a flatter recovery. Tune to the FVM noise floor.

# Everything below is read from disk (config.txt, header.txt, p_ref.bin) and depends on cwd=output/.
# It is loaded LAZILY on the first broadband_ir()/main() call, NOT at import — so `import allIRs` has
# NO side effects and can't crash merely because a run was interrupted (e.g. header.txt missing).
_ready = False
cfg = f_cutoff_sim = src_opt = dt = dt_half = fs_fv = f_cutoff = pulse_fs = sigma_s = lag = None

def _load_pulse():
    if not os.path.exists("../config/p_ref.bin"):
        return None
    q = np.fromfile("../config/p_ref.bin", np.float64)
    rr = Fraction(fs * dt / 2).limit_denominator(2000)     # 2*fs_fv -> fs
    return resample_poly(q, rr.numerator, rr.denominator)

def _init():
    """Read config/header/pulse once (cwd must be the project's output/). Populates module globals.
    Called at the top of broadband_ir()/main() so importing the module touches no files."""
    global _ready, cfg, f_cutoff_sim, src_opt, dt, dt_half, fs_fv, f_cutoff, pulse_fs, sigma_s, lag
    if _ready:
        return
    # speed of sound + cutoff from config; dt from the solver's output/header.txt
    cfg = np.loadtxt("../config/config.txt")
    f_cutoff_sim = cfg[2]
    src_opt = int(cfg[15]) if len(cfg) > 15 else 0   # config line 16: 0=PCS, 1=Ricker, 2=Sinusoid
    hp = open("../output/header.txt").read().split()   # Nx Ny Nz Nt Lx Ly Lz pbin anim dt scene h tol/h
    dt = float(hp[9])
    dt_half = 0.5 * dt
    fs_fv = 1 / dt
    # FVM<->GA crossover f_x. Fallback = 0.5*f_cutoff (PCS Gaussian roll-off); overridden below from the
    # actual source-pulse spectrum so it's also right for Ricker (bandpass) and any custom glob_f.
    f_cutoff = 0.5 * f_cutoff_sim                 # Hz (grid ceiling is f_cutoff_sim)
    # main3D writes the radiated pulse to ../config/p_ref.bin; deconvolving it recovers the bare room
    # IR so the FVM and GA bands share density and meet at f_cutoff. float64 at dt/2 -> resample to fs.
    pulse_fs = _load_pulse()
    if pulse_fs is None:
        print("[allIRs] !! ../config/p_ref.bin NOT FOUND -> source NOT deconvolved (fallback path). "
              "The FVM band still carries the source pulse; bands will NOT match. Re-run the FVM "
              "simulation with the current main3D to generate p_ref.bin.")
    else:
        print(f"[allIRs] p_ref.bin loaded ({pulse_fs.size} samples at fs) -> deconvolving source pulse.")
        _P = np.abs(np.fft.rfft(pulse_fs)); _f = np.fft.rfftfreq(pulse_fs.size, 1 / fs)
        _pk = int(_P.argmax()); _thr = _P[_pk] * 10 ** (-XOVER_DROP_DB / 20)
        _above = np.where((_f > _f[_pk]) & (_P >= _thr))[0]
        _fx = float(_f[_above[-1]]) if _above.size else float(_f[_pk])
        if 0 < _fx <= f_cutoff_sim:
            f_cutoff = _fx
        print(f"[allIRs] crossover f_x = {f_cutoff:.1f} Hz  (pulse peak {_f[_pk]:.1f} Hz)")
    # Fallback only (no p_ref.bin from an older run): source-pulse lag instead of deconv.
    if src_opt == 1:                                  # Ricker wavelet
        sigma_s = 1.0 / (np.sqrt(2) * np.pi * 0.3 * f_cutoff_sim)
    else:                                             # PCS (0) — Sinusoid (2) has no pulse lag
        sigma_s = 0.425 / f_cutoff_sim
    lag = int(round(np.ceil(8.5 * sigma_s / dt_half) * dt_half * fs))
    _ready = True

def broadband_ir(rec, src):
    """Broadband IR for one (rec, src) pair from the IR/HIR bins in cwd. Returns array at fs,
    or None if the low-freq IR for the pair is missing. Solver names them IR<rec>_<src>.bin."""
    _init()
    low_files = sorted(f for f in glob.glob("IR*.bin")     # IR<rec>_<src>.bin
                       if f[2:].split(".")[0].split("_")[:2] == [str(rec), str(src)])
    high_files = sorted(f for f in glob.glob("HIR*.bin")   # HIR<rec>_<src>_<band>.bin
                        if f[3:].split(".")[0].split("_")[:2] == [str(rec), str(src)])
    if not low_files:
        return None

    # low band: resample FVM rate -> fs, then deconvolve the source pulse -> bare room IR
    LIR = np.fromfile(low_files[0], np.float32)
    r = Fraction(fs / fs_fv).limit_denominator(2000)
    LIR = resample_poly(LIR, r.numerator, r.denominator)   # anti-aliased
    if pulse_fs is not None:
        n = LIR.size
        Q = np.fft.rfft(pulse_fs, n=n)
        eps = deconv_reg * np.max(np.abs(Q) ** 2)          # Tikhonov: caps gain where pulse is weak
        LIR = np.fft.irfft(np.fft.rfft(LIR) * np.conj(Q) / (np.abs(Q) ** 2 + eps), n=n)
    else:                                                  # older run without p_ref.bin
        LIR = np.roll(LIR * (fs_fv / fs), -lag)

    # ONE power-complementary filterbank: FVM is the band below f_cutoff, each GA octave a band
    # above it. Every input is filtered EXACTLY ONCE (the lowest GA band high-passes at f_cutoff and
    # fuses with the FVM low-pass), so the GA is never double-filtered at the crossover. sqrt() of the
    # amplitude-complementary cosine -> sum(w^2)=1 at every edge, so incoherent bands add flat in power.
    T = 2 ** (1 / 6)                                        # cosine transition half-width
    def _shoulder(f, e, hi):                                # hi=True -> high-pass shoulder at edge e
        x = np.clip((f - e / T) / (e * T - e / T), 0, 1)
        return 0.5 * (1 - np.cos(np.pi * x)) if hi else 0.5 * (1 + np.cos(np.pi * x))

    def _filt(ir, hp_e, lp_e):
        f = np.fft.rfftfreq(ir.size, 1 / fs)
        w = np.ones_like(f)
        if hp_e: w *= _shoulder(f, hp_e, True)
        if lp_e: w *= _shoulder(f, lp_e, False)
        return np.fft.irfft(np.fft.rfft(ir) * np.sqrt(w), n=ir.size)

    LIR = _filt(LIR, None, f_cutoff)                       # FVM: single low-pass at f_cutoff
    H_IRs = []
    for hf in high_files:
        fc = int(hf.split("_")[-1].split(".")[0])          # band from HIR..._<band>.bin
        if fc * 2 ** 0.5 <= f_cutoff:                      # octave fully below crossover -> FVM has it
            continue
        arr = np.fromfile(hf, np.float32)
        hp_e = max(f_cutoff, fc / 2 ** 0.5)                # lowest kept band high-passes AT f_cutoff
        lp_e = None if fc == octave_bands[-1] else fc * 2 ** 0.5   # top band runs to Nyquist
        H_IRs.append(_filt(arr, hp_e, lp_e))

    HIR = np.sum(H_IRs, axis=0) if H_IRs else np.zeros(0)
    N = max(LIR.size, HIR.size)
    LIRp = np.pad(LIR, (0, N - LIR.size))
    HIRp = np.pad(HIR, (0, N - HIR.size))
    # Match GA to FVM by SPECTRAL DENSITY across the crossover (peak-matching a delta to a
    # band-limited pulse leaves a ~bandwidth-ratio offset, ~40 dB here). Compare FVM just below
    # f_cutoff to GA just above; median over a wide octave is robust (no narrow-band blow-up).
    if HIRp.any():
        f = np.fft.rfftfreq(N, 1 / fs)
        lo = (f >= f_cutoff / 2) & (f < f_cutoff)          # FVM strong here
        hi = (f > f_cutoff) & (f <= f_cutoff * 2)          # GA strong here
        L = np.median(np.abs(np.fft.rfft(LIRp))[lo])
        H = np.median(np.abs(np.fft.rfft(HIRp))[hi])
        if H > 0:
            HIRp = HIRp * (L / H)
            print(f"[allIRs] rec{rec} src{src}: GA->FVM spectral match {20*np.log10(L/H):+.1f} dB")
    return LIRp + HIRp, LIRp, HIRp             # broadband, low-only, high-only (all length N)


def discover_pairs():
    """Every (rec, src) with a low-freq IR<rec>_<src>.bin present, sorted."""
    pairs = set()
    for f in glob.glob("IR*.bin"):
        m = re.match(r"IR(\d+)_(\d+)\.bin$", f)
        if m:
            pairs.add((int(m.group(1)), int(m.group(2))))   # (rec, src)
    return sorted(pairs)


def main():
    import traceback
    print("[allIRs] cwd:", os.getcwd())
    _init()
    pairs = discover_pairs()
    if not pairs:
        # nothing to do -> DON'T touch existing PNGs (deleting them here was wiping good output)
        print("[allIRs] no IR<rec>_<src>.bin found -> nothing to do; existing PNGs kept."); return
    print(f"[allIRs] pairs: {pairs}")

    # only now (we have pairs and will regenerate) clear stale per-source npy the GUI globs.
    # PNGs are left alone and simply overwritten below, so a mid-run crash can't wipe them.
    for old in glob.glob("IR_broad_*.npy"):
        try: os.remove(old)
        except OSError: pass

    by_rec = {}
    for rec, src in pairs:
        try:
            res = broadband_ir(rec, src)
        except Exception as ex:
            traceback.print_exc()
            print(f"[allIRs] rec{rec} src{src} FAILED: {ex} -> skipped"); continue
        if res is None:
            continue
        ir, LIRp, HIRp = res
        np.save(f"IR_broad_{rec}_{src}", ir)               # per-pair broadband IR (for the GUI)
        by_rec.setdefault(rec, []).append((src, ir, LIRp, HIRp))

    def _fr(x):                                            # (freq>0, magnitude dB)
        f = np.fft.rfftfreq(x.size, 1 / fs)
        return f[1:], 20 * np.log10(np.abs(np.fft.rfft(x))[1:] + 1e-12)

    # one figure per receiver: IR (top) + frequency response (bottom), a line per source
    for rec, items in sorted(by_rec.items()):
      try:
        fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(9, 7))
        for src, ir, _, _ in sorted(items):
            ax_t.plot(np.arange(ir.size) / fs, ir, lw=0.7, label=f"source {src}")
            f, mag = _fr(ir)
            ax_f.semilogx(f, mag, lw=0.7, label=f"source {src}")
        ax_t.set_title(f"Receiver {rec} — Impulse Response")
        ax_t.set_xlabel("Time [s]"); ax_t.set_ylabel("Amplitude")
        ax_t.grid(True); ax_t.legend()
        ax_f.set_title(f"Receiver {rec} — Frequency Response")
        ax_f.set_xlabel("Frequency [Hz]"); ax_f.set_ylabel("Magnitude [dB]")
        ax_f.set_xlim(20, fs / 2); ax_f.grid(True, which="both"); ax_f.legend()
        fig.tight_layout()
        fig.savefig(f"IR_rec{rec}.png", dpi=120)
        plt.close(fig)

        # DEBUG crossover: FVM (low) vs GA (high) FR overlaid, marker at f_cutoff — do the bands
        # meet in level there? A step/offset => calibration; smooth slope above => absorption.
        figc, axc = plt.subplots(figsize=(9, 4))
        for src, _, LIRp, HIRp in sorted(items):
            f, magL = _fr(LIRp); _, magH = _fr(HIRp)
            axc.semilogx(f, magL, lw=0.8, label=f"src {src} FVM (low)")
            axc.semilogx(f, magH, lw=0.8, ls="--", label=f"src {src} GA (high)")
        axc.axvline(f_cutoff, color=FG, lw=1, ls=":", label=f"f_cutoff {f_cutoff:.0f} Hz")
        axc.set_title(f"Receiver {rec} — crossover check (FVM vs GA level at f_cutoff)")
        axc.set_xlabel("Frequency [Hz]"); axc.set_ylabel("Magnitude [dB]")
        axc.set_xlim(20, fs / 2); axc.grid(True, which="both"); axc.legend(fontsize=8)
        figc.tight_layout()
        figc.savefig(f"IR_rec{rec}_crossover.png", dpi=120)
        plt.close(figc)
      except Exception as ex:
        traceback.print_exc()
        print(f"[allIRs] plotting receiver {rec} FAILED: {ex} -> skipped")
    print(f"[allIRs] {len(pairs)} pair(s), {len(by_rec)} receiver plot(s) written.")


def _selfcheck():
    # deconv(h * q, q) ~= h : the core of the root fix
    rng = np.random.default_rng(0)
    h = rng.standard_normal(4000)
    x = (np.arange(240) - 120) / 14.0
    q = (1 - x**2) * np.exp(-x**2 / 2)                     # Ricker-ish pulse centred at 120
    y = np.convolve(h, q)[:h.size]
    Q = np.fft.rfft(q, n=h.size)
    eps = 1e-9 * np.max(np.abs(Q) ** 2)
    hr = np.roll(np.fft.irfft(np.fft.rfft(y) * np.conj(Q) / (np.abs(Q) ** 2 + eps), n=h.size), -120)
    assert np.corrcoef(h, hr)[0, 1] > 0.98, "deconvolution self-check failed"


if __name__ == "__main__":
    try:
        _selfcheck()
    except Exception as ex:                 # never let a self-check block the actual plots
        print(f"[allIRs] self-check warning: {ex}")
    main()
