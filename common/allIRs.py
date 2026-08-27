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
from scipy import signal
from fractions import Fraction
from scipy.signal import resample_poly

octave_bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]

# speed of sound + cutoff from config; dt from the solver's output/header.txt
cfg = np.loadtxt("../config/config.txt")
f_cutoff_sim = cfg[2]

hp = open("../output/header.txt").read().split()   # Nx Ny Nz Nt Lx Ly Lz pbin anim dt scene h tol/h
dt = float(hp[9])

dt_half = 0.5 * dt
fs_fv   = 1 / dt
fs      = 44100                              # Hz — output sample rate
f_cutoff = f_cutoff_sim                       # Hz — low/high crossover

# PCS source lag (samples at fs)
sigma_s = 0.425 / f_cutoff_sim
ts_ = np.ceil(8.5 * sigma_s / dt_half)
lag = int(round(ts_ * dt_half * fs))

# zero-phase Butterworth split at the crossover
sos_lp = signal.butter(8, f_cutoff, "low",  fs=fs, output="sos")
sos_hp = signal.butter(8, f_cutoff, "high", fs=fs, output="sos")


def broadband_ir(rec, src):
    """Broadband IR for one (rec, src) pair from the IR/HIR bins in cwd. Returns array at fs,
    or None if the low-freq IR for the pair is missing. Solver names them IR<rec>_<src>.bin."""
    low_files = sorted(f for f in glob.glob("IR*.bin")     # IR<rec>_<src>.bin
                       if f[2:].split(".")[0].split("_")[:2] == [str(rec), str(src)])
    high_files = sorted(f for f in glob.glob("HIR*.bin")   # HIR<rec>_<src>_<band>.bin
                        if f[3:].split(".")[0].split("_")[:2] == [str(rec), str(src)])
    if not low_files:
        return None

    # low band: resample FVM rate -> fs, remove the source-pulse lag
    LIR = np.fromfile(low_files[0], np.float32)
    r = Fraction(fs / fs_fv).limit_denominator(2000)
    LIR = resample_poly(LIR, r.numerator, r.denominator)   # anti-aliased
    LIR = np.roll(LIR, -lag)

    # high bands: cosine octave bandpass per band, then sum
    H_IRs = []
    for hf in high_files:
        arr = np.fromfile(hf, np.float32)
        N = len(arr)
        f = np.fft.rfftfreq(N, 1 / fs)
        fc = int(hf.split("_")[-1].split(".")[0])          # band from HIR..._<band>.bin
        lo, hi = fc / 2**0.5, fc * 2**0.5                  # octave edges
        T = 2**(1 / 6)                                      # transition half-width
        xlo = np.clip((f - lo / T) / (lo * T - lo / T), 0, 1)
        w_hp = 0.5 * (1 - np.cos(np.pi * xlo))
        if fc == octave_bands[-1]:
            w_lp = np.ones_like(f)                          # top band runs to Nyquist
        else:
            xhi = np.clip((f - hi / T) / (hi * T - hi / T), 0, 1)
            w_lp = 0.5 * (1 + np.cos(np.pi * xhi))
        R = np.fft.rfft(arr)
        H_IRs.append(np.fft.irfft(R * w_hp * w_lp, n=N))

    LIR = signal.sosfiltfilt(sos_lp, LIR)
    if H_IRs:
        HIR = signal.sosfiltfilt(sos_hp, np.sum(H_IRs, axis=0))
    else:
        HIR = np.zeros(0)
    N = max(LIR.size, HIR.size)
    return np.pad(LIR, (0, N - LIR.size)) + np.pad(HIR, (0, N - HIR.size))


def discover_pairs():
    """Every (rec, src) with a low-freq IR<rec>_<src>.bin present, sorted."""
    pairs = set()
    for f in glob.glob("IR*.bin"):
        m = re.match(r"IR(\d+)_(\d+)\.bin$", f)
        if m:
            pairs.add((int(m.group(1)), int(m.group(2))))   # (rec, src)
    return sorted(pairs)


def main():
    # clear stale outputs so a removed/renamed pair can't leave a mislabeled file behind
    for old in glob.glob("IR_broad_*.npy") + glob.glob("IR_rec*.png"):
        try: os.remove(old)
        except OSError: pass

    pairs = discover_pairs()
    if not pairs:
        print("[allIRs] no IR<rec>_<src>.bin found in", os.getcwd()); return

    by_rec = {}
    for rec, src in pairs:
        ir = broadband_ir(rec, src)
        if ir is None:
            continue
        np.save(f"IR_broad_{rec}_{src}", ir)               # per-pair broadband IR (for the GUI)
        by_rec.setdefault(rec, []).append((src, ir))

    # one figure per receiver: IR (top) + frequency response (bottom), a line per source
    for rec, items in sorted(by_rec.items()):
        fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(9, 7))
        for src, ir in sorted(items):
            t = np.arange(ir.size) / fs
            ax_t.plot(t, ir, lw=0.7, label=f"source {src}")
            f = np.fft.rfftfreq(ir.size, 1 / fs)
            mag = 20 * np.log10(np.abs(np.fft.rfft(ir)) + 1e-12)
            ax_f.semilogx(f[1:], mag[1:], lw=0.7, label=f"source {src}")
        ax_t.set_title(f"Receiver {rec} — Impulse Response")
        ax_t.set_xlabel("Time [s]"); ax_t.set_ylabel("Amplitude")
        ax_t.grid(True); ax_t.legend()
        ax_f.set_title(f"Receiver {rec} — Frequency Response")
        ax_f.set_xlabel("Frequency [Hz]"); ax_f.set_ylabel("Magnitude [dB]")
        ax_f.set_xlim(20, fs / 2); ax_f.grid(True, which="both"); ax_f.legend()
        fig.tight_layout()
        fig.savefig(f"IR_rec{rec}.png", dpi=120)
        plt.close(fig)
    print(f"[allIRs] {len(pairs)} pair(s), {len(by_rec)} receiver plot(s) written.")


if __name__ == "__main__":
    main()
