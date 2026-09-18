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
pulse_xover_hint = None

def _load_pulse():
    if not os.path.exists("../config/p_ref.bin"):
        return None
    q = np.fromfile("../config/p_ref.bin", np.float64)
    rr = Fraction(fs * dt / 2).limit_denominator(2000)     # 2*fs_fv -> fs
    return resample_poly(q, rr.numerator, rr.denominator)

def _init():
    """Read config/header/pulse once (cwd must be the project's output/). Populates module globals.
    Called at the top of broadband_ir()/main() so importing the module touches no files."""
    global _ready, cfg, f_cutoff_sim, src_opt, dt, dt_half, fs_fv, f_cutoff, pulse_fs, sigma_s, lag, pulse_xover_hint
    if _ready:
        return
    # speed of sound + cutoff from config; dt from the solver's output/header.txt
    cfg = np.loadtxt("../config/config.txt")
    f_cutoff_sim = cfg[2]
    src_opt = int(cfg[15]) if len(cfg) > 15 else 0   # config line 16: 0=PCS, 1=Ricker, 2=Sinusoid
    # header.txt (dt) is written only when the FVM solver finishes. A GA-only run (f_cutoff_sim below
    # the FVM band, e.g. < 63 Hz) never runs FVM -> no header, no low IR. dt/fs_fv feed only the (absent)
    # low band, so tolerate the miss and let broadband_ir build from the GA octaves alone.
    if os.path.exists("../output/header.txt"):
        hp = open("../output/header.txt").read().split()   # Nx Ny Nz Nt Lx Ly Lz pbin anim dt scene h tol/h
        dt = float(hp[9])
        dt_half = 0.5 * dt
        fs_fv = 1 / dt
    else:
        dt = dt_half = fs_fv = None
        print("[allIRs] no ../output/header.txt (FVM did not run) -> GA-only: building from GA bands.")
    # FVM<->GA crossover f_x. Fallback = 0.5*f_cutoff (PCS Gaussian roll-off); overridden below from the
    # actual source-pulse spectrum so it's also right for Ricker (bandpass) and any custom glob_f.
    f_cutoff = 0.5 * f_cutoff_sim                 # Hz (grid ceiling is f_cutoff_sim)
    # main3D writes the radiated pulse to ../config/p_ref.bin; deconvolving it recovers the bare room
    # IR so the FVM and GA bands share density and meet at f_cutoff. float64 at dt/2 -> resample to fs.
    pulse_fs = _load_pulse() if dt is not None else None
    if pulse_fs is None:
        if dt is not None:               # only warn about a missing pulse when there IS a low band
            print("[allIRs] !! ../config/p_ref.bin NOT FOUND -> source NOT deconvolved (fallback path). "
                  "The FVM band still carries the source pulse; bands will NOT match. Re-run the FVM "
                  "simulation with the current main3D to generate p_ref.bin.")
    else:
        print(f"[allIRs] p_ref.bin loaded ({pulse_fs.size} samples at fs) -> deconvolving source pulse.")
        _P = np.abs(np.fft.rfft(pulse_fs)); _f = np.fft.rfftfreq(pulse_fs.size, 1 / fs)
        _pk = int(_P.argmax()); _thr = _P[_pk] * 10 ** (-XOVER_DROP_DB / 20)
        _above = np.where((_f > _f[_pk]) & (_P >= _thr))[0]
        _fx = float(_f[_above[-1]]) if _above.size else float(_f[_pk])
        # DIAGNOSTIC ONLY. This used to OVERRIDE the crossover, which decoupled it from main3D:
        # main3D emits GA bands with centre >= 0.5*f_cutoff (f_cutoff=512 -> 256 -> lowest GA band
        # 500, high-passed at 353 Hz), while this put the FVM low-pass at 203.5 Hz -> a HOLE from
        # 203 to 353 Hz that no band covered, and a match window half inside it. The crossover
        # is now taken from the GA bands actually present (broadband_ir); the pulse only tells you
        # how well the FVM is excited up to it.
        pulse_xover_hint = _fx
        print(f"[allIRs] pulse peak {_f[_pk]:.1f} Hz, {XOVER_DROP_DB:.0f} dB down by {_fx:.1f} Hz "
              f"(nominal crossover 0.5*f_cutoff = {f_cutoff:.1f} Hz)")
    # Fallback only (no p_ref.bin from an older run): source-pulse lag instead of deconv. Needs dt,
    # so skip it for a GA-only run (no low band to lag-align anyway).
    if dt is not None:
        if src_opt == 1:                              # Ricker wavelet
            sigma_s = 1.0 / (np.sqrt(2) * np.pi * 0.3 * f_cutoff_sim)
        else:                                         # PCS (0) — Sinusoid (2) has no pulse lag
            sigma_s = 0.425 / f_cutoff_sim
        lag = int(round(np.ceil(8.5 * sigma_s / dt_half) * dt_half * fs))
    else:
        sigma_s = lag = None
    _ready = True

def low_deconv(raw):
    """FVM-rate receiver trace -> bare room response at fs: resample, then deconvolve the source pulse.
    Same operator for pressure (IR) and the velocity channels (VEL), so they stay phase-aligned."""
    r = Fraction(fs / fs_fv).limit_denominator(2000)
    x = resample_poly(np.asarray(raw, float), r.numerator, r.denominator)   # anti-aliased
    if pulse_fs is not None:
        n = x.size
        Q = np.fft.rfft(pulse_fs, n=n)
        eps = deconv_reg * np.max(np.abs(Q) ** 2)              # Tikhonov: caps gain where pulse is weak
        return np.fft.irfft(np.fft.rfft(x) * np.conj(Q) / (np.abs(Q) ** 2 + eps), n=n)
    return np.roll(x * (fs_fv / fs), -lag)                     # older run without p_ref.bin


# ONE power-complementary filterbank: FVM is the band below f_cutoff, each GA octave a band
# above it. Every input is filtered EXACTLY ONCE (the lowest GA band high-passes at f_cutoff and
# fuses with the FVM low-pass), so the GA is never double-filtered at the crossover. sqrt() of the
# amplitude-complementary cosine -> sum(w^2)=1 at every edge, so incoherent bands add flat in power.
_T = 2 ** (1 / 6)                                           # cosine transition half-width
def _shoulder(f, e, hi):                                    # hi=True -> high-pass shoulder at edge e
    x = np.clip((f - e / _T) / (e * _T - e / _T), 0, 1)
    return 0.5 * (1 - np.cos(np.pi * x)) if hi else 0.5 * (1 + np.cos(np.pi * x))


def _filt(ir, hp_e, lp_e):
    f = np.fft.rfftfreq(ir.size, 1 / fs)
    w = np.ones_like(f)
    if hp_e: w *= _shoulder(f, hp_e, True)
    if lp_e: w *= _shoulder(f, lp_e, False)
    return np.fft.irfft(np.fft.rfft(ir) * np.sqrt(w), n=ir.size)


def broadband_ir(rec, src):
    """Broadband IR for one (rec, src) pair from the IR/HIR bins in cwd. Returns (broadband, low, high)
    arrays at fs, or None only if BOTH the low IR and the GA bands are missing. Solver names them
    IR<rec>_<src>.bin (low) and HIR<rec>_<src>_<band>.bin (high). A GA-only run (f_cutoff_sim below the
    FVM band, e.g. < 63 Hz) writes NO low IR -> the low stream comes back as zeros and the GA octaves
    alone form the broadband IR."""
    _init()
    low_files = sorted(f for f in glob.glob("IR*.bin")     # IR<rec>_<src>.bin
                       if f[2:].split(".")[0].split("_")[:2] == [str(rec), str(src)])
    high_files = sorted(f for f in glob.glob("HIR*.bin")   # HIR<rec>_<src>_<band>.bin
                        if f[3:].split(".")[0].split("_")[:2] == [str(rec), str(src)])
    if not low_files and not high_files:
        return None                                        # nothing for this pair

    # GA-only run: no FVM low band -> LIR stays empty (pads to zeros below) and the lowest present
    # octave runs to DC instead of high-passing at f_cutoff (nothing lives below it). The low band needs
    # dt/fs_fv from header.txt to resample+deconvolve; if the header is gone (FVM never finished) we
    # CAN'T use any stray IR*.bin, so treat the pair as GA-only regardless.
    have_low = bool(low_files) and fs_fv is not None
    # f_cutoff_sim >= 63 Hz means the user asked for a WAVE-based (FVM) low band. If it's genuinely
    # missing here the FVM run is incomplete -> warn LOUD so a broken run isn't silently downgraded to
    # GA-only. Below 63 Hz there IS no FVM band by design, so GA-only is expected and quiet.
    wave_expected = f_cutoff_sim >= octave_bands[0]
    if wave_expected and not have_low:
        why = "header.txt missing (FVM run not finished)" if low_files else "no IR<rec>_<src>.bin (low)"
        print(f"[allIRs] !! rec{rec} src{src}: f_cutoff={f_cutoff_sim:.0f} Hz expects a WAVE-based low "
              f"band but {why} -> FVM output incomplete; falling back to GA-only. Re-run the FVM sim "
              "for the wave band.")
    elif low_files and fs_fv is None:      # GA-only run with a stray old low IR + no header
        print(f"[allIRs] rec{rec} src{src}: IR (low) present but no header.txt (dt unknown) -> "
              "low band skipped, building from GA bands only.")
    LIR = low_deconv(np.fromfile(low_files[0], np.float32)) if have_low else np.zeros(0)

    present_fcs = [int(hf.split("_")[-1].split(".")[0]) for hf in high_files]
    lowest_fc = min(present_fcs) if present_fcs else None
    # THE crossover: where the lowest GA band main3D actually wrote begins (its lower edge). The
    # FVM owns everything below it -- including, at f_cutoff=512, the whole 250 Hz octave up to
    # 353 Hz, by design. Derived from the files, so FVM and GA meet exactly and no hole can open
    # whatever the pulse spectrum did. The pulse hint says how well the FVM is excited there.
    fx = (lowest_fc / 2 ** 0.5) if (have_low and lowest_fc) else f_cutoff
    globals()["f_cutoff"] = fx                             # the receiver plots mark the crossover from it
    if have_low and lowest_fc:
        hint = pulse_xover_hint
        note = (f"; pulse is {XOVER_DROP_DB:.0f} dB down by {hint:.0f} Hz -> FVM weakly excited "
                f"in {hint:.0f}-{fx:.0f} Hz" if (hint is not None and hint < fx) else "")
        print(f"[allIRs] rec{rec} src{src}: crossover f_x = {fx:.1f} Hz (lower edge of GA band {lowest_fc}){note}")
    if have_low:
        LIR = _filt(LIR, None, fx)                         # FVM: single low-pass at the crossover
    H_IRs = []
    for hf in high_files:
        fc = int(hf.split("_")[-1].split(".")[0])          # band from HIR..._<band>.bin
        if have_low and fc * 2 ** 0.5 <= fx:               # octave fully below crossover -> FVM has it
            continue
        arr = np.fromfile(hf, np.float32)
        if not have_low and fc == lowest_fc:               # GA-only: lowest octave runs to DC (no FVM below)
            hp_e = None
        else:
            hp_e = max(fx, fc / 2 ** 0.5)                  # lowest kept band high-passes AT the crossover
        lp_e = None if fc == octave_bands[-1] else fc * 2 ** 0.5   # top band runs to Nyquist
        H_IRs.append(_filt(arr, hp_e, lp_e))

    HIR = np.sum(H_IRs, axis=0) if H_IRs else np.zeros(0)
    N = max(LIR.size, HIR.size)
    LIRp = np.pad(LIR, (0, N - LIR.size))
    HIRp = np.pad(HIR, (0, N - HIR.size))
    # Match GA to FVM by SPECTRAL DENSITY across the crossover (peak-matching a delta to a
    # band-limited pulse leaves a ~bandwidth-ratio offset, ~40 dB here). Compare FVM just below
    # f_cutoff to GA just above; median over a wide octave is robust (no narrow-band blow-up).
    if have_low and LIRp.any() and HIRp.any():             # nothing to match against in a GA-only run
        f = np.fft.rfftfreq(N, 1 / fs)
        lo = (f >= fx / 2) & (f < fx)                      # FVM strong here
        hi = (f > fx) & (f <= fx * 2)                      # GA strong here
        L = np.median(np.abs(np.fft.rfft(LIRp))[lo])
        H = np.median(np.abs(np.fft.rfft(HIRp))[hi])
        # Where does the match factor come from? Print every term so a +300 dB match can be
        # attributed to the raw FVM level, the deconvolution gain, or a dead GA window.
        _Q = np.abs(np.fft.rfft(pulse_fs, n=N)) if pulse_fs is not None else None
        _rawt = np.fromfile(low_files[0], np.float32)
        _raw = np.abs(np.fft.rfft(_rawt))
        _fr = np.fft.rfftfreq(_rawt.size, 1 / fs_fv); _lo_raw = (_fr >= fx / 2) & (_fr < fx)
        print(f"[allIRs] rec{rec} src{src}: match terms  L(FVM deconv, {fx/2:.0f}-{fx:.0f} Hz)={L:.3e}"
              f"  H(GA, {fx:.0f}-{2*fx:.0f} Hz)={H:.3e}"
              f"  raw FVM in window={np.median(_raw[_lo_raw]) if _lo_raw.any() else float('nan'):.3e}"
              + (f"  |Q| max={_Q.max():.3e} median in window={np.median(_Q[lo]):.3e}" if _Q is not None else "  (no pulse)")
              + f"  max|LIR|={np.abs(LIRp).max():.3e}  max|HIR|={np.abs(HIRp).max():.3e}")
        if H > 0:
            HIRp = HIRp * (L / H)
            print(f"[allIRs] rec{rec} src{src}: GA->FVM spectral match {20*np.log10(L/H):+.1f} dB")
    return LIRp + HIRp, LIRp, HIRp             # broadband, low-only, high-only (all length N)


def discover_pairs():
    """Every (rec, src) with a low IR<rec>_<src>.bin OR a GA band HIR<rec>_<src>_<band>.bin present,
    sorted. GA-only runs (no low IR) are still discovered so their broadband IR gets built."""
    pairs = set()
    for f in glob.glob("IR*.bin"):
        m = re.match(r"IR(\d+)_(\d+)\.bin$", f)
        if m:
            pairs.add((int(m.group(1)), int(m.group(2))))   # (rec, src)
    for f in glob.glob("HIR*.bin"):
        m = re.match(r"HIR(\d+)_(\d+)_\d+\.bin$", f)
        if m:
            pairs.add((int(m.group(1)), int(m.group(2))))
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
        return f[1:], 20 * np.log10(np.abs(np.fft.rfft(x))[1:] / 2e-5 + 1e-12)   # dB re 20 uPa (IR in Pa per unit impulse)

    # one figure per receiver: a column per source, IR (top) + frequency response (bottom)
    for rec, items in sorted(by_rec.items()):
      try:
        items = sorted(items)
        fig, axes = plt.subplots(2, len(items), figsize=(6 + 3 * len(items), 7), squeeze=False)
        for c, (src, ir, _, _) in enumerate(items):
            ax_t, ax_f = axes[0, c], axes[1, c]
            ax_t.plot(np.arange(ir.size) / fs, ir, lw=0.7)
            f, mag = _fr(ir)
            ax_f.semilogx(f, mag, lw=0.7)
            ax_t.set_title(f"Receiver {rec}, source {src} — Impulse Response")
            ax_t.set_xlabel("Time [s]"); ax_t.set_ylabel("Amplitude"); ax_t.grid(True)
            ax_f.set_title(f"Receiver {rec}, source {src} — Frequency Response")
            ax_f.set_xlabel("Frequency [Hz]"); ax_f.set_ylabel("SPL [dB re 20 µPa]")
            ax_f.set_xlim(20, fs / 2); ax_f.grid(True, which="both")
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
        axc.set_xlabel("Frequency [Hz]"); axc.set_ylabel("SPL [dB re 20 µPa]")
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


def _selfcheck_ga_only():
    # GA-only run (f_cutoff_sim < 63 Hz -> no low IR AND no header.txt, since FVM never ran):
    # _init must tolerate the missing header and broadband_ir must build from the HIR bands alone,
    # returning zeros for the low stream instead of None / crashing.
    import tempfile, shutil
    global _ready
    rng = np.random.default_rng(0)
    cwd0 = os.getcwd()
    d = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(d, "config")); os.makedirs(os.path.join(d, "output"))
        np.savetxt(os.path.join(d, "config", "config.txt"),
                   np.array([0, 0, 50.0] + [0] * 13, float))          # cfg[2]=f_cutoff_sim=50 (<63)
        os.chdir(os.path.join(d, "output"))                           # NO header.txt
        for band in (500, 1000):                                      # two GA octaves only
            (rng.standard_normal(2000).astype(np.float32)).tofile(f"HIR0_0_{band}.bin")
        _ready = False
        res = broadband_ir(0, 0)
        assert res is not None, "GA-only pair dropped (should build from HIR alone)"
        bb, LIRp, HIRp = res
        assert not LIRp.any() and HIRp.any(), "GA-only: low stream zero, high stream carries the IR"
        assert np.allclose(bb, HIRp), "GA-only broadband == high band"
        assert (0, 0) in discover_pairs(), "GA-only pair not discovered"
        # stray low IR from an old FVM run + no header (dt unknown) -> must still fall back to GA-only,
        # not crash on fs_fv=None (the exact failure the user hit).
        (rng.standard_normal(2000).astype(np.float32)).tofile("IR0_0.bin")
        _ready = False
        bb2, LIRp2, HIRp2 = broadband_ir(0, 0)
        assert not LIRp2.any() and HIRp2.any(), "stray IR + no header -> low skipped, GA used"
        print("[allIRs] GA-only self-check OK")
    finally:
        os.chdir(cwd0); _ready = False; shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    for _chk in (_selfcheck, _selfcheck_ga_only):    # keep independent: a deconv flake can't mask GA-only
        try:
            _chk()
        except Exception as ex:             # never let a self-check block the actual plots
            print(f"[allIRs] self-check warning: {ex}")
    main()
