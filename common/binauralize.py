#!/usr/bin/env python3
"""Binaural (HRTF) auralization of the hybrid sim -> a stereo BRIR per (receiver, source).

HRTF applies per direction-of-arrival (DOA) in the RECEIVER'S HEAD FRAME. Head orientation is a
render-time parameter (a rotation), NOT baked into any solve -- so head motion is free and the
expensive field solve ran once. Three arrival classes, three treatments (see memory
hrtf-auralization-design):

  * FVM low band (LIRp, < crossover): one DOA = the direct receiver->source dir. Below the crossover
    the binaural cue is basically ITD, so convolving the whole low IR with the direct-DOA HRIR pair is
    faithful. ponytail: single direction over-localizes low reverb to the source; DirAC per-sample DOA
    from the exported VEL (p*v intensity) is the named upgrade.
  * ISM early reflections (ARR file, <= t_cross): discrete arrivals, one HRIR per image-source DOA,
    octave-gain-shaped, placed at its delay. The spatial meat.
  * Diffuse late tail (HIRp beyond the last arrival): isotropic -> decorrelated L/R (freq-dependent
    interaural coherence), NOT HRTF'd per ray.

Reuses allIRs.broadband_ir for the calibrated LIRp/HIRp (deconvolution + FVM<->GA crossover + spectral
match) so the binaural streams band-split IDENTICALLY to the mono broadband IR. Only direction is added.

Run from the project's output/ dir (where IR*/HIR*/ARR*/VEL* and allIRs.py's inputs live):
    python binauralize.py <hrtf.sofa> [az el roll]      # deg, per-receiver head pose (default 0,0,0)
Self-check (no SOFA needed):  python binauralize.py
"""
import glob, os, re, sys
import numpy as np
from scipy.signal import fftconvolve

OCTAVE_BANDS = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], float)
FS = 44100
C = 343.0
COH_FREQ = 700.0      # Hz: diffuse field stays interaurally coherent below this, decorrelated above
KLEN = 128            # taps of the per-arrival octave-gain kernel (linear phase, group delay KLEN/2)


# ---------------------------------------------------------------- head frame -------------------------
def head_basis(az, el, roll):
    """Head pose (radians) -> orthonormal (forward, left, up) in world coords. az=el=roll=0 ->
    forward=+x, left=+y, up=+z. az = yaw about world +z, el = pitch up, roll about forward axis."""
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    F = np.array([ce * ca, ce * sa, se])                       # nose direction
    L0 = np.cross([0, 0, 1.0], F)                              # world-up x forward -> points left
    n = np.linalg.norm(L0)
    L0 = L0 / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])     # F vertical -> pick +y
    U0 = np.cross(F, L0)
    cr, sr = np.cos(roll), np.sin(roll)                        # roll about F
    L = cr * L0 - sr * U0
    U = sr * L0 + cr * U0
    return F, L, U


def doa_to_azel(basis, d_world):
    """World DOA (receiver->source unit vec) -> (az, el) in the head frame, SOFA convention:
    az CCW from front (0=front, +90=left), el up positive. Radians."""
    F, L, U = basis
    d = np.asarray(d_world, float)
    nn = np.linalg.norm(d)
    if nn < 1e-12:
        return 0.0, 0.0
    d = d / nn
    xl, yl, zl = float(d @ F), float(d @ L), float(d @ U)
    return np.arctan2(yl, xl), np.arcsin(np.clip(zl, -1, 1))


# ---------------------------------------------------------------- HRTF set ---------------------------
def load_hrtf(sofa_path):
    """Read a SOFA HRIR set with pyfar -> {'az','el' (rad, N), 'hrir' (N,2,taps), 'fs'}. Resampled to
    FS if needed. az/el derived from the source cartesian coords (version-proof)."""
    import pyfar as pf
    try:
        audio, src, _ = pf.io.read_sofa(sofa_path)
    except ValueError:                                            # non-standard custom entries (e.g. KEMAR)
        audio, src, _ = pf.io.read_sofa(sofa_path, verify=False)
    xyz = src.cartesian if hasattr(src, "cartesian") else src.get_cart()
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    el = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))
    hrir = np.asarray(audio.time, float)
    hrir = hrir.reshape(hrir.shape[0], 2, hrir.shape[-1])      # (N,2,taps)
    fs = float(audio.sampling_rate)
    if abs(fs - FS) > 1:
        from scipy.signal import resample_poly
        from fractions import Fraction
        r = Fraction(int(FS), int(round(fs))).limit_denominator(1000)
        hrir = resample_poly(hrir, r.numerator, r.denominator, axis=-1)
    return {"az": az, "el": el, "hrir": hrir, "fs": FS,
            "unit": np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)}


def lookup_hrir(hrtf, az, el):
    """Nearest-DOA HRIR pair (hL, hR) by great-circle distance on the measurement grid."""
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    i = int(np.argmax(hrtf["unit"] @ d))                       # max dot = min angle
    return hrtf["hrir"][i, 0], hrtf["hrir"][i, 1]


# ---------------------------------------------------------------- octave-gain kernel -----------------
def _band_weights(freqs):
    """Power-complementary sqrt-cosine octave weights w_b(f), (n_bands, len(freqs)). SUM w^2 ~ 1.
    Same transition shape as allIRs' filterbank."""
    T = 2 ** (1 / 6)

    def shoulder(f, e, hi):
        x = np.clip((f - e / T) / (e * T - e / T), 0, 1)
        return 0.5 * (1 - np.cos(np.pi * x)) if hi else 0.5 * (1 + np.cos(np.pi * x))

    W = []
    for b, fc in enumerate(OCTAVE_BANDS):
        w = np.ones_like(freqs)
        if b > 0:
            w *= shoulder(freqs, fc / np.sqrt(2), True)        # high-pass at lower edge
        if b < len(OCTAVE_BANDS) - 1:
            w *= shoulder(freqs, fc * np.sqrt(2), False)       # low-pass at upper edge
        W.append(w)
    return np.array(W)


def arrival_kernel(g8, band_w):
    """Per-arrival linear-phase FIR (KLEN taps) whose octave magnitude = the 8 band gains g8.
    |H(f)| = sum_b g_b * w_b(f)^2  (flat g -> flat). Group delay KLEN/2, compensated at placement."""
    env = (np.asarray(g8, float)[:, None] * band_w ** 2).sum(0)   # (nfreq,)
    k = np.fft.irfft(env, n=KLEN)
    return np.fft.fftshift(k)                                     # zero-phase -> centered symmetric


# ---------------------------------------------------------------- diffuse decorrelation --------------
def decorrelate(x, seed):
    """Randomize phase above COH_FREQ (keep magnitude), coherent below -> an interaurally incoherent
    copy for the diffuse tail without coloring it. Below COH_FREQ ears stay coherent (long waves)."""
    n = x.size
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / FS)
    rng = np.random.default_rng(seed)
    ph = np.exp(1j * rng.uniform(-np.pi, np.pi, X.size))
    ph[f < COH_FREQ] = 1.0                                        # coherent low band
    ph[0] = 1.0                                                   # DC must stay real for irfft...
    if n % 2 == 0:
        ph[-1] = 1.0                                              # ...and the Nyquist bin when n is even
    return np.fft.irfft(X * ph, n=n)


# ---------------------------------------------------------------- arrivals I/O -----------------------
def read_arrivals(path):
    """ARR<rec>_<src>.bin -> (delay_s (n,), doa_world (n,3), g (n,8)). int32 count then 12 floats each."""
    with open(path, "rb") as f:
        n = int(np.fromfile(f, np.int32, 1)[0])
        data = np.fromfile(f, np.float32, n * 12).reshape(n, 12) if n else np.zeros((0, 12), np.float32)
    return data[:, 0], data[:, 1:4], data[:, 4:12]


# ---------------------------------------------------------------- BRIR assembly -----------------------
def _place(dst, ir, start):
    """Add ir into dst at sample `start` (clipped)."""
    a = max(0, start)
    b = min(dst.size, start + ir.size)
    if b > a:
        dst[a:b] += ir[a - start:b - start]


def brir(LIRp, HIRp, arrivals, hrtf, head_pose, band_w=None):
    """Stereo BRIR (2, N) from the calibrated mono streams + ISM arrivals + head pose (az,el,roll rad).
    LIRp = FVM low band, HIRp = GA high band (early+diffuse mixed), arrivals = (delay_s, doa, g8)."""
    basis = head_basis(*head_pose)
    band_w = _band_weights(np.fft.rfftfreq(KLEN, 1 / FS)) if band_w is None else band_w
    delay, doa, g = arrivals
    N = max(LIRp.size, HIRp.size) + 512
    bl, br = np.zeros(N), np.zeros(N)

    # --- FVM low band: convolve with the direct-DOA HRIR (earliest arrival = direct) ---
    if delay.size:
        d_dir = doa[int(np.argmin(delay))]
    else:                                                          # no ISM at all -> straight ahead
        d_dir = basis[0]
    az0, el0 = doa_to_azel(basis, d_dir)
    hL, hR = lookup_hrir(hrtf, az0, el0)
    _place(bl, fftconvolve(LIRp, hL), 0)
    _place(br, fftconvolve(LIRp, hR), 0)

    # --- ISM early reflections: per-arrival HRIR, octave-shaped, placed at its delay ---
    early_l, early_r = np.zeros(N), np.zeros(N)
    for k in range(delay.size):
        az, el = doa_to_azel(basis, doa[k])
        hL, hR = lookup_hrir(hrtf, az, el)
        ker = arrival_kernel(g[k], band_w)
        s = int(round(delay[k] * FS)) - KLEN // 2                 # compensate kernel group delay
        _place(early_l, fftconvolve(hL, ker), s)
        _place(early_r, fftconvolve(hR, ker), s)

    # calibrate the ARR-built early field to HIRp's early energy (ARR carries the RAW deposit gains;
    # allIRs re-scaled HIRp by its FVM<->GA spectral match -> one broadband scalar reconciles them).
    t_cross = int(round((delay.max() if delay.size else 0) * FS)) + KLEN
    e_arr = early_l[:t_cross] @ early_l[:t_cross] + early_r[:t_cross] @ early_r[:t_cross]
    hi_early = HIRp[:min(t_cross, HIRp.size)]
    e_hip = 2.0 * (hi_early @ hi_early)                           # mono energy split over two ears
    if e_arr > 0 and e_hip > 0:
        a = np.sqrt(e_hip / e_arr)
        early_l *= a
        early_r *= a
    bl += early_l
    br += early_r

    # --- diffuse late tail: HIRp beyond t_cross, decorrelated into two incoherent ears ---
    if HIRp.size > t_cross:
        tail = np.zeros(HIRp.size)
        tail[t_cross:] = HIRp[t_cross:]
        _place(bl, decorrelate(tail, 1), 0)
        _place(br, decorrelate(tail, 2), 0)

    return np.stack([bl, br])


def render(dry, brir_lr):
    """Convolve a mono dry signal with the stereo BRIR -> (2, M) binaural audio."""
    return np.stack([fftconvolve(dry, brir_lr[0]), fftconvolve(dry, brir_lr[1])])


# ---------------------------------------------------------------- one-receiver auralization ----------
def _read_audio(path):
    """Anechoic file -> mono float at FS (downmix + resample as needed)."""
    import soundfile as sf
    x, fs = sf.read(path)
    x = np.asarray(x, float)
    if x.ndim > 1:
        x = x.mean(1)
    if int(fs) != FS:
        from scipy.signal import resample_poly
        from fractions import Fraction
        r = Fraction(FS, int(fs)).limit_denominator(1000)
        x = resample_poly(x, r.numerator, r.denominator)
    return x


def _sources_for(rec):
    out = set()
    for f in glob.glob(f"ARR{rec}_*.bin"):
        m = re.match(rf"ARR{rec}_(\d+)\.bin$", os.path.basename(f))
        if m:
            out.add(int(m.group(1)))
    return sorted(out)


def auralize_receiver(rec, sofa, pose_deg, dry_path, out_wav):
    """Full binaural render for ONE receiver: SUM all its sources' BRIRs (convolution is linear, so one
    render on the summed BRIR == summing per-source renders, but far cheaper), convolve the dry anechoic
    mono file once, write a peak-normalized stereo wav. Run from the project's output/ dir."""
    import time
    import soundfile as sf
    try:
        import allIRs
        allIRs._init()                                         # reads config/header/pulse (cwd=output/)
    except FileNotFoundError as ex:
        raise SystemExit(f"[binauralize] missing solver output ({ex}). Re-run the sim to completion — "
                         "header.txt is deleted at startup and only rewritten at the END of a full run.")
    t0 = time.time()
    hrtf = load_hrtf(sofa)
    print(f"[binauralize] HRTF {hrtf['hrir'].shape} loaded ({time.time()-t0:.1f}s)", flush=True)
    pose = tuple(np.radians(pose_deg))
    dry = _read_audio(dry_path)
    print(f"[binauralize] dry {dry.size} samples ({dry.size/FS:.1f}s)", flush=True)
    B = None
    for src in _sources_for(rec):
        res = allIRs.broadband_ir(rec, src)
        if res is None:
            continue
        _, LIRp, HIRp = res
        arr = read_arrivals(f"ARR{rec}_{src}.bin")
        print(f"[binauralize]  src{src}: {arr[0].size} arrivals, building BRIR…", flush=True)
        b = brir(LIRp, HIRp, arr, hrtf, pose)
        if B is None:
            B = b
        else:                                                    # sum sources (zero-pad to the longer)
            n = max(B.shape[1], b.shape[1])
            B = np.pad(B, ((0, 0), (0, n - B.shape[1]))) + np.pad(b, ((0, 0), (0, n - b.shape[1])))
    if B is None:
        raise FileNotFoundError(f"no ARR/broadband data for receiver {rec} in {os.getcwd()}")
    print(f"[binauralize] BRIR {B.shape}, convolving dry signal…", flush=True)
    acc = render(dry, B)
    pk = np.abs(acc).max()
    if pk > 0:
        acc = 0.98 * acc / pk
    sf.write(out_wav, acc.T.astype(np.float32), FS)
    print(f"[binauralize] wrote {out_wav} ({time.time()-t0:.1f}s total)", flush=True)
    return out_wav


# ---------------------------------------------------------------- driver -----------------------------
def _pairs():
    out = set()
    for f in glob.glob("ARR*.bin"):
        m = re.match(r"ARR(\d+)_(\d+)\.bin$", f)
        if m:
            out.add((int(m.group(1)), int(m.group(2))))
    return sorted(out)


def main(sofa, pose_deg=(0, 0, 0)):
    import allIRs
    hrtf = load_hrtf(sofa)
    pose = tuple(np.radians(pose_deg))
    for rec, src in _pairs():
        res = allIRs.broadband_ir(rec, src)
        if res is None:
            print(f"[binauralize] rec{rec} src{src}: no low IR -> skip"); continue
        _, LIRp, HIRp = res
        arr = read_arrivals(f"ARR{rec}_{src}.bin")
        b = brir(LIRp, HIRp, arr, hrtf, pose)
        np.save(f"BRIR_{rec}_{src}.npy", b.astype(np.float32))
        print(f"[binauralize] rec{rec} src{src}: BRIR {b.shape} -> BRIR_{rec}_{src}.npy")


# ---------------------------------------------------------------- self-check -------------------------
def _fake_hrtf():
    """Synthetic HRIR grid: pure ITD + ILD (near ear earlier & louder). Enough to test the geometry,
    lookup, and BRIR assembly without a SOFA file."""
    az = np.radians(np.arange(0, 360, 10))
    el = np.zeros_like(az)
    taps = 64
    hrir = np.zeros((az.size, 2, taps))
    for i, a in enumerate(az):
        itd = int(round(8 * np.sin(a)))                          # +az (left) -> left ear earlier
        ild_l = 1.0 + 0.5 * np.sin(a)                            # left louder when source on left
        ild_r = 1.0 - 0.5 * np.sin(a)
        hrir[i, 0, max(0, 16 - itd)] = ild_l
        hrir[i, 1, max(0, 16 + itd)] = ild_r
    unit = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)
    return {"az": az, "el": el, "hrir": hrir, "fs": FS, "unit": unit}


def _selfcheck():
    rng = np.random.default_rng(0)
    # head-frame sanity: source dead ahead -> az=el=0; source to the left (+y) -> az=+90
    basis = head_basis(0, 0, 0)
    az, el = doa_to_azel(basis, [1, 0, 0]); assert abs(az) < 1e-6 and abs(el) < 1e-6, "front"
    az, el = doa_to_azel(basis, [0, 1, 0]); assert abs(az - np.pi / 2) < 1e-6, "left = +90"
    az, el = doa_to_azel(basis, [0, 0, 1]); assert abs(el - np.pi / 2) < 1e-6, "up = +90 el"
    # yaw 90deg: a source at world +y is now dead ahead
    b2 = head_basis(np.pi / 2, 0, 0)
    az, el = doa_to_azel(b2, [0, 1, 0]); assert abs(az) < 1e-6, "yaw follows head"

    # octave kernel: flat gains -> ~flat magnitude (allpass-ish), energy preserved
    bw = _band_weights(np.fft.rfftfreq(KLEN, 1 / FS))
    k = arrival_kernel(np.ones(8), bw); assert np.isfinite(k).all() and k.sum() > 0.5, "flat kernel"

    # decorrelation preserves magnitude spectrum, kills correlation above COH_FREQ
    x = rng.standard_normal(4096)
    a, b = decorrelate(x, 1), decorrelate(x, 2)
    assert np.allclose(np.abs(np.fft.rfft(a)), np.abs(np.fft.rfft(x)), atol=1e-6), "mag preserved"
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.5, "L/R decorrelated"

    # full BRIR: direct source hard LEFT -> left ear earlier and louder
    hrtf = _fake_hrtf()
    LIRp = np.zeros(2000); LIRp[10] = 1.0                        # low band: a direct spike
    HIRp = 0.01 * rng.standard_normal(4000) * np.exp(-np.arange(4000) / 800)  # decaying diffuse tail
    delay = np.array([0.01, 0.02, 0.03])                        # direct + 2 reflections (s)
    doa = np.array([[0, 1, 0.0], [1, 0, 0], [-1, 0, 0]])       # direct LEFT, others front/back
    g = np.ones((3, 8)); g[1:] *= 0.5
    B = brir(LIRp, HIRp, (delay, doa, g), hrtf, (0, 0, 0))
    assert B.shape[0] == 2 and np.isfinite(B).all(), "stereo finite"
    onL = np.argmax(np.abs(B[0]) > 1e-3); onR = np.argmax(np.abs(B[1]) > 1e-3)
    assert onL <= onR, "source on the left should reach the left ear first"
    assert np.abs(B[0]).max() >= np.abs(B[1]).max(), "left ear louder for a left source"

    # render shape
    y = render(rng.standard_normal(500), B); assert y.shape[0] == 2, "stereo render"
    print("binauralize self-check OK")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _selfcheck(); sys.exit()
    import argparse
    ap = argparse.ArgumentParser(description="Binaural auralization of the hybrid sim.")
    ap.add_argument("sofa", help="HRTF SOFA file")
    ap.add_argument("pose_pos", nargs="*", type=float, help="az el roll (deg) positional; default 0 0 0")
    ap.add_argument("--pose", nargs=3, type=float, metavar=("AZ", "EL", "ROLL"), help="az el roll (deg)")
    ap.add_argument("--rec", type=int, help="auralize ONE receiver to a wav (needs --dry --out)")
    ap.add_argument("--dry", help="anechoic mono audio in")
    ap.add_argument("--out", help="binaural wav out")
    a = ap.parse_args()
    pose = a.pose if a.pose is not None else (a.pose_pos + [0, 0, 0])[:3]
    if a.rec is not None:
        if not (a.dry and a.out):
            ap.error("--rec needs --dry AND --out (the --out path must live inside the project's output/)")
        # allIRs.py reads ../config and ../output RELATIVE to cwd, so run in the project's output/
        # (derived from --out). Lets manual runs work from anywhere; matches the GUI's cwd=output.
        a.sofa, a.dry, a.out = (os.path.abspath(a.sofa), os.path.abspath(a.dry), os.path.abspath(a.out))
        os.chdir(os.path.dirname(a.out))
        print(f"[binauralize] cwd -> {os.getcwd()}", flush=True)
        p = auralize_receiver(a.rec, a.sofa, pose, a.dry, a.out)
        print(f"[binauralize] receiver {a.rec} pose {pose} -> {p}")
    else:
        main(a.sofa, pose)                                       # batch: BRIR_*.npy for every pair
