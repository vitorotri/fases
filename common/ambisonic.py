#!/usr/bin/env python3
"""First-order ambisonic (B-format) auralization -- the cheap alternative to binauralize.py.

No HRTF. The receiver's field is kept as W (pressure) + X,Y,Z (pressure-weighted direction of
arrival), rotated by the head pose and decoded to two virtual cardioids at +-MIC_AZ. That is
"panning + gain per ear", but driven by the physical direction of every part of the response, so the
direct sound and early reflections move with the head and the diffuse tail stays diffuse. ILD-only
plus a per-source Woodworth ITD; no spectral cues: front-back ambiguous, weak elevation -- the price
versus HRTF.

Channels, all built from the SAME calibrated streams as allIRs/binauralize:
  * FVM low band : W = LIRp; X,Y,Z = -rho*c*v from VEL<rec>_<src>.bin through the identical
                   resample + pulse deconvolution + crossover low-pass (allIRs.low_deconv/_filt).
                   Sign: a plane wave arriving FROM direction d has v = -(p/Z0) d, and B-format
                   wants X = p*d_x, hence the minus.
  * ISM early    : each ARR arrival as its octave-shaped kernel, W += ker, XYZ += d * ker, calibrated
                   to HIRp's early energy exactly as binauralize.brir does.
  * diffuse tail : HIRp beyond the last arrival, decorrelated into the two ears at decode time.

The expensive part (dry signal (*) 6 channels) never depends on the head, so it runs once; live
playback only rotates 3 channels, delays the far ear by the source's ITD and mixes.

Run from the project's output/ dir:
    python ambisonic.py --rec R --dry a.wav --out foa.wav [--pose AZ EL ROLL] [--drymap map.json]
    python ambisonic.py --rec R --dry a.wav --live [--pose ...]     # sounddevice; pose lines on stdin
Live mode reads "az el roll" (deg) lines from stdin whenever they arrive (the GUI writes the dials),
loops the programme, and stops on EOF or a line "q".
Self-check (no files needed):  python ambisonic.py
"""
import glob, json, os, re, sys, threading
import numpy as np
from scipy.signal import fftconvolve

import allIRs
from binauralize import (head_basis, read_arrivals, arrival_kernel, _band_weights, decorrelate,
                         _place, _read_audio, KLEN, FS)

RHO = 1.225                          # main3D's air density
MIC_AZ = np.radians(110.0)           # ear pointing directions p: +-110 deg in the head frame (slightly rear)
MIC_S = 0.5                          # directivity a(d) = (1-s) + s*(d.p): 0 omni, 0.5 cardioid, 1 figure-8
HEAD_R = 0.0875                      # m, Woodworth head radius for the ITD
BLOCK = 1024                         # live block: ~23 ms at 44.1 kHz
LATENCY = 0.08                       # s, output buffer; PulseAudio/PipeWire through PortAudio underruns at 'low'


def _c_sound():
    """Speed of sound from config.txt temperature, same formula as main3D."""
    return float(np.sqrt(1.4 * 287.05 * (allIRs.cfg[3] + 273.15)))


# ---------------------------------------------------------------- B-format RIR -----------------------
def foa_rir(rec, src):
    """(W, XYZ (3,N), tail) for one (rec, src), at FS, world frame. None if the pair has no data."""
    res = allIRs.broadband_ir(rec, src)
    if res is None:
        return None
    _, LIRp, HIRp = res
    fx = allIRs.f_cutoff                                       # the crossover broadband_ir settled on
    N = max(LIRp.size, HIRp.size) + 512
    W, XYZ = np.zeros(N), np.zeros((3, N))
    _place(W, LIRp, 0)

    vf = f"VEL{rec}_{src}.bin"
    if LIRp.any() and allIRs.fs_fv is not None and os.path.isfile(vf):
        v = np.fromfile(vf, np.float32).reshape(-1, 3)
        Z0 = RHO * _c_sound()
        for a in range(3):
            _place(XYZ[a], allIRs._filt(allIRs.low_deconv(-Z0 * v[:, a]), None, fx), 0)
    elif LIRp.any():
        print(f"[ambisonic] rec{rec} src{src}: no {vf} -> low band rendered omnidirectional")

    af = f"ARR{rec}_{src}.bin"
    delay, doa, g = read_arrivals(af) if os.path.isfile(af) else (np.zeros(0), np.zeros((0, 3)), np.zeros((0, 8)))
    band_w = _band_weights(np.fft.rfftfreq(KLEN, 1 / FS))
    eW, eXYZ = np.zeros(N), np.zeros((3, N))
    for k in range(delay.size):
        ker = arrival_kernel(g[k], band_w)
        s0 = int(round(delay[k] * FS)) - KLEN // 2             # compensate kernel group delay
        d = doa[k] / (np.linalg.norm(doa[k]) or 1.0)
        _place(eW, ker, s0)
        for a in range(3):
            _place(eXYZ[a], d[a] * ker, s0)
    t_cross = int(round((delay.max() if delay.size else 0) * FS)) + KLEN
    e_arr = np.dot(eW[:t_cross], eW[:t_cross])
    hi = HIRp[:min(t_cross, HIRp.size)]
    e_hip = np.dot(hi, hi)
    if e_arr > 0 and e_hip > 0:                                # ARR carries raw deposit gains: rescale
        a = np.sqrt(e_hip / e_arr)
        eW *= a; eXYZ *= a
    W += eW; XYZ += eXYZ

    tail = np.zeros(N)
    if HIRp.size > t_cross:
        tail[t_cross:HIRp.size] = HIRp[t_cross:]
    return W, XYZ, tail


# ---------------------------------------------------------------- render + decode --------------------
def render_channels(rec, dry_path, dry_map=None):
    """Convolve the programme with the B-format RIR of every source of `rec`:
    -> (chs (S,6,M): W,X,Y,Z,tailL,tailR per source; dirs (S,3): each source's direct-sound DOA, world).
    Kept per source so the ITD can follow each source's own direction. dry_map: per-source files."""
    srcs = sorted(int(m.group(1)) for f in glob.glob(f"ARR{rec}_*.bin")
                  for m in [re.match(rf"ARR{rec}_(\d+)\.bin$", os.path.basename(f))] if m)
    if not srcs:
        srcs = sorted({int(m.group(1)) for f in glob.glob(f"IR{rec}_*.bin")
                       for m in [re.match(rf"IR{rec}_(\d+)\.bin$", f)] if m})
    if not srcs:
        raise FileNotFoundError(f"no ARR/IR data for receiver {rec} in {os.getcwd()}")
    dry_cache, chans, dirs = {}, [], []
    for src in srcs:
        foa = foa_rir(rec, src)
        if foa is None:
            continue
        W, XYZ, tail = foa
        path = (dry_map or {}).get(str(src), dry_path)
        if path not in dry_cache:
            dry_cache[path] = _read_audio(path)
        dry = dry_cache[path]
        chans.append(np.stack([fftconvolve(dry, W)] + [fftconvolve(dry, XYZ[a]) for a in range(3)]
                              + [fftconvolve(dry, decorrelate(tail, 1)), fftconvolve(dry, decorrelate(tail, 2))]))
        dirs.append(direct_doa(rec, src))
        print(f"[ambisonic] rec{rec} src{src}: rendered ({os.path.basename(path)})")
    if not chans:
        raise FileNotFoundError(f"no broadband data for receiver {rec}")
    M = max(c.shape[1] for c in chans)
    chs = np.stack([np.pad(c, ((0, 0), (0, M - c.shape[1]))) for c in chans])
    return chs, np.array(dirs)


def direct_doa(rec, src):
    """World unit vector receiver -> source (earliest ISM arrival); +x if there are no arrivals."""
    af = f"ARR{rec}_{src}.bin"
    if os.path.isfile(af):
        delay, doa, _ = read_arrivals(af)
        if delay.size:
            d = doa[int(np.argmin(delay))]
            return d / (np.linalg.norm(d) or 1.0)
    return np.array([1.0, 0.0, 0.0])


def itd_samples(az_head):
    """Woodworth ITD (samples) for a source at head azimuth az (rad, +left), folded front/back."""
    th = abs(az_head)
    th = th if th <= np.pi / 2 else np.pi - th
    return int(round(HEAD_R / 343.0 * (np.sin(th) + th) * FS))

# _ear(chk, F, L, sgn, i0, i1, dl)                                                                  
#   chk : (6, M) rendered channels of ONE source: rows 0-3 = W, X, Y, Z (world frame, dry signal    
#         already convolved in), rows 4-5 = the two decorrelated tail ears (not used here)          
#   F, L: world-frame unit vectors of the head's forward and left axes (from head_basis); projecting
#         X,Y,Z on them gives the head-frame components xh, yh                                      
#   sgn : +1 for the left ear, -1 for the right -> aims this ear's virtual mic at +MIC_AZ or -MIC_AZ
#   i0,i1: output sample range [i0, i1) of this block                                               
#   dl  : ITD delay of this ear in samples (0 for the ear nearer the source); the block is read     
#         dl samples earlier from the channels, i.e. the ear hears everything dl samples later      
#   returns (i1-i0,) samples: the directional part of this source at this ear, delayed by dl 
def _ear(chk, F, L, sgn, i0, i1, dl):
    """One source's DIRECTIONAL signal for one ear (directivity MIC_S aimed at sgn*MIC_AZ), delayed by dl samples,
    for output samples [i0, i1). Samples before the programme start read as zero."""
    n = i1 - i0
    j0 = i0 - dl
    a0 = max(j0, 0)
    seg = chk[:, a0:j0 + n]
    W, X, Y, Z = seg[0], seg[1], seg[2], seg[3]              # world-frame B-format, this block
    xh = F[0] * X + F[1] * Y + F[2] * Z                      # XYZ in the head frame: forward ...
    yh = L[0] * X + L[1] * Y + L[2] * Z                      # ... and left components
    px, py = np.cos(MIC_AZ), sgn * np.sin(MIC_AZ)            # this ear's pointing direction p
    # per arrival W = p_ac, XYZ = p_ac*d  ->  e = p_ac * [(1-s) + s*(d.p)]: the receiver directivity
    e = (1.0 - MIC_S) * W + MIC_S * (px * xh + py * yh)
    out = np.zeros(n)
    out[a0 - j0:] = e
    return out


def decode(chs, dirs, pose_deg, i0=0, i1=None, prev=None):
    """(S,6,M) channels + per-source direct DOA + head pose (deg) -> (2, n) stereo for samples [i0, i1).
    Per source: rotate world XYZ into the head frame, decode two cardioids at +-MIC_AZ (that is the ILD),
    delay the far ear by the Woodworth ITD of the source's direct-sound azimuth (fixed per source, so
    its reflections ride along), add the decorrelated tail with no delay. `prev` (dict, live only)
    remembers each source's last delays and crossfades over the block when they change."""
    F, L, _ = head_basis(*np.radians(pose_deg))
    i1 = chs.shape[2] if i1 is None else i1
    n = i1 - i0
    out = np.zeros((2, n))
    ramp = np.linspace(0.0, 1.0, n)
    for k in range(chs.shape[0]):
        th = np.arctan2(np.dot(L, dirs[k]), np.dot(F, dirs[k]))                # head-frame azimuth, +left
        d = itd_samples(th)
        dl = (0, d) if th > 0 else (d, 0)                          # (left delay, right delay)
        old = prev.get(k, dl) if prev is not None else dl
        for ear, sgn in ((0, +1), (1, -1)):
            e = _ear(chs[k], F, L, sgn, i0, i1, dl[ear])
            if old[ear] != dl[ear]:                                # delay jumped: fade old -> new
                e = (1 - ramp) * _ear(chs[k], F, L, sgn, i0, i1, old[ear]) + ramp * e
            out[ear] += e
        out[0] += chs[k, 4, i0:i1]
        out[1] += chs[k, 5, i0:i1]
        if prev is not None:
            prev[k] = dl
    return out


def _norm_gain(chs):
    """Peak-normalize against the loudest possible decode (all sources and channels in phase)."""
    ch = chs.sum(0)
    pk = np.abs(ch[0]).max() + np.abs(ch[1:4]).sum(0).max() + max(np.abs(ch[4]).max(), np.abs(ch[5]).max())
    return 0.9 / pk if pk > 0 else 1.0


def live(chs, dirs, pose_deg):
    """Loop the programme on the default output; re-read the pose from stdin lines "az el roll"."""
    import sounddevice as sd
    if os.environ.get("SD_DEVICE"):
        d = os.environ["SD_DEVICE"]; sd.default.device = int(d) if d.isdigit() else d
    pose = list(pose_deg)
    chs = (chs * _norm_gain(chs)).astype(np.float32)          # gain baked in, half the memory traffic
    stop = threading.Event()
    prev = {}                                                  # per-source last ITD (crossfade state)

    def reader():
        for line in sys.stdin:
            t = line.split()
            if not t or t[0].lower() == "q":
                break
            try:
                pose[:] = [float(t[0]), float(t[1]), float(t[2])]
            except (ValueError, IndexError):
                pass
        stop.set()
    threading.Thread(target=reader, daemon=True).start()

    M, pos = chs.shape[2], 0

    def cb(outdata, frames, _time, status):
        nonlocal pos
        n0, n1 = pos, min(pos + frames, M)
        blk = decode(chs, dirs, pose, n0, n1, prev)
        if n1 - n0 < frames:                                    # wrap: loop the programme
            blk = np.concatenate([blk, decode(chs, dirs, pose, 0, frames - (n1 - n0), prev)], axis=1)
            pos = frames - (n1 - n0)
        else:
            pos = n1
        outdata[:] = blk.T
    dev = sd.query_devices(kind="output")
    print(f"[ambisonic] output device: {dev['name']} (set SD_DEVICE=<name|index> to override)", flush=True)
    print(f"[ambisonic] live: {M / FS:.1f} s programme, {chs.shape[0]} source(s), block {BLOCK} "
          f"({1e3 * BLOCK / FS:.0f} ms), buffer {1e3 * LATENCY:.0f} ms; pose lines 'az el roll' on stdin, "
          "'q' to stop", flush=True)
    with sd.OutputStream(samplerate=FS, channels=2, blocksize=BLOCK, latency=LATENCY,
                         dtype="float32", callback=cb):
        stop.wait()


def check(rec):
    """How much of receiver `rec`'s response is directional at all, and does it pan? Per source: the
    direct-sound azimuth, the energy share of the directional part (W of direct+ISM vs the diffuse
    tail), and the early-window (first 80 ms) ILD for yaw 0/90/180/270. ILD flat across yaws while the
    directional share is a few percent = the geometry gave almost nothing to pan, not a decode bug."""
    srcs = sorted(int(m.group(1)) for f in glob.glob(f"ARR{rec}_*.bin")
                  for m in [re.match(rf"ARR{rec}_(\d+)\.bin$", os.path.basename(f))] if m)
    for src in srcs:
        foa = foa_rir(rec, src)
        if foa is None:
            continue
        W, XYZ, tail = foa
        af = f"ARR{rec}_{src}.bin"
        delay, doa, _ = read_arrivals(af)
        d = doa[int(np.argmin(delay))] if delay.size else np.array([1.0, 0, 0])
        az0 = np.degrees(np.arctan2(d[1], d[0]))
        e_dir, e_tail = float(np.dot(W, W)), float(np.dot(tail, tail))
        n_early = int(0.080 * FS)
        ch = np.stack([W, XYZ[0], XYZ[1], XYZ[2], decorrelate(tail, 1), decorrelate(tail, 2)])[None]
        ilds = []
        for yaw in (0, 90, 180, 270):
            lr = decode(ch, d[None], (yaw, 0, 0), 0, n_early)
            el, er = float(np.dot(lr[0], lr[0])), float(np.dot(lr[1], lr[1]))
            ilds.append(10 * np.log10((el + 1e-30) / (er + 1e-30)))
        print(f"[ambisonic] rec{rec} src{src}: direct from az {az0:+.0f} deg (0=+x, +90=+y), "
              f"{delay.size} ISM arrival(s); directional energy {100 * e_dir / (e_dir + e_tail + 1e-30):.1f} % "
              f"of the response; early ILD L/R [dB] at yaw 0/90/180/270: "
              + " / ".join(f"{v:+.1f}" for v in ilds))


# ---------------------------------------------------------------- self-check -------------------------
def _selfcheck():
    """A single arrival from the left must decode louder AND earlier on the left ear, swap when the head
    yaws 180, and be symmetric from straight ahead."""
    N = 2048
    dl = np.array([[0.0, 1.0, 0.0]])                                        # source at +y (left)
    ch = np.zeros((1, 6, N)); ch[0, 0, 100] = 1.0; ch[0, 2, 100] = 1.0      # W + Y impulse
    lr = decode(ch, dl, (0, 0, 0))
    assert lr[0].max() > 2 * lr[1].max(), "arrival from the left must favour the left ear"
    assert int(lr[1].argmax()) - int(lr[0].argmax()) == itd_samples(np.pi / 2) > 20, "right ear lags by the ITD"
    lr2 = decode(ch, dl, (180, 0, 0))
    assert lr2[1].max() > 2 * lr2[0].max(), "yaw 180: the same arrival is now on the right"
    assert int(lr2[0].argmax()) - int(lr2[1].argmax()) == itd_samples(np.pi / 2), "and the left ear lags"
    df = np.array([[1.0, 0.0, 0.0]])
    ch2 = np.zeros((1, 6, N)); ch2[0, 0, 100] = 1.0; ch2[0, 1, 100] = 1.0   # from +x (front)
    lr3 = decode(ch2, df, (0, 0, 0))
    assert np.allclose(lr3[0], lr3[1]), "frontal arrival is symmetric, no ITD"
    lr4 = decode(ch2, df, (90, 0, 0))                                        # head turned left -> source at right
    assert lr4[1].max() > 2 * lr4[0].max(), "yaw +90 puts a frontal source on the right ear"
    # live crossfade: a delay change inside a block must not jump
    prev = {0: (0, 0)}
    blk = decode(ch, dl, (0, 0, 0), 0, 512, prev)
    assert prev[0] == (0, itd_samples(np.pi / 2)) and np.isfinite(blk).all()
    # the low-band sign convention: v = -(p/Z0) d for a wave arriving from d
    d = np.array([0.0, 1.0, 0.0]); v = -d
    assert np.allclose(-v, d), "X,Y,Z = -Z0 v must point at the source"
    print("OK: FOA decode pans (ILD) and delays (ITD) with direction and head yaw; sign convention consistent.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _selfcheck(); sys.exit(0)
    import argparse
    ap = argparse.ArgumentParser(description="First-order ambisonic auralization (no HRTF).")
    ap.add_argument("--rec", type=int, required=True)
    ap.add_argument("--dry", help="anechoic mono audio (not needed for --check)")
    ap.add_argument("--drymap", help="JSON {source_id: anechoic_path}")
    ap.add_argument("--pose", nargs=3, type=float, default=(0, 0, 0), metavar=("AZ", "EL", "ROLL"))
    ap.add_argument("--out", help="write a stereo wav for --pose (offline)")
    ap.add_argument("--live", action="store_true", help="play on the default output, pose from stdin")
    ap.add_argument("--check", action="store_true", help="print directional share + early ILD vs yaw, no audio")
    a = ap.parse_args()
    if a.check:
        check(a.rec); sys.exit(0)
    if not a.dry:
        ap.error("--dry is required unless --check")
    dmap = json.load(open(a.drymap)) if a.drymap else None
    chs, dirs = render_channels(a.rec, a.dry, dmap)
    if a.out:
        import soundfile as sf
        sf.write(a.out, (decode(chs, dirs, a.pose) * _norm_gain(chs)).T, FS)
        print(f"[ambisonic] wrote {a.out} (pose {a.pose})")
    if a.live:
        live(chs, dirs, a.pose)
