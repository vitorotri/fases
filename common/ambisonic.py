#!/usr/bin/env python3
"""Ambisonic auralization with LIVE head rotation -- the interactive alternative to binauralize.py.

With --sofa (default order 3): the room response is encoded in real SH up to `order` (ISM arrivals at
full order from their exact DOAs; the wave-based low band at order 1 from p and v, all the solver
stores at the receiver point and all the head resolves below the seam), rotated per block to the head
pose and decoded binaurally with SH-domain HRTFs fitted from the SOFA set (LS below ~N c/(2 pi r),
MagLS above). The diffuse tail stays decorrelated per ear. HRTF gives the front/back and elevation
cues the no-HRTF mode lacks. Decoder cached next to the SOFA (<sofa>.<FS>.hoa<N>.npz).

Without --sofa, first-order B-format as below:

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
    python ambisonic.py --rec R --dry a.wav --live --sofa h.sofa [--order 3]   # HOA + HRTF
Live mode reads "az el roll" (deg) lines from stdin whenever they arrive (the GUI writes the dials),
loops the programme, and stops on EOF or a line "q".
Self-check (no files needed):  python ambisonic.py
"""
import glob, json, os, re, sys, threading
import numpy as np
from scipy.signal import fftconvolve

import allIRs
from binauralize import (head_basis, read_arrivals, arrival_kernel, _band_weights, decorrelate,
                         _place, _read_audio, load_hrtf, KLEN, FS, C as C_HEAD)

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
def _rir_parts(rec, src):
    """Pieces shared by foa_rir and hoa_rir, at FS, world frame, or None if the pair has no data:
    (N, LIRp, XYZl (3,N) low-band -Z0*v or None, arrivals [(start, kernel, unit doa)], a, tail,
    t_cross, e_hip, has_rt) -- the last three are always (0, 0.0, True) now.
    Needs the HRT files (allIRs.rt_only_ir): a = 1 (the ARR gains ARE the deposited pressures) and
    tail = the RT-only stream at ALL times -- nothing counted twice. Runs older than the HRT split are
    refused (None): their RT share cannot be separated from HIRp."""
    res = allIRs.broadband_ir(rec, src)
    if res is None:
        return None
    _, LIRp, HIRp = res
    RTp = allIRs.rt_only_ir(rec, src)                          # None for runs without HRT files
    if RTp is None:
        print(f"[ambisonic] !! rec{rec} src{src}: no HRT files (run older than the HRT split) -> skipped. "
              "Re-run the GA (GA-only reuses the FVM) to auralize it.")
        return None
    fx = allIRs.f_cutoff                                      # the crossover broadband_ir settled on
    N = max(LIRp.size, HIRp.size, RTp.size) + 512
    XYZl = None
    vf = f"VEL{rec}_{src}.bin"
    if LIRp.any() and allIRs.fs_fv is not None and os.path.isfile(vf):
        v = np.fromfile(vf, np.float32).reshape(-1, 3)
        Z0 = RHO * _c_sound()
        XYZl = np.zeros((3, N))
        for k in range(3):
            _place(XYZl[k], allIRs._filt(allIRs.low_deconv(-Z0 * v[:, k]), None, fx), 0)
    elif LIRp.any():
        print(f"[ambisonic] rec{rec} src{src}: no {vf} -> low band rendered omnidirectional")

    af = f"ARR{rec}_{src}.bin"
    delay, doa, g = read_arrivals(af) if os.path.isfile(af) else (np.zeros(0), np.zeros((0, 3)), np.zeros((0, 8)))
    band_w = _band_weights(np.fft.rfftfreq(KLEN, 1 / FS))
    arr = []
    for k in range(delay.size):
        ker = arrival_kernel(g[k], band_w)
        s0 = int(round(delay[k] * FS)) - KLEN // 2             # compensate kernel group delay
        arr.append((s0, ker, doa[k] / (np.linalg.norm(doa[k]) or 1.0)))
    tail = np.zeros(N)                                         # exact arrivals + RT-only tail at all times
    tail[:RTp.size] = RTp
    return N, LIRp, XYZl, arr, 1.0, tail, 0, 0.0, True


def foa_rir(rec, src):
    """(W, XYZ (3,N), tail) for one (rec, src), at FS, world frame. None if the pair has no data."""
    parts = _rir_parts(rec, src)
    if parts is None:
        return None
    N, LIRp, XYZl, arr, a, tail, _, _, _ = parts
    W, XYZ = np.zeros(N), np.zeros((3, N))
    _place(W, LIRp, 0)
    if XYZl is not None:
        XYZ += XYZl
    for s0, ker, d in arr:
        _place(W, a * ker, s0)
        for k in range(3):
            _place(XYZ[k], a * d[k] * ker, s0)
    return W, XYZ, tail


# ---------------------------------------------------------------- higher order (HOA) -----------------
# Real, ORTHONORMAL spherical harmonics (integral of Y^2 over the sphere = 1), ACN order (n^2+n+m), no
# Condon-Shortley phase, orders <= 3 as Cartesian polynomials of the unit direction (x front, y left,
# z up in the head frame; world x, y, z before rotation). A plane wave p from direction d encodes to
# p * Y(d); the ear signal is sum_nm s_nm * G_nm with G the SH-domain HRTF (hoa_decoder).
def sh(d, order=3):
    """(K,3) unit directions -> (K, (order+1)^2) real orthonormal SH, ACN."""
    d = np.atleast_2d(np.asarray(d, float))
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    c = [np.full_like(x, 0.28209479177387814)]
    if order >= 1:
        k1 = 0.4886025119029199
        c += [k1 * y, k1 * z, k1 * x]
    if order >= 2:
        c += [1.0925484305920792 * x * y, 1.0925484305920792 * y * z,
              0.31539156525252005 * (3 * z * z - 1), 1.0925484305920792 * x * z,
              0.5462742152960396 * (x * x - y * y)]
    if order >= 3:
        c += [0.5900435899266435 * y * (3 * x * x - y * y), 2.890611442640554 * x * y * z,
              0.4570457994644658 * y * (5 * z * z - 1), 0.3731763325901154 * z * (5 * z * z - 3),
              0.4570457994644658 * x * (5 * z * z - 1), 1.445305721320277 * z * (x * x - y * y),
              0.5900435899266435 * x * (x * x - 3 * y * y)]
    if order > 3:
        raise ValueError("orders above 3 are not implemented")
    return np.stack(c, 1)


def _fib(n):
    """n near-uniform unit directions (Fibonacci sphere)."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    ph = i * np.pi * (3.0 - np.sqrt(5.0))
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], 1)


_ROT_GRID = {}


def rot_matrix(pose_deg, order=3):
    """SH-domain rotation world -> head frame for head pose (deg): s_head = M @ s_world, i.e.
    M @ Y(d) = Y(B d) with B = rows (forward, left, up) of head_basis. Solved by least squares on a
    fixed direction grid -- exact for these orders, since rotation keeps each order's subspace."""
    if order not in _ROT_GRID:
        g = _fib(100)
        _ROT_GRID[order] = (g, np.linalg.pinv(sh(g, order)))
    g, P = _ROT_GRID[order]
    B = np.stack(head_basis(*np.radians(pose_deg)))           # rows F, L, U
    return sh(g @ B.T, order).T @ P.T


def hoa_rir(rec, src, order=3, g=None, pose=(0.0, 0.0, 0.0)):
    """(S ((order+1)^2, N) world-frame SH room response, tail) or None. Low band = 1st order only (the
    solver stores p and v at the receiver point; below the seam the head is small against the
    wavelength, so order 1 is what the ears can resolve anyway); ISM arrivals at full `order`."""
    parts = _rir_parts(rec, src)
    if parts is None:
        return None
    N, LIRp, XYZl, arr, a, tail, t_cross, e_hip, has_rt = parts
    S = np.zeros(((order + 1) ** 2, N))
    k0, k1 = 0.28209479177387814, 0.4886025119029199          # Y00, order-1 factor
    _place(S[0], k0 * LIRp, 0)
    if XYZl is not None and order >= 1:                        # plane-wave encode: p*Y(d), XYZ = p*d
        S[1] += k1 * XYZl[1]; S[2] += k1 * XYZl[2]; S[3] += k1 * XYZl[0]
    Se = np.zeros_like(S)                                     # early arrivals, kept apart for the level
    for s0, ker, d in arr:
        y = sh(d, order)[0]
        for n in range(y.size):
            _place(Se[n], a * y[n] * ker, s0)
    if g is not None and has_rt:
        # Arrivals are exact (a = 1); remove only the HRTF set's own gain, as binauralize.hrtf_diffuse_gain does.
        # Diffuse-field energy of the SH-domain decoder: with orthonormal SH, a plane wave p*Y(d) reaches an ear
        # as sum_n Y_n(d) G_n, whose energy averaged over the sphere is sum_n ||G_n||^2 / (4 pi); /2 averages ears.
        g_df = float(np.sum(np.asarray(g, float) ** 2)) / (8.0 * np.pi)
        if g_df > 0:
            Se /= np.sqrt(g_df)
    return S + Se, tail


def render_hoa(rec, dry_path, dry_map=None, order=3, g=None, pose=(0.0, 0.0, 0.0)):
    """Programme (*) SH room response, summed over all sources of `rec` (one rotation + one binaural
    decode serves them all) -> (S (nsh, M) float32 world frame, tails (2, M) float32, Y (2, M) or None).

    Fast path (all exact, convolution is linear):
      * sources that play the SAME audio file are summed BEFORE convolving: one 16-channel convolution
        per audio file, not per source;
      * the audio is FFT'd ONCE per file and reused for all 16 channels + tails (+ the 2 below);
      * with the decoder `g`: Y = the binaural output at `pose`, from the audio (*) the 2-channel binaural
        room response at that pose (decode of the SHORT room response) -- exactly what the live decode
        plays at a fixed pose; used for the level and for offline --out instead of a full 16-ch decode."""
    import time
    from scipy import fft as sfft
    groups = {}                                                # audio path -> [S sum, tail sum, sources]
    for src in _sources_of(rec):
        r = hoa_rir(rec, src, order, g, pose)
        if r is None:
            continue
        S, tail = r
        path = (dry_map or {}).get(str(src), dry_path)
        grp = groups.setdefault(path, [None, None, []])
        grp[0] = _acc(grp[0], S); grp[1] = _acc(grp[1], tail[None]); grp[2].append(src)
    if not groups:
        raise FileNotFoundError(f"no broadband data for receiver {rec}")
    S_out = T_out = Y_out = None
    L = g.shape[-1] if g is not None else 0
    for path, (S, tail, srcs) in groups.items():
        t0 = time.time()
        dry = _read_audio(path)
        n_out = dry.size + S.shape[1] + L - 1
        nfft = sfft.next_fast_len(n_out, real=True)
        print(f"[ambisonic] rec{rec} src{srcs}: convolving {S.shape[0]} SH channels with "
              f"{dry.size / FS:.1f} s of audio ({os.path.basename(path)})...", flush=True)
        X = sfft.rfft(dry, nfft, workers=-1)                   # the audio's spectrum, once

        def conv(h):
            return sfft.irfft(X * sfft.rfft(h, nfft, workers=-1), nfft, workers=-1)[:n_out]
        Sc = np.empty((S.shape[0], n_out), np.float32)
        for k in range(S.shape[0]):
            Sc[k] = conv(S[k])
        t = tail[0]
        Tc = np.stack([conv(decorrelate(t, 1)), conv(decorrelate(t, 2))]).astype(np.float32)
        S_out, T_out = _acc(S_out, Sc), _acc(T_out, Tc)
        if g is not None:                                      # 2-ch binaural room response at `pose`
            Sp = np.pad(S, ((0, 0), (0, L)))
            brir = hoa_decode(Sp, np.zeros((2, Sp.shape[1])), g, pose, order)
            Y_out = _acc(Y_out, np.stack([conv(brir[0]), conv(brir[1])]) + Tc)
        print(f"[ambisonic] rec{rec} src{srcs}: rendered order {order} in {time.time() - t0:.1f} s, "
              f"response peak {np.abs(S).max():.3g}", flush=True)
    M = max(S_out.shape[1], T_out.shape[1], Y_out.shape[1] if Y_out is not None else 0)
    S_out = _acc(np.zeros((S_out.shape[0], M), np.float32), S_out)
    T_out = _acc(np.zeros((2, M), np.float32), T_out)
    if Y_out is not None:
        Y_out = _acc(np.zeros((2, M)), Y_out)
    return S_out, T_out, Y_out


def _acc(acc, x):
    """acc + x along the last axis, growing acc to fit."""
    if acc is None:
        return x.copy()
    if x.shape[1] > acc.shape[1]:
        acc = np.pad(acc, ((0, 0), (0, x.shape[1] - acc.shape[1])))
    acc[:, :x.shape[1]] += x
    return acc


def hoa_decoder(hrtf, order=3, f_cut=None, reg=1e-2, max_dirs=1200):
    """SH-domain binaural filters g ((order+1)^2, 2, L) from an HRIR set (binauralize.load_hrtf).
    Least squares fit of the HRTFs onto the SH basis below f_cut; above it MagLS (Schoerkhuber et al.
    2018): fit only the MAGNITUDE, phase carried from the previous bin -- order N cannot hold the HRTF
    phase above ~N*c/(2 pi r), and fitting it there blurs the image. Dense grids are thinned to a near-
    uniform subset (nearest HRTF direction to a Fibonacci grid) so plain LS is not biased to the poles."""
    U = np.asarray(hrtf["unit"], float); h = np.asarray(hrtf["hrir"], float)
    if U.shape[0] > max_dirs:
        tg = _fib(max_dirs)
        idx = np.unique(np.concatenate([np.argmax(U @ tg[i:i + 200].T, axis=0)
                                        for i in range(0, max_dirs, 200)]))
        U, h = U[idx], h[idx]
    Y = sh(U, order)
    nsh = Y.shape[1]
    YtY = Y.T @ Y
    P = np.linalg.solve(YtY + reg * np.trace(YtY) / nsh * np.eye(nsh), Y.T)   # (nsh, K) regularised pinv
    T = h.shape[-1]
    nfft = max(256, 1 << int(np.ceil(np.log2(2 * T))))
    H = np.fft.rfft(h, nfft, axis=-1)                          # (K, 2, F)
    f = np.fft.rfftfreq(nfft, 1.0 / FS)
    f_cut = order * C_HEAD / (2 * np.pi * HEAD_R) if f_cut is None else f_cut
    kc = max(1, int(np.searchsorted(f, f_cut)))
    G = np.zeros((nsh, 2, f.size), complex)
    G[:, :, :kc] = np.einsum('nk,kef->nef', P, H[:, :, :kc])
    for b in range(kc, f.size):                                # MagLS
        ph = np.angle(Y @ G[:, :, b - 1])                      # (K, 2) phase of the previous bin's fit
        G[:, :, b] = P @ (np.abs(H[:, :, b]) * np.exp(1j * ph))
    g = np.fft.irfft(G, nfft, axis=-1)
    nf = nfft // 8                                             # fade the end: no wrap-around clicks
    g[..., -nf:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, nf)))
    return g


def hoa_decoder_cached(sofa_path, order=3):
    """hoa_decoder of a SOFA file, cached next to it as <sofa>.<FS>.hoa<order>.npz (like load_hrtf)."""
    cache = f"{sofa_path}.{FS}.hoa{order}.npz"
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(sofa_path):
        return np.load(cache)["g"]
    g = hoa_decoder(load_hrtf(sofa_path), order)
    try:
        np.savez(cache, g=g)
    except OSError:
        pass
    return g


class HoaBinaural:
    """Streaming SH -> binaural: process() takes head-frame SH blocks of ANY length, convolves with the
    SH-domain HRTFs by FFT and carries the overlap to the next block (overlap-add), so consecutive
    blocks equal one long convolution."""
    def __init__(self, g):
        self.g = np.asarray(g, float)
        self.L = self.g.shape[-1]
        self._G = {}
        self.tail = np.zeros((2, self.L - 1))

    def process(self, s):
        n = s.shape[1]
        nfft = 1 << int(np.ceil(np.log2(n + self.L - 1)))
        G = self._G.get(nfft)
        if G is None:
            G = self._G[nfft] = np.fft.rfft(self.g, nfft, axis=-1)
        y = np.fft.irfft(np.einsum('nf,nef->ef', np.fft.rfft(s, nfft, axis=-1), G), nfft, axis=-1)
        buf = y[:, :n + self.L - 1]
        buf[:, :self.L - 1] += self.tail
        self.tail = buf[:, n:].copy()
        return buf[:, :n]


def hoa_decode(S, tails, g, pose_deg, order=3, chunk=65536):
    """Whole programme at one fixed pose -> (2, M) (offline --out and gain estimate)."""
    st, Mr = HoaBinaural(g), rot_matrix(pose_deg, order)
    out = np.zeros((2, S.shape[1]))
    for i in range(0, S.shape[1], chunk):
        out[:, i:i + chunk] = st.process(Mr @ S[:, i:i + chunk]) + tails[:, i:i + chunk]
    return out


# ---------------------------------------------------------------- render + decode --------------------
def render_channels(rec, dry_path, dry_map=None):
    """Convolve the programme with the B-format RIR of every source of `rec`:
    -> (chs (S,6,M): W,X,Y,Z,tailL,tailR per source; dirs (S,3): each source's direct-sound DOA, world).
    Kept per source so the ITD can follow each source's own direction. dry_map: per-source files."""
    srcs = _sources_of(rec)
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


def _sources_of(rec):
    """Source ids with data for receiver `rec` (ARR files, else low-band IR files)."""
    srcs = sorted(int(m.group(1)) for f in glob.glob(f"ARR{rec}_*.bin")
                  for m in [re.match(rf"ARR{rec}_(\d+)\.bin$", os.path.basename(f))] if m)
    if not srcs:
        srcs = sorted({int(m.group(1)) for f in glob.glob(f"IR{rec}_*.bin")
                       for m in [re.match(rf"IR{rec}_(\d+)\.bin$", f)] if m})
    if not srcs:
        raise FileNotFoundError(f"no ARR/IR data for receiver {rec} in {os.getcwd()}")
    return srcs


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


def _play(render, M, pose):
    """Loop a programme of M samples on the default output. render(n0, n1) -> (2, n) block, reading the
    shared `pose` list, which a stdin reader thread updates from lines "az el roll" (deg)."""
    import sounddevice as sd
    if os.environ.get("SD_DEVICE"):
        d = os.environ["SD_DEVICE"]; sd.default.device = int(d) if d.isdigit() else d
    stop = threading.Event()

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

    pos = 0

    stats = {"calls": 0, "under": 0}

    def cb(outdata, frames, _time, status):
        nonlocal pos
        try:
            if status.output_underflow:
                stats["under"] += 1
            n0, n1 = pos, min(pos + frames, M)
            blk = render(n0, n1)
            if n1 - n0 < frames:                                # wrap: loop the programme
                blk = np.concatenate([blk, render(0, frames - (n1 - n0))], axis=1)
                pos = frames - (n1 - n0)
            else:
                pos = n1
            outdata[:] = blk.T
            stats["calls"] += 1
            if stats["calls"] == 1:
                print(f"[ambisonic] first block out, peak {np.abs(blk).max():.3g}", flush=True)
        except Exception:                                       # sounddevice would swallow it: silence
            import traceback
            traceback.print_exc(); sys.stderr.flush()
            outdata.fill(0); stop.set()
    dev = sd.query_devices(kind="output")
    print(f"[ambisonic] output device: {dev['name']} (set SD_DEVICE=<name|index> to override)", flush=True)
    print(f"[ambisonic] live: {M / FS:.1f} s programme, block {BLOCK} ({1e3 * BLOCK / FS:.0f} ms), buffer "
          f"{1e3 * LATENCY:.0f} ms; pose lines 'az el roll' on stdin, 'q' to stop", flush=True)
    with sd.OutputStream(samplerate=FS, channels=2, blocksize=BLOCK, latency=LATENCY,
                         dtype="float32", callback=cb):
        print("[ambisonic] stream started", flush=True)
        while not stop.wait(5.0):                               # heartbeat: proves the callback runs
            print(f"[ambisonic] playing: {stats['calls']} blocks, {stats['under']} underrun(s)", flush=True)
    print(f"[ambisonic] stopped after {stats['calls']} blocks, {stats['under']} underrun(s)", flush=True)


def live(chs, dirs, pose_deg):
    """1st order, no HRTF: loop the programme; per block rotate + two virtual mics + per-source ITD."""
    pose = list(pose_deg)
    chs = (chs * _norm_gain(chs)).astype(np.float32)          # gain baked in, half the memory traffic
    prev = {}                                                  # per-source last ITD (crossfade state)
    print(f"[ambisonic] 1st order, no HRTF, {chs.shape[0]} source(s)", flush=True)
    _play(lambda n0, n1: decode(chs, dirs, pose, n0, n1, prev), chs.shape[2], pose)


def live_hoa(S, tails, g, pose_deg, order=3, peak=None):
    """HOA + HRTF: per block rotate the SH stream to the head pose (matrix crossfaded over the block
    when the pose moved) and decode binaurally with the SH-domain HRTFs (streaming convolution)."""
    pose = list(pose_deg)
    if peak is None:                                            # render_hoa normally hands it over
        print("[ambisonic] estimating the output level (one offline decode)...", flush=True)
        peak = np.abs(hoa_decode(S, tails, g, pose, order)).max()
    pk = peak
    gain = 0.7 / pk if pk > 0 else 1.0                          # ~3 dB headroom for other poses
    print(f"[ambisonic] offline decode peak {pk:.3g} -> gain {gain:.3g}", flush=True)
    S = (S * gain).astype(np.float32); tails = (tails * gain).astype(np.float32)
    st, state = HoaBinaural(g), {"M": None, "pose": None}
    ramp_cache = {}

    def render(n0, n1):
        p = list(pose)                                          # one consistent snapshot (stdin thread writes)
        if state["pose"] != p:
            state["pose"] = p
            Mn = rot_matrix(p, order)
        else:
            Mn = state["M"]
        Mo = state["M"] if state["M"] is not None else Mn
        blk = S[:, n0:n1]
        if Mo is Mn:
            sh_ = Mn @ blk
        else:                                                   # pose moved: crossfade the rotation
            n = n1 - n0
            r = ramp_cache.get(n)
            if r is None:
                r = ramp_cache[n] = np.linspace(0.0, 1.0, n)
            sh_ = (1 - r) * (Mo @ blk) + r * (Mn @ blk)
        state["M"] = Mn
        return st.process(sh_) + tails[:, n0:n1]
    print(f"[ambisonic] order {order} + HRTF, {S.shape[0]} SH channels", flush=True)
    _play(render, S.shape[1], pose)


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



def _lag_lf(lr, f_max=1000.0, max_lag=30):
    """Samples by which the RIGHT ear lags the left, from the <f_max part (cross-correlation peak)."""
    n = 1 << int(np.ceil(np.log2(lr.shape[1] + 2 * max_lag)))
    X = np.fft.rfft(lr, n, axis=-1)
    X[:, np.fft.rfftfreq(n, 1.0 / FS) > f_max] = 0
    xl, xr = np.fft.irfft(X, n, axis=-1)
    lags = np.arange(-max_lag, max_lag + 1)
    return int(lags[np.argmax([np.dot(xr, np.roll(xl, k)) for k in lags])])


def _selfcheck_hoa():
    """SH basis orthonormal; rotation exact; a synthetic HRTF set decoded at order 3 pans left/right
    with the head; streaming blocks == one long convolution; FOA -> SH mapping = plane-wave encode."""
    rng = np.random.default_rng(1)
    g = _fib(4000)
    Y = sh(g, 3)
    assert np.allclose(Y.T @ Y * 4 * np.pi / len(g), np.eye(16), atol=2e-2), "SH not orthonormal"
    d = rng.normal(size=(50, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    for pose in ((30, 20, 10), (200, -60, 80), (0, 0, 0)):
        Mr = rot_matrix(pose, 3)
        B = np.stack(head_basis(*np.radians(pose)))
        assert np.allclose(Mr @ sh(d, 3).T, sh(d @ B.T, 3).T, atol=1e-8), f"rotation wrong at {pose}"
    # synthetic HRTF: gain + delay per ear from the direction's left component
    U = _fib(600); h = np.zeros((600, 2, 64))
    for i, u in enumerate(U):
        h[i, 0, 10 + int(round(4 * (1 - u[1])))] = 1.0 + 0.6 * u[1]      # left ear
        h[i, 1, 10 + int(round(4 * (1 + u[1])))] = 1.0 - 0.6 * u[1]      # right ear
    gf = hoa_decoder({"unit": U, "hrir": h}, 3)
    x = np.zeros(512); x[50] = 1.0
    S = sh([0.0, 1.0, 0.0], 3).T * x                                     # plane wave from world +y
    tails = np.zeros((2, 512))
    lr = hoa_decode(S, tails, gf, (0, 0, 0))
    el, er = (lr[0] ** 2).sum(), (lr[1] ** 2).sum()
    assert el > 2 * er, f"source on the left must be louder left ({el:.3g} vs {er:.3g})"
    assert _lag_lf(lr) > 0, "right ear must lag (ITD, judged below 1 kHz: MagLS drops phase above ~1.9 kHz)"
    lr2 = hoa_decode(S, tails, gf, (180, 0, 0))
    assert (lr2[1] ** 2).sum() > 2 * (lr2[0] ** 2).sum(), "yaw 180 must swap the ears"
    Sf = sh([1.0, 0.0, 0.0], 3).T * x                                    # from the front: symmetric
    lr3 = hoa_decode(Sf, tails, gf, (0, 0, 0))
    assert abs((lr3[0] ** 2).sum() / (lr3[1] ** 2).sum() - 1) < 0.1, "frontal source must be balanced"
    # streaming in odd blocks == whole
    s_in = rng.normal(size=(16, 3000))
    whole = HoaBinaural(gf).process(s_in)
    st, parts, i = HoaBinaural(gf), [], 0
    for n in (1, 700, 33, 1024, 1242):
        parts.append(st.process(s_in[:, i:i + n])); i += n
    assert np.allclose(np.concatenate(parts, 1), whole, atol=1e-9), "block streaming != long convolution"
    # render_hoa's level/--out shortcut: audio (*) (2-ch binaural room response at a pose) must equal
    # the streamed rotation + decode of audio (*) 16-ch room response (linearity)
    Sr, dry = rng.normal(size=(16, 200)), rng.normal(size=500)
    pose, Lg = (40, 10, 5), gf.shape[-1]
    prog = np.stack([np.convolve(dry, Sr[n]) for n in range(16)])                   # (16, 699)
    A = hoa_decode(prog, np.zeros((2, prog.shape[1])), gf, pose)
    brir = hoa_decode(np.pad(Sr, ((0, 0), (0, Lg))), np.zeros((2, 200 + Lg)), gf, pose)
    B = np.stack([np.convolve(dry, brir[e]) for e in range(2)])[:, :prog.shape[1]]
    assert np.allclose(A, B, atol=1e-9), "2-ch binaural room response shortcut != streamed decode"
    # decoder diffuse-field gain used by hoa_rir (HRT mode): the decoded energy of a unit plane wave, averaged
    # over the sphere and both ears, must equal sum ||G_n||^2 / (8 pi) (orthonormal SH)
    Yd = sh(_fib(3000), 3)                                               # (K, 16)
    e_dirs = np.sum(np.einsum('kn,net->ket', Yd, gf) ** 2, axis=(1, 2)) / 2.0
    assert abs(e_dirs.mean() / (np.sum(gf ** 2) / (8 * np.pi)) - 1) < 2e-2, "decoder diffuse-field gain formula"
    # FOA (W = p, XYZ = p d) -> SH order 1 equals the plane-wave encoding p Y(d)
    dd = np.array([0.36, -0.48, 0.8])
    k0, k1 = 0.28209479177387814, 0.4886025119029199
    assert np.allclose([k0, k1 * dd[1], k1 * dd[2], k1 * dd[0]], sh(dd, 1)[0]), "FOA -> SH mapping"
    print("OK: HOA -- SH orthonormal, rotation exact, order-3 HRTF decode pans with the head, "
          "streaming exact, binaural-room-response shortcut exact, FOA/SH mapping consistent.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _selfcheck(); _selfcheck_hoa(); sys.exit(0)
    import argparse
    ap = argparse.ArgumentParser(description="Ambisonic auralization: order 1..3 + SOFA HRTF, "
                                             "or 1st order without HRTF when no --sofa is given.")
    ap.add_argument("--rec", type=int, required=True)
    ap.add_argument("--dry", help="anechoic mono audio (not needed for --check)")
    ap.add_argument("--drymap", help="JSON {source_id: anechoic_path}")
    ap.add_argument("--pose", nargs=3, type=float, default=(0, 0, 0), metavar=("AZ", "EL", "ROLL"))
    ap.add_argument("--out", help="write a stereo wav for --pose (offline)")
    ap.add_argument("--live", action="store_true", help="play on the default output, pose from stdin")
    ap.add_argument("--check", action="store_true", help="print directional share + early ILD vs yaw, no audio")
    ap.add_argument("--sofa", help="HRTF set: enables the higher-order binaural decode (else 1st order, no HRTF)")
    ap.add_argument("--order", type=int, default=3, choices=(1, 2, 3), help="ambisonic order with --sofa")
    a = ap.parse_args()
    if a.check:
        check(a.rec); sys.exit(0)
    if not a.dry:
        ap.error("--dry is required unless --check")
    dmap = json.load(open(a.drymap)) if a.drymap else None
    if a.sofa:
        print(f"[ambisonic] HRTF decoder (order {a.order}) from {os.path.basename(a.sofa)}...", flush=True)
        g = hoa_decoder_cached(a.sofa, a.order)
        print(f"[ambisonic] decoder ready: {g.shape[0]} SH x 2 ears x {g.shape[-1]} taps, "
              f"peak {np.abs(g).max():.3g}", flush=True)
        S, tails, y = render_hoa(a.rec, a.dry, dmap, a.order, g, a.pose)   # y = output at --pose
        if a.out:
            import soundfile as sf
            sf.write(a.out, (y * (0.9 / (np.abs(y).max() or 1.0))).T, FS)
            print(f"[ambisonic] wrote {a.out} (order {a.order} + HRTF, pose {a.pose})")
        if a.live:
            live_hoa(S, tails, g, a.pose, a.order, peak=float(np.abs(y).max()))
        sys.exit(0)
    chs, dirs = render_channels(a.rec, a.dry, dmap)
    if a.out:
        import soundfile as sf
        sf.write(a.out, (decode(chs, dirs, a.pose) * _norm_gain(chs)).T, FS)
        print(f"[ambisonic] wrote {a.out} (pose {a.pose})")
    if a.live:
        live(chs, dirs, a.pose)
