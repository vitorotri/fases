#!/usr/bin/env python3
"""ISO 3382 room-acoustic parameters, per octave band, per receiver.

Reads the broadband IRs allIRs.py writes (IR_broad_<rec>_<src>.npy in the output/ dir), octave-band
filters each with pyfar, and lets pyrato compute the noise-compensated EDC (Chu+Lundeby) and the
parameters from it: T20/T30/EDT, C50/C80, D50 -- all per band. Writes params.json for the GUI to
plot as param-vs-band curves per receiver.

Run from the project's output/ directory, same as allIRs.py.
"""
import glob, re, os, json
import numpy as np
import pyfar as pf
import pyrato as ra

fs = 44100
OCTAVE_BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000]


def receiver_irs():
    """{rec: broadband IR summed over its sources} from IR_broad_<rec>_<src>.npy in cwd.
    ponytail: summed sources = the combined field, matching the GUI IR overlay; split per (rec,src)
    later if you need source-separated parameters."""
    groups = {}
    for f in glob.glob("IR_broad_*.npy"):
        m = re.match(r"IR_broad_(\d+)_(\d+)\.npy$", f)
        if m:
            groups.setdefault(int(m.group(1)), []).append(np.load(f))
    out = {}
    for rec, arrs in sorted(groups.items()):
        n = max(a.size for a in arrs)
        ir = np.zeros(n, float)
        for a in arrs:
            ir[:a.size] += a
        out[rec] = ir
    return out


def _lst(a):
    """Per-band values -> JSON list; NaN/inf/None -> null. Handles pyrato arrays and plain lists."""
    out = []
    for v in np.atleast_1d(a).ravel():
        try:
            fv = float(v)
            out.append(fv if np.isfinite(fv) else None)
        except (TypeError, ValueError):
            out.append(None)
    return out


def _onset(e):
    """Direct-sound sample: first point within -20 dB of the peak (start of the C/D integration)."""
    pk = int(e.argmax())
    hit = np.nonzero(e[:pk + 1] >= e[pk] * 1e-2)[0]
    return int(hit[0]) if hit.size else 0


def _clarity(h):
    """C50, C80 [dB] and D50 [0..1] for one band IR (this pyrato build lacks clarity()/definition())."""
    e = np.asarray(h, float) ** 2
    tot = e.sum()
    if tot <= 0:
        return None, None, None
    d = _onset(e)
    n50, n80 = int(0.050 * fs), int(0.080 * fs)
    e50, e80 = e[d:d + n50].sum(), e[d:d + n80].sum()
    l50, l80 = e[d + n50:].sum(), e[d + n80:].sum()
    c50 = float(10 * np.log10(e50 / l50)) if e50 > 0 and l50 > 0 else None
    c80 = float(10 * np.log10(e80 / l80)) if e80 > 0 and l80 > 0 else None
    return c50, c80, float(e50 / tot)


def params_for(ir):
    """Per-octave-band ISO 3382 params for one receiver IR (pyfar filterbank + pyrato RT, C/D direct)."""
    sig = pf.Signal(np.atleast_2d(ir), fs)
    bands = pf.dsp.filter.fractional_octave_bands(
        sig, num_fractions=1, frequency_range=(OCTAVE_BANDS[0], OCTAVE_BANDS[-1]))
    bt = bands.time.reshape(-1, bands.time.shape[-1])           # (n_bands, n_samples)
    # plain Schroeder integral (no noise-floor detection): simulated IRs decay to ~0, they have no
    # noise plateau, so the Chu/Lundeby truncation methods fail ("Low SNR"). Schroeder is correct here.
    edc = ra.edc.schroeder_integration(bands)
    rt = ra.reverberation_time_linear_regression
    C50, C80, D50 = [], [], []
    for i in range(len(OCTAVE_BANDS)):
        c50, c80, d50 = _clarity(bt[i]) if i < bt.shape[0] else (None, None, None)
        C50.append(c50); C80.append(c80); D50.append(d50)
    return {
        "bands": OCTAVE_BANDS,
        "T20": _lst(rt(edc, 'T20')),
        "T30": _lst(rt(edc, 'T30')),
        "EDT": _lst(rt(edc, 'EDT')),
        "C50": _lst(C50), "C80": _lst(C80), "D50": _lst(D50),
    }


def main():
    recs = receiver_irs()
    if not recs:
        print("[acouparams] no IR_broad_*.npy found in", os.getcwd()); return
    out = {}
    for rec, ir in recs.items():
        try:
            out[str(rec)] = params_for(ir)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"[acouparams] rec{rec} FAILED: {ex} -> skipped")
    with open("params.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"[acouparams] wrote params.json for {len(out)} receiver(s): {sorted(out)}")


if __name__ == "__main__":
    main()
