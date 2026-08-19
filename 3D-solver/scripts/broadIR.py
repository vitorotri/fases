#!/usr/bin/env python3

# Creates single broadband IR for given source-receiver pair

import glob, numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from fractions import Fraction
from scipy.signal import resample_poly


octave_bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]

dt = 0.000050986
fs_fv = 1/dt
fs = 44100 # Hz
f_cutoff = 1000 # Hz
src = 0 # source index
rec = 0 # rec index

# Read only the files for THIS source-receiver pair
low_files = sorted(f for f in glob.glob('IR*.bin')   # IR<source>_<receiver>.bin
                   if f[2:].split('.')[0].split('_')[:2] == [str(src), str(rec)])
low_arrays = [np.fromfile(file, np.float32) for file in low_files]

high_files = sorted(f for f in glob.glob('HIR*.bin') # HIR<source>_<receiver>_<band>.bin
                    if f[3:].split('.')[0].split('_')[:2] == [str(src), str(rec)])
high_arrays = [np.fromfile(file, np.float32) for file in high_files]

H_IRs = []

plt.figure()
plt.plot(high_arrays[0])

# filter each array on high-frequency bands
for h in range(len(high_arrays)):
	
	N = len(high_arrays[h])
	f = np.fft.rfftfreq(N, 1/fs) # frequency of each FFT bin [Hz]

	fc = int(high_files[h].split('_')[-1].split('.')[0]) # band from HIR..._<band>.bin
	
	lo, hi = fc/2**0.5, fc*2**0.5 # octave edges
	T = 2**(1/6) # transition half-width

	# lower edge -> high-pass: 0 below lo, cosine up to 1 above
	xlo = np.clip((f - lo/T) / (lo*T - lo/T), 0, 1)
	w_hp = 0.5 * (1 - np.cos(np.pi * xlo))

	# upper edge -> low-pass: 1 below hi, cosine down to 0; top band runs to Nyquist
	if fc == octave_bands[-1]:
		w_lp = np.ones_like(f)
	else:
		xhi = np.clip((f - hi/T) / (hi*T - hi/T), 0, 1)
		w_lp = 0.5 * (1 + np.cos(np.pi * xhi))

	w_band = w_hp * w_lp # octave bandpass

	R = np.fft.rfft(high_arrays[h]) # IR -> spectrum
	ir_band = np.fft.irfft(R * w_band, n=N) # back to time, same length
	
	H_IRs.append(np.array(ir_band))

# sum filtered high-frequency arrays
HIR = np.sum(H_IRs, axis=0)

# low IR is at fs_fv; resample to 44100 BEFORE filtering (filters designed at fs)
LIR = low_arrays[0]
r = Fraction(fs/fs_fv).limit_denominator(2000)
LIR = resample_poly(LIR, r.numerator, r.denominator)   # anti-aliased

# Butterworth filters (digital, zero-phase) for low- and high-pass
sos_lp = signal.butter(8, f_cutoff, 'low', fs=fs, output='sos')
sos_hp = signal.butter(8, f_cutoff, 'high', fs=fs, output='sos')

LIR = signal.sosfiltfilt(sos_lp, LIR)
HIR = signal.sosfiltfilt(sos_hp, HIR)

for sos, name in [(sos_lp, 'low'), (sos_hp, 'high')]:
	w, h = signal.sosfreqz(sos, worN=2048, fs=fs)
	plt.figure()
	plt.semilogx(w, 20 * np.log10(abs(h)))
	plt.title(f'Butterworth {name}-pass response')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude [dB]')
	plt.margins(0, 0.1)
	plt.grid(which='both', axis='both')
	plt.axvline(f_cutoff, color='green') # cutoff frequency

N = max(LIR.size, HIR.size)
IR_ = np.pad(LIR, (0, N-LIR.size)) + np.pad(HIR, (0, N-HIR.size))

plt.figure()
plt.plot(IR_)
plt.show()

