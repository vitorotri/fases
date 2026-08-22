#!/usr/bin/env python3

# Creates single broadband IR for given source-receiver pair

import glob, numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from fractions import Fraction
from scipy.signal import resample_poly

def first_peak(x, thr=0.05, win=50):
    a = np.abs(x)
    i0 = np.argmax(a > thr * a.max())        # onset: first sample above threshold
    return a[i0:i0+win].max()                # peak within the arrival


octave_bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]

# speed of sound + cutoff from config; dt + room dims from the solver's output/header.txt
cfg = np.loadtxt('../config/config.txt')
f_cutoff_sim, T_C = cfg[2], cfg[3]
c = (1.4 * 287.05 * (T_C + 273.15))**0.5    # speed of sound [m/s]

hp = open('../output/header.txt').read().split()   # Nx Ny Nz Nt Lx Ly Lz pbin anim dt scene h tol/h
Lx, Ly, Lz = float(hp[4]), float(hp[5]), float(hp[6])
dt, h      = float(hp[9]), float(hp[11])
Lx, Ly, Lz = Lx-12*h, Ly-12*h, Lz-12*h  # padded -> real acoustic room: 6h ghost pad + 6h immersed wall inset (~3h/face)

dt_half = 0.5 * dt
fs_fv = 1/dt
fs = 44100 # Hz
f_cutoff = f_cutoff_sim # Hz — script crossover (can be lower than the simulation's f_cutoff_sim)
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

LIR = low_arrays[0]
r = Fraction(fs/fs_fv).limit_denominator(2000)
LIR = resample_poly(LIR, r.numerator, r.denominator)   # anti-aliased
'''
# Ricker Wavelet lag
sigma_s = 1/(np.sqrt(2)*np.pi*0.3*f_cutoff_sim)
ts_ = np.ceil(8.5*sigma_s/dt_half)
lag = int(round(ts_*dt_half*fs))
'''
# PCS lag
sigma_s = 0.425 / f_cutoff_sim
ts_ = np.ceil(8.5 * sigma_s / dt_half)
lag = int(round(ts_ * dt_half * fs)) # ts_ on the dt_half grid -> samples at 44100

# apply lag
LIR = np.roll(LIR, -lag)

plt.figure()
plt.plot(LIR)
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

# sum filtered high-frequency arrays (all equal length now)
HIR = np.sum(H_IRs, axis=0)

# Butterworth filters (digital, zero-phase) for low- and high-pass
sos_lp = signal.butter(8, f_cutoff, 'low', fs=fs, output='sos')
sos_hp = signal.butter(8, f_cutoff, 'high', fs=fs, output='sos')

LIR = signal.sosfiltfilt(sos_lp, LIR)
HIR = signal.sosfiltfilt(sos_hp, HIR)

# Re-scale to correct convention if sim not calibrated
#HIR *= 4*np.pi
#g = first_peak(LIR) / first_peak(HIR) # if materials are the same only (comparison below on frequency domain)
#HIR *= g

for sos, name in [(sos_lp, 'low'), (sos_hp, 'high')]:
	w, h = signal.sosfreqz(sos, worN=2048, fs=fs)
	plt.figure()
	plt.semilogx(w, 20 * np.log10(abs(h) + 1e-30))
	plt.title(f'Butterworth {name}-pass response')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude [dB]')
	plt.margins(0, 0.1)
	plt.grid(which='both', axis='both')
	plt.axvline(f_cutoff, color='green') # cutoff frequency

plt.figure()
plt.plot(LIR,'b')
plt.plot(HIR,'g')

N = max(LIR.size, HIR.size)
IR_ = np.pad(LIR, (0, N-LIR.size)) + np.pad(HIR, (0, N-HIR.size))

plt.figure()
plt.plot(IR_)

# --- wave-based vs GA-based, both LP-filtered to f_cutoff for fair shape ---
# Use more ISM than RT here
# Make IRs long enough for good representation in the spectral domain

ga_t   = signal.sosfiltfilt(sos_lp, high_arrays[0])   # GA IR low-passed to f_cutoff (same band as FVM)
wave_t = LIR                                          # FVM, already resampled + LP

Nt   = max(wave_t.size, ga_t.size)
wave_t = np.pad(wave_t, (0, Nt-wave_t.size))
ga_t   = np.pad(ga_t,   (0, Nt-ga_t.size))
ga_t  *= first_peak(wave_t) / first_peak(ga_t)        # match level on the direct peak
ft     = np.fft.rfftfreq(Nt, 1/fs)

spl = lambda x: 20*np.log10(np.abs(np.fft.rfft(x))/2e-5 + 1e-30)

plt.figure()
# axial room modes only (one of l,m,n nonzero): f = c*i/(2L) per axis
nmax  = int(np.ceil(2*f_cutoff*max(Lx, Ly, Lz)/c)) + 1
modes = sorted({0.5*c*i/L for L in (Lx, Ly, Lz) for i in range(1, nmax)})
for fm in modes:
    if 20 <= fm <= f_cutoff:
        plt.axvline(fm, color='0.7', lw=0.5, zorder=0)
plt.plot(ft, spl(wave_t), 'b', label='wave-based (FVM)')
plt.plot(ft, spl(ga_t),   'g', label='GA-based (ISM/RT)')
plt.axvline(f_cutoff, color='r'); plt.xlim(20, f_cutoff)
plt.xlabel('Hz'); plt.ylabel('SPL [dB]'); plt.legend(); plt.grid(which='both')

# --- VALIDATION: IR_ vs modes of the real acoustic room (Lx,Ly,Lz already effective) ---
fmax_plot = 200 # Hz
nv = int(np.ceil(2*fmax_plot*max(Lx, Ly, Lz)/c)) + 1
modes = sorted({0.5*c*np.sqrt((l/Lx)**2 + (m/Ly)**2 + (n/Lz)**2)
                for l in range(nv) for m in range(nv) for n in range(nv)} - {0.0})
# axial only:
#modes = sorted({0.5*c*i/L for L in (Lx, Ly, Lz) for i in range(1, nv)})
ft_ir = np.fft.rfftfreq(IR_.size, 1/fs)
plt.figure()
for fm in modes:
    if 20 <= fm <= fmax_plot:
        plt.axvline(fm, color='0.6', lw=0.5, zorder=0)
plt.plot(ft_ir, spl(IR_), 'k', label='broadband IR')
plt.axvline(f_cutoff, color='r'); plt.xlim(20, fmax_plot)
plt.xlabel('Hz'); plt.ylabel('SPL [dB]'); plt.legend(); plt.grid(which='both')
plt.title('Modal check: IR peaks vs f_lmn (effective room)')

plt.show()



