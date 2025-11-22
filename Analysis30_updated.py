# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 15:13:58 2025

@author: Asus
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, c

data_ref = np.loadtxt('air_wg30_delay_2.txt', skiprows=1)
data_sam = np.loadtxt('sam_wg30_delay_2.txt', skiprows=1)

# Extract time (ps) and E-field amplitudes
time_ps_ref, S_ref = data_ref[:, 1], data_ref[:, 7]
time_ps_sam, S_sam = data_sam[:, 1], data_sam[:, 7]

# Convert time to seconds for FFT

t1 = np.min(time_ps_sam)
idx = np.argmin(np.abs(time_ps_ref - t1))

t_add=time_ps_ref[0:idx+1]

min_val1=S_sam[0]
min_val2=S_ref[-1]

t_ps_sam=np.concatenate((t_add,time_ps_sam))
S_sam2 = np.concatenate((min_val1*np.ones(len(t_add)), S_sam))


t2 = np.max(time_ps_ref)
idx2 = np.argmin(np.abs(time_ps_sam - t2))
t_add2=time_ps_sam[idx2:]
t_ps_ref=np.concatenate((time_ps_ref,t_add2))
S_ref2 = np.concatenate((S_ref,min_val2*np.ones(len(t_add))))



i0 = np.where((time_ps_ref > 431) & (time_ps_ref < 432))
zero_cross_idx1=np.argmin(np.abs(S_ref[i0]-0))
zero_cross1=time_ps_ref[i0][zero_cross_idx1]
print(zero_cross1)

i1 = np.where((time_ps_sam > 436) & (time_ps_sam < 437))
zero_cross_idx2=np.argmin(np.abs(S_sam[i1]-0))
zero_cross2=time_ps_sam[i1][zero_cross_idx2]
print(zero_cross2)



plt.figure(figsize=(10, 4))
plt.plot(t_ps_ref, S_ref2, label='Reference (Air)', color='blue')
plt.plot(t_ps_sam, S_sam2, label='Sample', color='red')
#plt.axvline(x=zero_cross1, color='blue', linestyle='--')
#plt.axvline(x=zero_cross2, color='red', linestyle='--')
plt.title('Time-domain terahertz waveform ($30^\circ$)')
plt.xlabel('Time (ps)')
plt.ylabel('Signal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


normalized_S_ref=S_ref2/max(S_ref2)
normalized_S_sam=S_sam2/max(S_ref2)


plt.figure(figsize=(10, 4))
plt.plot(t_ps_ref, normalized_S_ref, label='Reference (Air)', color='blue')
plt.plot(t_ps_sam, normalized_S_sam, label='Sample', color='red')
#plt.axvline(x=zero_cross1, color='blue', linestyle='--')
#plt.axvline(x=zero_cross2, color='red', linestyle='--')
plt.title('Normalized Signals')
plt.xlabel('Time (ps)')
plt.ylabel('Normalized Signal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

peak_voltage=1.78


R = 5730   #V/W
fr=1000    #Hz
T_cb = 0.7223
T_pump=0.9338
T=T_cb*T_pump


print('Total T =',T)

W_max=peak_voltage/(T*R*fr)
print('max W',W_max)


w_x=0.804*1e-3
w_y=0.828*1e-3

A_eff=np.pi*w_x*w_y/2

dt=(time_ps_ref[1]-time_ps_ref[0]) * 1e-12
tau_eff = sum((x**2) * dt for x in normalized_S_ref)
E_0=np.sqrt(W_max/(epsilon_0*c*A_eff*tau_eff))




print(E_0)


E_ref=normalized_S_ref*E_0
E_sam=normalized_S_sam*E_0


plt.figure(figsize=(10, 4))
plt.plot(t_ps_ref, E_ref*1e-5, label='Reference (Air)', color='blue')
plt.plot(t_ps_sam, E_sam*1e-5, label='Sample', color='red')
#plt.axvline(x=zero_cross1, color='blue', linestyle='--')
#plt.axvline(x=zero_cross2, color='red', linestyle='--')
#plt.title('Elctric field')
plt.xlabel('Time (ps)')
plt.ylabel('Electric field (KV/cm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


time_s_ref = time_ps_ref * 1e-12
time_s_sam = time_ps_sam * 1e-12

t_s_ref = t_ps_ref * 1e-12
t_s_sam = t_ps_sam * 1e-12




def compute_fft(time_s, signal, N_new=None):
    N = len(signal)
    dt = time_s[1] - time_s[0]
    
    # If N_new is specified and larger than N, pad the signal
    if N_new and N_new > N:
        # Create a new array of size N_new and fill the first N elements with the signal
        # The rest are automatically zero (if initialized correctly, e.g., with np.zeros)
        padded_signal = np.zeros(N_new)
        padded_signal[:N] = signal
        signal_to_fft = padded_signal
        N_used = N_new
    else:
        signal_to_fft = signal
        N_used = N

    fft_vals = np.fft.fft(signal_to_fft)
    freqs = np.fft.fftfreq(N_used, dt)
    half = N_used // 2
    return freqs[:half], fft_vals[:half]

# --- Example of use ---
# Let's say your current N is about 4000. Choose a larger power of 2, like 16384.
N_FFT = 16384 


freq, Eref_w = compute_fft(time_s_ref, S_ref, N_new=N_FFT) 
_, Esam_w = compute_fft(time_s_sam, S_sam, N_new=N_FFT) 



freq_THz = freq / 1e12 

amp_ref = np.abs(Eref_w)
amp_sam = np.abs(Esam_w)









"""
plt.figure(figsize=(10, 4))
plt.plot(freq_THz, Eref_w, label='Air', color='blue')
plt.plot(freq_THz, Esam_w, label='Sample', color='red')
plt.title('Amplitude Spectrum (Frequency Domain)')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""
#plt.figure(figsize=(10, 4))
plt.plot(freq_THz, np.abs(Eref_w), label='Reference', color='blue')
plt.plot(freq_THz, np.abs(Esam_w), label='Sample', color='red')
plt.yscale('log') # Log scale is often better for spectra
plt.title('Amplitude Spectrum')
plt.xlabel('Frequency (THz)')
plt.ylabel('|E($\omega$)|')
plt.xlim(0, 4) # Limit x-axis to relevant THz range
plt.legend()
plt.grid(True, which="both", ls="-")
plt.tight_layout()
plt.show()

T_w = Esam_w / Eref_w   # Complex ratio


#freq_THz=freq_THz[1:]
#freq=freq[1:]
#T_w=T_w[1:]


phi_w=np.unwrap(np.angle(T_w)) #phase
T_amp = np.abs(T_w)     # Magnitude

# ------------------- Material Parameters --------------------
d = 0.5e-3  # 

omega = 2 * np.pi * freq

# ------------------- Refractive Index --------------------
n_w = 1 + (phi_w * c) / (omega *d)

#print(n_w)
print(max(n_w[1:]))

# ------------------- Absorption Coefficient --------------------
alpha_w = -(2 / d) * np.log(((n_w + 1)**2 / (4 * n_w)) * T_amp)


# ------------------- Plot Refractive Index --------------------
plt.figure(figsize=(8, 5))
plt.plot(freq_THz, n_w, color='purple')
plt.xlabel('Frequency (THz)')
plt.ylabel('Refractive Index n(ω)')
plt.title('Refractive Index vs Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------- Plot Absorption Coefficient --------------------
plt.figure(figsize=(8, 5))
plt.plot(freq_THz, alpha_w, color='green')
plt.xlabel('Frequency (THz)')
plt.ylabel('Absorption Coefficient α(ω) (1/m)')
plt.title('Absorption Coefficient vs Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()


# ... [Previous calculation parts remain the same] ...

# 1. Auto-detect Cutoff (Logic remains the same)
high_freq_threshold = 3.5
noise_indices = np.where(freq_THz > high_freq_threshold)[0]
if len(noise_indices) > 0:
    noise_level = np.mean(amp_ref[noise_indices])
else:
    noise_level = np.mean(amp_ref[-int(len(amp_ref)*0.1):])

# 2. Dynamic Range Limit Calculation (Remains the same)
dynamic_range = amp_ref / noise_level
# Avoid division by zero in n_w if present by using a mask or ignoring index 0 later
T_fr = (4 * n_w) / ((n_w + 1)**2)
valid_dr_mask = (dynamic_range * T_fr) > 1
alpha_max_detectable = np.zeros_like(freq_THz)
# Use 'where' to avoid log warnings on invalid data
alpha_max_detectable[valid_dr_mask] = (2 / d) * np.log(dynamic_range[valid_dr_mask] * T_fr[valid_dr_mask])

# 3. Cutoff Index
snr_threshold = 3.0
# Find where signal drops below threshold *after* the peak
peak_idx = np.argmax(amp_ref)
low_snr_region = np.where(amp_ref[peak_idx:] < (noise_level * snr_threshold))[0]

if len(low_snr_region) > 0:
    cutoff_idx = peak_idx + low_snr_region[0]
else:
    cutoff_idx = len(freq_THz)

print(f"Auto-detected Cutoff Frequency: {freq_THz[cutoff_idx]:.2f} THz")

# --- FIX 1: Slice from index 1 to avoid omega=0 (NaN) issues ---
freq_clean = freq_THz[1:cutoff_idx]
n_clean = n_w[1:cutoff_idx]
alpha_clean = alpha_w[1:cutoff_idx]
alpha_max_clean = alpha_max_detectable[1:cutoff_idx]

# =============================================================================
#  PLOTTING THE CLEANED RESULTS
# =============================================================================

# Plot 1: Refractive Index
plt.figure(figsize=(8, 5))
plt.plot(freq_clean, n_clean, color='purple', label=r'Refractive Index $n(\omega)$')
plt.xlabel('Frequency (THz)')
plt.ylabel('Refractive Index')
plt.title('Refractive Index (Reliable Bandwidth)')
plt.grid(True)
plt.xlim(0, freq_clean[-1])
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Absorption Coefficient
plt.figure(figsize=(8, 5))
plt.plot(freq_clean, alpha_clean, color='green', label=r'Measured $\alpha(\omega)$')
plt.plot(freq_clean, alpha_max_clean, color='gray', linestyle='--', label='Max Detectable Limit')

plt.xlabel('Frequency (THz)')
# --- FIX 2: Use raw string r'...' for LaTeX symbols ---
plt.ylabel(r'Absorption Coefficient $\alpha(\omega)$ (1/m)')
plt.title('Absorption Coefficient vs Frequency')
plt.xlim(0, freq_clean[-1])

# Use nanmin/nanmax to be safe against any remaining NaNs
plt.ylim(np.nanmin(alpha_clean), np.nanmax(alpha_clean) * 1.2)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()