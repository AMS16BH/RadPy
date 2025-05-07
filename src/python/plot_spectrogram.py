import numpy as np
import matplotlib.pyplot as plt
import wave
import os

filename = '..'

# === Fixed Parameters ===
hydrogen_freq = 1420.405751e6

S11_dB = -16.88
S21_dB = -44.86
z_real = 66.3
z_imag = 13.6

# load and extract data
if not os.path.exists(filename):
  raise FileNotFoundError(f'File not found: {filename}')

with wave.open(filename, 'rb') as wav_file:
  sample_rate = wav_file.getframerate()
  
with open(filename, 'rb') as f:
  f.read(44)
  raw_data = np.fromfile(f,dtype=np.float32)


# Data Structure

iq_data = raw_data.reshape(-1, 2)
i = iq_data[:,0]
q = iq_data[:,1]
iq = i + 1j*q
iq[~np.isfinite(iq)] = 0.0
duration_sec = len(iq) / sample_rate

# VNA Corrections
gamma_mag = 10 ** (S11_db / 20)
efficiency = 1 - gamma_mag**2

s21_gain = 10 ** (-S21_dB / 10)
gain_correction_dB = -10 * np.log10(s21_gain)

Z = Z_real + 1j * z_imag
gamma_complex = (Z-50) / (Z+50)
phase_correction = np.exp(-1j *np.angle(gamma_complex))
iq_corrected = iq * phase_correction
iq_corrected[np.abs(iq_corrected) > 1e5] = 0.0
                          
