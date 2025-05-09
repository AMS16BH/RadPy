import numpy as np
import matplotlib.pyplot as plt
import wave
import os

# === Parameters ===
filename = r"..."
hydrogen_freq = 1420.405751e6  # Hz

# VNA Measurements
S11_dB = -16.88
S21_dB = -44.86
Z_real = 66.3
Z_imag = 13.6

# === Load WAV file and extract I/Q ===
if not os.path.exists(filename):
    raise FileNotFoundError(f"File not found: {filename}")

with wave.open(filename, 'rb') as wav_file:
    sample_rate = wav_file.getframerate()

with open(filename, 'rb') as f:
    f.read(44)  # skip header
    raw_data = np.fromfile(f, dtype=np.float32)

iq_data = raw_data.reshape(-1, 2)
i = iq_data[:, 0]
q = iq_data[:, 1]
iq = i + 1j * q
iq[~np.isfinite(iq)] = 0.0  # clean NaNs
duration_sec = len(iq) / sample_rate


gamma_mag = 10 ** (S11_dB / 20)
efficiency = 1 - gamma_mag**2

s21_gain = 10 ** (-S21_dB / 10)
gain_correction_dB = -10 * np.log10(s21_gain)

Z = Z_real + 1j * Z_imag
gamma_complex = (Z - 50) / (Z + 50)
phase_correction = np.exp(-1j * np.angle(gamma_complex))
iq_corrected = iq * phase_correction
iq_corrected[np.abs(iq_corrected) > 1e5] = 0.0  # remove outliers

# === Compute Spectrogram ===
def compute_spectrogram(signal, fs, window_size=4096, overlap=3024):
    step = window_size - overlap
    segments = (len(signal) - window_size) // step
    spectrogram = []
    for i in range(segments):
        start = i * step
        windowed = signal[start:start+window_size] * np.hamming(window_size)
        if not np.any(np.isfinite(windowed)):
            continue
        spectrum = np.fft.fftshift(np.fft.fft(windowed))
        ref_level = np.median(np.abs(spectrum[np.isfinite(spectrum)])) + 1e-10
        magnitude = np.abs(spectrum) / ref_level
        corrected_power = magnitude + gain_correction_dB + 10 * np.log10(efficiency)
        spectrogram.append(corrected_power)
    return np.array(spectrogram).T

spec = compute_spectrogram(iq_corrected, sample_rate)

# === Time/Frequency Axes ===
time_axis = np.linspace(0, duration_sec, spec.shape[1])
freq_axis = np.linspace(-sample_rate, sample_rate, spec.shape[0]) + hydrogen_freq
freq_mhz = freq_axis / 1e6

# === Plot (no masking) ===
plt.figure(figsize=(12, 6))
plt.imshow(spec, aspect='auto',interpolation='hamming',
           extent=[time_axis[0], time_axis[-1], freq_mhz[0], freq_mhz[-1]],
           cmap='inferno', vmin=np.nanpercentile(spec, 10), vmax=np.nanpercentile(spec, 99))
plt.xlabel("Time [s]")
plt.ylabel("Frequency [MHz]")
plt.title("Hydrogen Line Spectrogram (Full Frequency Range)")
plt.colorbar(label="Power [dB]")
plt.tight_layout()
plt.show()
