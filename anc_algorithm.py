#!/usr/bin/env python3
"""
ANC Hackathon — Your Algorithm
================================

This is your main working file.  Implement both phases here.

Usage:
    python anc_algorithm.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

# ═══════════════════════════════════════════════════
# 0.  Load Data
# ═══════════════════════════════════════════════════

FS = 16_000  # sample rate (Hz)

x = np.load("data/x_ref.npy")            # reference mic signal
d = np.load("data/d_target.npy")          # noise at eardrum (through P(z))
s_path = np.load("data/s_path_impulse.npy")  # secondary path impulse response

N = len(x)
print(f"Signal length: {N} samples ({N/FS:.1f} s)")
print(f"Secondary path: {len(s_path)} taps")


# ═══════════════════════════════════════════════════
# Phase I:  Standard LMS  (assume S(z) = 1)
# ═══════════════════════════════════════════════════

# TODO: Implement the LMS adaptive filter
# TODO: Plot MSE convergence and find the optimal step size

# --- Your LMS implementation here ---
# Your code should produce:
#   e_lms : np.ndarray of shape (N,) — the error (residual) signal

e_lms = np.zeros(N)  # ← replace with your result



# ── Phase I Evaluation (DO NOT MODIFY) ────────────────────
print("\n" + "=" * 55)
print("  Phase I Evaluation — LMS (S(z) = 1)")
print("=" * 55)
_tail = int(0.2 * N)
_p_d = np.mean(d[-_tail:] ** 2)
_p_e1 = np.mean(e_lms[-_tail:] ** 2)
_att1 = 10 * np.log10(_p_d / (_p_e1 + 1e-30))
print(f"  Noise power  (d):  {10*np.log10(_p_d):.1f} dB")
print(f"  Residual     (e):  {10*np.log10(_p_e1 + 1e-30):.1f} dB")
print(f"  ★ Attenuation:     {_att1:.1f} dB")
print("=" * 55)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
_t = np.arange(N) / FS
axes[0].plot(_t, d, lw=0.3, color="coral")
axes[0].set(ylabel="Amp", title="d[n] — Noise at eardrum")
axes[1].plot(_t, e_lms, lw=0.3, color="green")
axes[1].set(ylabel="Amp", xlabel="Time (s)",
            title=f"e[n] — LMS residual  (attenuation = {_att1:.1f} dB)")
plt.tight_layout()
plt.savefig("phase1_result.png", dpi=150)
plt.close()
print("  Saved: phase1_result.png\n")


# ═══════════════════════════════════════════════════
# Phase II:  Filtered-x LMS (FxLMS)
# ═══════════════════════════════════════════════════

# TODO: Show that standard LMS diverges when S(z) is introduced
# TODO: Implement the FxLMS algorithm to restore stability

# --- Your FxLMS implementation here ---
# Your code should produce:
#   e_fxlms : np.ndarray of shape (N,) — the FxLMS error signal

e_fxlms = np.zeros(N)  # ← replace with your result



# ── Phase II Evaluation (DO NOT MODIFY) ───────────────────
print("=" * 55)
print("  Phase II Evaluation — FxLMS")
print("=" * 55)
_p_e2 = np.mean(e_fxlms[-_tail:] ** 2)
_att2 = 10 * np.log10(_p_d / (_p_e2 + 1e-30))
print(f"  Noise power  (d):  {10*np.log10(_p_d):.1f} dB")
print(f"  Residual     (e):  {10*np.log10(_p_e2 + 1e-30):.1f} dB")
print(f"  ★ Attenuation:     {_att2:.1f} dB")
print("=" * 55)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(_t, d, lw=0.3, color="coral")
axes[0].set(ylabel="Amp", title="d[n] — Noise at eardrum")
axes[1].plot(_t, e_fxlms, lw=0.3, color="green")
axes[1].set(ylabel="Amp", xlabel="Time (s)",
            title=f"e[n] — FxLMS residual  (attenuation = {_att2:.1f} dB)")
plt.tight_layout()
plt.savefig("phase2_result.png", dpi=150)
plt.close()
print("  Saved: phase2_result.png\n")

# ── Spectrum Comparison ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
_f, _Pd = sig.welch(d, fs=FS, nperseg=1024)
_f, _Pe1 = sig.welch(e_lms, fs=FS, nperseg=1024)
_f, _Pe2 = sig.welch(e_fxlms, fs=FS, nperseg=1024)
ax.semilogy(_f, _Pd,  color="coral",     lw=1.2, label="d[n] noise (before ANC)")
ax.semilogy(_f, _Pe1, color="steelblue", lw=1.2, label=f"LMS residual ({_att1:.1f} dB)")
ax.semilogy(_f, _Pe2, color="green",     lw=1.2, label=f"FxLMS residual ({_att2:.1f} dB)")
ax.set(xlabel="Frequency (Hz)", ylabel="PSD",
       title="Noise Spectrum — Before vs After ANC")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("spectrum_comparison.png", dpi=150)
plt.close()
print("  Saved: spectrum_comparison.png")

# ── Audio Demo ────────────────────────────────────────────
try:
    import sounddevice as sd
    _music_path = "data/music.npy"
    if os.path.exists(_music_path):
        _music = np.load(_music_path)
        _music_ear = sig.fftconvolve(_music, s_path, mode="full")[:N]
        _n = min(N, len(_music_ear))
        _vol = 0.7

        _ear_before = d[:_n] + _music_ear[:_n]
        _ear_after  = e_fxlms[:_n] + _music_ear[:_n]

        def _play(s, label):
            s = s / (np.max(np.abs(s)) + 1e-12) * _vol
            print(f"\n  ▶  {label}  ({len(s)/FS:.1f} s)")
            try:
                sd.play(s.astype(np.float32), samplerate=FS); sd.wait()
            except KeyboardInterrupt:
                sd.stop(); print("     ⏹  Skipped.")

        print(f"\n{'='*55}")
        print("  Audio Demo")
        print(f"{'='*55}")
        _play(_ear_before, "BEFORE ANC — noise + music")
        _play(_ear_after,  "AFTER  ANC — FxLMS active")
        print("\n  Done.\n")
except ImportError:
    print("  (sounddevice not installed — skipping audio demo)")
