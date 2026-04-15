#!/usr/bin/env python3
"""
ANC Hackathon — Audio Playback
================================

Listen to the raw signals before you start coding.

Usage:
    python play_audio.py                # play all signals
    python play_audio.py --signal x     # reference noise only
    python play_audio.py --signal d     # noise at eardrum only
    python play_audio.py --signal music # music source only
    python play_audio.py --signal ear   # what the ear hears (noise + music)
"""

import argparse
import os

import numpy as np
import sounddevice as sd
from scipy import signal as sig

DATA_DIR = "data"
FS = 16_000


def load_signal(name, data_dir):
    path = os.path.join(data_dir, name)
    arr = np.load(path)
    print(f"  Loaded {name:25s}  {len(arr)} samples  "
          f"({len(arr)/FS:.2f} s)  RMS {np.sqrt(np.mean(arr**2)):.4f}")
    return arr


def play(signal, fs, label):
    duration = len(signal) / fs
    print(f"\n  ▶  Playing: {label}  ({duration:.1f} s)")
    print("     Press Ctrl+C to skip.\n")
    try:
        sd.play(signal.astype(np.float32), samplerate=fs)
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        print("     ⏹  Skipped.")


def main():
    ap = argparse.ArgumentParser(description="ANC Hackathon Audio Player")
    ap.add_argument("--signal",
                    choices=["x", "d", "music", "ear", "all"],
                    default="all")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--volume", type=float, default=0.5)
    args = ap.parse_args()

    fs = FS
    vol = np.clip(args.volume, 0.0, 1.0)

    print("=" * 50)
    print("  ANC Hackathon — Audio Player")
    print("=" * 50)
    print(f"  Sample rate : {fs} Hz")
    print(f"  Volume      : {vol:.0%}")
    print()

    x = load_signal("x_ref.npy", args.data_dir)
    d = load_signal("d_target.npy", args.data_dir)

    music_path = os.path.join(args.data_dir, "music.npy")
    s_path = os.path.join(args.data_dir, "s_path_impulse.npy")
    has_music = os.path.exists(music_path)
    has_s = os.path.exists(s_path)

    if has_music:
        m = load_signal("music.npy", args.data_dir)
    if has_s:
        s_imp = load_signal("s_path_impulse.npy", args.data_dir)

    x = x * vol
    d = d * vol

    if args.signal in ("x", "all"):
        play(x, fs, "x[n] — Reference mic (ambient noise outside headphone)")

    if args.signal in ("d", "all"):
        play(d, fs, "d[n] — Noise at eardrum (through primary path P(z))")

    if args.signal in ("music", "all") and has_music:
        play(m * vol, fs, "music[n] — Music source")

    if args.signal in ("ear", "all") and has_music and has_s:
        music_at_ear = sig.fftconvolve(m, s_imp, mode="full")[:len(d)]
        n = min(len(d), len(music_at_ear))
        ear = d[:n] + music_at_ear[:n]
        ear *= vol / (np.max(np.abs(ear)) + 1e-12)
        play(ear, fs,
             "ear[n] — What you hear: noise + music, NO ANC")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
