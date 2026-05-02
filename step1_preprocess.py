"""
Step 1: Preprocessing - Fixed version
Removes broken dB filter (paper uses raw amplitude scale, not dB scale)
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

RAW_DIR        = "data/raw"
OUT_DIR        = "data/processed"
SR_TARGET      = 8000
MIN_DUR        = 1.5
SILENCE_WIN    = 100
SILENCE_THRESH = 0.001


def trim_silence(signal, win=SILENCE_WIN, thresh=SILENCE_THRESH):
    energy = np.array([
        np.sum(signal[i:i+win] ** 2)
        for i in range(0, len(signal) - win, win)
    ])
    mask = energy > thresh
    if not np.any(mask):
        return signal
    first = np.argmax(mask) * win
    last  = (len(mask) - np.argmax(mask[::-1])) * win
    return signal[first:last]


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    processed, skipped = 0, 0

    for wav_path in sorted(Path(RAW_DIR).glob("*.wav")):
        signal, sr = librosa.load(str(wav_path), sr=SR_TARGET, mono=True)

        # rescale to [-1, 1]
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val

        # trim silence
        signal = trim_silence(signal)

        # discard if < 1.5s
        if len(signal) / sr < MIN_DUR:
            print(f"  [SKIP] {wav_path.name} — {len(signal)/sr:.2f}s < 1.5s")
            skipped += 1
            continue

        sf.write(f"{OUT_DIR}/{wav_path.name}", signal, SR_TARGET)
        processed += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    run()
