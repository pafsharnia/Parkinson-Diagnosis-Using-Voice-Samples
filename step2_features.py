"""
Step 2: Feature Extraction
Replicating Iyer et al. (2023) - Scientific Reports

Part A — Acoustic Signal Features (23 features via Parselmouth/Praat):
  - Mean & SD of F0 (fundamental frequency)
  - Mean & SD of formants F1–F4
  - HNR (harmonics-to-noise ratio)
  - Jitter (5 variants)
  - Shimmer (6 variants)
  - Duration

Part B — Spectral Features (LPC, LAR, Cepstral, MFCC):
  - 32ms sliding window, 50% overlap
  - Mean and variance vectors per feature type
"""

import os
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import pandas as pd
from pathlib import Path
from scipy.signal import lfilter
from scipy.linalg import solve_toeplitz


# ── Config ───────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
LABELS_CSV    = "data/labels.csv"   # columns: filename, label (PD=1, HC=0), sex
OUT_CSV       = "data/features.csv"

SR      = 8000
WIN_S   = 0.032          # 32ms window for spectral features
HOP_R   = 0.5            # 50% overlap
LPC_ORD = 10             # autoregressive model order
N_MFCC  = 10             # number of MFCC coefficients
# ─────────────────────────────────────────────────────────────────────────────


# ── Part A: Acoustic Signal Features ─────────────────────────────────────────

def extract_acoustic_features(wav_path):
    """
    Extract 23 acoustic features using Parselmouth (Python interface to Praat).
    Matches Table in Iyer et al. Methods section exactly.
    """
    snd = parselmouth.Sound(str(wav_path))
    duration = snd.duration

    # ── Pitch (F0) ──
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
    f0_sd   = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    f0_mean = f0_mean if not np.isnan(f0_mean) else 0.0
    f0_sd   = f0_sd   if not np.isnan(f0_sd)   else 0.0

    # ── Formants F1–F4 ──
    formants = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    formant_features = {}
    for fi in range(1, 5):
        fmean = call(formants, "Get mean", fi, 0, 0, "Hertz")
        fsd   = call(formants, "Get standard deviation", fi, 0, 0, "Hertz")
        formant_features[f"f{fi}_mean"] = fmean if not np.isnan(fmean) else 0.0
        formant_features[f"f{fi}_sd"]   = fsd   if not np.isnan(fsd)   else 0.0

    # ── HNR ──
    hnr_obj = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr     = call(hnr_obj, "Get mean", 0, 0)
    hnr     = hnr if not np.isnan(hnr) else 0.0

    # ── PointProcess (needed for jitter & shimmer) ──
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)

    # ── Jitter (5 variants) ──
    jitter_local      = call(point_process, "Get jitter (local)",          0, 0, 0.0001, 0.02, 1.3)
    jitter_local_abs  = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap        = call(point_process, "Get jitter (rap)",             0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5       = call(point_process, "Get jitter (ppq5)",            0, 0, 0.0001, 0.02, 1.3)
    jitter_ddp        = call(point_process, "Get jitter (ddp)",             0, 0, 0.0001, 0.02, 1.3)
    jitters = [jitter_local, jitter_local_abs, jitter_rap, jitter_ppq5, jitter_ddp]
    jitters = [0.0 if (v is None or np.isnan(v)) else v for v in jitters]

    # ── Shimmer (6 variants) ──
    shimmer_local     = call([snd, point_process], "Get shimmer (local)",         0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_local_dB  = call([snd, point_process], "Get shimmer (local_dB)",     0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3      = call([snd, point_process], "Get shimmer (apq3)",           0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5      = call([snd, point_process], "Get shimmer (apq5)",           0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq11     = call([snd, point_process], "Get shimmer (apq11)",          0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_dda       = call([snd, point_process], "Get shimmer (dda)",            0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmers = [shimmer_local, shimmer_local_dB, shimmer_apq3, shimmer_apq5, shimmer_apq11, shimmer_dda]
    shimmers = [0.0 if (v is None or np.isnan(v)) else v for v in shimmers]

    features = {
        "duration":          duration,
        "f0_mean":           f0_mean,
        "f0_sd":             f0_sd,
        **formant_features,
        "hnr":               hnr,
        "jitter_local":      jitters[0],
        "jitter_local_abs":  jitters[1],
        "jitter_rap":        jitters[2],
        "jitter_ppq5":       jitters[3],
        "jitter_ddp":        jitters[4],
        "shimmer_local":     shimmers[0],
        "shimmer_local_dB":  shimmers[1],
        "shimmer_apq3":      shimmers[2],
        "shimmer_apq5":      shimmers[3],
        "shimmer_apq11":     shimmers[4],
        "shimmer_dda":       shimmers[5],
    }
    return features   # 23 features total


# ── Part B: Spectral Features ─────────────────────────────────────────────────

def levinson_durbin(r, order):
    """
    Levinson-Durbin recursion to solve Yule-Walker equations.
    Returns LPC coefficients and partial correlation (PARCOR) coefficients.
    """
    a = np.zeros(order)
    k = np.zeros(order)
    e = r[0]
    for i in range(order):
        k[i] = -np.dot(a[:i], r[i-1::-1]) - r[i+1] if i > 0 else -r[1] / r[0]
        k[i] /= e if e != 0 else 1e-10
        a_new = a.copy()
        a_new[i] = k[i]
        for j in range(i):
            a_new[j] = a[j] + k[i] * a[i-1-j]
        a = a_new
        e *= (1 - k[i] ** 2)
    return a, k


def lpc_to_lar(parcor):
    """Convert PARCOR coefficients to Log-Area Ratios (LAR)."""
    parcor = np.clip(parcor, -1 + 1e-10, 1 - 1e-10)
    return np.log((1 - parcor) / (1 + parcor))


def lpc_to_cepstrum(lpc, order):
    """Recursively compute Cepstral Coefficients from LPC coefficients."""
    cep = np.zeros(order)
    cep[0] = -lpc[0]
    for n in range(1, order):
        cep[n] = -lpc[n] - sum((k+1)/n * cep[k] * lpc[n-1-k] for k in range(n))
    return cep


def extract_spectral_features(wav_path):
    """
    Extract LPC, LAR, Cepstral, and MFCC variance vectors from a recording.
    Uses 32ms sliding window with 50% overlap as per the paper.
    Returns dict of mean and variance vectors for each feature type.
    """
    signal, sr = librosa.load(str(wav_path), sr=SR, mono=True)
    n_fft  = int(WIN_S * sr)          # 256 samples at 8kHz
    hop    = int(n_fft * (1 - HOP_R)) # 128 samples (50% overlap)

    lpc_frames, lar_frames, cep_frames, mfcc_frames = [], [], [], []

    # sliding window
    for start in range(0, len(signal) - n_fft, hop):
        frame = signal[start:start + n_fft]
        frame = frame * np.hanning(len(frame))

        # autocorrelation
        r = np.correlate(frame, frame, mode='full')
        r = r[len(r)//2:len(r)//2 + LPC_ORD + 1]
        r[0] += 1e-6  # regularisation

        # LPC & PARCOR via Levinson-Durbin
        lpc, parcor = levinson_durbin(r, LPC_ORD)
        lar = lpc_to_lar(parcor)
        cep = lpc_to_cepstrum(lpc, LPC_ORD)

        lpc_frames.append(lpc)
        lar_frames.append(lar)
        cep_frames.append(cep)

    # MFCC via librosa
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=n_fft, hop_length=hop)
    mfcc_frames = mfcc.T  # shape: (frames, n_mfcc)

    def mean_var(frames):
        arr = np.array(frames)
        return np.mean(arr, axis=0), np.var(arr, axis=0)

    lpc_m, lpc_v = mean_var(lpc_frames)
    lar_m, lar_v = mean_var(lar_frames)
    cep_m, cep_v = mean_var(cep_frames)
    mfc_m, mfc_v = mean_var(mfcc_frames)

    features = {}
    for i, v in enumerate(lpc_m):  features[f"lpc_mean_{i}"] = v
    for i, v in enumerate(lpc_v):  features[f"lpc_var_{i}"]  = v
    for i, v in enumerate(lar_m):  features[f"lar_mean_{i}"] = v
    for i, v in enumerate(lar_v):  features[f"lar_var_{i}"]  = v
    for i, v in enumerate(cep_m):  features[f"cep_mean_{i}"] = v
    for i, v in enumerate(cep_v):  features[f"cep_var_{i}"]  = v
    for i, v in enumerate(mfc_m):  features[f"mfcc_mean_{i}"] = v
    for i, v in enumerate(mfc_v):  features[f"mfcc_var_{i}"]  = v

    return features


# ── Main runner ───────────────────────────────────────────────────────────────

def run():
    labels_df = pd.read_csv(LABELS_CSV)   # columns: filename, label, sex
    label_map = dict(zip(labels_df["filename"], labels_df["label"]))

    rows = []
    for wav_path in sorted(Path(PROCESSED_DIR).glob("*.wav")):
        label = label_map.get(wav_path.name)
        if label is None:
            print(f"  [WARN] No label for {wav_path.name}, skipping")
            continue

        print(f"  Extracting: {wav_path.name}")
        try:
            acoustic = extract_acoustic_features(wav_path)
            spectral = extract_spectral_features(wav_path)
            row = {"filename": wav_path.name, "label": label, **acoustic, **spectral}
            rows.append(row)
        except Exception as e:
            print(f"  [ERROR] {wav_path.name}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    run()
