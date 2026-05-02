"""
Step 4 (Optional): CNN with Transfer Learning
Replicating Iyer et al. (2023) - Scientific Reports

- Generate 600x600 spectrogram images from 1.5s of the sustained vowel /a/
- Fine-tune Inception V3 (pretrained on ImageNet) with custom classifier head
- 70/30 split repeated 100 times, AUC reported each iteration
- Paper reports mean AUC = 0.97 (color), 0.96 (grayscale)

Requirements: pip install tensorflow matplotlib librosa
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for image saving

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR  = "data/processed"
LABELS_CSV     = "data/labels.csv"
SPEC_DIR       = "data/spectrograms"   # where spectrogram images are saved
IMG_SIZE       = (600, 600)
DURATION       = 1.5                   # seconds — all recordings trimmed to this
SR             = 8000
N_FFT          = int(0.032 * SR)       # 256 samples (32ms)
HOP_LENGTH     = N_FFT // 2            # 50% overlap
N_ITER         = 100
TEST_SIZE      = 0.30
EPOCHS         = 10
BATCH_SIZE     = 4
LR             = 0.001
# ─────────────────────────────────────────────────────────────────────────────


# ── Spectrogram Generation ───────────────────────────────────────────────────

def generate_spectrograms(color=True):
    """
    Generate 600x600 spectrogram images from the first 1.5s of each recording.
    Saves as JPG to SPEC_DIR/color/ or SPEC_DIR/gray/
    """
    labels_df = pd.read_csv(LABELS_CSV)
    label_map = dict(zip(labels_df["filename"], labels_df["label"]))

    mode    = "color" if color else "gray"
    out_dir = Path(SPEC_DIR) / mode
    os.makedirs(out_dir, exist_ok=True)

    paths, labels = [], []

    for wav_path in sorted(Path(PROCESSED_DIR).glob("*.wav")):
        label = label_map.get(wav_path.name)
        if label is None:
            continue

        out_path = out_dir / (wav_path.stem + ".jpg")
        if not out_path.exists():
            signal, sr = librosa.load(str(wav_path), sr=SR, duration=DURATION)

            # pad if shorter than DURATION
            target_len = int(DURATION * SR)
            if len(signal) < target_len:
                signal = np.pad(signal, (0, target_len - len(signal)))
            else:
                signal = signal[:target_len]

            # STFT spectrogram
            D = librosa.stft(signal, n_fft=1024, hop_length=HOP_LENGTH,
                              win_length=N_FFT)
            S_db = 10 * np.log10(np.abs(D) / np.max(np.abs(D)) + 1e-10)

            # render to image
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            if color:
                img = librosa.display.specshow(S_db, sr=SR, hop_length=HOP_LENGTH,
                                               x_axis="time", y_axis="hz",
                                               ax=ax, cmap="viridis")
            else:
                img = librosa.display.specshow(S_db, sr=SR, hop_length=HOP_LENGTH,
                                               x_axis="time", y_axis="hz",
                                               ax=ax, cmap="gray")
            ax.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(str(out_path), format="jpg", bbox_inches="tight",
                        pad_inches=0, dpi=100)
            plt.close()

        paths.append(str(out_path))
        labels.append(int(label))

    print(f"  Generated {len(paths)} spectrogram images → {out_dir}")
    return np.array(paths), np.array(labels)


# ── CNN Model ─────────────────────────────────────────────────────────────────

def build_inception_model(input_shape=(600, 600, 3)):
    """
    Inception V3 pretrained on ImageNet with custom classification head.
    Matches paper architecture: BatchNorm → Dense → Dropout → Dense(sigmoid)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import InceptionV3

    base = InceptionV3(include_top=False, weights="imagenet",
                       input_shape=input_shape, pooling="avg")
    base.trainable = False   # freeze pretrained weights

    x = base.output
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def load_image(path, color=True):
    """Load and normalise a spectrogram image to [0,1]."""
    import tensorflow as tf
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    if not color:
        img = tf.image.rgb_to_grayscale(img)
        img = tf.repeat(img, 3, axis=-1)   # replicate to 3ch for InceptionV3
    img = img / 255.0
    return img.numpy()


def run_cnn(color=True, n_iter=N_ITER):
    """
    Run the CNN evaluation: n_iter random 70/30 splits, report mean AUC.
    """
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    mode = "color" if color else "grayscale"
    print(f"\nRunning CNN ({mode} spectrograms, {n_iter} iterations)...")

    paths, labels = generate_spectrograms(color=color)

    # pre-load all images into memory (81 × 600 × 600 × 3 ≈ manageable)
    print("  Loading images into memory...")
    X = np.stack([load_image(p, color=color) for p in paths])
    y = labels

    aucs = []
    for i in range(n_iter):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=i, stratify=y
        )
        model = build_inception_model()
        model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        probs = model.predict(X_te, verbose=0).flatten()
        auc   = roc_auc_score(y_te, probs)
        aucs.append(auc)
        if (i + 1) % 10 == 0:
            print(f"  Iter {i+1:>3}/{n_iter} — running mean AUC: {np.mean(aucs):.3f}")

        # save model checkpoint
        os.makedirs("results/checkpoints", exist_ok=True)
        model.save_weights(f"results/checkpoints/iter_{i:03d}.h5")

    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)
    print(f"\n  {mode.capitalize()} CNN — Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"  Paper reports: {'0.97' if color else '0.96'}")

    # save AUC log
    os.makedirs("results", exist_ok=True)
    pd.DataFrame({"auc": aucs}).to_csv(f"results/cnn_{mode}_aucs.csv", index=False)
    return aucs


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run color CNN first (paper's primary result)
    # Set n_iter=10 for a quick test; use 100 for full replication
    color_aucs = run_cnn(color=True,  n_iter=10)
    gray_aucs  = run_cnn(color=False, n_iter=10)

    print("\n── Final Summary ────────────────────────────────────────────────")
    print(f"  Color CNN     : {np.mean(color_aucs):.3f} ± {np.std(color_aucs):.3f}  (paper: 0.97)")
    print(f"  Grayscale CNN : {np.mean(gray_aucs):.3f}  ± {np.std(gray_aucs):.3f}  (paper: 0.96)")
