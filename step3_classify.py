"""
Step 3: Classification
Replicating Iyer et al. (2023) - Scientific Reports

- Random Forest (1000 trees) and Logistic Regression classifiers
- 70/30 train/test split, repeated 100 times
- AUC reported per iteration and averaged
- Feature importance via mean decrease Gini (RF)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_CSV  = "data/features.csv"
N_ITER        = 100      # number of random train/test splits
TEST_SIZE     = 0.30     # 30% test set
N_TREES       = 1000     # RF trees
MIN_LEAF      = 5        # RF min samples per leaf
MAX_FEAT      = 6        # RF features per split (paper: sqrt of ~23 ≈ 6)
RANDOM_STATE  = 42
# ─────────────────────────────────────────────────────────────────────────────


def load_feature_sets(csv_path):
    """
    Load features and split into:
      - acoustic_feats: 23 Praat-based features
      - spectral_feats: LPC/LAR/Cep/MFCC mean+variance vectors
    """
    df = pd.read_csv(csv_path).dropna()
    y  = df["label"].values

    acoustic_cols = [
        "duration", "f0_mean", "f0_sd",
        "f1_mean", "f1_sd", "f2_mean", "f2_sd",
        "f3_mean", "f3_sd", "f4_mean", "f4_sd",
        "hnr",
        "jitter_local", "jitter_local_abs", "jitter_rap", "jitter_ppq5", "jitter_ddp",
        "shimmer_local", "shimmer_local_dB", "shimmer_apq3",
        "shimmer_apq5", "shimmer_apq11", "shimmer_dda",
    ]
    spectral_prefixes = ["lpc_mean", "lpc_var", "lar_mean", "lar_var",
                         "cep_mean", "cep_var", "mfcc_mean", "mfcc_var"]
    spectral_cols = [c for c in df.columns
                     if any(c.startswith(p) for p in spectral_prefixes)]

    X_acoustic = df[acoustic_cols].values
    X_spectral = df[spectral_cols].values

    return X_acoustic, X_spectral, y, acoustic_cols, spectral_cols


def evaluate_classifier(clf, X, y, n_iter=N_ITER, test_size=TEST_SIZE):
    """
    Run n_iter random 70/30 splits, return list of AUC scores.
    Mirrors paper's 100-iteration evaluation strategy.
    """
    aucs = []
    for seed in range(n_iter):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, prob))
    return aucs


def build_rf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=N_TREES,
            max_features=MAX_FEAT,
            min_samples_leaf=MIN_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])


def build_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE
        ))
    ])


def get_feature_importance(X, y, feature_names):
    """Train RF on full data and return feature importances (mean decrease Gini)."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    rf = RandomForestClassifier(
        n_estimators=N_TREES, max_features=MAX_FEAT,
        min_samples_leaf=MIN_LEAF, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_s, y)
    return rf.feature_importances_


def plot_results(results: dict, save_path="results/auc_boxplot.png"):
    """Boxplot of AUC distributions across all classifiers — mirrors Fig 1 in paper."""
    import os
    os.makedirs("results", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(results.keys())
    data   = [results[k] for k in labels]

    bp = ax.boxplot(data, patch_artist=True, notch=False)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
              "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]
    for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("AUC")
    ax.set_title("Classification AUC over 100 iterations (Iyer et al. 2023 replication)")
    ax.axhline(0.5, linestyle="--", color="grey", linewidth=0.8, label="Chance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Plot saved → {save_path}")
    plt.show()


def plot_feature_importance(importances, feature_names,
                             save_path="results/feature_importance.png", top_n=15):
    """Bar chart of top-N most important features."""
    import os
    os.makedirs("results", exist_ok=True)

    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(top_n), importances[idx], color="#4C72B0", alpha=0.8)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean Decrease Gini")
    ax.set_title("Top Feature Importances (RF on Acoustic Features)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Plot saved → {save_path}")
    plt.show()


def run():
    print("Loading features...")
    X_ac, X_sp, y, ac_names, sp_names = load_feature_sets(FEATURES_CSV)
    print(f"  Acoustic features : {X_ac.shape}")
    print(f"  Spectral features : {X_sp.shape}")
    print(f"  Labels            : {np.bincount(y)} (HC / PD)")

    results = {}

    # ── Acoustic: Random Forest ──────────────────────────────────────────────
    print("\n[1/4] RF on Acoustic Signal Features...")
    aucs = evaluate_classifier(build_rf(), X_ac, y)
    results["Acoustic\n(RF)"] = aucs
    print(f"  Mean AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  (paper: 0.72)")

    # ── Acoustic: Logistic Regression ────────────────────────────────────────
    print("[2/4] LR on Acoustic Signal Features...")
    aucs = evaluate_classifier(build_lr(), X_ac, y)
    results["Acoustic\n(LR)"] = aucs
    print(f"  Mean AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  (paper: 0.60)")

    # ── Spectral variance: RF (MFCC var — best performer in paper) ───────────
    mfcc_var_cols = [i for i, n in enumerate(sp_names) if "mfcc_var" in n]
    X_mfcc_var = X_sp[:, mfcc_var_cols]
    print("[3/4] RF on MFCC Variance Features...")
    aucs = evaluate_classifier(build_rf(), X_mfcc_var, y)
    results["MFCC Var\n(RF)"] = aucs
    print(f"  Mean AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  (paper: 0.73)")

    # ── Spectral variance: LR (MFCC var) ─────────────────────────────────────
    print("[4/4] LR on MFCC Variance Features...")
    aucs = evaluate_classifier(build_lr(), X_mfcc_var, y)
    results["MFCC Var\n(LR)"] = aucs
    print(f"  Mean AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  (paper: 0.73)")

    # ── Feature importance ───────────────────────────────────────────────────
    print("\nComputing feature importances...")
    importances = get_feature_importance(X_ac, y, ac_names)
    top_idx = np.argsort(importances)[::-1][:5]
    print("  Top 5 features:")
    for i in top_idx:
        print(f"    {ac_names[i]:<25} {importances[i]:.4f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Summary ───────────────────────────────────────────────────────")
    print(f"{'Feature Set':<25} {'Classifier':<6} {'Mean AUC':<10} {'SD'}")
    for k, v in results.items():
        label = k.replace("\n", " ")
        print(f"  {label:<23} {np.mean(v):.3f}      ±{np.std(v):.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_results(results)
    plot_feature_importance(importances, ac_names)


if __name__ == "__main__":
    run()
