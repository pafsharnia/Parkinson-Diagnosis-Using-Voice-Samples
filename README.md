# Parkinson's Disease Detection from Voice — Replication of Iyer et al. (2023)
# COMP5405 Project

## Paper
Iyer, A., Kemp, A., Rahmatallah, Y., et al.
"A machine learning method to process voice samples for identification of Parkinson's disease"
Scientific Reports, 13, 20615 (2023)
https://doi.org/10.1038/s41598-023-47568-w

---

## Dataset
Download from figshare:
https://doi.org/10.6084/m9.figshare.23849127

Place all .wav files in:  data/raw/

Create data/labels.csv with columns:
  filename, label, sex
  e.g.:
  subject_001.wav,1,M
  subject_002.wav,0,F
  (label: 1=PD, 0=HC — sex: M/F)

---

## Setup

pip install -r requirements.txt

---

## Run

# Step 1 — Preprocess audio
python step1_preprocess.py

# Step 2 — Extract features
python step2_features.py

# Step 3 — Classify (RF + LR, 100 iterations)
python step3_classify.py

# Step 4 — CNN with Inception V3 (optional, GPU recommended)
python step4_cnn.py

---

## Expected Results (from paper)
| Method                  | Mean AUC |
|-------------------------|----------|
| Acoustic features + RF  | 0.72     |
| Acoustic features + LR  | 0.60     |
| MFCC variance + RF      | 0.73     |
| MFCC variance + LR      | 0.73     |
| CNN (color spectrograms)| 0.97     |
| CNN (gray spectrograms) | 0.96     |

---

## File Structure
pd_detection/
├── step1_preprocess.py     # audio cleaning
├── step2_features.py       # feature extraction (Praat + spectral)
├── step3_classify.py       # RF + LR classification
├── step4_cnn.py            # Inception V3 CNN (optional)
├── requirements.txt
├── data/
│   ├── raw/                ← put downloaded .wav files here
│   ├── processed/          ← cleaned wavs (auto-created)
│   ├── labels.csv          ← you create this
│   ├── features.csv        ← auto-created by step2
│   └── spectrograms/       ← auto-created by step4
└── results/                ← plots and AUC logs (auto-created)
