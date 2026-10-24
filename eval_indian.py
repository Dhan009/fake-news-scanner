"""
Test the saved models on indian_holdout.csv (rows that were never used for training).
Run:  python eval_indian.py
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences

import FakeNewsDetection as fnd

path = sys.argv[1] if len(sys.argv) > 1 else "indian_holdout.csv"
df = pd.read_csv(path)
y = df["label"].astype(int).values
cleaned = df["text"].astype(str).apply(fnd.clean_text)

art = fnd.load_artifacts()
X = art["tfidf"].transform(cleaned)
preds = {name: model.predict(X) for name, model in art["ml"].items()}

seq = pad_sequences(art["tokenizer"].texts_to_sequences(cleaned), maxlen=fnd.MAX_LEN)
lstm_raw = art["lstm"].predict(seq, verbose=0).reshape(-1, 1)
if "lstm_calibrator" in art:                      # new version: use the calibrated LSTM probability
    lstm_p = art["lstm_calibrator"].predict_proba(lstm_raw)[:, 1]
else:
    lstm_p = lstm_raw.ravel()
preds["LSTM"] = (lstm_p >= 0.5).astype(int)

print(f"\nResults on {len(df)} Indian articles never used for training (1 = fake):\n")
print(f"{'Model':22s} {'Accuracy':>9s} {'Precision':>10s} {'Recall':>8s} {'F1':>7s}")
for name, p in preds.items():
    print(f"{name:22s} {accuracy_score(y, p):9.4f} {precision_score(y, p):10.4f} "
          f"{recall_score(y, p):8.4f} {f1_score(y, p):7.4f}")
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    print(f"{'':22s} TN={tn} FP={fp} FN={fn} TP={tp}")
