"""
False-positive test on FULL-LENGTH real Indian articles (Indian Express).
Every article in this file is real news, so any article the models call "fake" is a mistake.

Run:   python eval_real_indian.py
       python eval_real_indian.py --sample 500       (faster)
"""
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

import FakeNewsDetection as fnd
import signals

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="News_Articles_Indian_Express.csv")
ap.add_argument("--text-col", default="articles")
ap.add_argument("--sample", type=int, default=2000, help="how many articles to test (random)")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

try:
    df = pd.read_csv(args.csv, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(args.csv, encoding="latin-1")
if args.text_col not in df.columns:
    raise SystemExit(f"Column '{args.text_col}' not found. Columns: {list(df.columns)}")

texts = df[args.text_col].dropna().astype(str)
texts = texts[texts.str.split().str.len() >= 40]            # full articles only
texts = texts.sample(n=min(args.sample, len(texts)), random_state=args.seed)
texts = texts.str.slice(0, fnd.MAX_CHARS)                    # same cut-off as the server
print(f"Testing {len(texts)} real articles from {args.csv}")

cleaned = texts.apply(fnd.clean_text)
art = fnd.load_artifacts()

X = art["tfidf"].transform(cleaned)
probs = {name: m.predict_proba(X)[:, 1] for name, m in art["ml"].items()}
seq = pad_sequences(art["tokenizer"].texts_to_sequences(cleaned), maxlen=fnd.MAX_LEN)
lstm_raw = art["lstm"].predict(seq, verbose=0).reshape(-1, 1)
if "lstm_calibrator" in art:                      # new version: use the calibrated LSTM probability
    probs["LSTM"] = art["lstm_calibrator"].predict_proba(lstm_raw)[:, 1]
else:
    probs["LSTM"] = lstm_raw.ravel()

n = len(texts)
print("\nShare of these REAL articles wrongly called fake (false-positive rate):\n")
print(f"{'Model':22s} {'Called fake':>12s} {'Correct (real)':>15s}")
for name, p in probs.items():
    fp = float((p > 0.5).mean())
    print(f"{name:22s} {fp*100:11.2f}% {100 - fp*100:14.2f}%")

avg = np.mean(list(probs.values()), axis=0) * 100
print(f"\nAverage style score (what the extension shows as 'x% fake-like'): mean {avg.mean():.1f}%, median {np.median(avg):.1f}%")
print(f"Articles with style score >= {signals.STYLE_FAKE_AT}% (would read as fake-like): "
      f"{(avg >= signals.STYLE_FAKE_AT).mean()*100:.2f}%")
print(f"Articles with style score < {signals.STYLE_REAL_BELOW}% (read as real-like): "
      f"{(avg < signals.STYLE_REAL_BELOW).mean()*100:.2f}%")
print("\nAll articles here are real, so accuracy on this set = 100% - 'Called fake'.")
