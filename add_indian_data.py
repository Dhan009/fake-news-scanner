"""
Merge an Indian news dataset into FakeNews.csv.

Example:
  python add_indian_data.py --indian ifnd.csv --text-col Statement --label-col Label ^
         --fake-values FAKE --real-values TRUE

Creates:
  FakeNews_combined.csv   original data + 80% of the Indian data  (train on this)
  indian_holdout.csv      the other 20% of the Indian data         (test on this, never trained on)
"""
import argparse
import re
import sys
import pandas as pd

# Text that gives away the source instead of the writing style. Removed before training.
GIVEAWAYS = [
    r"^\s*[A-Z][A-Za-z .,/-]{0,40}\(\s*Reuters\s*\)\s*[-–—]?\s*",   # "WASHINGTON (Reuters) -"
    r"\(\s*Reuters\s*\)",
    r"^\s*fact[\s-]*check\s*[:\-–—]\s*",
    r"^\s*(claim|rating|verdict)\s*[:\-–—]\s*",
    r"\b(pti|ani|ians)\s*[-–—:]\s*",                                   # agency tags
]

def strip_giveaways(text):
    for pat in GIVEAWAYS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text.strip()

def to_label(value, fake_values, real_values):
    v = str(value).strip().lower()
    if v in fake_values:
        return 1
    if v in real_values:
        return 0
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indian", required=True, help="path to the Indian dataset CSV")
    ap.add_argument("--text-col", nargs="+", required=True, help="column(s) holding the news text (several are joined)")
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--fake-values", required=True, help="comma-separated label values meaning FAKE, e.g. FAKE,false")
    ap.add_argument("--real-values", required=True, help="comma-separated label values meaning REAL, e.g. TRUE,real")
    ap.add_argument("--original", default="FakeNews.csv")
    ap.add_argument("--min-words", type=int, default=5, help="drop rows with fewer words than this")
    ap.add_argument("--holdout", type=float, default=0.2, help="share of Indian rows kept for separate testing")
    ap.add_argument("--balance", action="store_true", help="downsample the bigger Indian class so both classes match")
    args = ap.parse_args()

    fake_values = {v.strip().lower() for v in args.fake_values.split(",")}
    real_values = {v.strip().lower() for v in args.real_values.split(",")}

    try:
        ind = pd.read_csv(args.indian)
    except UnicodeDecodeError:
        ind = pd.read_csv(args.indian, encoding="latin-1")
    for c in args.text_col + [args.label_col]:
        if c not in ind.columns:
            sys.exit(f"Column '{c}' not found. Columns in your file: {list(ind.columns)}")

    print(f"Indian file rows: {len(ind)}")
    df = pd.DataFrame()
    df["text"] = ind[args.text_col].fillna("").astype(str).agg(" ".join, axis=1)
    df["label"] = ind[args.label_col].apply(lambda v: to_label(v, fake_values, real_values))

    unmapped = df["label"].isna().sum()
    if unmapped:
        print(f"Dropped {unmapped} rows whose label was not in your fake/real lists "
              f"(values seen: {sorted(ind[args.label_col].astype(str).unique())[:10]})")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["text"] = df["text"].apply(strip_giveaways)
    before = len(df)
    df = df[df["text"].str.split().str.len() >= args.min_words]
    df = df.drop_duplicates(subset="text")
    print(f"After cleaning: {len(df)} rows (removed {before - len(df)} short or duplicate rows)")

    if args.balance and df["label"].nunique() == 2:
        n = df["label"].value_counts().min()
        df = df.groupby("label", group_keys=False).sample(n=n, random_state=42)
        print(f"Balanced to {n} rows per class")

    print("Indian class counts (0 = real, 1 = fake):")
    print(df["label"].value_counts().to_string())
    avg = df.groupby("label")["text"].apply(lambda s: s.str.split().str.len().mean()).round(1)
    print("Average words per article by class:", avg.to_dict())
    if avg.max() > 3 * max(avg.min(), 1):
        print("WARNING: fake and real texts differ a lot in length. The model may learn length, not fakeness.")

    # separate test set that is never used in training
    hold = df.sample(frac=args.holdout, random_state=42)
    train_part = df.drop(hold.index)
    hold.to_csv("indian_holdout.csv", index=False)

    orig = pd.read_csv(args.original)
    if "text" not in orig.columns or "label" not in orig.columns:
        sys.exit(f"{args.original} must have 'text' and 'label' columns. It has: {list(orig.columns)}")
    combined = pd.concat([orig[["text", "label"]], train_part], ignore_index=True)
    combined = combined.drop_duplicates(subset="text").sample(frac=1, random_state=42)
    combined.to_csv("FakeNews_combined.csv", index=False)

    print(f"\nSaved FakeNews_combined.csv  ({len(combined)} rows: {len(orig)} original + {len(train_part)} Indian)")
    print(f"Saved indian_holdout.csv     ({len(hold)} rows, for testing only)")

if __name__ == "__main__":
    main()
