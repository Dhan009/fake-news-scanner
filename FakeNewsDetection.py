# ==========================================================
# Fake News Detection using AI  (+ live "select & scan" API)
# ==========================================================
# Usage:
#   pip install -r requirements.txt
#   python FakeNewsDetection.py train --csv FakeNews.csv      # train + save models
#   python FakeNewsDetection.py serve                         # start API for the browser extension
#   python FakeNewsDetection.py predict --text "some news"    # quick test from terminal
# ==========================================================

import os
import re
import sys
import html
import pickle
import traceback
import argparse

import numpy as np
import pandas as pd
import joblib

from text_checks import check_text_usable, news_likeness, MAX_CHARS
from source_check import verify_source
from signals import assess_signals

MODEL_DIR = "saved_models"
MAX_WORDS = 10000      # LSTM vocabulary size
MAX_LEN = 200          # LSTM sequence length
MIN_WORDS_FOR_RELIABLE = 40   # shorter text -> low-confidence warning

# ---- Highlighting settings
HIGHLIGHT_THRESHOLD = 0.70    # sentence fake-probability needed to be highlighted
MIN_WORDS_PER_SENTENCE = 6    # shorter sentences are too noisy to score
MAX_SENTENCES = 300           # safety cap for very long selections


# ==============================
# Text preprocessing (shared by training and prediction)
# ==============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ==============================
# TRAINING
# ==============================
def train(csv_path, show_plots=True):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import nltk
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense

    nltk.download('stopwords', quiet=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---- Load dataset (label: 0 = Real, 1 = Fake)
    df = pd.read_csv(csv_path)
    df = df[['text', 'label']].dropna()
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 0]

    X, y = df['text'], df['label'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Traditional ML models
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"\n{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        if show_plots:
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix: {name}')
            plt.xlabel('Predicted'); plt.ylabel('Actual')
            plt.show()

    # ---- LSTM
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_train)
    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
    X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN)

    lstm_model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=128),
        LSTM(128, dropout=0.2),          # recurrent_dropout removed: it disables fast GPU kernels
        Dense(1, activation='sigmoid'),
    ])
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = lstm_model.fit(
        X_train_pad, y_train.values,
        epochs=5, batch_size=64, validation_split=0.1
    )

    y_pred_lstm = (lstm_model.predict(X_test_pad) > 0.5).astype("int32")
    acc_lstm = accuracy_score(y_test, y_pred_lstm)
    results['LSTM'] = acc_lstm
    print(f"\nLSTM Accuracy: {acc_lstm:.4f}")
    print(classification_report(y_test, y_pred_lstm))

    if show_plots:
        sns.heatmap(confusion_matrix(y_test, y_pred_lstm), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix: LSTM'); plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.show()

        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(results.keys()), y=list(results.values()))
        plt.title('Model Accuracy Comparison'); plt.ylim(0, 1); plt.show()

        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend(); plt.title('LSTM Accuracy'); plt.show()

        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend(); plt.title('LSTM Loss'); plt.show()

    # ---- Save everything needed for live prediction
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf.joblib"))
    for name, model in models.items():
        joblib.dump(model, os.path.join(MODEL_DIR, name.replace(" ", "_") + ".joblib"), compress=3)
    lstm_model.save(os.path.join(MODEL_DIR, "lstm.keras"))
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(MODEL_DIR, "accuracies.pkl"), "wb") as f:
        pickle.dump(results, f)
    print(f"\nModels saved to ./{MODEL_DIR}/  -> now run: python FakeNewsDetection.py serve")


# ==============================
# PREDICTION (percentage of fakeness)
# ==============================
_artifacts = None

def load_artifacts():
    global _artifacts
    if _artifacts is not None:
        return _artifacts
    from tensorflow.keras.models import load_model

    if not os.path.exists(os.path.join(MODEL_DIR, "tfidf.joblib")):
        sys.exit("No saved models found. Run:  python FakeNewsDetection.py train --csv FakeNews.csv")

    ml_models = {}
    for name in ["Logistic Regression", "Naive Bayes", "Random Forest"]:
        ml_models[name] = joblib.load(os.path.join(MODEL_DIR, name.replace(" ", "_") + ".joblib"))
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    _artifacts = {
        "tfidf": joblib.load(os.path.join(MODEL_DIR, "tfidf.joblib")),
        "ml": ml_models,
        "tokenizer": tokenizer,
        "lstm": load_model(os.path.join(MODEL_DIR, "lstm.keras")),
    }
    return _artifacts


# ==============================
# HIGHLIGHTING (find suspicious sentences)
# ==============================
def find_suspicious_sentences(text, art):
    """
    Split the ORIGINAL text into sentences, score each one with the same
    ensemble used for the whole text, and return:
      - a list of suspicious sentences (with character offsets into `text`)
      - an HTML version of the text with suspicious sentences wrapped in <mark>
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # 1) split into sentences while remembering where each one sits in the text
    spans = []   # (start, end, sentence_text)
    for m in re.finditer(r'[^.!?\n]+[.!?]*', text):
        raw = m.group()
        stripped = raw.strip()
        if not stripped:
            continue
        start = m.start() + (len(raw) - len(raw.lstrip()))
        spans.append((start, start + len(stripped), stripped))
    spans = spans[:MAX_SENTENCES]

    # 2) score every sentence that is long enough to be meaningful (batched = fast)
    scorable = [i for i, (_, _, s) in enumerate(spans)
                if len(clean_text(s).split()) >= MIN_WORDS_PER_SENTENCE]
    scores = {}
    if scorable:
        cleaned = [clean_text(spans[i][2]) for i in scorable]
        vec = art["tfidf"].transform(cleaned)
        probs = [model.predict_proba(vec)[:, 1] for model in art["ml"].values()]
        seq = pad_sequences(art["tokenizer"].texts_to_sequences(cleaned), maxlen=MAX_LEN)
        probs.append(art["lstm"].predict(seq, verbose=0).reshape(-1))
        avg = np.mean(probs, axis=0)
        scores = {idx: float(p) for idx, p in zip(scorable, avg)}

    # 3) collect suspicious sentences
    suspicious = []
    flagged_idx = set()
    for i, p in scores.items():
        if p >= HIGHLIGHT_THRESHOLD:
            start, end, s = spans[i]
            suspicious.append({
                "text": s,
                "start": start,
                "end": end,
                "fake_percent": round(p * 100, 1),
            })
            flagged_idx.add(i)

    # 4) build highlighted HTML (everything is escaped, so it is safe to inject)
    def esc(t):
        return html.escape(t).replace("\n", "<br>")

    parts, cursor = [], 0
    for i, (start, end, s) in enumerate(spans):
        parts.append(esc(text[cursor:start]))
        if i in flagged_idx:
            pct = round(scores[i] * 100, 1)
            parts.append(
                f'<mark class="fn-suspicious" title="{pct}% likely fake" '
                f'style="background:#ffd54f;color:#000;border-radius:3px;padding:0 2px;">'
                f'{esc(s)}</mark>'
            )
        else:
            parts.append(esc(s))
        cursor = end
    parts.append(esc(text[cursor:]))

    return suspicious, "".join(parts)


def predict_fakeness(text, page_host=None, page_protocol=None):
    """Return the writing-style percentage, the source check, and a verdict with an
    agreement status. The signals are shown separately, never merged into one number."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    art = load_artifacts()
    cleaned = clean_text(text)
    word_count = len(cleaned.split())

    vec = art["tfidf"].transform([cleaned])
    per_model = {}
    for name, model in art["ml"].items():
        per_model[name] = float(model.predict_proba(vec)[0][1])   # P(class 1 = fake)

    seq = pad_sequences(art["tokenizer"].texts_to_sequences([cleaned]), maxlen=MAX_LEN)
    per_model["LSTM"] = float(art["lstm"].predict(seq, verbose=0)[0][0])

    fake_prob = float(np.mean(list(per_model.values())))   # average of the 4 models
    style_percent = round(fake_prob * 100, 1)               # signal 1: writing style only

    source = verify_source(page_host, page_protocol)        # signal 2: website
    assessment = assess_signals(style_percent, word_count, source)

    warning = None
    if word_count < MIN_WORDS_FOR_RELIABLE:
        warning = (f"Only {word_count} words selected. The models were trained on full articles, "
                   "so short text gives low-confidence results. Select more of the article.")

    suspicious_sentences, highlighted_html = find_suspicious_sentences(str(text), art)

    return {
        "verdict": assessment["verdict"],
        "verdict_level": assessment["verdict_level"],
        "agreement": assessment["agreement"],
        "style_percent": style_percent,
        "style_stance": assessment["style_stance"],
        "style_label": assessment["style_label"],
        "source": source,
        "word_count": word_count,
        "warning": warning,
        "models": {k: round(v * 100, 1) for k, v in per_model.items()},
        "suspicious_sentences": suspicious_sentences,
        "highlighted_html": highlighted_html,
    }


# ==============================
# API SERVER (used by the browser extension)
# ==============================
def serve(host="127.0.0.1", port=5000):
    from flask import Flask, request, jsonify

    app = Flask(__name__)
    load_artifacts()   # load once at startup

    @app.after_request
    def add_cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict():
        if request.method == "OPTIONS":
            return ("", 204)
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()

        # 1) is there analysable text at all?
        ok, message = check_text_usable(text)
        if not ok:
            return jsonify({"error": message}), 400

        # 2) is it news? If not, skip the meaningless fake/real score.
        #    The user can override with "Scan anyway" (force=true).
        if not data.get("force"):
            nl = news_likeness(text)
            if not nl["is_news"]:
                return jsonify({
                    "not_news": True,
                    "message": "This doesn't look like a news article, so a fake/real score "
                               "would not be meaningful.",
                    "reasons": nl["reasons"],
                })

        # 3) score it; any error is reported instead of leaving the card waiting
        try:
            return jsonify(predict_fakeness(text[:MAX_CHARS],
                                            page_host=data.get("host"),
                                            page_protocol=data.get("protocol")))
        except Exception:
            traceback.print_exc()          # full error appears in the server terminal
            return jsonify({"error": "The server could not analyse this text. "
                                     "Check the server terminal for the error."}), 500

    print(f"Fake news API running on http://{host}:{port}")
    app.run(host=host, port=port)


# ==============================
# CLI
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--csv", default="FakeNews.csv")
    p_train.add_argument("--no-plots", action="store_true")

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--port", type=int, default=5000)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--text", required=True)
    p_pred.add_argument("--url", default=None, help="website the text came from (optional)")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args.csv, show_plots=not args.no_plots)
    elif args.cmd == "serve":
        serve(port=args.port)
    elif args.cmd == "predict":
        import json
        print(json.dumps(predict_fakeness(args.text, page_host=args.url), indent=2))
