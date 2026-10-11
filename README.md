# Fake News Scanner

A Chrome extension and local Python backend that checks news text you select on any web page. Instead of merging everything into one score, it shows **two separate signals** and tells you whether they agree.

## Problem Statement

Fake news spreads quickly, and readers rarely have time to verify each article they see. Most detectors give a single score, but a single score can hide important information. For example, a false article published on a reputable website can have its score pulled toward the middle, so the disagreement between the signals is lost.

This project addresses that in two ways:

1. **Select-and-scan.** Select text on any page and click the scan button. No copying or pasting is needed, and the extension knows which website the text came from.
2. **Separate signals with an agreement status.** The result keeps two independent signals apart:
   - **Signal 1, writing style:** a fake-like percentage from four trained models (Logistic Regression, Naive Bayes, Random Forest, LSTM), plus the per-model scores and the suspicious sentences, which are highlighted.
   - **Signal 2, source:** a check of the website against a source list (reliable, unreliable, satire, social platform, look-alike of a reliable domain, or unknown).
   - **Agreement status:** "Signals agree", or "Signals conflict, verify manually". A conflict is useful information, so it is reported and not averaged away.

The tool is an estimate from writing patterns and a source list. It is **not a fact-check**, so always verify with trusted sources.

## How It Works

```
Chrome extension (content.js)  -->  background.js  -->  Flask API (/predict)
   select text, click scan                              FakeNewsDetection.py
                                                         |-- text_checks.py   (is it usable news text?)
                                                         |-- 4 models         (writing-style signal)
                                                         |-- source_check.py  (source signal)
                                                         `-- signals.py       (agreement status)
```

## Project Structure

```
.
├── FakeNewsDetection.py   # train models, run the API server, CLI prediction
├── signals.py             # combines the signals into a verdict + agreement status
├── source_check.py        # website / source verification
├── text_checks.py         # selection validity and "does this look like news?" checks
├── combine.py             # prepares the dataset
├── show_accuracy.py       # shows saved model accuracies
├── requirements.txt
└── extension/             # Chrome extension (load this folder in Chrome)
    ├── manifest.json
    ├── background.js
    ├── content.js
    └── content.css
```

## Dataset

The dataset is **not included in this repository** because of its size (`FakeNews.csv` is about 92 MB).

- **Dataset link:** [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Place the final file in the project root as `FakeNews.csv`.
- The training code expects two columns: `text` and `label` (`0` = real, `1` = fake).
- If you download separate `Fake.csv` and `True.csv` files, use `combine.py` to produce `FakeNews.csv`.

## How to Run

### 1. Set up the environment

```bash
git clone https://github.com/YOUR-USERNAME/fake-news-scanner.git
cd fake-news-scanner

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2. Add the dataset and train the models

Put `FakeNews.csv` in the project folder, then run:

```bash
python FakeNewsDetection.py train --csv FakeNews.csv
```

Add `--no-plots` to skip the charts. The trained models are saved to `saved_models/`. To see the saved accuracies afterwards, run `python show_accuracy.py`.

### 3. Start the API server

```bash
python FakeNewsDetection.py serve
```

The server runs at `http://127.0.0.1:5000`. Keep this terminal open while you use the extension.

### 4. Load the extension in Chrome

1. Open `chrome://extensions`.
2. Turn on **Developer mode**.
3. Click **Load unpacked** and select the `extension` folder.
4. Open any news page (reload it if it was already open) and select a few lines of article text.
5. Click **🔍 Scan for fake news**. The result card shows both signals and the agreement status.

### Quick test from the terminal (optional)

```bash
python FakeNewsDetection.py predict --text "paste some news text here" --url reuters.com
```

## Customising the Source List

`source_check.py` ships with a small **starter** list. It is not an authoritative rating. To add or override sites, create a `sources_custom.json` file in the project folder:

```json
{
  "example-news.com": "reliable",
  "fakesite.xyz": "unreliable"
}
```

Allowed categories: `reliable`, `unreliable`, `satire`, `social`.

## Troubleshooting

| Problem | Fix |
|---|---|
| Scan button doesn't appear | Reload the extension on `chrome://extensions`, then refresh the web page. Select at least a few words of English text. |
| "Cannot reach the local server" | Start the API with `python FakeNewsDetection.py serve`. |
| "No saved models found" | Run the training step first. |
| Extension doesn't work on some pages | Chrome blocks extensions on `chrome://` pages and the Chrome Web Store. |

## Limitations

- English text only, because the models were trained on English articles.
- Very short selections give low-confidence results. Select more of the article.
- Writing style depends on the training dataset and can miss new kinds of misinformation.
- The source list is small and needs to be reviewed and extended.
- This is a decision aid, not a fact-checker.
