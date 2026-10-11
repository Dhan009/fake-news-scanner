# Checks whether a selection is real, analysable text before the models run.
import re

MAX_CHARS = 10000          # longer selections are cut (the LSTM only reads ~200 words anyway)
MIN_WORDS = 3              # minimum number of real words


def check_text_usable(text):
    """Return (ok, message). message explains why the text can't be analysed."""
    text = (text or "").strip()
    if not text:
        return False, "No text was selected. Select the article text with your mouse."

    letters = sum(ch.isalpha() for ch in text)
    non_space = sum(not ch.isspace() for ch in text)
    if letters == 0:
        return False, "The selection has no readable text (only images, numbers or symbols)."
    if letters / max(non_space, 1) < 0.5:
        return False, "The selection is mostly numbers or symbols (a table, code or data?). Select article text instead."

    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    if ascii_letters / letters < 0.7:
        return False, "Only English text is supported, because the models were trained on English articles."

    if len(re.findall(r"[A-Za-z]{2,}", text)) < MIN_WORDS:
        return False, "Select at least a few words of article text."

    return True, ""


# ==========================================================
# "Does this look like a news article?"  (heuristic, not a trained model)
# ==========================================================
NEWS_MIN_SCORE = 3        # out of 5 signals; below this the text is treated as not news-like

_REPORTING = re.compile(
    r"\b(said|says|told|according to|reported|reports|announced|announce|stated|confirmed|"
    r"officials?|spokes(?:man|woman|person)|minister|government|police|court|president|"
    r"parliament|election|authorities|investigation|statement|press conference)\b", re.I)
_MONTHS = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday|yesterday|today)\b", re.I)
_SECOND_PERSON = re.compile(r"\b(you|your|you're|yours)\b", re.I)


def news_likeness(text):
    """
    Score 0-5 from simple signals found in news writing. Returns
    {"score": int, "is_news": bool, "reasons": [str, ...]} where reasons explain what is missing.
    This is a rough heuristic: it can wrongly reject short or opinion-style real news.
    """
    text = (text or "").strip()
    words = re.findall(r"[A-Za-z']+", text)
    n = len(words)
    score, reasons = 0, []

    # 1. long enough to be a news passage
    if n >= 40:
        score += 1
    else:
        reasons.append(f"Only {n} words; news passages are usually longer.")

    # 2. reporting language (said, according to, officials, ...)
    if _REPORTING.search(text):
        score += 1
    else:
        reasons.append("No reporting language (e.g. 'said', 'according to', 'officials').")

    # 3. names of people / places / organisations (capitalised words not starting a sentence)
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", text) if s.strip()]
    mid_caps = 0
    for s in sentences:
        toks = re.findall(r"[A-Za-z']+", s)
        mid_caps += sum(1 for t in toks[1:] if t[0].isupper() and not t.isupper())
    if n and mid_caps / n >= 0.04:
        score += 1
    else:
        reasons.append("Few names of people, places or organisations.")

    # 4. dates, days or figures
    if re.search(r"\d", text) or _MONTHS.search(text):
        score += 1
    else:
        reasons.append("No dates, days or figures.")

    # 5. sentence structure typical of prose
    avg_len = n / max(len(sentences), 1)
    if len(sentences) >= 2 and 8 <= avg_len <= 45:
        score += 1
    else:
        reasons.append("Sentence structure doesn't look like a news paragraph.")

    # penalties
    if n and len(_SECOND_PERSON.findall(text)) / n > 0.03:
        score -= 1
        reasons.append("Addresses the reader directly ('you'), which suggests a guide, ad or chat.")
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) >= 5 and sum(len(l.split()) for l in lines) / len(lines) < 5:
        score -= 1
        reasons.append("Looks like a menu or list of short items.")

    score = max(score, 0)
    return {"score": score, "is_news": score >= NEWS_MIN_SCORE, "reasons": reasons}
