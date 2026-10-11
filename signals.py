# ==========================================================
# Combines SEPARATE signals into a verdict + agreement status,
# without merging them into one number.
#   Signal 1: writing style  (percentage from the 4 models)
#   Signal 2: source         (category of the website)
# ==========================================================

STYLE_FAKE_AT = 70          # style percent at/above this -> "fake-like"
STYLE_REAL_BELOW = 40       # style percent below this     -> "real-like"
MIN_WORDS_FOR_STYLE = 40    # fewer words -> style signal is inconclusive

# how each source category counts: good = credible record, bad = poor record, none = no signal
SOURCE_STANCE = {
    "reliable": "good", "unreliable": "bad", "satire": "bad",
    "lookalike": "bad", "social": "none", "unknown": "none",
}


def _style_stance(percent, word_count):
    if word_count < MIN_WORDS_FOR_STYLE:
        return "inconclusive", f"Too little text ({word_count} words) for a reliable reading"
    if percent >= STYLE_FAKE_AT:
        return "fake", "Reads like articles labelled fake"
    if percent < STYLE_REAL_BELOW:
        return "real", "Reads like articles labelled real"
    return "inconclusive", "Falls between fake-like and real-like"


def assess_signals(style_percent, word_count, source):
    """Return verdict, verdict_level, per-signal stances and an agreement status."""
    s_stance, s_label = _style_stance(style_percent, word_count)
    src_stance = SOURCE_STANCE.get(source.get("category"), "none")

    out = {
        "style_stance": s_stance,
        "style_label": s_label,
        "source_stance": src_stance,
    }

    # ---- satire is a special case: the writing-style score means nothing there
    if source.get("category") == "satire":
        out.update(verdict="Satire source", verdict_level="neutral",
                   agreement={"status": "single",
                              "message": "This is a satire website. Its articles imitate news but are not "
                                         "meant to be factual, so the style score is not meaningful."})
        return out

    # ---- no usable source signal: rely on style alone and say so
    if src_stance == "none":
        if s_stance == "fake":
            verdict, level = "Likely FAKE (writing style only)", "fake"
        elif s_stance == "real":
            verdict, level = "Likely REAL (writing style only)", "real"
        else:
            verdict, level = "Insufficient evidence", "uncertain"
        out.update(verdict=verdict, verdict_level=level,
                   agreement={"status": "single",
                              "message": "Only one signal available: this website is not in the source list, "
                                         "so the result rests on writing style alone."})
        return out

    # ---- both signals available
    if src_stance == "good":
        if s_stance == "fake":
            out.update(verdict="Conflicting signals", verdict_level="conflict",
                       agreement={"status": "conflict",
                                  "message": "Signals conflict, verify manually: the source is established, "
                                             "but the writing reads like fake articles."})
        elif s_stance == "real":
            out.update(verdict="Likely REAL", verdict_level="real",
                       agreement={"status": "agree",
                                  "message": "Signals agree: established source and real-like writing."})
        else:
            out.update(verdict="Uncertain", verdict_level="uncertain",
                       agreement={"status": "inconclusive",
                                  "message": "The source is established, but the writing style is "
                                             "inconclusive. Check the specific claims."})
    else:  # bad
        if s_stance == "fake":
            out.update(verdict="Likely FAKE", verdict_level="fake",
                       agreement={"status": "agree",
                                  "message": "Signals agree: low-credibility source and fake-like writing."})
        elif s_stance == "real":
            out.update(verdict="Conflicting signals", verdict_level="conflict",
                       agreement={"status": "conflict",
                                  "message": "Signals conflict, verify manually: the source has a poor "
                                             "record, but the writing reads like real articles."})
        else:
            out.update(verdict="Be cautious", verdict_level="uncertain",
                       agreement={"status": "inconclusive",
                                  "message": "The source has a poor record and the writing style is "
                                             "inconclusive. Verify before trusting it."})
    return out
