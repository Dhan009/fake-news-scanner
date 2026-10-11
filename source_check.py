# ==========================================================
# Source verification: how trustworthy is the website the
# selected text came from?  (used by FakeNewsDetection.py)
# ==========================================================
# IMPORTANT: DEFAULT_SOURCES is a small STARTER list. It is not an
# authoritative rating. Review it, and extend/override it by creating
# a file called  sources_custom.json  next to this script, e.g.
#   { "example-news.com": "reliable", "fakesite.xyz": "unreliable" }
# Allowed categories: reliable | unreliable | satire | social
# ==========================================================

import os
import json
import difflib
from urllib.parse import urlparse

CUSTOM_FILE = "sources_custom.json"
LOOKALIKE_RATIO = 0.86    # similarity needed to flag a possible look-alike domain

DEFAULT_SOURCES = {
    # ---- established news agencies / newspapers / broadcasters
    "reuters.com": "reliable", "apnews.com": "reliable",
    "bbc.com": "reliable", "bbc.co.uk": "reliable",
    "theguardian.com": "reliable", "nytimes.com": "reliable",
    "washingtonpost.com": "reliable", "npr.org": "reliable",
    "thehindu.com": "reliable", "indianexpress.com": "reliable",
    "hindustantimes.com": "reliable", "ndtv.com": "reliable",
    "timesofindia.indiatimes.com": "reliable", "vox.com": "reliable",
    "aljazeera.com": "reliable", "pti.in": "reliable",
    # ---- official / scientific bodies
    "who.int": "reliable", "cdc.gov": "reliable", "nih.gov": "reliable",
    "pib.gov.in": "reliable", "nature.com": "reliable", "science.org": "reliable",
    # ---- fact-checkers
    "snopes.com": "reliable", "politifact.com": "reliable",
    "factcheck.org": "reliable", "boomlive.in": "reliable",
    "altnews.in": "reliable", "factly.in": "reliable",
    # ---- widely documented fake / hoax / conspiracy sites
    "infowars.com": "unreliable", "naturalnews.com": "unreliable",
    "beforeitsnews.com": "unreliable", "worldnewsdailyreport.com": "unreliable",
    "empirenews.net": "unreliable", "nationalreport.net": "unreliable",
    "newspunch.com": "unreliable", "yournewswire.com": "unreliable",
    # ---- satire (not meant to be factual)
    "theonion.com": "satire", "babylonbee.com": "satire", "clickhole.com": "satire",
    # ---- user-generated platforms: the platform says nothing about the post
    "facebook.com": "social", "x.com": "social", "twitter.com": "social",
    "instagram.com": "social", "youtube.com": "social", "reddit.com": "social",
    "whatsapp.com": "social", "t.me": "social", "medium.com": "social",
}

CATEGORY_INFO = {
    "reliable":   {"label": "Established / reliable source"},
    "unreliable": {"label": "Known unreliable source"},
    "satire":     {"label": "Satire website"},
    "social":     {"label": "User-generated platform"},
    "lookalike":  {"label": "Possible look-alike of a reliable site"},
    "unknown":    {"label": "Source not in the list"},
}


def _load_sources():
    sources = dict(DEFAULT_SOURCES)
    if os.path.exists(CUSTOM_FILE):
        try:
            with open(CUSTOM_FILE, "r", encoding="utf-8") as f:
                custom = json.load(f)
            for dom, cat in custom.items():
                if cat in ("reliable", "unreliable", "satire", "social"):
                    sources[normalize_host(dom)] = cat
        except Exception as e:
            print(f"[source_check] could not read {CUSTOM_FILE}: {e}")
    return sources


def normalize_host(value):
    """Accept a URL or a bare hostname; return a lowercase hostname without www./m./amp."""
    value = (value or "").strip().lower()
    if not value:
        return ""
    if "//" not in value:
        value = "//" + value
    host = urlparse(value).hostname or ""
    for prefix in ("www.", "m.", "amp."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    return host


def _lookup(host, sources):
    """Match the host or any parent domain (most specific first)."""
    parts = host.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in sources:
            return candidate, sources[candidate]
    return None, None


def _lookalike_of(host, sources):
    """Return the reliable domain this host appears to imitate, or None."""
    for dom, cat in sources.items():
        if cat != "reliable" or host == dom or host.endswith("." + dom):
            continue
        if host.startswith(dom + "."):                       # reuters.com.co
            return dom
        if difflib.SequenceMatcher(None, host, dom).ratio() >= LOOKALIKE_RATIO:   # reuter.com
            return dom
    return None


def verify_source(page_host, protocol=None):
    """Return a dict describing the source of the selected text."""
    sources = _load_sources()
    host = normalize_host(page_host)
    warnings = []

    if not host or host in ("localhost", "127.0.0.1"):
        info = CATEGORY_INFO["unknown"]
        return {"host": host or "unknown", "category": "unknown",
                "label": info["label"], "warnings": []}

    matched, category = _lookup(host, sources)
    if category is None:
        imitated = _lookalike_of(host, sources)
        if imitated:
            category = "lookalike"
            warnings.append(f"This address looks similar to {imitated} but is not it.")
        else:
            category = "unknown"

    if protocol and protocol.lower().startswith("http:") and not protocol.lower().startswith("https"):
        warnings.append("This page is not using HTTPS.")
    if category == "satire":
        warnings.append("Satire is written to look like news but is not meant to be factual.")
    if category == "social":
        warnings.append("Posts on this platform can come from anyone; the platform itself says nothing about accuracy.")

    info = CATEGORY_INFO[category]
    return {"host": host, "category": category, "label": info["label"],
            "warnings": warnings}
