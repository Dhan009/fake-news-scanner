// Shows a "Scan" button when text is selected; clicking it shows the result card.
const MAX_CHARS = 10000;          // same limit as the server (text_checks.py)
const SCAN_TIMEOUT_MS = 30000;    // slightly longer than background.js (25 s) so its message shows first
const MIN_SELECTED_WORDS = 3;     // ignore tiny selections

// Quick check so the button only appears for real text (the server does the full check)
function hasReadableText(text) {
  if (!text) return false;
  const words = text.match(/[A-Za-z]{2,}/g) || [];
  return words.length >= MIN_SELECTED_WORDS;
}

let btn = null, card = null;

function removeEl(el) { if (el && el.parentNode) el.parentNode.removeChild(el); }
function clearAll() { removeEl(btn); removeEl(card); btn = card = null; }

// Fixed positioning, kept inside the visible window. Works on pages that scroll
// inside an inner container (chat apps, web apps) where page coordinates don't.
function place(el, rect) {
  const m = 8;
  const h = el.offsetHeight || 0, w = el.offsetWidth || 0;
  let top = rect.bottom + m;
  if (top + h > window.innerHeight - m) {
    const above = rect.top - h - m;
    top = above >= m ? above : Math.max(m, window.innerHeight - h - m);
  }
  const left = Math.min(Math.max(m, rect.left), Math.max(m, window.innerWidth - w - m));
  el.style.top = top + "px";
  el.style.left = left + "px";
}

function colorFor(pct) { return pct >= 70 ? "#d93025" : pct >= 40 ? "#f9ab00" : "#188038"; }


function row(label, value) {
  const r = document.createElement("div"); r.className = "fns-row";
  const a = document.createElement("span"); a.textContent = label;
  const b = document.createElement("span"); b.textContent = value;
  r.append(a, b); return r;
}

function showCard(rect, content) {
  removeEl(card);
  card = document.createElement("div"); card.id = "fns-card";
  const close = document.createElement("span");
  close.className = "fns-close"; close.textContent = "×"; close.onclick = clearAll;
  card.append(close, content);
  document.body.appendChild(card);
  place(card, rect);
}

function simpleMsg(text) {
  const d = document.createElement("div"); d.textContent = text; return d;
}

const LEVEL_COLORS = {
  fake: "#d93025", real: "#188038", uncertain: "#f9ab00", conflict: "#e8710a", neutral: "#5f6368"
};
const SOURCE_COLORS = { good: "#188038", bad: "#d93025", none: "#5f6368" };

function box(title) {
  const b = document.createElement("div"); b.className = "fns-src";
  const t = document.createElement("div"); t.className = "fns-small"; t.style.marginTop = "0";
  t.textContent = title; b.appendChild(t); return b;
}

function line(text, color, bold) {
  const d = document.createElement("div"); d.textContent = text;
  if (color) d.style.color = color;
  if (bold) d.style.fontWeight = "700";
  return d;
}

function renderResult(d) {
  const wrap = document.createElement("div");
  const levelColor = LEVEL_COLORS[d.verdict_level] || "#5f6368";

  // ---- overall verdict + agreement status (signals are NOT merged into one number)
  const v = document.createElement("div"); v.className = "fns-verdict";
  v.textContent = d.verdict; v.style.color = levelColor;
  wrap.appendChild(v);

  const ag = d.agreement || {};
  const agBox = document.createElement("div"); agBox.className = "fns-src";
  agBox.style.borderLeft = "4px solid " + levelColor;
  agBox.style.marginTop = "0";
  agBox.appendChild(line(ag.message || "", null, false));
  wrap.appendChild(agBox);

  // ---- signal 1: writing style
  const s1 = box("Signal 1: Writing style");
  const styleColor = d.style_stance === "fake" ? LEVEL_COLORS.fake
                   : d.style_stance === "real" ? LEVEL_COLORS.real : LEVEL_COLORS.uncertain;
  s1.appendChild(line(d.style_percent + "% fake-like", styleColor, true));
  const bar = document.createElement("div"); bar.className = "fns-bar";
  const fill = document.createElement("div"); fill.className = "fns-fill";
  fill.style.width = d.style_percent + "%"; fill.style.background = styleColor;
  bar.appendChild(fill); s1.appendChild(bar);
  s1.appendChild(line(d.style_label || ""));
  wrap.appendChild(s1);

  // ---- signal 2: source
  const src = d.source || {};
  const s2 = box("Signal 2: Source");
  s2.appendChild(line(src.host || "unknown", null, true));
  s2.appendChild(line(src.label || "", SOURCE_COLORS[d.source_stance] || "#5f6368", true));
  (src.warnings || []).forEach((w) => s2.appendChild(line("• " + w)));
  wrap.appendChild(s2);

  // ---- details of the style signal
  const hh = document.createElement("div"); hh.className = "fns-small"; hh.textContent = "Per-model fakeness:";
  wrap.appendChild(hh);
  for (const [name, pct] of Object.entries(d.models)) wrap.appendChild(row(name, pct + "%"));
  wrap.appendChild(row("Words analysed", d.word_count));

  // ---- suspicious sentences (highlighted)
  const susp = d.suspicious_sentences || [];
  const sh = document.createElement("div"); sh.className = "fns-small";
  sh.textContent = susp.length
    ? "Suspicious sentences (" + susp.length + ") highlighted:"
    : "No individual sentence scored high enough to highlight.";
  wrap.appendChild(sh);
  if (susp.length && d.highlighted_html) {
    const hl = document.createElement("div"); hl.className = "fns-hl";
    hl.innerHTML = d.highlighted_html;   // server escapes all text before adding <mark> tags
    wrap.appendChild(hl);
  }

  if (d.warning) {
    const w = document.createElement("div"); w.className = "fns-warn"; w.textContent = d.warning;
    wrap.appendChild(w);
  }
  const note = document.createElement("div"); note.className = "fns-small";
  note.textContent = "Estimate from writing patterns and a source list, not a fact-check. Verify with trusted sources.";
  wrap.appendChild(note);
  return wrap;
}

function renderNotNews(d, rect, text) {
  const wrap = document.createElement("div");
  const v = document.createElement("div"); v.className = "fns-verdict";
  v.textContent = "Doesn't look like news"; v.style.color = "#5f6368";
  const m = document.createElement("div"); m.textContent = d.message;
  wrap.append(v, m);
  (d.reasons || []).forEach((r) => {
    const rd = document.createElement("div"); rd.className = "fns-small"; rd.textContent = "• " + r;
    wrap.appendChild(rd);
  });
  const b = document.createElement("button");
  b.textContent = "Scan anyway";
  Object.assign(b.style, { marginTop: "10px", padding: "6px 12px", border: "0", borderRadius: "14px",
    background: "#1a73e8", color: "#fff", cursor: "pointer", fontSize: "12px" });
  b.onclick = () => startScan(rect, text, true);
  wrap.appendChild(b);
  return wrap;
}

function startScan(rect, text, force) {
  showCard(rect, simpleMsg("Scanning… (script v8)"));

  // Whatever happens, the card must end with a result or a message, never "Scanning…" forever.
  let finished = false;
  const finish = (render) => {
    if (finished) return;
    finished = true;
    clearTimeout(timer);
    if (!card) return;           // card is gone if the user closed it
    try {
      render();
    } catch (e) {                // a drawing error must never leave the card on "Scanning…"
      console.error("[Fake News Scanner v8] could not display result:", e);
      showCard(rect, simpleMsg("Could not display the result: " + e.message));
    }
  };
  const timer = setTimeout(() => finish(() => showCard(rect,
    simpleMsg("The scan took too long. Check that the server is running, or select less text."))),
    SCAN_TIMEOUT_MS);

  console.log("[Fake News Scanner v8] scan started, characters:", text.length);
  try {
    chrome.runtime.sendMessage(
      { type: "SCAN", text: text.slice(0, MAX_CHARS), host: location.hostname,
        protocol: location.protocol, force: !!force },
      (resp) => finish(() => {
        console.log("[Fake News Scanner v8] reply received:", resp);
        if (chrome.runtime.lastError || !resp)
          return showCard(rect, simpleMsg("Extension error. Reload this page (F5) and try again."));
        if (!resp.ok) return showCard(rect, simpleMsg(resp.error));
        if (resp.data.not_news) return showCard(rect, renderNotNews(resp.data, rect, text));
        showCard(rect, renderResult(resp.data));
      })
    );
  } catch (err) {
    // happens when the extension was reloaded while this tab stayed open
    finish(() => showCard(rect,
      simpleMsg("The extension was reloaded. Reload this page (F5) and try again.")));
  }
}

document.addEventListener("mouseup", (e) => {
  if (!chrome.runtime || !chrome.runtime.id) return;   // orphaned copy after an extension reload
  if ((btn && btn.contains(e.target)) || (card && card.contains(e.target))) return;
  setTimeout(() => {
    const sel = window.getSelection();
    const text = sel ? sel.toString().trim() : "";
    clearAll();
    if (!hasReadableText(text) || sel.rangeCount === 0) return;
    const rect = sel.getRangeAt(0).getBoundingClientRect();

    btn = document.createElement("div"); btn.id = "fns-btn";
    btn.textContent = "🔍 Scan for fake news";
    document.body.appendChild(btn);
    place(btn, rect);

    btn.addEventListener("mousedown", (ev) => ev.preventDefault());
    btn.addEventListener("click", () => {
      removeEl(btn); btn = null;
      startScan(rect, text, false);
    });
  }, 10);
});
