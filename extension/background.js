// Relays requests from the page to the local Python API, with a time limit.
const API_URL = "http://127.0.0.1:5000/predict";
const TIMEOUT_MS = 25000;

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type !== "SCAN") return;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: msg.text, host: msg.host, protocol: msg.protocol, force: !!msg.force }),
    signal: controller.signal
  })
    .then(async (r) => {
      const data = await r.json().catch(() => null);   // server errors may not be JSON
      if (r.ok && data) sendResponse({ ok: true, data });
      else sendResponse({ ok: false,
        error: (data && data.error) || ("Server error (status " + r.status + "). Check the server terminal.") });
    })
    .catch((err) => sendResponse({
      ok: false,
      error: err.name === "AbortError"
        ? "The scan took too long and was stopped. Try selecting less text."
        : "Cannot reach the local server. Run: python FakeNewsDetection.py serve"
    }))
    .finally(() => clearTimeout(timer));

  return true;
});
