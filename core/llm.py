"""Gemini REST client using only the standard library (no `requests`, no SDK).

Why: fewer dependencies = fewer ways for the Sunday-night deploy to break.
"""
import json
import os
import re
import urllib.error
import urllib.request

BASE = "https://generativelanguage.googleapis.com/v1beta"

# Fallback order only. resolve_model() normally picks the newest Flash the key can serve,
# because Google retires model names faster than a hackathon repo gets updated.
PREFERRED_MODELS = [
    "gemini-3.6-flash",
    "gemini-3-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
]

# Set to whatever last worked, so the UI can show the truth rather than what we asked for.
LAST_WORKING_MODEL = ""


class LLMError(RuntimeError):
    def __init__(self, message: str, suggested_model: str = ""):
        super().__init__(message)
        self.suggested_model = suggested_model


def _suggested_from(body: str) -> str:
    """Google's 404 usually names the replacement: 'Please update your code to use models/X'."""
    m = re.search(r"use\s+models/([A-Za-z0-9._-]+)", body)
    return m.group(1) if m else ""


def _post(url: str, payload: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")[:600]
        if e.code == 429:
            raise LLMError("Rate limit hit (429). Wait a minute, or switch to your second API key.") from e
        if e.code in (401, 403):
            raise LLMError(f"API key rejected ({e.code}). Check the key. Details: {body}") from e
        if e.code == 404:
            alt = _suggested_from(body)
            hint = f" Google suggests {alt}." if alt else " Pick another model in the sidebar."
            raise LLMError(f"Model not available (404).{hint} Details: {body}", suggested_model=alt) from e
        raise LLMError(f"Gemini HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"Network error calling Gemini: {e.reason}") from e


def list_models(api_key: str) -> list:
    """Model names this key can actually call with generateContent."""
    url = f"{BASE}/models?key={api_key}&pageSize=100"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 - surfaced to the UI, never fatal
        raise LLMError(f"Could not list models: {e}") from e
    out = []
    for m in data.get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            out.append(m["name"].split("/")[-1])
    return out


def model_rank(name: str) -> tuple:
    """Sort key: newest stable Flash first. Higher tuple wins.

    Model names come and go (gemini-2.5-flash was retired for new keys mid-2026), so we rank
    by version number rather than trusting a hard-coded list.
    """
    n = name.lower()
    version = 0.0
    m = re.search(r"(\d+(?:\.\d+)?)", n)
    if m:
        try:
            version = float(m.group(1))
        except ValueError:
            version = 0.0
    is_flash = 1 if "flash" in n else 0
    not_lite = 0 if ("lite" in n or "8b" in n) else 1
    stable = 0 if any(t in n for t in ("preview", "exp", "thinking", "tuning")) else 1
    return (is_flash, stable, not_lite, version)


def resolve_model(api_key: str, wanted: str = "") -> str:
    """Return a model name this key can actually serve, preferring the newest Flash."""
    if wanted:
        return wanted
    try:
        available = list_models(api_key)
    except LLMError:
        return PREFERRED_MODELS[0]
    usable = [m for m in available if "flash" in m.lower()] or available
    if usable:
        return sorted(usable, key=model_rank, reverse=True)[0]
    return PREFERRED_MODELS[0]


def generate_json(prompt: str, api_key: str, model: str, schema: dict | None = None,
                  temperature: float = 0.0, timeout: int = 180) -> object:
    """Call Gemini and return parsed JSON. Raises LLMError with a human-readable message."""
    if not api_key:
        raise LLMError("No API key set.")
    cfg = {"temperature": temperature, "responseMimeType": "application/json"}
    if schema:
        cfg["responseSchema"] = schema
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": cfg}
    data = _call_with_fallback(payload, api_key, model, timeout)
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        fb = json.dumps(data)[:400]
        raise LLMError(f"Unexpected Gemini response: {fb}") from e
    return _loads_lenient(text)


def _call_with_fallback(payload: dict, api_key: str, model: str, timeout: int) -> dict:
    """Call the model; if it has been retired, switch to the replacement Google names and retry.

    A retired model name is the most likely way this app breaks on demo day, so it heals itself
    instead of stopping at an error the user cannot act on.
    """
    global LAST_WORKING_MODEL
    tried = []
    candidate = model
    for _ in range(3):
        try:
            data = _post(f"{BASE}/models/{candidate}:generateContent?key={api_key}", payload, timeout)
            LAST_WORKING_MODEL = candidate
            return data
        except LLMError as e:
            tried.append(candidate)
            nxt = e.suggested_model or next((m for m in PREFERRED_MODELS if m not in tried), "")
            if not nxt or nxt in tried:
                raise
            candidate = nxt
    raise LLMError(f"No usable model. Tried: {', '.join(tried)}")


def generate_text(prompt: str, api_key: str, model: str, temperature: float = 0.3,
                  timeout: int = 120) -> str:
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": temperature}}
    data = _call_with_fallback(payload, api_key, model, timeout)
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise LLMError("Empty response from Gemini.") from e


def _loads_lenient(text: str) -> object:
    """Models sometimes wrap JSON in prose or fences. Recover instead of failing the demo."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"Model did not return valid JSON. First 300 chars: {text[:300]}")


def key_from_env() -> str:
    return os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
