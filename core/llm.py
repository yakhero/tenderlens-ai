"""Gemini REST client using only the standard library (no `requests`, no SDK).

Why: fewer dependencies = fewer ways for the Sunday-night deploy to break.
"""
import json
import os
import urllib.error
import urllib.request

BASE = "https://generativelanguage.googleapis.com/v1beta"

# First one that the key actually supports wins (see resolve_model).
PREFERRED_MODELS = [
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
]


class LLMError(RuntimeError):
    pass


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
            raise LLMError(f"Model not found (404). Pick another model in the sidebar. Details: {body}") from e
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


def resolve_model(api_key: str, wanted: str = "") -> str:
    """Return a model name that exists for this key. Never guesses blindly."""
    if wanted:
        return wanted
    try:
        available = list_models(api_key)
    except LLMError:
        return PREFERRED_MODELS[-1]
    for p in PREFERRED_MODELS:
        if p in available:
            return p
    flashes = [m for m in available if "flash" in m and "thinking" not in m]
    return flashes[0] if flashes else (available[0] if available else PREFERRED_MODELS[-1])


def generate_json(prompt: str, api_key: str, model: str, schema: dict | None = None,
                  temperature: float = 0.0, timeout: int = 180) -> object:
    """Call Gemini and return parsed JSON. Raises LLMError with a human-readable message."""
    if not api_key:
        raise LLMError("No API key set.")
    cfg = {"temperature": temperature, "responseMimeType": "application/json"}
    if schema:
        cfg["responseSchema"] = schema
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": cfg}
    data = _post(f"{BASE}/models/{model}:generateContent?key={api_key}", payload, timeout)
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        fb = json.dumps(data)[:400]
        raise LLMError(f"Unexpected Gemini response: {fb}") from e
    return _loads_lenient(text)


def generate_text(prompt: str, api_key: str, model: str, temperature: float = 0.3,
                  timeout: int = 120) -> str:
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": temperature}}
    data = _post(f"{BASE}/models/{model}:generateContent?key={api_key}", payload, timeout)
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
