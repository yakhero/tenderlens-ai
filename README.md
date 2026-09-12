# TenderLens AI

**Read the tender. Know if you qualify. Before you bid.**

A human-in-the-loop compliance copilot for Pakistani public-procurement bidding documents
(PPRA / e-PADS). Upload a tender PDF: AI agents extract every requirement with a page
citation, code verifies each citation, you review and edit, and a tested rules engine
decides **BID / NO-BID / CONDITIONAL** against your company profile.

Built for the HEC–NCEAC & PEC Generative & Agentic AI Training, Cohort 11 mid-term hackathon.

## The design rule

> The LLM reads documents and writes prose. **Every number, comparison and verdict is
> computed in Python that we unit-test.**

An LLM asked to compare "Rs 79,000 × 2 kits" with "Rs 85,000 × 1 kit" gets it wrong often
enough to matter, so it is never asked. It extracts; code decides.

## Pipeline

| # | Step | Type | What it does |
|---|------|------|--------------|
| 1 | `core/ingest.py` | code | PDF → per-page text (PyMuPDF), chunking with `[PAGE n]` markers |
| 2 | `core/extract.py` | **LLM** | pages → structured requirements (category, verbatim quote, page, mandatory flag, mapped profile field, threshold) |
| 3 | `core/verify.py` | code | checks each quote really appears on the cited page; corrects the page or flags the row |
| — | Human gate 1 | **human** | edit, delete, add requirements in the UI |
| 4 | `core/compliance.py` | code | Met / Not Met / Needs Review per requirement, then the decision |
| 5 | `core/draft.py` | **LLM** | clarification email from facts computed in step 4 |
| — | Human gate 2 | **human** | edit and approve before export |

## Run it locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-from-aistudio.google.com"

python run_tests.py                                   # 19 unit tests, no API key needed
python run_cli.py data/tenders/sample_tender.txt      # end-to-end in the terminal
streamlit run app.py                                  # the UI
```

The app also runs with **no API key**: pick a bundled tender and click
*Load saved extraction* to replay a cached run. That is the demo safety net if the API
rate-limits during judging.

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub (public).
2. share.streamlit.io → New app → pick the repo → main file `app.py`.
3. Settings → Secrets:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
4. Open the app once about 10 minutes before judging: free apps sleep when idle.

## Repo layout

```
app.py              Streamlit UI (5 steps, 2 human gates)
run_cli.py          terminal end-to-end run
run_tests.py        test runner (works without pytest)
core/               ingest · extract · verify · compliance · draft · exports · llm · schemas
data/tenders/       tender documents (swap in real PPRA PDFs)
data/profiles/      two fictional company profiles, A fails and B passes
data/cache/         saved extractions, keyed by file hash
tests/test_core.py  unit tests for the rules engine and the verifier
```

## Notes and limits

- `data/tenders/sample_tender.txt` is a **synthetic test fixture**, not a real tender. Use a
  real PPRA document for the demo.
- Scanned (image-only) PDFs are detected and warned about, not OCR'd.
- The free Gemini tier may use inputs to improve Google's models, so only public tender
  documents are used here. Never upload a real company's private documents.
- Company profiles are fictional.
