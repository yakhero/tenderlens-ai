"""Command-line end-to-end run. Use this to test extraction before touching the UI.

    export GEMINI_API_KEY="your key"
    python run_cli.py data/tenders/sample_tender.txt data/profiles/profile_a.json
"""
import json
import sys
from pathlib import Path

from core import compliance, draft, extract, ingest, llm, verify
from core.schemas import CompanyProfile


def main(tender_path: str, profile_path: str = "data/profiles/profile_a.json") -> int:
    key = llm.key_from_env()
    if not key:
        print("Set GEMINI_API_KEY first."), sys.exit(2)

    pages = ingest.read_any(tender_path)
    print(f"Read {len(pages)} pages from {tender_path}")
    if ingest.is_scanned(pages):
        print("WARNING: looks like a scanned PDF - little selectable text.")

    model = llm.resolve_model(key)
    print(f"Model: {model}")
    reqs = extract.extract_requirements(pages, key, model,
                                        progress=lambda i, n: print(f"  chunk {i}/{n}"))
    reqs = verify.verify_requirements(reqs, pages)
    print(f"{len(reqs)} requirements · {verify.verification_rate(reqs)}% citations verified")

    out = Path("data/cache") / f"{ingest.file_hash(Path(tender_path).read_bytes())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.to_dict() for r in reqs], indent=1), encoding="utf-8")
    print(f"Saved extraction to {out} (the app can load this without calling the API)")

    profile = CompanyProfile.from_dict(json.loads(Path(profile_path).read_text(encoding="utf-8")))
    result = compliance.evaluate(reqs, profile)
    print("\n" + "=" * 70)
    print(f"{profile.company_name}: {result['decision']}")
    print(draft.offline_note(result))
    print("=" * 70)
    for r in result["rows"][:40]:
        flag = "" if r["verified"] else "  [citation unverified]"
        print(f'{r["id"]:>6} p{r["page"]:<4}{r["status"]:<13}{r["requirement"][:62]}{flag}')
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(*sys.argv[1:3]))
