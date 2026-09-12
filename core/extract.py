"""Extraction Agent: tender pages -> list[Requirement]. The model extracts; it never judges."""
from typing import Callable, List, Optional

from core import llm
from core.ingest import Page, make_chunks
from core.schemas import CATEGORIES, PROFILE_FIELDS, Requirement, coerce_requirement

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "requirements": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "category": {"type": "STRING", "enum": CATEGORIES},
                    "text": {"type": "STRING"},
                    "quote": {"type": "STRING"},
                    "page": {"type": "INTEGER"},
                    "mandatory": {"type": "BOOLEAN"},
                    "profile_field": {"type": "STRING", "enum": PROFILE_FIELDS},
                    "threshold": {"type": "STRING"},
                },
                "required": ["category", "text", "quote", "page", "mandatory", "profile_field"],
            },
        }
    },
    "required": ["requirements"],
}

PROMPT = """You extract requirements from Pakistani public-procurement bidding documents
(PPRA / e-PADS). You are an extractor, not a judge.

RULES
- One item per distinct thing the BIDDER must satisfy, hold, or submit.
- "quote": copy the sentence VERBATIM from the text. Never paraphrase inside "quote".
- "page": the number in the nearest preceding [PAGE n] marker for that quote.
- "text": your own plain-English summary, max 20 words.
- "mandatory": true if the bidder is disqualified without it ("shall", "must", "required"),
  false if it is optional or merely preferred.
- "profile_field": one of the allowed values. Meaning:
    ntn / sales_tax_registered / on_fbr_atl  -> tax registration and FBR Active Taxpayer List
    secp_incorporated / years_incorporated   -> company registration and its age
    pec_registration                         -> Pakistan Engineering Council category (e.g. "C-4")
    years_experience                         -> minimum years of relevant experience
    similar_contracts_count                  -> number of similar contracts/clients required
    similar_contract_max_value_pkr           -> value of a single similar contract, in PKR
    avg_annual_turnover_pkr                  -> minimum annual turnover, in PKR
    bank_account_scheduled_bank / not_blacklisted
    certifications                           -> ISO, OEM authorisation, licences
    documents_on_hand                        -> a document that must be attached
    bid_security                             -> bid security / earnest money
    deadline                                 -> submission or opening date and time
    other                                    -> anything you cannot map with confidence
- "threshold": the number or code the requirement demands, as a plain string
  ("5", "5000000", "C-4", "2026-10-15"). Convert "Rs 5 million" to "5000000".
  Use "" when there is no threshold. NEVER invent one.
- Skip instructions aimed at the procuring agency, boilerplate, and page headers.
- Do NOT compute, compare, score, or decide eligibility. Extraction only.
- If a requirement is unclear, still extract it and set profile_field to "other".

TENDER TEXT:
{chunk}
"""


def extract_requirements(
    pages: List[Page],
    api_key: str,
    model: str,
    max_chars: int = 45000,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Requirement]:
    chunks = make_chunks(pages, max_chars=max_chars)
    raw_items: List[dict] = []
    for i, chunk in enumerate(chunks, start=1):
        if progress:
            progress(i, len(chunks))
        data = llm.generate_json(PROMPT.format(chunk=chunk), api_key, model, RESPONSE_SCHEMA)
        items = data.get("requirements", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        raw_items.extend(items)
    return dedupe([r for r in (coerce_requirement(x, i) for i, x in enumerate(raw_items, 1)) if r])


def dedupe(reqs: List[Requirement]) -> List[Requirement]:
    """Chunk boundaries can repeat a requirement. Keep the first, renumber ids."""
    seen, out = set(), []
    for r in reqs:
        key = (r.profile_field, _norm(r.text)[:60], str(r.threshold))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    for i, r in enumerate(out, start=1):
        r.id = f"R-{i:03d}"
    return out


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())
