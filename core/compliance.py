"""Compliance Engine. Pure Python, unit-tested, no LLM anywhere in this file.

Every number, comparison and verdict the user sees is produced here.
"""
import re
from datetime import date, datetime
from typing import List, Optional

from core.schemas import (BOOL_FIELDS, NUMERIC_FIELDS, PEC_ORDER, CompanyProfile, Requirement)

MET, NOT_MET, REVIEW = "Met", "Not Met", "Needs Review"
BID, NO_BID, CONDITIONAL = "BID", "NO-BID", "CONDITIONAL"

FIELD_LABEL = {
    "ntn": "NTN / income-tax registration",
    "on_fbr_atl": "FBR Active Taxpayer List",
    "sales_tax_registered": "Sales tax (GST) registration",
    "secp_incorporated": "SECP incorporation",
    "years_incorporated": "Years since incorporation",
    "pec_registration": "PEC registration category",
    "years_experience": "Years of relevant experience",
    "similar_contracts_count": "Number of similar contracts",
    "similar_contract_max_value_pkr": "Largest similar contract (PKR)",
    "avg_annual_turnover_pkr": "Average annual turnover (PKR)",
    "bank_account_scheduled_bank": "Account in a scheduled bank",
    "not_blacklisted": "Not blacklisted",
    "certifications": "Certifications",
    "documents_on_hand": "Document to attach",
    "bid_security": "Bid security",
    "deadline": "Submission deadline",
    "other": "Unmapped requirement",
}


def parse_number(value) -> Optional[float]:
    """'Rs 5 million' / '5,000,000' / '5' -> 5000000.0 / 5.0. None when there is no number."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).lower().replace(",", "")
    mult = 1.0
    if "million" in s or re.search(r"\bm\b", s):
        mult = 1_000_000.0
    if "billion" in s:
        mult = 1_000_000_000.0
    if "lac" in s or "lakh" in s:
        mult = 100_000.0
    if "crore" in s:
        mult = 10_000_000.0
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m:
        return None
    n = float(m.group()) * mult
    # "5000000 million" would be a model error; keep the raw number in that case.
    return n if n < 1e15 else float(m.group())


def parse_pec(value) -> Optional[int]:
    if not value:
        return None
    m = re.search(r"c\s*-?\s*([1-6ab])", str(value).lower())
    if not m:
        return None
    code = f"C-{m.group(1).upper()}"
    return PEC_ORDER.index(code) if code in PEC_ORDER else None


def _doc_match(requirement_text: str, docs: List[str]) -> bool:
    words = set(re.findall(r"[a-z]{4,}", requirement_text.lower()))
    for d in docs:
        dwords = set(re.findall(r"[a-z]{4,}", str(d).lower()))
        if dwords and len(words & dwords) >= max(1, min(2, len(dwords))):
            return True
    return False


def evaluate_one(req: Requirement, profile: CompanyProfile) -> dict:
    """One requirement -> {status, evidence}. Unknown is always Needs Review, never a guess."""
    f, thr = req.profile_field, req.threshold
    row = {
        "id": req.id, "category": req.category, "requirement": req.text,
        "page": req.page, "mandatory": req.mandatory, "field": FIELD_LABEL.get(f, f),
        "quote": req.quote, "verified": req.verified,
    }

    if f in BOOL_FIELDS:
        have = bool(getattr(profile, f))
        row.update(status=MET if have else NOT_MET,
                   evidence="Yes, on file" if have else "Company profile says no")
    elif f in NUMERIC_FIELDS:
        need, have = parse_number(thr), float(getattr(profile, f) or 0)
        if need is None:
            row.update(status=REVIEW, evidence=f"No numeric threshold extracted (company has {have:,.0f})")
        else:
            row.update(status=MET if have >= need else NOT_MET,
                       evidence=f"Required {need:,.0f} / company has {have:,.0f}")
    elif f == "pec_registration":
        need, have = parse_pec(thr), parse_pec(profile.pec_registration)
        if need is None or have is None:
            row.update(status=REVIEW, evidence=f"Required {thr or 'unclear'} / company has {profile.pec_registration or 'none'}")
        else:
            row.update(status=MET if have >= need else NOT_MET,
                       evidence=f"Required {thr} / company has {profile.pec_registration}")
    elif f == "certifications":
        needle = str(thr or req.text).lower()
        hit = next((c for c in profile.certifications if str(c).lower() in needle or needle in str(c).lower()), None)
        row.update(status=MET if hit else NOT_MET,
                   evidence=f"Holds {hit}" if hit else "Not in company certifications")
    elif f == "documents_on_hand":
        have = _doc_match(req.text + " " + str(thr or ""), profile.documents_on_hand)
        row.update(status=MET if have else NOT_MET,
                   evidence="Document available" if have else "Document not marked as available")
    elif f == "bid_security":
        amount = parse_number(thr)
        row.update(status=REVIEW,
                   evidence=f"Bid security {amount:,.0f} PKR - arrange pay order" if amount else "Bid security required - amount to confirm")
    elif f == "deadline":
        row.update(status=REVIEW, evidence=f"Deadline: {thr or 'see quote'}")
    else:
        row.update(status=REVIEW, evidence="Could not map to a profile field - human review")

    if not req.verified and row["status"] != REVIEW:
        row["status"] = REVIEW
        row["evidence"] = f"{row['evidence']} (citation unverified)"
    return row


def evaluate(reqs: List[Requirement], profile: CompanyProfile) -> dict:
    rows = [evaluate_one(r, profile) for r in reqs]
    counts = {MET: 0, NOT_MET: 0, REVIEW: 0}
    for r in rows:
        counts[r["status"]] += 1
    # The decision is about QUALIFICATION. The submission deadline is a countdown every
    # bidder faces, not an eligibility gate, so it informs the user without blocking.
    blockers = [r for r in rows if r["mandatory"] and r["status"] == NOT_MET
                and r["field"] != FIELD_LABEL["deadline"]]
    checks = [r for r in rows if r["mandatory"] and r["status"] == REVIEW
              and r["field"] != FIELD_LABEL["deadline"]]
    decision = NO_BID if blockers else (CONDITIONAL if checks else BID)
    return {
        "rows": rows, "counts": counts, "decision": decision,
        "blockers": blockers, "checks": checks,
        "missing_documents": missing_documents(rows),
    }


def missing_documents(rows: List[dict]) -> List[str]:
    out = []
    for r in rows:
        if r["status"] == NOT_MET or (r["status"] == REVIEW and r["field"] in ("Bid security", "Document to attach")):
            out.append(f"{r['requirement']} (page {r['page']})")
    return out


def days_left(deadline_text: str, today: Optional[date] = None) -> Optional[int]:
    """Deadline arithmetic in code, never in the model."""
    today = today or date.today()
    s = str(deadline_text or "")
    for fmt, pat in (("%Y-%m-%d", r"\d{4}-\d{2}-\d{2}"),
                     ("%d-%m-%Y", r"\d{2}-\d{2}-\d{4}"),
                     ("%d/%m/%Y", r"\d{2}/\d{2}/\d{4}"),
                     ("%d %B %Y", r"\d{1,2} [A-Za-z]+ \d{4}"),
                     ("%B %d, %Y", r"[A-Za-z]+ \d{1,2}, \d{4}")):
        m = re.search(pat, s)
        if m:
            try:
                return (datetime.strptime(m.group(), fmt).date() - today).days
            except ValueError:
                continue
    return None
