"""Citation verifier: does the quoted sentence really exist on the cited page?

This is the anti-hallucination layer, and it runs in code - the model never sets `verified`.
"""
import difflib
import re
from typing import List

from core.ingest import Page
from core.schemas import Requirement

MIN_RATIO = 0.82
MIN_LEN = 12


def norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9%]+", " ", s)
    return " ".join(s.split())


MIN_TOKEN_OVERLAP = 0.85
MIN_LCS_RATIO = 0.40


def _found(quote_n: str, page_n: str) -> bool:
    """Is this quote really on this page?

    Exact substring is the happy path. PDF extraction breaks lines and hyphenates, so we
    also accept a near match, but only when BOTH (a) almost every word of the quote appears
    on the page and (b) a long contiguous run of the quote appears verbatim. Word overlap
    alone would pass an invented sentence built from common tender vocabulary.
    """
    if not quote_n or not page_n:
        return False
    if quote_n in page_n:
        return True
    q_tokens = quote_n.split()
    if not q_tokens:
        return False
    p_tokens = set(page_n.split())
    overlap = sum(1 for t in q_tokens if t in p_tokens) / len(q_tokens)
    if overlap < MIN_TOKEN_OVERLAP:
        return False
    match = difflib.SequenceMatcher(None, page_n, quote_n, autojunk=False).find_longest_match(
        0, len(page_n), 0, len(quote_n))
    return (match.size / len(quote_n)) >= MIN_LCS_RATIO


def verify_requirements(reqs: List[Requirement], pages: List[Page], search_radius: int = 2) -> List[Requirement]:
    """Mark each requirement verified, and silently fix the page number if the quote is nearby."""
    text_by_page = {no: norm(t) for no, t in pages}
    page_numbers = sorted(text_by_page)
    for r in reqs:
        q = norm(r.quote)
        if len(q) < MIN_LEN:
            r.verified, r.verify_note = False, "quote too short to verify"
            continue
        if _found(q, text_by_page.get(r.page, "")):
            r.verified, r.verify_note = True, ""
            continue
        moved = None
        for delta in range(1, search_radius + 1):
            for cand in (r.page - delta, r.page + delta):
                if cand in text_by_page and _found(q, text_by_page[cand]):
                    moved = cand
                    break
            if moved:
                break
        if moved:
            r.verify_note = f"citation corrected from page {r.page} to {moved}"
            r.page, r.verified = moved, True
        else:
            hits = [p for p in page_numbers if _found(q, text_by_page[p])]
            if hits:
                r.verify_note = f"citation corrected from page {r.page} to {hits[0]}"
                r.page, r.verified = hits[0], True
            else:
                r.verified, r.verify_note = False, "quote not found in document - review manually"
    return reqs


def verification_rate(reqs: List[Requirement]) -> float:
    return round(100.0 * sum(1 for r in reqs if r.verified) / len(reqs), 1) if reqs else 0.0
