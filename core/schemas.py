"""Data shapes for TenderLens AI. Stdlib only - frozen contract for the whole team."""
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

CATEGORIES = ["eligibility", "technical", "financial", "document", "deadline", "evaluation"]

# The ONLY values the LLM may put in `profile_field`.
PROFILE_FIELDS = [
    "ntn", "on_fbr_atl", "sales_tax_registered", "secp_incorporated",
    "years_incorporated", "pec_registration", "years_experience",
    "similar_contracts_count", "similar_contract_max_value_pkr",
    "avg_annual_turnover_pkr", "bank_account_scheduled_bank", "not_blacklisted",
    "certifications", "documents_on_hand", "bid_security", "deadline", "other",
]

BOOL_FIELDS = ["ntn", "on_fbr_atl", "sales_tax_registered", "secp_incorporated",
               "bank_account_scheduled_bank", "not_blacklisted"]
NUMERIC_FIELDS = ["years_incorporated", "years_experience", "similar_contracts_count",
                  "similar_contract_max_value_pkr", "avg_annual_turnover_pkr"]

# PEC categories, lowest capacity first. C-A is the highest.
PEC_ORDER = ["C-6", "C-5", "C-4", "C-3", "C-2", "C-1", "C-B", "C-A"]


@dataclass
class Requirement:
    id: str
    category: str
    text: str
    quote: str
    page: int
    mandatory: bool = True
    profile_field: str = "other"
    threshold: Any = None
    verified: bool = False           # set by verify.py, never by the model
    verify_note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompanyProfile:
    company_name: str = ""
    city: str = ""
    ntn: bool = False
    on_fbr_atl: bool = False
    sales_tax_registered: bool = False
    secp_incorporated: bool = False
    years_incorporated: int = 0
    pec_registration: Optional[str] = None
    years_experience: int = 0
    similar_contracts_count: int = 0
    similar_contract_max_value_pkr: int = 0
    avg_annual_turnover_pkr: int = 0
    bank_account_scheduled_bank: bool = False
    not_blacklisted: bool = False
    certifications: list = field(default_factory=list)
    documents_on_hand: list = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "CompanyProfile":
        known = {k: v for k, v in d.items() if k in CompanyProfile.__annotations__}
        return CompanyProfile(**known)

    def to_dict(self) -> dict:
        return asdict(self)


def _as_bool(v, default=True) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "mandatory")
    return default


def _as_int(v, default=0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def coerce_requirement(raw: dict, idx: int) -> Optional[Requirement]:
    """Turn one raw LLM object into a Requirement, or None if unusable.

    The model is not trusted: anything out of range is pulled back to a safe value
    rather than crashing the app mid-demo.
    """
    if not isinstance(raw, dict):
        return None
    quote = str(raw.get("quote") or "").strip()
    text = str(raw.get("text") or "").strip() or quote[:120]
    if not text:
        return None
    cat = str(raw.get("category") or "").strip().lower()
    if cat not in CATEGORIES:
        cat = "eligibility"
    pf = str(raw.get("profile_field") or "other").strip().lower()
    if pf not in PROFILE_FIELDS:
        pf = "other"
    thr = raw.get("threshold", None)
    if isinstance(thr, str) and thr.strip().lower() in ("", "null", "none", "n/a"):
        thr = None
    return Requirement(
        id=str(raw.get("id") or f"R-{idx:03d}"),
        category=cat,
        text=text[:300],
        quote=quote[:1200],
        page=max(1, _as_int(raw.get("page"), 1)),
        mandatory=_as_bool(raw.get("mandatory"), True),
        profile_field=pf,
        threshold=thr,
    )
