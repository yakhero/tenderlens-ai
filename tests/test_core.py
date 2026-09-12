import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import compliance, exports, extract, ingest, verify  # noqa: E402
from core.schemas import CompanyProfile, Requirement, coerce_requirement  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def req(**kw):
    base = dict(id="R-001", category="eligibility", text="t", quote="q", page=1,
                mandatory=True, profile_field="other", threshold=None, verified=True)
    base.update(kw)
    return Requirement(**base)


def profile(name="a"):
    p = ROOT / "data" / "profiles" / f"profile_{name}.json"
    return CompanyProfile.from_dict(json.loads(p.read_text()))


# ---------------------------------------------------------------- parsing
def test_parse_number_handles_pakistani_notation():
    assert compliance.parse_number("Rs. 5 million") == 5_000_000
    assert compliance.parse_number("5,000,000") == 5_000_000
    assert compliance.parse_number("3") == 3
    assert compliance.parse_number("2 lac") == 200_000
    assert compliance.parse_number("") is None
    assert compliance.parse_number(None) is None


def test_parse_pec_ordering():
    assert compliance.parse_pec("C-4") < compliance.parse_pec("C-1")
    assert compliance.parse_pec("c 3") == compliance.parse_pec("C-3")
    assert compliance.parse_pec("none") is None


# ---------------------------------------------------------------- rules
def test_boolean_requirement():
    a = profile("a")
    assert compliance.evaluate_one(req(profile_field="ntn"), a)["status"] == compliance.MET
    assert compliance.evaluate_one(req(profile_field="sales_tax_registered"), a)["status"] == compliance.NOT_MET


def test_numeric_threshold_both_ways():
    a, b = profile("a"), profile("b")
    r = req(profile_field="avg_annual_turnover_pkr", threshold="5000000")
    assert compliance.evaluate_one(r, a)["status"] == compliance.NOT_MET   # 3M < 5M
    assert compliance.evaluate_one(r, b)["status"] == compliance.MET       # 60M >= 5M


def test_missing_threshold_is_review_not_a_guess():
    r = req(profile_field="years_experience", threshold=None)
    assert compliance.evaluate_one(r, profile("a"))["status"] == compliance.REVIEW


def test_pec_category_comparison():
    r = req(profile_field="pec_registration", threshold="C-4")
    assert compliance.evaluate_one(r, profile("a"))["status"] == compliance.NOT_MET  # C-6 < C-4
    assert compliance.evaluate_one(r, profile("b"))["status"] == compliance.MET      # C-3 > C-4


def test_document_and_certification_matching():
    b = profile("b")
    assert compliance.evaluate_one(req(profile_field="documents_on_hand",
                                       text="Attach audited financial statements"), b)["status"] == compliance.MET
    assert compliance.evaluate_one(req(profile_field="certifications", threshold="ISO 9001"),
                                   b)["status"] == compliance.MET
    assert compliance.evaluate_one(req(profile_field="certifications", threshold="ISO 27001"),
                                   b)["status"] == compliance.NOT_MET


def test_unverified_citation_is_downgraded_to_review():
    r = req(profile_field="ntn", verified=False)
    assert compliance.evaluate_one(r, profile("b"))["status"] == compliance.REVIEW


# ---------------------------------------------------------------- decision
def test_decision_no_bid_conditional_bid():
    b = profile("b")
    ok = req(profile_field="ntn")
    blocker = req(id="R-002", profile_field="avg_annual_turnover_pkr", threshold="900000000")
    unclear = req(id="R-003", profile_field="other")
    assert compliance.evaluate([ok], b)["decision"] == compliance.BID
    assert compliance.evaluate([ok, unclear], b)["decision"] == compliance.CONDITIONAL
    assert compliance.evaluate([ok, unclear, blocker], b)["decision"] == compliance.NO_BID


def test_optional_requirement_never_blocks():
    b = profile("b")
    optional_fail = req(profile_field="certifications", threshold="ISO 14001", mandatory=False)
    assert compliance.evaluate([optional_fail], b)["decision"] == compliance.BID


def test_counts_and_missing_documents():
    res = compliance.evaluate([req(profile_field="ntn"),
                               req(id="R-002", profile_field="sales_tax_registered")], profile("a"))
    assert res["counts"][compliance.MET] == 1 and res["counts"][compliance.NOT_MET] == 1
    assert res["missing_documents"]


def test_days_left():
    assert compliance.days_left("2026-10-28", today=date(2026, 10, 8)) == 20
    assert compliance.days_left("28-10-2026", today=date(2026, 10, 8)) == 20
    assert compliance.days_left("no date here") is None


# ---------------------------------------------------------------- ingest + verify
def test_chunking_keeps_page_markers():
    pages = [(i, "x" * 500) for i in range(1, 21)]
    chunks = ingest.make_chunks(pages, max_chars=2000)
    assert len(chunks) > 1
    assert "[PAGE 1]" in chunks[0] and "[PAGE 20]" in chunks[-1]


def test_scanned_detection():
    assert ingest.is_scanned([(1, ""), (2, " "), (3, "x")])
    assert not ingest.is_scanned([(1, "a" * 300), (2, "b" * 300)])


def test_verifier_confirms_corrects_and_rejects():
    pages = ingest.read_any(str(ROOT / "data" / "tenders" / "sample_tender.txt"))
    good = req(quote="The bidder shall be registered for sales tax", page=2)
    wrong_page = req(id="R-002", quote="Each bid shall be accompanied by a bid security of Rs. 150,000", page=2)
    fake = req(id="R-003", quote="The bidder must own a helicopter and two submarines", page=2)
    out = verify.verify_requirements([good, wrong_page, fake], pages)
    assert out[0].verified is True
    assert out[1].verified is True and out[1].page == 3 and "corrected" in out[1].verify_note
    assert out[2].verified is False
    assert verify.verification_rate(out) > 60


# ---------------------------------------------------------------- schema + misc
def test_coerce_rejects_model_junk():
    assert coerce_requirement({"text": ""}, 1) is None
    r = coerce_requirement({"text": "x", "category": "nonsense", "profile_field": "hacked",
                            "page": "seven", "mandatory": "yes"}, 3)
    assert r.category == "eligibility" and r.profile_field == "other" and r.page == 1 and r.mandatory is True


def test_dedupe_and_renumber():
    a = req(text="Bidder shall hold NTN", profile_field="ntn")
    b = req(id="R-009", text="bidder shall hold ntn", profile_field="ntn")
    out = extract.dedupe([a, b])
    assert len(out) == 1 and out[0].id == "R-001"


def test_csv_export_has_header_and_rows():
    res = compliance.evaluate([req(profile_field="ntn")], profile("b"))
    csv_bytes = exports.rows_to_csv(res["rows"])
    assert csv_bytes.startswith(b"id,status") and b"Met" in csv_bytes


def test_end_to_end_offline_sample():
    """Sample tender + hand-written requirements: A must fail, B must pass."""
    pages = ingest.read_any(str(ROOT / "data" / "tenders" / "sample_tender.txt"))
    reqs = [
        req(id="R-001", quote="The bidder shall possess a valid National Tax Number", page=2, profile_field="ntn"),
        req(id="R-002", quote="The bidder shall be registered for sales tax", page=2,
            profile_field="sales_tax_registered"),
        req(id="R-003", quote="shall have been in continuous operation for at least five (05) years", page=2,
            profile_field="years_incorporated", threshold="5"),
        req(id="R-004", quote="minimum average annual turnover of Rs. 5 million", page=2,
            profile_field="avg_annual_turnover_pkr", threshold="Rs. 5 million"),
    ]
    reqs = verify.verify_requirements(reqs, pages)
    assert all(r.verified for r in reqs)
    assert compliance.evaluate(reqs, profile("a"))["decision"] == compliance.NO_BID
    assert compliance.evaluate(reqs, profile("b"))["decision"] == compliance.BID
