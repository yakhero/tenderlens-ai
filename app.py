"""TenderLens AI - Streamlit UI.

Flow: upload tender -> AI extracts requirements (cited) -> human edits -> code decides
Bid / No-Bid against a company profile -> exports.
"""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from core import compliance, draft, exports, extract, ingest, llm, verify
from core.schemas import PROFILE_FIELDS, CompanyProfile, Requirement, coerce_requirement

ROOT = Path(__file__).parent
DATA, CACHE = ROOT / "data", ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="TenderLens AI", page_icon="📄", layout="wide")

STATUS_COLOR = {"Met": "#1a7f4b", "Not Met": "#c0392b", "Needs Review": "#b8860b"}
DECISION_STYLE = {
    "BID": ("#e8f5ee", "#1a7f4b", "BID - you meet every mandatory requirement on file"),
    "NO-BID": ("#fdecec", "#c0392b", "NO-BID - a mandatory requirement is not met"),
    "CONDITIONAL": ("#fff4e0", "#b8860b", "CONDITIONAL - clear the open items before you commit"),
}


# ---------------------------------------------------------------- helpers
def load_profiles() -> dict:
    out = {}
    for p in sorted((DATA / "profiles").glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out[d.get("company_name", p.stem)] = CompanyProfile.from_dict(d)
        except Exception as e:  # noqa: BLE001
            st.sidebar.warning(f"Could not read {p.name}: {e}")
    return out


def reqs_to_df(reqs) -> pd.DataFrame:
    return pd.DataFrame([{
        "keep": True, "id": r.id, "category": r.category, "requirement": r.text,
        "field": r.profile_field, "threshold": "" if r.threshold is None else str(r.threshold),
        "mandatory": r.mandatory, "page": r.page,
        "cited": "yes" if r.verified else "CHECK", "quote": r.quote, "note": r.verify_note,
    } for r in reqs])


def df_to_reqs(df: pd.DataFrame):
    out = []
    for i, row in enumerate(df.to_dict("records"), start=1):
        if not row.get("keep", True):
            continue
        r = coerce_requirement({
            "id": row.get("id") or f"R-{i:03d}", "category": row.get("category"),
            "text": row.get("requirement"), "quote": row.get("quote", ""),
            "page": row.get("page", 1), "mandatory": row.get("mandatory", True),
            "profile_field": row.get("field", "other"), "threshold": row.get("threshold") or None,
        }, i)
        if r:
            r.verified = str(row.get("cited", "")).lower() == "yes"
            out.append(r)
    return out


def cache_path(h: str) -> Path:
    return CACHE / f"{h}.json"


def save_cache(h: str, reqs):
    cache_path(h).write_text(json.dumps([r.to_dict() for r in reqs], indent=1), encoding="utf-8")


def load_cache(h: str):
    p = cache_path(h)
    if not p.exists():
        return None
    return [Requirement(**d) for d in json.loads(p.read_text(encoding="utf-8"))]


# ---------------------------------------------------------------- sidebar
st.sidebar.title("TenderLens AI")
st.sidebar.caption("Read the tender. Know if you qualify. Before you bid.")

api_key = (st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") and st.secrets else "") or llm.key_from_env()
api_key = st.sidebar.text_input("Gemini API key", value=api_key, type="password",
                                help="Get a free key at aistudio.google.com. Or set it in Streamlit secrets.")
model = st.sidebar.text_input("Model", value=st.session_state.get("model", ""),
                              placeholder="auto-detect", help="Leave blank to auto-pick a Flash model your key supports.")
if st.sidebar.button("Check key / list models", use_container_width=True):
    try:
        names = llm.list_models(api_key)
        picked = llm.resolve_model(api_key)
        st.session_state["model"] = picked
        st.sidebar.success(f"Key works. Using: {picked}")
        st.sidebar.caption("Available: " + ", ".join(names[:12]))
    except llm.LLMError as e:
        st.sidebar.error(str(e))

st.sidebar.divider()
st.sidebar.markdown("**How it works**")
st.sidebar.markdown(
    "1. AI extracts requirements with page citations\n"
    "2. Code verifies every citation\n"
    "3. **You** review and edit\n"
    "4. Code decides Bid / No-Bid\n"
    "5. You approve the outputs",
)
st.sidebar.info("The model reads and writes. Every number and verdict is computed in tested Python.")

# ---------------------------------------------------------------- step 1
st.title("Tender compliance copilot")
st.caption("PPRA / e-PADS bidding documents, checked against your company profile.")

st.header("1. Tender document")
samples = sorted(list((DATA / "tenders").glob("*.pdf")) + list((DATA / "tenders").glob("*.txt")))
c1, c2 = st.columns([3, 2])
with c1:
    up = st.file_uploader("Upload a tender PDF", type=["pdf", "txt"])
with c2:
    choice = st.selectbox("...or use a bundled tender", ["-"] + [p.name for p in samples])

raw, source_name = None, ""
if up is not None:
    raw, source_name = up.getvalue(), up.name
elif choice != "-":
    p = DATA / "tenders" / choice
    raw, source_name = p.read_bytes(), choice

if raw:
    h = ingest.file_hash(raw)
    try:
        pages = ingest.read_pdf_bytes(raw) if source_name.lower().endswith(".pdf") else ingest.read_txt_bytes(raw)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    st.success(f"**{source_name}** · {len(pages)} pages · {sum(len(t) for _, t in pages):,} characters")
    if ingest.is_scanned(pages):
        st.warning("This looks like a scanned PDF with little selectable text. Extraction quality will be poor.")

    cached = load_cache(h)
    b1, b2 = st.columns([1, 1])
    run = b1.button("Extract requirements", type="primary", use_container_width=True)
    if cached and b2.button(f"Load saved extraction ({len(cached)} requirements)", use_container_width=True):
        st.session_state["reqs"] = cached
        st.session_state["confirmed"] = None

    if run:
        if not api_key:
            st.error("Add your Gemini API key in the sidebar first.")
        else:
            use_model = model or st.session_state.get("model") or llm.resolve_model(api_key)
            st.session_state["model"] = use_model
            bar = st.progress(0.0, text="Reading the tender...")
            try:
                reqs = extract.extract_requirements(
                    pages, api_key, use_model,
                    progress=lambda i, n: bar.progress(i / n, text=f"Extracting, chunk {i} of {n} ({use_model})"))
                reqs = verify.verify_requirements(reqs, pages)
                st.session_state["reqs"] = reqs
                st.session_state["confirmed"] = None
                save_cache(h, reqs)
                bar.empty()
                st.success(f"{len(reqs)} requirements extracted · {verify.verification_rate(reqs)}% of citations auto-verified")
            except llm.LLMError as e:
                bar.empty()
                st.error(f"Extraction failed: {e}")

# ---------------------------------------------------------------- step 2
reqs = st.session_state.get("reqs")
if reqs:
    st.header("2. Review what the AI found")
    st.caption("Human gate 1. Rows marked **CHECK** could not be matched to the cited page - read those first. "
               "Untick *keep* to drop a row, or fix any field.")
    edited = st.data_editor(
        reqs_to_df(reqs), use_container_width=True, hide_index=True, num_rows="dynamic",
        column_config={
            "keep": st.column_config.CheckboxColumn("keep", width="small"),
            "field": st.column_config.SelectboxColumn("maps to", options=PROFILE_FIELDS, width="medium"),
            "mandatory": st.column_config.CheckboxColumn("must", width="small"),
            "page": st.column_config.NumberColumn("page", width="small"),
            "cited": st.column_config.TextColumn("cited", width="small", disabled=True),
            "quote": st.column_config.TextColumn("verbatim quote", width="large"),
            "note": st.column_config.TextColumn("note", disabled=True),
        },
        key="editor",
    )
    if st.button("Confirm requirements", type="primary"):
        st.session_state["confirmed"] = df_to_reqs(edited)

# ---------------------------------------------------------------- step 3
confirmed = st.session_state.get("confirmed")
if confirmed:
    st.header("3. Company profile")
    profiles = load_profiles()
    names = list(profiles)
    pick = st.selectbox("Profile", names) if names else None
    profile = profiles.get(pick, CompanyProfile())
    with st.expander("Edit this profile"):
        d = profile.to_dict()
        cols = st.columns(3)
        for i, (k, v) in enumerate(d.items()):
            col = cols[i % 3]
            if isinstance(v, bool):
                d[k] = col.checkbox(k, value=v)
            elif isinstance(v, int):
                d[k] = col.number_input(k, value=int(v), step=1)
            elif isinstance(v, list):
                d[k] = [s.strip() for s in col.text_area(k, value="\n".join(map(str, v)), height=90).split("\n") if s.strip()]
            else:
                d[k] = col.text_input(k, value="" if v is None else str(v))
        profile = CompanyProfile.from_dict(d)

    result = compliance.evaluate(confirmed, profile)

    st.header("4. Decision")
    bg, fg, msg = DECISION_STYLE[result["decision"]]
    st.markdown(
        f"<div style='background:{bg};border-left:10px solid {fg};padding:14px 18px;border-radius:6px'>"
        f"<span style='color:{fg};font-size:30px;font-weight:700'>{result['decision']}</span>"
        f"<div style='color:#333;font-size:15px;margin-top:2px'>{msg}</div></div>",
        unsafe_allow_html=True)

    c = result["counts"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Requirements", sum(c.values()))
    m2.metric("Met", c["Met"])
    m3.metric("Not met", c["Not Met"])
    m4.metric("Needs review", c["Needs Review"])

    deadline_row = next((r for r in result["rows"] if r["field"] == "Submission deadline"), None)
    if deadline_row:
        dl = compliance.days_left(deadline_row["evidence"] + " " + deadline_row["quote"])
        if dl is not None:
            st.info(f"Submission deadline in **{dl} days** (computed in code from page {deadline_row['page']}).")

    if result["blockers"]:
        st.error("**Blocking:** " + " · ".join(f"{r['requirement']} (p{r['page']})" for r in result["blockers"]))
    if result["checks"]:
        st.warning("**To check:** " + " · ".join(f"{r['requirement']} (p{r['page']})" for r in result["checks"]))

    st.subheader("Compliance matrix")
    df = pd.DataFrame(result["rows"])[["id", "status", "mandatory", "category", "field",
                                       "requirement", "evidence", "page", "verified", "quote"]]
    st.dataframe(
        df.style.map(lambda v: f"color:{STATUS_COLOR.get(v, '')};font-weight:600" if v in STATUS_COLOR else "",
                     subset=["status"]),
        use_container_width=True, hide_index=True, height=420)

    st.subheader("5. Action pack")
    if result["missing_documents"]:
        st.markdown("**Before you bid, arrange:**")
        for d_ in result["missing_documents"]:
            st.markdown(f"- {d_}")
    else:
        st.markdown("Nothing outstanding on this profile.")

    e1, e2, e3 = st.columns(3)
    xlsx = exports.rows_to_xlsx(result["rows"], result["decision"])
    if xlsx:
        e1.download_button("Download matrix (Excel)", xlsx, "compliance_matrix.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    e2.download_button("Download matrix (CSV)", exports.rows_to_csv(result["rows"]),
                       "compliance_matrix.csv", "text/csv", use_container_width=True)
    if e3.button("Draft clarification email", use_container_width=True):
        use_model = model or st.session_state.get("model") or "gemini-2.0-flash"
        try:
            st.session_state["letter"] = draft.clarification_email(
                result, profile.company_name, source_name if raw else "", api_key, use_model)
        except llm.LLMError as e:
            st.session_state["letter"] = f"(Could not reach the model: {e})\n\n" + draft.offline_note(result)

    if st.session_state.get("letter"):
        st.text_area("Human gate 2: edit before sending", st.session_state["letter"], height=260)
        st.download_button("Download email (.txt)", st.session_state["letter"].encode("utf-8"),
                           "clarification_email.txt", "text/plain")
