"""Drafting Agent: turns the code-produced result into text a human edits and sends.

The model writes prose only. Every fact in the prompt was computed by compliance.py.
"""
from core import llm
from core.compliance import NO_BID

LETTER_PROMPT = """Write a short, formal clarification email from a bidder to a Pakistani
procuring agency. Plain business English, no flattery, under 180 words.

Company: {company}
Tender: {tender}
Ask the agency to clarify ONLY these points, quoting the clause and page number for each:
{points}

End with a request for a written response before the bid submission deadline.
Output the email only: a subject line, then the body. Invent no facts.
"""

SUMMARY_PROMPT = """Write a 3-sentence internal note for the bid manager. Facts only, no advice
beyond what is given.

Decision: {decision}
Requirements checked: {total} ({met} met, {not_met} not met, {review} need review)
Blocking items: {blockers}
Items to check: {checks}
"""


def clarification_email(result: dict, company: str, tender: str, api_key: str, model: str) -> str:
    points = [f'- Page {r["page"]}, {r["requirement"]}: "{r["quote"][:180]}"'
              for r in (result["checks"] + result["blockers"])[:6]]
    if not points:
        return "No ambiguous or blocking clauses were found, so no clarification is needed."
    return llm.generate_text(
        LETTER_PROMPT.format(company=company or "our company", tender=tender or "the tender",
                             points="\n".join(points)),
        api_key, model)


def internal_note(result: dict, api_key: str, model: str) -> str:
    c = result["counts"]
    return llm.generate_text(SUMMARY_PROMPT.format(
        decision=result["decision"], total=sum(c.values()),
        met=c["Met"], not_met=c["Not Met"], review=c["Needs Review"],
        blockers="; ".join(r["requirement"] for r in result["blockers"]) or "none",
        checks="; ".join(r["requirement"] for r in result["checks"]) or "none",
    ), api_key, model)


def offline_note(result: dict) -> str:
    """Used when there is no API key: same facts, no model."""
    c = result["counts"]
    head = {NO_BID: "Do not bid as things stand."}.get(result["decision"], "")
    lines = [f"Decision: {result['decision']}. {head}".strip(),
             f"{sum(c.values())} requirements checked: {c['Met']} met, {c['Not Met']} not met, {c['Needs Review']} need review."]
    if result["blockers"]:
        lines.append("Blocking: " + "; ".join(r["requirement"] for r in result["blockers"]))
    if result["checks"]:
        lines.append("To check: " + "; ".join(r["requirement"] for r in result["checks"]))
    return "\n".join(lines)
