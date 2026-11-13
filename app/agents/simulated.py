"""
Simulated LLM stages for the public demo (DEMO_MODE=true with no GROQ_API_KEY).

The real pipeline uses an LLM to plan, critique and write. The demo has no
paid key, so these functions stand in with simple, deterministic rules over
the real retrieved passages. Their output is labelled as simulated everywhere
it is shown: the planner uses fixed question templates, the critic reports
only what can be counted, and the writer quotes sentences instead of
summarising. Retrieval (Wikipedia, arXiv) and ranking are not simulated.
"""
import re

SIMULATED_NOTE = (
    "LLM stages are simulated in this demo (no API key): sub-questions come from templates, "
    "the critique only counts sources, and the report quotes the top passages instead of summarising them."
)

_LEAD = re.compile(
    r"^(please\s+)?(what|how|why|when|who|which|where|explain|describe|tell me about|give me|is|are|does|do|can)\b"
    r"(\s+(is|are|was|were|does|do|did|an?|the)\b)*\s*",
    re.IGNORECASE,
)
_TRAILING_VERB = re.compile(r"\s+(work|works|mean|means|happen|happens|function|functions)$", re.IGNORECASE)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def topic_of(query: str) -> str:
    """'How does transformer attention work?' -> 'transformer attention'."""
    t = _LEAD.sub("", query.strip()).rstrip("?.! ")
    t = _TRAILING_VERB.sub("", t)
    return t or query.strip().rstrip("?.! ")


def plan(query: str) -> list[str]:
    topic = topic_of(query)
    subs = [query.strip(), f"{topic} overview", f"{topic} applications", f"{topic} limitations"]
    seen, out = set(), []
    for s in subs:
        if s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def critique(passages: list[dict]) -> dict:
    sources = {p.get("source", "unknown") for p in passages}
    gaps = []
    if not passages:
        gaps.append("No passages met the relevance threshold.")
    if "arxiv" not in sources:
        gaps.append("No arXiv paper among the top passages: no research-level source.")
    if "wikipedia" not in sources:
        gaps.append("No encyclopedia overview among the top passages.")
    if 0 < len(passages) < 3:
        gaps.append(f"Only {len(passages)} relevant passage(s) found.")
    avg = sum(p.get("similarity", 0.0) for p in passages) / len(passages) if passages else 0.0
    quality = "high" if avg >= 0.35 and len(passages) >= 5 else "medium" if avg >= 0.2 else "low"
    return {
        "contradictions": [],
        "gaps": gaps,
        "unreliable_sources": [],
        "overall_quality": quality,
        "simulated": True,
        "note": "Contradictions are not checked without an LLM.",
    }


def first_sentences(text: str, n: int = 2, max_chars: int = 420) -> str:
    sents = _SENT.split(" ".join(text.split()))
    out = " ".join(sents[:n])
    return out if len(out) <= max_chars else out[: max_chars - 3].rsplit(" ", 1)[0] + "..."


def synthesize(query: str, passages: list[dict]) -> dict:
    if not passages:
        return {"summary": f"No relevant sources were found for: {query}", "key_findings": [], "sections": []}
    top = passages[:5]
    summary = first_sentences(top[0].get("text", ""), n=2)
    findings = [
        f"{first_sentences(p.get('text', ''), n=1, max_chars=240)} ({p.get('title', 'source')})" for p in top[:4]
    ]
    sections = []
    for source, title in (("wikipedia", "Background (Wikipedia)"), ("arxiv", "Research (arXiv)"),
                          ("duckduckgo", "Web")):
        group = [p for p in passages[:10] if p.get("source") == source][:3]
        if group:
            sections.append({
                "title": title,
                "content": " ".join(first_sentences(p.get("text", ""), n=2, max_chars=300) for p in group),
                "sources": [p["url"] for p in group if p.get("url")],
            })
    return {"summary": summary, "key_findings": findings, "sections": sections}
