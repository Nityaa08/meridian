"""
Grounding check: is each sentence of the report backed by the passages it cites?

No LLM is used. A sentence counts as supported when at least
SUPPORT_THRESHOLD of its content words (lowercased, 4+ letters, not stop
words) appear in one of the passages it may cite: a section's own sources,
or, for the summary and key findings, the passages given to the writer.

It is a lexical check, so it catches invented content (names, numbers,
claims with no matching words in the sources) but not a sentence that
reuses the source's words to say something different. In the simulated
demo the writer quotes passages, so the score is high by construction;
with a real LLM writer it is a real signal.
"""
import re

SUPPORT_THRESHOLD = 0.6
# The writer is shown the first 500 characters of each passage, so only those count.
PASSAGE_CHARS = 500
_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z][a-z0-9'-]{3,}")
STOP = frozenset("""
    about above after again against also among because been before being below between both could does doing
    down during each from further have having here into itself just more most other over same should some
    such than that their them then there these they this those through under until very were what when where
    which while will with would your within without across using used upon
""".split())


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in STOP}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(" ".join(str(text).split())) if len(content_words(s)) >= 3]


def support(sentence: str, passages: list[str]) -> float:
    """Best share of the sentence's content words found in any one passage."""
    words = content_words(sentence)
    if not words:
        return 1.0
    return max((len(words & content_words(p)) / len(words) for p in passages), default=0.0)


def strip_citation(text: str, titles: list[str]) -> str:
    """Drop a trailing "(source title)" citation, so the title is not checked as if it were a claim.

    The demo writer ends each key finding with the title of the passage it quotes. The sentence
    splitter made that title a sentence of its own, and a title whose words are not in its
    passage's text (a search engine's title, for one) was reported as unsupported.
    """
    t = str(text).rstrip()
    for title in sorted({x.strip() for x in titles if x and x.strip()}, key=len, reverse=True):
        cite = f"({title})"
        if t.endswith(cite):
            return t[: -len(cite)].rstrip()
    return t


def check_report(report: dict, passages: list[dict]) -> dict:
    if not passages:
        # Nothing was retrieved, so the report is a "no sources found" note, not claims.
        return {"sentences_checked": 0, "supported": 0, "score": None, "threshold": SUPPORT_THRESHOLD,
                "unsupported": [], "method": "skipped: no passages"}
    by_url = {p.get("url"): p.get("text", "")[:PASSAGE_CHARS] for p in passages if p.get("url")}
    all_texts = [p.get("text", "")[:PASSAGE_CHARS] for p in passages]
    titles = [" ".join(str(p.get("title", "")).split()) for p in passages] + ["source"]

    parts: list[tuple[str, str, list[str]]] = [("summary", report.get("summary", ""), all_texts)]
    parts += [("key finding", strip_citation(" ".join(f.split()), titles), all_texts)
              for f in report.get("key_findings", []) if isinstance(f, str)]
    for s in report.get("sections", []):
        cited = [by_url[u] for u in s.get("sources", []) if u in by_url]
        parts.append((f"section: {s.get('title', '')}", s.get("content", ""), cited))

    checked, unsupported = 0, []
    for where, text, texts in parts:
        for sent in sentences(text):
            checked += 1
            score = support(sent, texts)
            if score < SUPPORT_THRESHOLD:
                unsupported.append({"where": where, "sentence": sent, "support": round(score, 2)})
    supported = checked - len(unsupported)
    return {
        "sentences_checked": checked,
        "supported": supported,
        "score": round(supported / checked, 4) if checked else None,
        "threshold": SUPPORT_THRESHOLD,
        "unsupported": unsupported[:10],
        "method": "lexical: share of a sentence's content words found in a cited passage",
    }
