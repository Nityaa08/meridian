"""Grounding check: each report sentence against the passages it may cite."""
from app.agents import simulated
from app.pipeline.grounding import check_report, sentences, strip_citation, support, PASSAGE_CHARS

ATTN = {
    "url": "https://en.wikipedia.org/wiki/Attention_(machine_learning)",
    "title": "Attention (machine learning)",
    "source": "wikipedia",
    "text": "Attention is a machine learning method that determines the relative importance of each "
            "component in a sequence relative to the other components in that sequence. Transformers "
            "use scaled dot-product attention computed over queries, keys and values.",
}
LONG = {
    "url": "https://arxiv.org/abs/2004.05150",
    "title": "Longformer",
    "source": "arxiv",
    "text": "Transformer-based models are unable to process long sequences due to their self-attention "
            "operation, which scales quadratically with the sequence length.",
}


def test_a_quoted_sentence_is_supported_and_an_invented_one_is_not():
    passages = [ATTN["text"]]
    assert support("Transformers use scaled dot-product attention over queries, keys and values.", passages) >= 0.6
    assert support("Attention was invented in 1987 by Margaret Hamilton at Bell Labs.", passages) < 0.6


def test_short_fragments_are_not_counted_as_sentences():
    assert sentences("See [1]. Attention weighs each component of a sequence.") == [
        "Attention weighs each component of a sequence."
    ]


def test_simulated_report_is_fully_grounded():
    report = simulated.synthesize("How does attention work?", [ATTN, LONG])
    g = check_report(report, [ATTN, LONG])
    assert g["sentences_checked"] > 0
    assert g["score"] == 1.0 and g["unsupported"] == []


def test_a_section_must_be_backed_by_its_own_sources():
    # True to the Longformer passage, but the section cites only the Wikipedia page.
    report = {
        "summary": "",
        "key_findings": [],
        "sections": [{
            "title": "Scaling",
            "content": "Self-attention scales quadratically with the sequence length, so long sequences are hard.",
            "sources": [ATTN["url"]],
        }],
    }
    g = check_report(report, [ATTN, LONG])
    assert g["supported"] == 0
    assert g["unsupported"][0]["where"] == "section: Scaling"

    report["sections"][0]["sources"] = [LONG["url"]]
    assert check_report(report, [ATTN, LONG])["supported"] == 1


def test_an_invented_finding_is_reported_with_its_support():
    report = {
        "summary": ATTN["text"].split(". ")[0] + ".",
        "key_findings": ["Attention reduces training cost by 94 percent according to Google's 2031 benchmark study."],
        "sections": [],
    }
    g = check_report(report, [ATTN, LONG])
    assert g["sentences_checked"] == 2 and g["supported"] == 1
    assert g["score"] == 0.5
    [bad] = g["unsupported"]
    assert bad["where"] == "key finding" and bad["support"] < 0.6


def test_only_the_part_of_a_passage_the_writer_saw_counts():
    hidden = {"url": "u", "text": "x " * PASSAGE_CHARS + "Quantum gravity unifies relativity with quantum mechanics."}
    report = {"summary": "Quantum gravity unifies relativity with quantum mechanics.", "key_findings": [], "sections": []}
    assert check_report(report, [hidden])["supported"] == 0


def test_an_empty_report_has_no_score():
    assert check_report({"summary": "", "key_findings": [], "sections": []}, [ATTN])["score"] is None


def test_no_passages_means_no_score_not_zero():
    report = simulated.synthesize("anything", [])
    g = check_report(report, [])
    assert g["sentences_checked"] == 0 and g["score"] is None


# A web result as the live demo got it on 2026-11-18: the search engine's title shares no
# words with the snippet, so the old check reported the title itself as an unsupported sentence.
WEB = {
    "url": "https://arxiv.org/abs/2412.11854",
    "title": "[2412.11854] Towards Understanding Systems Trade-offs in ...",
    "source": "duckduckgo",
    "text": "In this section, we explore the bottlenecks and trade-offs associated with introducing "
            "retrieval augmentation into a traditional language model pipeline.",
}


def test_a_findings_citation_is_not_checked_as_a_sentence():
    report = simulated.synthesize("What are the trade-offs of RAG?", [WEB])
    assert report["key_findings"][0].endswith(f"({WEB['title']})")
    g = check_report(report, [WEB])
    assert all("Towards Understanding" not in u["sentence"] for u in g["unsupported"])
    assert g["score"] == 1.0
    # Only the quoted sentence of the finding is counted, not the citation after it.
    finding_sentences = [s for s in sentences(report["key_findings"][0]) if "(" not in s]
    assert g["sentences_checked"] == len(sentences(report["summary"])) + len(finding_sentences) + 1


def test_strip_citation_only_removes_a_known_title_at_the_end():
    titles = [WEB["title"], "Attention (machine learning)"]
    assert strip_citation("Claim one. (Attention (machine learning))", titles) == "Claim one."
    assert strip_citation("Claim (an aside) here.", titles) == "Claim (an aside) here."
    assert strip_citation("Claim. (Some other title)", titles) == "Claim. (Some other title)"


def test_an_invented_finding_still_fails_after_its_citation_is_removed():
    report = {"summary": "", "sections": [],
              "key_findings": [f"Retrieval cuts hallucinations by 94 percent in every benchmark. ({WEB['title']})"]}
    g = check_report(report, [WEB])
    assert g["sentences_checked"] == 1 and g["supported"] == 0
