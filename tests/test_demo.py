"""
Tests for the public demo (app/demo_app.py), the hashing embedder and the
simulated LLM stages. Same fakes as the rest of the suite: no network, no key,
no PyTorch.
"""
import json
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.agents import simulated
from app.config import settings
from app.demo_app import RateLimiter, create_demo_app
from app.embedders import HashingEmbedder, load_embedder


def events(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


@pytest.fixture
def demo(services, monkeypatch):
    """Demo mode with no Groq key: LLM stages simulated."""
    monkeypatch.setattr(settings, "demo_mode", True)
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "embedder_backend", "hashing")
    return TestClient(create_demo_app())


class TestHashingEmbedder:
    def test_vectors_are_unit_length(self):
        v = HashingEmbedder().encode(["attention weights tokens", "graph neural networks"])
        assert v.shape == (2, 2048)
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)

    def test_related_text_ranks_above_unrelated(self):
        e = HashingEmbedder()
        q, rel, unrel = e.encode([
            "How does transformer attention work?",
            "The transformer architecture relies on self-attention to weigh tokens.",
            "Photosynthesis converts light energy into chemical energy in plants.",
        ])
        assert float(q @ rel) > float(q @ unrel)

    def test_falls_back_when_sentence_transformers_is_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import raises ImportError
        assert isinstance(load_embedder("sentence-transformers", "all-MiniLM-L6-v2", "cpu"), HashingEmbedder)


class TestSimulatedStages:
    def test_plan_uses_the_topic(self):
        subs = simulated.plan("How does transformer attention work?")
        assert subs[0] == "How does transformer attention work?"
        assert "transformer attention overview" in subs
        assert simulated.topic_of("How does attention work?") == "attention"
        assert simulated.topic_of("What is a large language model") == "large language model"
        assert len(subs) == len({s.lower() for s in subs})

    def test_critique_only_reports_what_it_can_count(self):
        c = simulated.critique([{"source": "wikipedia", "similarity": 0.5, "url": "u"}])
        assert c["contradictions"] == [] and c["simulated"] is True
        assert any("arXiv" in g for g in c["gaps"])

    def test_synthesis_quotes_passages_and_cites_their_urls(self):
        passages = [
            {"source": "wikipedia", "title": "A", "url": "https://w/a", "text": "First fact. Second fact. Third.", "similarity": .6},
            {"source": "arxiv", "title": "B", "url": "https://x/b", "text": "Paper claim. More.", "similarity": .4},
        ]
        s = simulated.synthesize("q", passages)
        assert s["summary"] == "First fact. Second fact."
        assert {u for sec in s["sections"] for u in sec["sources"]} == {"https://w/a", "https://x/b"}


class TestDemoApp:
    def test_home_is_the_demo_page(self, demo):
        r = demo.get("/")
        assert r.status_code == 200 and "Meridian" in r.text and "/api/research" in r.text

    def test_health_says_llm_is_simulated(self, demo):
        h = demo.get("/health").json()
        assert h["mode"] == "demo" and h["llm"].startswith("simulated")

    def test_stream_runs_every_stage_and_ends_with_a_report(self, demo, services):
        r = demo.get("/api/research", params={"q": "How does transformer attention work?"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        evs = events(r.text)
        assert evs[0]["type"] == "START" and evs[0]["data"]["simulated_llm"] is True
        agents = {e["agent"] for e in evs}
        assert {"PlannerAgent", "ResearcherAgent", "ExtractorAgent", "CriticAgent", "SynthesizerAgent"} <= agents
        final = evs[-1]
        assert final["type"] == "REPORT"
        result = final["data"]
        assert result["status"] == "completed"
        report = result["report"]
        assert report["simulated_llm"] is True and report["summary"]
        retrieved = {s["url"] for s in report["sources"]}
        assert all(u in retrieved for sec in report["sections"] for u in sec["sources"])
        # No LLM call was made: only Wikipedia and arXiv were contacted.
        assert {req.url.host for req in services.requests} <= {"en.wikipedia.org", "export.arxiv.org"}
        assert len(result["metadata"]["sub_questions"]) == 4

    def test_with_a_key_the_llm_stages_are_real(self, services, monkeypatch):
        monkeypatch.setattr(settings, "demo_mode", True)
        monkeypatch.setattr(settings, "groq_api_key", "test-key")
        monkeypatch.setattr(settings, "embedder_backend", "hashing")
        client = TestClient(create_demo_app())
        evs = events(client.get("/api/research", params={"q": "attention"}).text)
        assert evs[0]["data"]["simulated_llm"] is False
        assert evs[-1]["data"]["report"]["summary"] == "Attention weights inputs by relevance."
        assert any(req.url.host == "api.groq.com" for req in services.requests)

    def test_empty_and_too_long_questions_are_rejected(self, demo):
        assert demo.get("/api/research").status_code == 400
        assert demo.get("/api/research", params={"q": "x" * 201}).status_code == 413

    def test_rate_limited_per_visitor(self, services, monkeypatch):
        monkeypatch.setattr(settings, "demo_mode", True)
        monkeypatch.setattr(settings, "groq_api_key", "")
        monkeypatch.setattr(settings, "embedder_backend", "hashing")
        monkeypatch.setattr(settings, "demo_rate_limit_per_hour", 1)
        client = TestClient(create_demo_app())
        assert client.get("/api/research", params={"q": "attention"}).status_code == 200
        blocked = client.get("/api/research", params={"q": "attention"})
        assert blocked.status_code == 429 and "Retry-After" in blocked.headers


def test_rate_limiter_window_resets():
    t = [0.0]
    rl = RateLimiter(limit=2, window_s=10, clock=lambda: t[0])
    assert rl.allow("a")[0] and rl.allow("a")[0]
    assert not rl.allow("a")[0]
    assert rl.allow("b")[0]
    t[0] = 11
    assert rl.allow("a")[0]
