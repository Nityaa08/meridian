import asyncio
import json
import sys
import types

import pytest

from app.agents.base import AgentEvent
from app.agents.extractor import ExtractorAgent
from app.agents.researcher import ResearcherAgent, parse_wikipedia_response
from app.agents.synthesizer import filter_citations
from app.config import Settings
from app.pipeline.events import EventQueue, active_session_count, release_queue

from conftest import WIKI_RESPONSE


def run(coro):
    return asyncio.run(coro)


# ─── Config ───────────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_model_is_not_retired(self):
        # llama-3.1-8b-instant was shut down for free/dev tiers on 2026-08-16
        assert Settings.model_fields["groq_model"].default == "openai/gpt-oss-20b"

    def test_embedder_defaults_to_cpu(self):
        assert Settings.model_fields["embedder_device"].default == "cpu"

    def test_load_embedder_passes_device(self, monkeypatch):
        seen = {}

        class FakeST:
            def __init__(self, name, device=None):
                seen["name"], seen["device"] = name, device

        monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
        import main
        main.load_embedder()
        assert seen == {"name": "all-MiniLM-L6-v2", "device": "cpu"}


# ─── Researcher ───────────────────────────────────────────────────────────────

class TestResearcher:
    def test_parse_wikipedia_keeps_search_rank_order(self):
        passages = parse_wikipedia_response(WIKI_RESPONSE)
        assert [p["title"] for p in passages] == [
            "Attention (machine learning)",
            "Transformer (deep learning architecture)",
        ]
        assert passages[0]["url"] == "https://en.wikipedia.org/wiki/Attention_(machine_learning)"
        assert all(p["source"] == "wikipedia" for p in passages)

    def test_parse_wikipedia_skips_empty_extracts_and_handles_no_results(self):
        assert parse_wikipedia_response({"batchcomplete": True}) == []
        data = {"query": {"pages": [{"title": "X", "index": 1, "extract": ""}]}}
        assert parse_wikipedia_response(data) == []

    def test_wikipedia_uses_action_api(self, services):
        passages = run(ResearcherAgent()._search_wikipedia("transformer attention"))
        assert len(passages) == 2
        wiki = [r for r in services.requests if r.url.host == "en.wikipedia.org"]
        assert wiki[0].url.path == "/w/api.php"
        assert wiki[0].url.params["gsrsearch"] == "transformer attention"

    def test_all_three_sources_return_passages(self, services):
        r = ResearcherAgent()
        assert run(r._search_wikipedia("attention"))
        assert run(r._search_arxiv("attention"))
        ddg = run(r._search_duckduckgo("attention"))
        assert ddg == [{
            "title": "DDG result", "url": "https://example.org/ddg",
            "text": "Web result about attention", "source": "duckduckgo",
        }]

    def test_duckduckgo_failure_returns_empty(self, services, monkeypatch):
        import ddgs

        class Broken:
            def text(self, *a, **k):
                raise RuntimeError("blocked")

        monkeypatch.setattr(ddgs, "DDGS", Broken)
        assert run(ResearcherAgent()._search_duckduckgo("x")) == []


# ─── Extractor ────────────────────────────────────────────────────────────────

PASSAGES = [
    {"title": "a", "url": "u1", "text": "attention weights tokens by relevance", "source": "wikipedia"},
    {"title": "b", "url": "u2", "text": "transformers use self attention layers", "source": "arxiv"},
    {"title": "c", "url": "u3", "text": "bananas are yellow fruit", "source": "duckduckgo"},
]


class TestExtractor:
    def test_single_encode_call_for_query_and_passages(self, embedder):
        top = run(ExtractorAgent(embedder).run("how does attention work", PASSAGES, asyncio.Queue()))
        assert embedder.calls == [["how does attention work"] + [p["text"] for p in PASSAGES]]
        assert [p["url"] for p in top][:1] == ["u1"]
        assert "u3" not in [p["url"] for p in top]  # below similarity threshold

    def test_concurrent_sessions_never_encode_in_parallel(self, embedder):
        async def many():
            agents = [ExtractorAgent(embedder) for _ in range(4)]
            await asyncio.gather(*(a.run("attention", PASSAGES, asyncio.Queue()) for a in agents))

        run(many())
        assert len(embedder.calls) == 4
        assert embedder.max_active == 1


# ─── Synthesizer citations ────────────────────────────────────────────────────

class TestCitations:
    def test_unknown_urls_removed(self):
        sections = [
            {"title": "A", "content": "x", "sources": ["u1", "https://invented.example"]},
            {"title": "B", "content": "y", "sources": "u2"},
            "not a section",
        ]
        cleaned, removed = filter_citations(sections, {"u1", "u2"})
        assert cleaned == [
            {"title": "A", "content": "x", "sources": ["u1"]},
            {"title": "B", "content": "y", "sources": ["u2"]},
        ]
        assert removed == 1

    def test_non_list_sections(self):
        assert filter_citations(None, {"u1"}) == ([], 0)


# ─── Event queues ─────────────────────────────────────────────────────────────

class TestEventQueue:
    def test_progress_is_saved_while_running(self):
        from app.pipeline import events as ev
        from app.agents.base import AgentEvent

        saved: list[list[dict]] = []

        async def on_change(history):
            saved.append(history)

        async def run():
            ev.PROGRESS_SAVE_INTERVAL_S = 0.01
            q = ev.EventQueue(on_change=on_change)
            for i in range(3):  # a burst becomes one write
                await q.put(AgentEvent(agent="researcher", type="SEARCHING", message=f"m{i}"))
            await asyncio.sleep(0.05)
            await q.put(AgentEvent(agent="orchestrator", type="DONE", message="done"))
            await asyncio.sleep(0.05)

        asyncio.run(run())
        assert len(saved) == 2
        assert [e["message"] for e in saved[-1]] == ["m0", "m1", "m2", "done"]

    def test_records_history_and_release(self):
        async def go():
            queues = {}
            q = EventQueue()
            queues["s"] = q
            await q.put(AgentEvent(agent="A", type="PLANNING", message="m"))
            assert q.history[0]["type"] == "PLANNING"
            assert active_session_count(queues) == 1
            release_queue(queues, "s", q, ttl_s=0)
            assert queues == {}
            assert active_session_count({"t": q}) == 0

        run(go())


# ─── API end to end (fake network) ────────────────────────────────────────────

def _events(sse_text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in sse_text.splitlines() if line.startswith("data: ")]


class TestApi:
    def test_research_happy_path(self, client, services):
        r = client.post("/research", json={"query": "How does transformer attention work?"})
        assert r.status_code == 200
        sid = r.json()["session_id"]

        report = client.get(f"/research/{sid}").json()
        assert report["status"] == "completed", report["report"]["warnings"]
        rep = report["report"]
        assert rep["warnings"] == []
        urls = {s["url"] for s in rep["sources"]}
        assert "http://arxiv.org/abs/1706.03762v7" in urls
        assert any(u.startswith("https://en.wikipedia.org/wiki/") for u in urls)
        allowed = {s["url"] for s in rep["sources"]}
        for section in rep["sections"]:
            assert set(section["sources"]) <= allowed
        assert report["metadata"]["sources_searched"] >= 3

    def test_llm_failure_marks_session_degraded(self, client, services):
        services.llm_down = True
        sid = client.post("/research", json={"query": "transformer attention"}).json()["session_id"]
        report = client.get(f"/research/{sid}").json()
        assert report["status"] == "degraded"
        warnings = " ".join(report["report"]["warnings"])
        assert "Planner" in warnings and "Critic" in warnings and "Synthesizer" in warnings

    def test_invented_citation_removed_and_reported(self, client, services):
        services.bogus_citation = True
        sid = client.post("/research", json={"query": "transformer attention"}).json()["session_id"]
        report = client.get(f"/research/{sid}").json()
        rep = report["report"]
        assert report["status"] == "degraded"
        assert all("made-up.example.com" not in u for s in rep["sections"] for u in s["sources"])
        assert any("Removed 1 section citation" in w for w in rep["warnings"])

    def test_stream_live_then_replay_and_queue_cleanup(self, client):
        sid = client.post("/research", json={"query": "transformer attention"}).json()["session_id"]

        live = _events(client.get(f"/research/{sid}/stream").text)
        assert live[-1]["type"] == "DONE"
        types_seen = {e["type"] for e in live}
        assert {"PLANNING", "SEARCHING", "EXTRACTING", "CRITIQUING", "SYNTHESIZING", "DONE"} <= types_seen

        # Live queue is gone; a second request replays the saved log instead of hanging.
        assert sid not in client.app.state.event_queues
        replay = _events(client.get(f"/research/{sid}/stream").text)
        assert replay == live

        assert client.get("/health").json()["active_sessions"] == 0

    def test_full_event_log_saved(self, client):
        import main  # noqa: F401
        from sqlalchemy import select
        from app.database import AsyncSessionLocal
        from app.models.session import ResearchSession

        sid = client.post("/research", json={"query": "transformer attention"}).json()["session_id"]

        async def load():
            async with AsyncSessionLocal() as db:
                res = await db.execute(select(ResearchSession).where(ResearchSession.id == sid))
                return res.scalar_one().agent_events

        events = client.portal.call(load) if hasattr(client, "portal") and client.portal else None
        if events is None:
            pytest.skip("TestClient portal not available")
        assert len(events) > 5
        assert events[0]["type"] == "PLANNING" and events[-1]["type"] == "DONE"

    def test_followup_allowed_after_degraded(self, client, services):
        services.llm_down = True
        sid = client.post("/research", json={"query": "transformer attention"}).json()["session_id"]
        services.llm_down = False
        r = client.post(f"/research/{sid}/followup", json={"question": "What is multi-head attention?"})
        assert r.status_code == 200
        report = client.get(f"/research/{sid}").json()
        assert report["report"]["followup_question"] == "What is multi-head attention?"

    def test_unknown_stream_404(self, client):
        assert client.get("/research/does-not-exist/stream").status_code == 404

    def test_stream_follows_a_session_running_on_another_worker(self, client):
        """No local queue, session still running elsewhere: stream it from the DB log."""
        import uuid
        from sqlalchemy import update
        from app.database import AsyncSessionLocal
        from app.models.session import ResearchSession
        from app.api import stream as stream_mod

        if not (hasattr(client, "portal") and client.portal):
            pytest.skip("TestClient portal not available")

        sid = str(uuid.uuid4())
        first = [{"type": "PLANNING", "message": "planning"}, {"type": "SEARCHING", "message": "searching"}]

        class FakeRequest:
            async def is_disconnected(self):
                return False

        async def scenario():
            stream_mod.FOLLOW_POLL_S = 0.01
            async with AsyncSessionLocal() as db:
                db.add(ResearchSession(id=sid, query="q", status="running", agent_events=first))
                await db.commit()
            gen = stream_mod._follow_from_db(sid, FakeRequest())
            got = [json.loads((await gen.__anext__())["data"]) for _ in range(2)]
            # The owning worker finishes the session.
            async with AsyncSessionLocal() as db:
                await db.execute(update(ResearchSession).where(ResearchSession.id == sid).values(
                    status="completed", agent_events=first + [{"type": "DONE", "message": "done"}]))
                await db.commit()
            got += [json.loads(e["data"]) async for e in gen]
            return got

        events = client.portal.call(scenario)
        assert [e["type"] for e in events] == ["PLANNING", "SEARCHING", "DONE"]
        assert client.get(f"/research/{sid}/stream").status_code == 200

    def test_running_session_without_local_queue_is_not_404(self, client):
        import uuid
        from app.database import AsyncSessionLocal
        from app.models.session import ResearchSession

        if not (hasattr(client, "portal") and client.portal):
            pytest.skip("TestClient portal not available")
        sid = str(uuid.uuid4())

        async def add():
            async with AsyncSessionLocal() as db:
                db.add(ResearchSession(id=sid, query="q", status="failed",
                                       agent_events=[{"type": "ERROR", "message": "boom"}]))
                await db.commit()
        client.portal.call(add)
        assert _events(client.get(f"/research/{sid}/stream").text)[-1]["type"] == "ERROR"

    def test_health(self, client):
        body = client.get("/health").json()
        assert body["model"] == "openai/gpt-oss-20b"
        assert body["embedder"] == "loaded"
