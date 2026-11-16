"""
One deadline per research run and per follow-up (SESSION_TIMEOUT_S). Each HTTP call
had its own timeout, but a run made many of them, so a slow run could take minutes
with no single limit. These tests stall the synthesizer and check that the run stops
at the deadline, says why, and (for a follow-up) keeps the earlier report.
"""
import asyncio
import json
import time

import pytest

from app.config import settings
from app.pipeline import orchestrator


def _events(sse_text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in sse_text.splitlines() if line.startswith("data: ")]


@pytest.fixture
def stalled_synthesizer(monkeypatch):
    async def stall(self, *a, **k):
        await asyncio.sleep(30)

    monkeypatch.setattr("app.agents.synthesizer.SynthesizerAgent.run", stall)


def test_default_deadline_is_on_and_zero_turns_it_off(monkeypatch):
    assert settings.session_timeout_s == 180
    assert orchestrator._deadline_s() == 180
    monkeypatch.setattr(settings, "session_timeout_s", 0)
    assert orchestrator._deadline_s() is None


def test_slow_research_stops_at_the_deadline_and_says_why(client, services, monkeypatch, request):
    sid_ok = client.post("/research", json={"query": "transformer attention"}).json()["session_id"]
    assert client.get(f"/research/{sid_ok}").json()["status"] == "completed"

    monkeypatch.setattr(settings, "session_timeout_s", 0.3)
    request.getfixturevalue("stalled_synthesizer")

    started = time.monotonic()
    sid = client.post("/research", json={"query": "transformer attention"}).json()["session_id"]
    assert time.monotonic() - started < 5  # not the 30 s the stalled call would take

    body = client.get(f"/research/{sid}").json()
    assert body["status"] == "failed"
    assert "SESSION_TIMEOUT_S" in body["error_message"]
    assert body["report"] is None
    last = _events(client.get(f"/research/{sid}/stream").text)[-1]
    assert last["type"] == "ERROR" and "0.3 s" in last["message"]


def test_slow_followup_stops_and_keeps_the_earlier_report(client, services, monkeypatch, request):
    sid = client.post("/research", json={"query": "transformer attention"}).json()["session_id"]
    before = client.get(f"/research/{sid}").json()
    assert before["status"] == "completed"

    monkeypatch.setattr(settings, "session_timeout_s", 0.3)
    request.getfixturevalue("stalled_synthesizer")

    started = time.monotonic()
    assert client.post(f"/research/{sid}/followup", json={"question": "multi-head?"}).status_code == 200
    assert time.monotonic() - started < 5

    after = client.get(f"/research/{sid}").json()
    assert after["status"] == "completed"  # back to what it was, so another follow-up can be asked
    assert "SESSION_TIMEOUT_S" in after["error_message"]
    assert after["report"] == before["report"]


def test_a_timeout_inside_an_agent_is_not_reported_as_the_deadline(client, services, monkeypatch):
    async def own_timeout(self, *a, **k):
        raise TimeoutError("upstream read timed out")
    monkeypatch.setattr("app.agents.synthesizer.SynthesizerAgent.run", own_timeout)

    sid = client.post("/research", json={"query": "transformer attention"}).json()["session_id"]
    body = client.get(f"/research/{sid}").json()
    assert body["status"] == "failed"
    assert body["error_message"] == "upstream read timed out"
