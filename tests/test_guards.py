"""
Guards on the full service: optional API key, question length cap and per-IP
research limit. Before these, /sessions listed every caller's questions to anyone,
and /research accepted a question of any length, as often as a client liked.
"""
import pytest

from app.config import settings
from app.limits import RateLimiter

KEY = "s3cret-test-key"


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setattr(settings, "api_key", KEY)


class TestApiKey:
    def test_no_key_configured_keeps_the_api_open(self, client):
        assert client.get("/sessions").status_code == 200

    def test_protected_endpoints_reject_missing_or_wrong_key(self, client, keyed):
        for method, path, body in [
            ("get", "/sessions", None),
            ("post", "/research", {"query": "transformer attention"}),
            ("post", "/research/any-id/followup", {"question": "why?"}),
            ("delete", "/research/any-id", None),
        ]:
            kwargs = {"json": body} if body else {}
            r = getattr(client, method)(path, **kwargs)
            assert r.status_code == 401, (method, path)
            assert r.headers["www-authenticate"] == "Bearer"
            r = getattr(client, method)(path, headers={"Authorization": "Bearer wrong"}, **kwargs)
            assert r.status_code == 401, (method, path)

    def test_rejected_request_starts_no_session(self, client, keyed):
        before = len(client.get("/sessions", headers={"X-API-Key": KEY}).json()["sessions"])
        assert client.post("/research", json={"query": "transformer attention"}).status_code == 401
        after = len(client.get("/sessions", headers={"X-API-Key": KEY}).json()["sessions"])
        assert after == before

    def test_bearer_and_x_api_key_both_accepted(self, client, keyed):
        r = client.post("/research", json={"query": "transformer attention"},
                        headers={"Authorization": f"Bearer {KEY}"})
        assert r.status_code == 200
        sid = r.json()["session_id"]
        listed = client.get("/sessions", headers={"X-API-Key": KEY}).json()["sessions"]
        assert sid in {s["session_id"] for s in listed}

    def test_session_id_is_enough_to_read_a_report_and_its_stream(self, client, keyed):
        # EventSource in a browser cannot send headers, so reading one session and its
        # stream needs only the unguessable session ID returned to the key holder.
        sid = client.post("/research", json={"query": "transformer attention"},
                          headers={"X-API-Key": KEY}).json()["session_id"]
        assert client.get(f"/research/{sid}").status_code == 200
        assert client.get(f"/research/{sid}/stream").status_code == 200

    def test_health_and_root_stay_public(self, client, keyed):
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200


class TestLength:
    def test_query_over_the_limit_is_rejected_before_any_work(self, client, monkeypatch):
        monkeypatch.setattr(settings, "max_query_chars", 50)
        before = len(client.get("/sessions").json()["sessions"])
        r = client.post("/research", json={"query": "x" * 51})
        assert r.status_code == 400
        assert "51 characters; the limit is 50" in r.json()["detail"]
        assert len(client.get("/sessions").json()["sessions"]) == before

    def test_limit_counts_the_stripped_question(self, client, monkeypatch):
        monkeypatch.setattr(settings, "max_query_chars", 20)
        r = client.post("/research", json={"query": "   " + "a" * 20 + "   "})
        assert r.status_code == 200  # 20 characters once stripped

    def test_followup_question_is_capped_too(self, client, monkeypatch):
        sid = client.post("/research", json={"query": "transformer attention"}).json()["session_id"]
        monkeypatch.setattr(settings, "max_query_chars", 10)
        r = client.post(f"/research/{sid}/followup", json={"question": "what about self-attention?"})
        assert r.status_code == 400
        # The rejected follow-up did not claim the session.
        assert client.get(f"/research/{sid}").json()["status"] == "completed"

    def test_empty_query_still_rejected(self, client):
        assert client.post("/research", json={"query": "   "}).json()["detail"] == "Query cannot be empty"


class TestResearchRateLimit:
    def test_third_run_in_an_hour_gets_429_with_retry_after(self, client, monkeypatch):
        monkeypatch.setattr(settings, "research_rate_limit_per_hour", 2)
        sid = client.post("/research", json={"query": "transformer attention"}).json()["session_id"]
        # A follow-up is a research run too.
        assert client.post(f"/research/{sid}/followup", json={"question": "and self-attention?"}).status_code == 200
        r = client.post("/research", json={"query": "transformer attention"})
        assert r.status_code == 429
        assert 0 < int(r.headers["retry-after"]) <= 3600

    def test_invalid_requests_do_not_use_up_the_limit(self, client, monkeypatch):
        monkeypatch.setattr(settings, "research_rate_limit_per_hour", 1)
        monkeypatch.setattr(settings, "max_query_chars", 30)
        for _ in range(3):
            assert client.post("/research", json={"query": "y" * 31}).status_code == 400
        assert client.post("/research", json={"query": "transformer attention"}).status_code == 200

    def test_zero_turns_the_limit_off(self, client, monkeypatch):
        monkeypatch.setattr(settings, "research_rate_limit_per_hour", 0)
        for _ in range(3):
            assert client.post("/research", json={"query": "transformer attention"}).status_code == 200

    def test_limits_are_per_client(self):
        rl = RateLimiter(limit=1)
        assert rl.allow("1.1.1.1")[0]
        assert not rl.allow("1.1.1.1")[0]
        assert rl.allow("2.2.2.2")[0]
