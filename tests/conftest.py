"""
Test fixtures. No network, no Groq key and no PyTorch needed:
- every httpx call made by the agents goes to an in-process fake (Groq, Wikipedia, arXiv),
- DuckDuckGo is replaced by a fake ddgs.DDGS,
- the sentence-transformers model is replaced by a small bag-of-words embedder.
"""
import json
import os
import re
import sys
import tempfile
import threading
import time
import zlib
from pathlib import Path

import httpx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Must be set before app.config is imported.
_db_dir = tempfile.mkdtemp(prefix="meridian-test-")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_db_dir}/test.db"
os.environ["GROQ_API_KEY"] = "test-key"


# ─── Fake embedder ────────────────────────────────────────────────────────────

class FakeEmbedder:
    """Hashes words into a fixed-size vector. Tracks calls and concurrency."""

    DIM = 128

    def __init__(self):
        self.calls: list[list[str]] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def encode(self, texts):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.01)  # widen the window for overlapping calls
            self.calls.append(list(texts))
            out = np.zeros((len(texts), self.DIM), dtype=np.float32)
            for i, t in enumerate(texts):
                for w in re.findall(r"[a-z]+", t.lower()):
                    out[i, zlib.crc32(w.encode()) % self.DIM] += 1.0
            return out
        finally:
            with self._lock:
                self.active -= 1


# ─── Fake HTTP services ───────────────────────────────────────────────────────

WIKI_RESPONSE = {
    "batchcomplete": True,
    "query": {
        "pages": [
            {
                "pageid": 2, "title": "Transformer (deep learning architecture)", "index": 2,
                "fullurl": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
                "extract": "A transformer is a deep learning architecture based on attention.",
            },
            {
                "pageid": 1, "title": "Attention (machine learning)", "index": 1,
                "fullurl": "https://en.wikipedia.org/wiki/Attention_(machine_learning)",
                "extract": "Attention in machine learning weights parts of the input by relevance.",
            },
        ]
    },
}

ARXIV_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <title>Attention Is All You Need</title>
    <summary>We propose the Transformer, based solely on attention mechanisms.</summary>
  </entry>
</feed>"""


class FakeServices:
    def __init__(self):
        self.llm_down = False
        self.bogus_citation = False
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.url.host
        if host == "api.groq.com":
            return self._groq(request)
        if host == "en.wikipedia.org":
            if request.url.path == "/w/api.php":
                return httpx.Response(200, json=WIKI_RESPONSE)
            return httpx.Response(404)
        if host == "export.arxiv.org":
            return httpx.Response(200, text=ARXIV_RESPONSE)
        return httpx.Response(404)

    def _groq(self, request: httpx.Request) -> httpx.Response:
        if self.llm_down:
            return httpx.Response(404, json={"error": {"message": "model not found"}})
        body = json.loads(request.content)
        system = body["messages"][0]["content"]
        user = body["messages"][1]["content"]
        if "planning" in system:
            content = {"sub_questions": ["What is attention in transformers?", "How does self-attention work?"]}
        elif "critical" in system:
            content = {"contradictions": [], "gaps": [], "unreliable_sources": [], "overall_quality": "high"}
        else:
            urls = re.findall(r"Source: (\S+)", user)
            sources = urls[:2] + (["https://made-up.example.com/paper"] if self.bogus_citation else [])
            content = {
                "summary": "Attention weights inputs by relevance.",
                "key_findings": ["Transformers use attention."],
                "sections": [{"title": "Overview", "content": "...", "sources": sources}],
            }
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}]})


class FakeDDGS:
    calls: list[str] = []

    def text(self, query, max_results=3, **kwargs):
        FakeDDGS.calls.append(query)
        return [{"title": "DDG result", "href": "https://example.org/ddg", "body": f"Web result about {query}"}]


@pytest.fixture
def services(monkeypatch):
    svc = FakeServices()
    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(svc.handler)
        return real_client(*args, **kwargs)

    for mod in ("planner", "critic", "synthesizer", "researcher"):
        monkeypatch.setattr(f"app.agents.{mod}.httpx.AsyncClient", client_factory)

    import ddgs
    FakeDDGS.calls = []
    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    return svc


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def client(services, embedder, monkeypatch):
    import main
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, "load_embedder", lambda: embedder)
    with TestClient(main.app) as c:
        yield c
