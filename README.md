# Meridian — Multi-Agent Research Intelligence API

> A five-stage research pipeline: an LLM splits your question into sub-questions, the sub-questions are searched concurrently, local embeddings rank the passages, an LLM reviews them for gaps and contradictions, and an LLM writes a structured report — with every stage streamed live over Server-Sent Events.

---

## The Problem

A basic retrieval system fires one query and returns one chunk. Meridian tries a broader approach: it splits the question, searches each part, ranks the evidence, asks an LLM to note gaps and contradictions, and shows each step as it happens. It has not been evaluated for answer quality yet.

---

## How It Works

```
POST /research  →  query arrives
                        │
              ┌─────────▼──────────┐
              │   PLANNER AGENT    │  LLM decomposes → 3-5 sub-questions
              └─────────┬──────────┘
                        │ sub-questions (parallel)
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
  ┌────────────┐ ┌────────────┐ ┌────────────┐
  │ RESEARCHER │ │ RESEARCHER │ │ RESEARCHER │  Wikipedia + arXiv + DuckDuckGo
  └──────┬─────┘ └──────┬─────┘ └──────┬─────┘
         └──────────────┼──────────────┘
                        │ all raw passages
              ┌─────────▼──────────┐
              │  EXTRACTOR AGENT   │  sentence-transformers cosine ranking
              └─────────┬──────────┘
                        │ top-k passages
              ┌─────────▼──────────┐
              │   CRITIC AGENT     │  LLM finds contradictions + gaps
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │ SYNTHESIZER AGENT  │  LLM builds final structured report
              └─────────┬──────────┘
                        │
              GET /research/{id}  →  full JSON report
```

Each step streams live to `GET /research/{id}/stream` as Server-Sent Events.

---

## Quick Start

```bash
git clone https://github.com/Nityaa08/meridian.git
cd meridian
pip install -r requirements.txt

# Get a free Groq API key at https://console.groq.com
cp .env.example .env
# edit .env and set GROQ_API_KEY=your_key_here
# and set GROQ_MODEL to a current Groq model (the default was retired; see Known Issues)

uvicorn main:app --reload
```

Before you run it, read [Known Issues](#known-issues-checked-2026-10-28): on macOS the server can crash on the first request.

---

## Example Usage

```bash
# 1. Start a research session
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "How does transformer attention work?"}'
# → {"session_id": "abc-123", "status": "running", "stream_url": "/research/abc-123/stream"}

# 2. Stream live agent events
curl -N http://localhost:8000/research/abc-123/stream
# → data: {"agent":"PlannerAgent","type":"PLANNING","message":"Identified 4 sub-questions",...}
# → data: {"agent":"ResearcherAgent","type":"SEARCHING","message":"Found 8 passages for: ...",...}
# → data: {"agent":"ExtractorAgent","type":"EXTRACTING","message":"Selected top 10 passages",...}
# → data: {"agent":"CriticAgent","type":"CRITIQUING","message":"Found 1 contradictions, 2 gaps",...}
# → data: {"agent":"SynthesizerAgent","type":"SYNTHESIZING","message":"Report complete. Confidence: 0.84",...}
#   ("Confidence" in this message is the retrieval similarity described below)
# → data: {"agent":"Orchestrator","type":"DONE",...}

# 3. Get the full report
curl http://localhost:8000/research/abc-123

# 4. Ask a follow-up (reuses already-fetched sources)
curl -X POST http://localhost:8000/research/abc-123/followup \
  -H "Content-Type: application/json" \
  -d '{"question": "How does multi-head attention differ from single-head?"}'

# 5. Health check
curl http://localhost:8000/health
# → {"status":"ok","embedder":"loaded","db":"ok","model":"llama-3.1-8b-instant"}
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/research` | Start a new research session |
| `GET` | `/research/{id}/stream` | SSE stream of live agent events |
| `GET` | `/research/{id}` | Get status + final report |
| `POST` | `/research/{id}/followup` | Ask follow-up (reuses fetched sources) |
| `GET` | `/sessions` | List all sessions |
| `DELETE` | `/research/{id}` | Delete a session |
| `GET` | `/health` | System health + embedder status |
| `GET` | `/docs` | Interactive API docs (Swagger) |

---

## Final Report Structure

`confidence_score` is the retrieval similarity: the mean cosine similarity of the passages used. It is not a measure of correctness.

```json
{
  "session_id": "uuid",
  "query": "original question",
  "status": "completed",
  "report": {
    "summary": "2-3 sentence overview",
    "key_findings": ["finding 1", "finding 2", "finding 3"],
    "sections": [
      {"title": "...", "content": "...", "sources": ["url1", "url2"]}
    ],
    "sources": [
      {"title": "...", "url": "...", "relevance_score": 0.91}
    ],
    "confidence_score": 0.87,
    "gaps_identified": ["aspect not covered by available sources"],
    "contradictions": []
  },
  "metadata": {
    "agents_used": 5,
    "sources_searched": 14,
    "sources_used": 8,
    "total_time_ms": 8234
  }
}
```

---

## Agent Descriptions

| Agent | Role |
|-------|------|
| **PlannerAgent** | Uses Groq LLM to decompose the query into 3-5 focused sub-questions, ensuring full coverage of the topic. |
| **ResearcherAgent** | Runs once per sub-question, all sub-questions concurrently. Each run searches Wikipedia, then arXiv, then DuckDuckGo only if fewer than 3 passages were found. Plain code, no LLM. |
| **ExtractorAgent** | Uses `sentence-transformers` to embed query + passages and rank them by cosine similarity; keeps passages with similarity at least 0.15, up to 10. Plain code, no LLM. |
| **CriticAgent** | Asks the Groq LLM to list contradictions, knowledge gaps and potentially unreliable sources in up to 8 passages. Its accuracy has not been evaluated. |
| **SynthesizerAgent** | Uses Groq LLM to build a structured report with summary, key findings, and cited sections. Retrieval similarity (the `confidence_score` field) = mean cosine similarity of the passages used. It measures how close the passages are to the question, not whether the report is correct. |

---

## Tech Stack

| Component | Tool | Why Free |
|-----------|------|----------|
| API framework | FastAPI + asyncio | Open source |
| LLM | Groq, OpenAI-compatible API (default `llama-3.1-8b-instant`, retired 16 Aug 2026 — set `GROQ_MODEL`) | Free tier, no credit card |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Local, open source, 90MB |
| Search | Wikipedia REST API + arXiv API | Official, no key needed |
| Search fallback | duckduckgo-search | Free scraper library |
| Database | SQLite + aiosqlite | No external service |
| Streaming | sse-starlette | Open source |

---

## Architecture Notes

**Why SSE instead of WebSockets?** SSE is unidirectional (server → client), which matches this pattern exactly. Simpler to implement, works with `curl -N`, no handshake overhead.

**Why sentence-transformers locally?** Zero latency, no API quota, no cost. The `all-MiniLM-L6-v2` model (90MB) provides strong semantic search performance. Loaded once at startup via lifespan.

**Why multi-source search?** Wikipedia for factual overviews, arXiv for cutting-edge research/technical topics, DuckDuckGo as a fallback for recent or niche topics. No single source covers everything.

**Why asyncio.gather with return_exceptions=True?** One failing API call (DuckDuckGo rate limit, arXiv timeout) must not kill the entire pipeline. Today the fallbacks are silent: the final status does not show which stages failed (see Known Issues).

**Why sentence-transformers in asyncio.to_thread?** `encode()` is synchronous and CPU-bound. Running it directly in an async function blocks the entire event loop. `to_thread()` offloads it to a thread pool.

---

## Known Issues (checked 2026-10-28)

These were found by re-running the project with a fresh install. They are not fixed yet.

- **macOS crash on the first research request.** The Extractor runs two `SentenceTransformer.encode` calls at the same time in two threads. With the library versions installed on 2026-10-28 (torch 2.14.1, sentence-transformers 6.2.0), this segfaults the server process (exit code 139). Embedding the query and passages in one call avoids it. `requirements.txt` is unpinned, so installs differ over time.
- **Wikipedia returns no results.** The code calls `https://en.wikipedia.org/api/rest_v1/page/search/title`, which returns HTTP 404. Wikipedia's title search is at `/w/rest.php/v1/search/title`.
- **DuckDuckGo search errors.** `duckduckgo-search==6.2.6` fails with `BuilderError Invalid impersonate: "chrome_119"` against its current dependencies.
- **Only arXiv returns passages today**, so non-technical questions may get no sources.
- **Default model retired.** Groq shut down `llama-3.1-8b-instant` for free and developer tiers on 16 August 2026. Set `GROQ_MODEL` in `.env` to a current model, for example `openai/gpt-oss-20b` (Groq's suggested replacement).
- **Fallbacks are silent.** If every LLM call fails, the session still ends with `status: completed` and a `confidence_score`. Only the stream shows the fallback messages.
- **Citations are not checked.** URLs the Synthesizer puts in `sections[].sources` are not validated against the retrieved sources.
- **Event log and queues.** Only the final `DONE` event is saved in `agent_events`. Event queues are not removed after a session ends, so `active_sessions` in `/health` keeps growing and re-opening a finished stream waits with no data.
- **Not deployable on Vercel serverless.** `sentence-transformers` pulls in PyTorch, so the install bundle is about 5.6 GB, above Vercel's 500 MB function limit. Run it locally or in a container on a VM instead.
- **No authentication or rate limit.** `/sessions` lists every user's queries.

## Tests

There are no automated tests yet.
