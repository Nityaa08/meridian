"""
The public demo (DEMO_MODE=true), built to run on a serverless function.

Differences from the full app in main.py:
- One stateless endpoint, GET /api/research?q=..., runs the whole pipeline
  inside the request and streams each stage as Server-Sent Events, ending with
  a REPORT event. No database, no background tasks, no session list (the full
  app's /sessions would show every visitor's queries).
- The embedder is chosen by EMBEDDER_BACKEND; the demo uses "hashing" because
  PyTorch does not fit in a 500 MB function.
- With no GROQ_API_KEY the planner, critic and writer are simulated
  (app/agents/simulated.py) and every event says so. With a key they are real.
- Queries are length-capped and rate limited per client IP (per instance).
"""
import asyncio
import json
import os
import time
from dataclasses import asdict

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.agents.base import AgentEvent
from app.agents.critic import CriticAgent
from app.agents.extractor import ExtractorAgent
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearcherAgent
from app.agents.simulated import SIMULATED_NOTE
from app.agents.synthesizer import SynthesizerAgent
from app.config import settings
from app.embedders import load_embedder
from app.pipeline.orchestrator import STATUS_COMPLETED, STATUS_DEGRADED, collect_warnings

_STATIC = os.path.join(os.path.dirname(__file__), "static")


class RateLimiter:
    """Fixed one-hour window per key, in memory (so per serverless instance)."""

    def __init__(self, limit: int, window_s: float = 3600.0, clock=time.monotonic):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self.hits: dict[str, tuple[int, float]] = {}

    def allow(self, key: str) -> tuple[bool, int]:
        now = self.clock()
        count, reset_at = self.hits.get(key, (0, now + self.window_s))
        if reset_at <= now:
            count, reset_at = 0, now + self.window_s
        if len(self.hits) > 10_000:
            self.hits = {k: v for k, v in self.hits.items() if v[1] > now}
        if count >= self.limit:
            return False, int(reset_at - now)
        self.hits[key] = (count + 1, reset_at)
        return True, 0


def client_ip(request: Request) -> str:
    # Vercel sets these itself; elsewhere fall back to the socket address.
    real = request.headers.get("x-real-ip")
    if real and os.environ.get("VERCEL"):
        return real
    return request.client.host if request.client else "unknown"


def dedupe(passages: list[dict]) -> list[dict]:
    seen, out = set(), []
    for p in passages:
        url = p.get("url", "")
        if url and url in seen:
            continue
        if url:
            seen.add(url)
        out.append(p)
    return out


async def research(query: str, embedder, queue: asyncio.Queue) -> dict:
    """The five stages, in memory. Returns {status, report, metadata}."""
    start = time.time()
    planner = PlannerAgent()
    sub_queries = await planner.run(query, queue)

    researcher = ResearcherAgent()
    results = await asyncio.gather(*(researcher.run(sq, queue) for sq in sub_queries), return_exceptions=True)
    passages = dedupe([p for r in results if not isinstance(r, Exception) for p in r])

    top = await ExtractorAgent(embedder).run(query, passages, queue)
    critic = CriticAgent()
    critique = await critic.run(query, top, queue)
    synthesizer = SynthesizerAgent()
    report = await synthesizer.run(query, top, critique, queue)

    warnings = collect_warnings(planner=planner, sources_searched=len(passages), top_passages=top,
                                critic=critic, synthesizer=synthesizer)
    report.update(query=query, warnings=warnings, simulated_llm=settings.simulate_llm,
                  embedder=getattr(embedder, "name", settings.embedder_model))
    return {
        "status": STATUS_DEGRADED if warnings else STATUS_COMPLETED,
        "report": report,
        "metadata": {
            "sub_questions": sub_queries,
            "sources_searched": len(passages),
            "sources_used": len(top),
            "total_time_ms": int((time.time() - start) * 1000),
        },
    }


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def create_demo_app() -> FastAPI:
    app = FastAPI(
        title="Meridian demo",
        description="Stateless demo of the Meridian research pipeline. See /api/research.",
        version="1.0.0",
    )
    register_demo_routes(app)
    return app


def register_demo_routes(app: FastAPI) -> None:
    """Add the demo page, /health and /api/research to an app (used by main.py in DEMO_MODE)."""
    app.state.embedder = None
    app.state.limiter = RateLimiter(settings.demo_rate_limit_per_hour)

    def embedder():
        if app.state.embedder is None:
            app.state.embedder = load_embedder(settings.embedder_backend, settings.embedder_model,
                                               settings.embedder_device or None)
        return app.state.embedder

    @app.get("/")
    async def home():
        return FileResponse(os.path.join(_STATIC, "demo.html"))

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "mode": "demo",
            "llm": "simulated (no API key)" if settings.simulate_llm else f"groq:{settings.groq_model}",
            "embedder": settings.embedder_backend,
            "rate_limit_per_hour": settings.demo_rate_limit_per_hour,
        }

    @app.get("/api/research")
    async def research_stream(request: Request, q: str = Query("", description="The research question")):
        query = " ".join(q.split())
        if not query:
            return JSONResponse({"error": "Add a question with ?q=..."}, status_code=400)
        if len(query) > settings.demo_max_query_chars:
            return JSONResponse(
                {"error": f"Questions are limited to {settings.demo_max_query_chars} characters in the demo."},
                status_code=413)
        ok, retry = app.state.limiter.allow(client_ip(request))
        if not ok:
            return JSONResponse(
                {"error": f"The demo allows {settings.demo_rate_limit_per_hour} questions an hour per visitor. "
                          f"Try again in {max(1, retry // 60)} minutes."},
                status_code=429, headers={"Retry-After": str(retry)})

        queue: asyncio.Queue = asyncio.Queue()

        async def stream():
            yield _sse({"agent": "Orchestrator", "type": "START", "message": f"Researching: {query}",
                        "data": {"simulated_llm": settings.simulate_llm, "note": SIMULATED_NOTE
                                 if settings.simulate_llm else "", "embedder": settings.embedder_backend}})
            task = asyncio.create_task(research(query, embedder(), queue))
            while True:
                getter = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    yield _sse(asdict(getter.result()))
                    continue
                getter.cancel()
                while not queue.empty():
                    yield _sse(asdict(queue.get_nowait()))
                break
            try:
                result = task.result()
                yield _sse({"agent": "Orchestrator", "type": "REPORT", "message": "Research complete",
                            "data": result})
            except Exception as e:  # report the failure in-band; headers are already sent
                yield _sse(asdict(AgentEvent(agent="Orchestrator", type="ERROR", message=str(e))))

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
