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
from app.errors import describe_pipeline_error, log_failure
from app.limits import RateLimiter, client_ip  # noqa: F401  (RateLimiter re-exported for tests)
from app.pipeline.orchestrator import (STATUS_COMPLETED, STATUS_DEGRADED, _deadline_s, _timeout_message,
                                      collect_warnings)

_STATIC = os.path.join(os.path.dirname(__file__), "static")


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


class DeadlineExceeded(Exception):
    """The whole demo run passed SESSION_TIMEOUT_S."""


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

        async def run_with_deadline() -> dict:
            # The same SESSION_TIMEOUT_S deadline as the full service. Only the deadline
            # itself is reported as one; an agent's own TimeoutError is an ordinary failure.
            limit = _deadline_s()
            if limit is None:
                return await research(query, embedder(), queue)
            try:
                async with asyncio.timeout(limit) as deadline:
                    return await research(query, embedder(), queue)
            except TimeoutError:
                if deadline.expired():
                    raise DeadlineExceeded() from None
                raise

        async def stream():
            yield _sse({"agent": "Orchestrator", "type": "START", "message": f"Researching: {query}",
                        "data": {"simulated_llm": settings.simulate_llm, "note": SIMULATED_NOTE
                                 if settings.simulate_llm else "", "embedder": settings.embedder_backend}})
            task = asyncio.create_task(run_with_deadline())
            getter = None
            try:
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
                except DeadlineExceeded:
                    yield _sse(asdict(AgentEvent(agent="Orchestrator", type="ERROR", message=_timeout_message())))
                except Exception as e:  # report the failure in-band; headers are already sent
                    log_failure("Demo research run", e)
                    yield _sse(asdict(AgentEvent(agent="Orchestrator", type="ERROR",
                                                 message=describe_pipeline_error(e))))
            finally:
                # The visitor closed the tab (or the stream ended): stop the run, so an
                # abandoned question does not keep spending searches and LLM calls.
                if getter is not None:
                    getter.cancel()
                task.cancel()

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
