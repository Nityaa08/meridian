import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import text

from app.config import settings
from app.database import engine, Base
from app.api.research import router as research_router
from app.api.stream import router as stream_router
from app.api.sessions import router as sessions_router
from app.pipeline.events import active_session_count


def load_embedder():
    """Load the embedder (imported lazily so tests and the demo can skip torch)."""
    from app.embedders import load_embedder as _load

    # device=None lets the library choose (MPS on Apple Silicon, which crashed).
    return _load(settings.embedder_backend, settings.embedder_model, settings.embedder_device or None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload embedder to avoid cold-start on first request
    print("Loading sentence-transformers embedder...")
    app.state.embedder = load_embedder()
    print(f"Embedder loaded: {getattr(app.state.embedder, 'name', settings.embedder_model)} "
          f"(device={settings.embedder_device or 'auto'})")

    # Initialize SQLite tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database initialized.")

    # Per-session SSE event queues
    app.state.event_queues: dict[str, asyncio.Queue] = {}

    yield

    # Cleanup
    await engine.dispose()


if settings.demo_mode:
    # Public demo: one stateless streaming endpoint, no database (see app/demo_app.py).
    from app.demo_app import create_demo_app

    app = create_demo_app()
else:
    app = FastAPI(
        title="Meridian — Multi-Agent Research Intelligence API",
        description=(
            "5 specialized AI agents decompose your query, search Wikipedia + arXiv + DuckDuckGo in parallel, "
            "rank passages by semantic similarity, detect contradictions, and synthesize a structured report — "
            "all streamed live via SSE."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(research_router)
    app.include_router(stream_router)
    app.include_router(sessions_router)

    @app.get("/")
    async def root():
        return {
            "name": "Meridian",
            "tagline": "Multi-agent research intelligence — parallel search, semantic ranking, live SSE streaming",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health")
    async def health():
        embedder_status = "loaded" if hasattr(app.state, "embedder") and app.state.embedder else "not loaded"
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            db_status = "ok"
        except Exception:
            db_status = "error"

        return {
            "status": "ok",
            "embedder": embedder_status,
            "db": db_status,
            "model": settings.groq_model,
            "active_sessions": active_session_count(getattr(app.state, "event_queues", {})),
        }
