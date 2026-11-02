import asyncio
import json
from dataclasses import asdict
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from app.database import AsyncSessionLocal
from app.models.session import ResearchSession
from app.pipeline.orchestrator import FINISHED_STATUSES

router = APIRouter()

TERMINAL_EVENTS = ("DONE", "ERROR")


@router.get("/research/{session_id}/stream")
async def stream_research(session_id: str, request: Request):
    queues = request.app.state.event_queues
    queue = queues.get(session_id)

    # Finished session whose live queue was consumed or dropped:
    # replay the saved event log instead of waiting for events that never come.
    if queue is None or (getattr(queue, "finished", False) and queue.empty()):
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(ResearchSession).where(ResearchSession.id == session_id))
            session = result.scalar_one_or_none()
        if session is None or session.status not in FINISHED_STATUSES:
            raise HTTPException(status_code=404, detail="No active stream for this session")
        saved = list(session.agent_events or [])

        async def replay_generator():
            for event in saved:
                yield {"data": json.dumps(event)}

        return EventSourceResponse(replay_generator())

    async def event_generator():
        while True:
            try:
                if await request.is_disconnected():
                    break
                event = await asyncio.wait_for(queue.get(), timeout=60.0)
                yield {"data": json.dumps(asdict(event))}
                if event.type in TERMINAL_EVENTS:
                    # The live stream has been delivered; later requests replay from the DB.
                    if queues.get(session_id) is queue and getattr(queue, "finished", True):
                        queues.pop(session_id, None)
                    break
            except asyncio.TimeoutError:
                yield {"data": json.dumps({"type": "HEARTBEAT", "message": "still running..."})}
            except Exception:
                break

    return EventSourceResponse(event_generator())
