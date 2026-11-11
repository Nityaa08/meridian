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

# Following a session that runs on another worker: how often to check the DB,
# how often to send a heartbeat, and when to give up on a session that never finishes.
FOLLOW_POLL_S = 0.5
FOLLOW_HEARTBEAT_S = 15.0
FOLLOW_MAX_S = 30 * 60.0


async def _load(session_id: str):
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(ResearchSession).where(ResearchSession.id == session_id))
        return result.scalar_one_or_none()


async def _follow_from_db(session_id: str, request: Request):
    """
    Streams a session that is running on another worker by reading its saved event
    log, which the owning worker updates as events happen. Sends each new event once,
    and stops after a DONE/ERROR event or when the session's status is finished.
    """
    loop = asyncio.get_running_loop()
    started = last_beat = loop.time()
    sent = 0
    while True:
        if await request.is_disconnected():
            return
        session = await _load(session_id)
        if session is None:
            return
        events = list(session.agent_events or [])
        for event in events[sent:]:
            yield {"data": json.dumps(event)}
            if event.get("type") in TERMINAL_EVENTS:
                return
        sent = max(sent, len(events))
        if session.status in FINISHED_STATUSES:
            return
        now = loop.time()
        if now - started > FOLLOW_MAX_S:
            yield {"data": json.dumps({"type": "ERROR", "message": "stream timed out waiting for the session"})}
            return
        if now - last_beat >= FOLLOW_HEARTBEAT_S:
            last_beat = now
            yield {"data": json.dumps({"type": "HEARTBEAT", "message": "still running..."})}
        await asyncio.sleep(FOLLOW_POLL_S)


@router.get("/research/{session_id}/stream")
async def stream_research(session_id: str, request: Request):
    queues = request.app.state.event_queues
    queue = queues.get(session_id)

    # No live queue here: either the session finished and its queue was dropped, or it is
    # running on another worker process. Both are served from the event log in the DB.
    if queue is None or (getattr(queue, "finished", False) and queue.empty()):
        session = await _load(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="No active stream for this session")

        if session.status in FINISHED_STATUSES:
            saved = list(session.agent_events or [])

            async def replay_generator():
                for event in saved:
                    yield {"data": json.dumps(event)}

            return EventSourceResponse(replay_generator())

        return EventSourceResponse(_follow_from_db(session_id, request))

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
