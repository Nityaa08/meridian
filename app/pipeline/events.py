"""
Per-session event queues for the SSE stream.

EventQueue records every event it receives, so the orchestrator can save the
full event log and a finished session's stream can be replayed from the DB.
"""
import asyncio
from dataclasses import asdict

# Seconds a finished session's queue is kept for a late stream client before
# it is dropped (after that the stream is replayed from the saved event log).
FINISHED_QUEUE_TTL_S = 300.0


class EventQueue(asyncio.Queue):
    def __init__(self) -> None:
        super().__init__()
        self.history: list[dict] = []
        self.finished = False

    async def put(self, item) -> None:
        self.history.append(asdict(item))
        await super().put(item)


def active_session_count(queues: dict) -> int:
    return sum(1 for q in queues.values() if not getattr(q, "finished", False))


def release_queue(queues: dict, session_id: str, queue: asyncio.Queue,
                  ttl_s: float = FINISHED_QUEUE_TTL_S) -> None:
    """Mark a session's queue finished and drop it from the registry after ttl_s."""
    if isinstance(queue, EventQueue):
        queue.finished = True

    def _drop() -> None:
        if queues.get(session_id) is queue:
            queues.pop(session_id, None)

    if ttl_s <= 0:
        _drop()
    else:
        asyncio.get_running_loop().call_later(ttl_s, _drop)
