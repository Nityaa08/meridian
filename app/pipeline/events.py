"""
Per-session event queues for the SSE stream.

EventQueue records every event it receives, so the orchestrator can save the
full event log and a finished session's stream can be replayed from the DB.

With more than one worker process, a stream request can land on a worker that
doesn't hold the session's queue. So the event log is also saved to the DB while
the session runs (see EventQueue.on_change), and the stream endpoint can follow a
running session from the DB instead of returning 404.
"""
import asyncio
from dataclasses import asdict
from typing import Awaitable, Callable

# How often a running session's event log is written to the DB, at most.
PROGRESS_SAVE_INTERVAL_S = 0.5

# Seconds a finished session's queue is kept for a late stream client before
# it is dropped (after that the stream is replayed from the saved event log).
FINISHED_QUEUE_TTL_S = 300.0


class EventQueue(asyncio.Queue):
    def __init__(self, on_change: Callable[[list[dict]], Awaitable[None]] | None = None) -> None:
        super().__init__()
        self.history: list[dict] = []
        self.finished = False
        # Called with the event log so far, at most every PROGRESS_SAVE_INTERVAL_S.
        self._on_change = on_change
        self._save_task: asyncio.Task | None = None

    async def put(self, item) -> None:
        self.history.append(asdict(item))
        await super().put(item)
        if self._on_change and (self._save_task is None or self._save_task.done()):
            self._save_task = asyncio.get_running_loop().create_task(self._save_soon())

    async def _save_soon(self) -> None:
        # Batches a burst of events into one write.
        await asyncio.sleep(PROGRESS_SAVE_INTERVAL_S)
        try:
            await self._on_change(list(self.history))
        except Exception:
            pass  # progress saving is best effort; the final save still happens


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
