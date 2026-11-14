"""
Per-session event queues for the SSE stream.

EventQueue records every event it receives, so the orchestrator can save the
full event log and a finished session's stream can be replayed from the DB.

It is a broadcast log, not a work queue: each stream client keeps its own
position in `history` and waits for new events with wait_for_event(). So two
clients on the same session (two tabs, or an EventSource that reconnected)
each get every event from the first one, and each sees DONE/ERROR. (It used to
be an asyncio.Queue that clients consumed with get(), which split the events
between them and left the client that missed DONE waiting forever.)

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


class EventQueue:
    def __init__(self, on_change: Callable[[list[dict]], Awaitable[None]] | None = None) -> None:
        self.history: list[dict] = []
        self.finished = False
        # Called with the event log so far, at most every PROGRESS_SAVE_INTERVAL_S.
        self._on_change = on_change
        self._save_task: asyncio.Task | None = None
        # Replaced on every put/finish; waiters hold the old one, which is then set.
        self._changed = asyncio.Event()

    def _notify(self) -> None:
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    async def put(self, item) -> None:
        self.history.append(asdict(item))
        self._notify()
        if self._on_change and (self._save_task is None or self._save_task.done()):
            self._save_task = asyncio.get_running_loop().create_task(self._save_soon())

    def finish(self) -> None:
        """No more events will be added; wakes stream clients so they can stop."""
        self.finished = True
        self._notify()

    async def wait_for_event(self, seen: int, timeout: float) -> None:
        """
        Return once history has more than `seen` events or the queue is finished.
        Raises asyncio.TimeoutError after `timeout` seconds with nothing new.
        """
        if len(self.history) > seen or self.finished:
            return
        await asyncio.wait_for(self._changed.wait(), timeout)

    async def _save_soon(self) -> None:
        # Batches a burst of events into one write.
        await asyncio.sleep(PROGRESS_SAVE_INTERVAL_S)
        try:
            await self._on_change(list(self.history))
        except Exception:
            pass  # progress saving is best effort; the final save still happens


def active_session_count(queues: dict) -> int:
    return sum(1 for q in queues.values() if not getattr(q, "finished", False))


def release_queue(queues: dict, session_id: str, queue: "EventQueue",
                  ttl_s: float = FINISHED_QUEUE_TTL_S) -> None:
    """Mark a session's queue finished and drop it from the registry after ttl_s."""
    if isinstance(queue, EventQueue):
        queue.finish()

    def _drop() -> None:
        if queues.get(session_id) is queue:
            queues.pop(session_id, None)

    if ttl_s <= 0:
        _drop()
    else:
        asyncio.get_running_loop().call_later(ttl_s, _drop)
