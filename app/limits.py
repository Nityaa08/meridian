"""
Request guards shared by the full service and the demo.

- RateLimiter: a fixed one-hour window per key, kept in memory (so per process).
- client_ip: the caller's IP, trusting Vercel's X-Real-IP header only on Vercel.
- require_api_key: when API_KEY is set, the endpoints that start work or list
  other people's queries need `Authorization: Bearer <key>` or `X-API-Key: <key>`.
- limit_research: per-IP hourly limit on starting research or follow-ups.
- check_length: rejects questions longer than MAX_QUERY_CHARS.
"""
import hmac
import os
import time

from fastapi import HTTPException, Request

from app.config import settings


class RateLimiter:
    """Fixed one-hour window per key, in memory (so per process or serverless instance)."""

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


def _presented_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


async def require_api_key(request: Request) -> None:
    """FastAPI dependency. No-op when API_KEY is unset (local development)."""
    expected = settings.api_key
    if not expected:
        return
    # compare_digest so the time taken does not reveal how much of the key matched.
    if not hmac.compare_digest(_presented_key(request).encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail="Missing or wrong API key. Send 'Authorization: Bearer <key>' or 'X-API-Key: <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def limit_research(request: Request) -> None:
    """FastAPI dependency: RESEARCH_RATE_LIMIT_PER_HOUR runs per client IP (0 turns it off)."""
    limit = settings.research_rate_limit_per_hour
    if limit <= 0:
        return
    limiter = getattr(request.app.state, "research_limiter", None)
    if limiter is None or limiter.limit != limit:
        limiter = request.app.state.research_limiter = RateLimiter(limit)
    ok, retry = limiter.allow(client_ip(request))
    if not ok:
        raise HTTPException(
            status_code=429,
            detail=f"Limit of {limit} research runs an hour reached. Try again in {retry} seconds.",
            headers={"Retry-After": str(retry)},
        )


def check_length(text: str, field: str) -> str:
    """Strips the text and rejects it if empty or longer than MAX_QUERY_CHARS (HTTP 400)."""
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail=f"{field} cannot be empty")
    if len(text) > settings.max_query_chars:
        raise HTTPException(
            status_code=400,
            detail=f"{field} is {len(text)} characters; the limit is {settings.max_query_chars}.",
        )
    return text
