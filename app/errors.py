"""What a user is told when something fails, and what only the server log sees.

Exception text is not shown to users: an httpx error quotes the provider URL,
a JSON or KeyError quotes part of the model's output, and anything raised deep
in a library can say more about the server than a visitor needs. Users get the
kind of failure; the full exception goes to the log.
"""
from __future__ import annotations

import json
import logging

import httpx

log = logging.getLogger("meridian.errors")


def describe_llm_error(exc: BaseException) -> str:
    """A short, safe reason an LLM call failed, for events and report warnings."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            return f"HTTP {status}: the API key was refused"
        if status == 429:
            return "HTTP 429: the provider's rate limit was reached"
        if status >= 500:
            return f"HTTP {status}: the provider had an error"
        return f"HTTP {status} from the provider"
    if isinstance(exc, httpx.TimeoutException):
        return "the provider did not answer in time"
    if isinstance(exc, httpx.HTTPError):
        return "could not reach the provider"
    if isinstance(exc, json.JSONDecodeError):
        return "the model returned output that is not JSON"
    if isinstance(exc, (KeyError, IndexError, TypeError)):
        return "the provider's response had an unexpected shape"
    return f"unexpected error ({exc.__class__.__name__})"


def describe_pipeline_error(exc: BaseException) -> str:
    """The message for a run that failed outright."""
    return (f"Research failed with an internal error ({exc.__class__.__name__}); "
            "the details are in the server log. Please try again.")


def log_failure(where: str, exc: BaseException) -> None:
    log.warning("%s failed: %s", where, exc.__class__.__name__, exc_info=exc)
