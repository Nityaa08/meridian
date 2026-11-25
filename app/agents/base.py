import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class AgentEvent:
    agent: str
    type: str  # PLANNING / SEARCHING / EXTRACTING / CRITIQUING / SYNTHESIZING / DONE / ERROR
    message: str
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        # Set when an LLM-backed agent fell back to a default result.
        self.error: str | None = None

    async def emit(self, queue: asyncio.Queue, event_type: str, message: str, data: dict = None):
        event = AgentEvent(
            agent=self.name,
            type=event_type,
            message=message,
            data=data or {},
        )
        await queue.put(event)
        return event

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        pass


def json_object(content: str) -> dict:
    """Parse a model's JSON reply, which must be one object (a list or a string is not)."""
    import json

    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise TypeError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def list_field(parsed: dict, key: str, strings: bool = False) -> list:
    """A list field from a model's JSON reply: null or a wrong type becomes [].

    Models in JSON mode still return `"gaps": null` or a bare string now and
    then. Before 2026-11-25 that reached len() and failed the whole run.
    """
    value = parsed.get(key)
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    if strings:
        return [str(v) for v in value if isinstance(v, (str, int, float)) and str(v).strip()]
    return [v for v in value if v is not None]
