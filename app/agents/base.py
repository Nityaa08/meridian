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
