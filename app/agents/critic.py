import asyncio
import httpx
from app.agents.base import BaseAgent, json_object, list_field
from app.config import settings
from app.errors import describe_llm_error, log_failure
from app.agents import simulated


class CriticAgent(BaseAgent):
    def __init__(self):
        super().__init__("CriticAgent")

    async def run(self, query: str, passages: list[dict], queue: asyncio.Queue) -> dict:
        await self.emit(queue, "CRITIQUING", "Analyzing sources for contradictions and gaps")

        passages_text = "\n\n".join(
            f"[{i+1}] ({p.get('source', 'unknown')}) {p.get('title', '')}: {p.get('text', '')[:400]}"
            for i, p in enumerate(passages[:8])
        )

        system_prompt = (
            "You are a critical research analyst. Review the provided passages and identify: "
            "contradictions between sources, significant knowledge gaps, and potentially unreliable sources. "
            "Return a JSON object with keys: "
            "'contradictions' (list of strings), "
            "'gaps' (list of strings), "
            "'unreliable_sources' (list of strings), "
            "'overall_quality' (string: high/medium/low)."
        )
        user_prompt = (
            f"Research question: {query}\n\n"
            f"Passages:\n{passages_text}"
        )

        if settings.simulate_llm:
            critique = simulated.critique(passages)
        else:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        f"{settings.groq_base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                        json={
                            "model": settings.groq_model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "response_format": {"type": "json_object"},
                            "temperature": 0.2,
                        },
                    )
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]
                    raw = json_object(content)
                    critique = {
                        "contradictions": list_field(raw, "contradictions"),
                        "gaps": list_field(raw, "gaps", strings=True),
                        "unreliable_sources": list_field(raw, "unreliable_sources"),
                        "overall_quality": raw.get("overall_quality") if isinstance(raw.get("overall_quality"), str) else "unknown",
                    }
            except Exception as e:
                log_failure("Critic LLM call", e)
                self.error = describe_llm_error(e)
                await self.emit(queue, "CRITIQUING", f"Critic fallback: {self.error}", {"error": self.error})
                critique = {
                    "contradictions": [],
                    "gaps": ["Unable to perform full critique"],
                    "unreliable_sources": [],
                    "overall_quality": "medium",
                }

        await self.emit(
            queue, "CRITIQUING",
            f"Found {len(critique.get('contradictions', []))} contradictions, "
            f"{len(critique.get('gaps', []))} gaps. Quality: {critique.get('overall_quality', 'unknown')}",
            critique,
        )
        return critique
