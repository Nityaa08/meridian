import asyncio
import json
import httpx
from app.agents.base import BaseAgent
from app.config import settings


class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("PlannerAgent")

    async def run(self, query: str, queue: asyncio.Queue) -> list[str]:
        await self.emit(queue, "PLANNING", f"Decomposing query: '{query}'")

        system_prompt = (
            "You are a research planning assistant. Given a research question, "
            "decompose it into 3-5 specific sub-questions that together fully cover the topic. "
            "Return a JSON object with a single key 'sub_questions' containing a list of strings."
        )
        user_prompt = f"Research question: {query}"

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
                        "temperature": 0.3,
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                sub_questions = parsed.get("sub_questions", [query])
        except Exception as e:
            self.error = f"{e.__class__.__name__}: {e}"
            await self.emit(queue, "PLANNING", f"Planner fallback: {e}", {"error": str(e)})
            sub_questions = [query]

        if not isinstance(sub_questions, list) or not sub_questions:
            sub_questions = [query]
        sub_questions = [str(q) for q in sub_questions][:5]
        await self.emit(
            queue, "PLANNING",
            f"Identified {len(sub_questions)} sub-questions",
            {"sub_questions": sub_questions},
        )
        return sub_questions
