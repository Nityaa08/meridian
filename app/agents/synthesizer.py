import asyncio
import json
import numpy as np
import httpx
from app.agents.base import BaseAgent
from app.config import settings


def filter_citations(sections: list, allowed_urls: set[str]) -> tuple[list[dict], int]:
    """
    Keep only section sources that are URLs of passages given to the LLM.
    Returns (sections, number of citations removed).
    """
    cleaned, removed = [], 0
    for section in sections if isinstance(sections, list) else []:
        if not isinstance(section, dict):
            continue
        sources = section.get("sources") or []
        if not isinstance(sources, list):
            sources = [sources]
        kept = [u for u in sources if isinstance(u, str) and u.strip() in allowed_urls]
        removed += len(sources) - len(kept)
        cleaned.append({**section, "sources": [u.strip() for u in kept]})
    return cleaned, removed


class SynthesizerAgent(BaseAgent):
    def __init__(self):
        super().__init__("SynthesizerAgent")
        self.removed_citations = 0

    async def run(
        self,
        query: str,
        passages: list[dict],
        critique: dict,
        queue: asyncio.Queue,
        followup_context: str = None,
    ) -> dict:
        await self.emit(queue, "SYNTHESIZING", "Building structured research report")

        passages_text = "\n\n".join(
            f"[{i+1}] Source: {p.get('url', 'unknown')} | {p.get('title', '')}\n{p.get('text', '')[:500]}"
            for i, p in enumerate(passages[:10])
        )
        sources_list = [
            {"title": p.get("title", ""), "url": p.get("url", ""), "relevance_score": round(p.get("similarity", 0.0), 4)}
            for p in passages[:10]
        ]

        gaps = critique.get("gaps", [])
        contradictions = critique.get("contradictions", [])

        followup_note = f"\n\nFollow-up question to address: {followup_context}" if followup_context else ""

        system_prompt = (
            "You are a research synthesis expert. Build a comprehensive, structured research report. "
            "Return a JSON object with keys: "
            "'summary' (2-3 sentence overview), "
            "'key_findings' (list of 3-5 specific findings as strings), "
            "'sections' (list of objects with 'title', 'content', 'sources' list of URLs). "
            "Be specific and cite evidence from the passages provided."
        )
        user_prompt = (
            f"Research question: {query}{followup_note}\n\n"
            f"Identified gaps: {json.dumps(gaps)}\n"
            f"Contradictions: {json.dumps(contradictions)}\n\n"
            f"Passages:\n{passages_text}"
        )

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
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
                        "temperature": 0.4,
                        # Room for reasoning tokens: gpt-oss models count them here.
                        "max_tokens": 4096,
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                synthesis = json.loads(content)
        except Exception as e:
            self.error = f"{e.__class__.__name__}: {e}"
            await self.emit(queue, "SYNTHESIZING", f"Synthesizer fallback: {e}", {"error": str(e)})
            synthesis = {
                "summary": f"Research on: {query}",
                "key_findings": ["Synthesis failed — raw passages available"],
                "sections": [],
            }

        confidence = float(np.mean([p.get("similarity", 0.5) for p in passages])) if passages else 0.0

        sections, self.removed_citations = filter_citations(
            synthesis.get("sections", []),
            {p.get("url", "") for p in passages[:10] if p.get("url")},
        )

        report = {
            "summary": synthesis.get("summary", ""),
            "key_findings": synthesis.get("key_findings", []),
            "sections": sections,
            "sources": sources_list,
            "confidence_score": round(confidence, 4),
            "gaps_identified": gaps,
            "contradictions": contradictions,
        }

        await self.emit(
            queue, "SYNTHESIZING",
            f"Report complete. Confidence: {confidence:.2f}. "
            f"{len(synthesis.get('key_findings', []))} key findings.",
            {"confidence_score": round(confidence, 4)},
        )
        return report
