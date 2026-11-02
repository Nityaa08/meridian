import asyncio
import time
from datetime import datetime
from dataclasses import asdict

from sqlalchemy import select

from app.agents.base import AgentEvent
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearcherAgent
from app.agents.extractor import ExtractorAgent
from app.agents.critic import CriticAgent
from app.agents.synthesizer import SynthesizerAgent
from app.database import AsyncSessionLocal
from app.models.session import ResearchSession

# Session status values: pending / running / completed / degraded / failed.
# "degraded" = a report was produced, but a stage fell back or found nothing;
# report["warnings"] says which.
STATUS_COMPLETED = "completed"
STATUS_DEGRADED = "degraded"
FINISHED_STATUSES = ("completed", "degraded", "failed")


def _history(queue: asyncio.Queue) -> list[dict]:
    return list(getattr(queue, "history", []))


def collect_warnings(
    *,
    planner: PlannerAgent | None,
    sources_searched: int,
    top_passages: list[dict],
    critic: CriticAgent,
    synthesizer: SynthesizerAgent,
) -> list[str]:
    warnings = []
    if planner is not None and planner.error:
        warnings.append(f"Planner LLM call failed; only the original question was searched ({planner.error})")
    if sources_searched == 0:
        warnings.append("No search source returned any passages")
    elif not top_passages:
        warnings.append("No passages met the relevance threshold")
    if critic.error:
        warnings.append(f"Critic LLM call failed; contradictions and gaps were not analysed ({critic.error})")
    if synthesizer.error:
        warnings.append(f"Synthesizer LLM call failed; the report has no real summary ({synthesizer.error})")
    if synthesizer.removed_citations:
        warnings.append(
            f"Removed {synthesizer.removed_citations} section citation(s) that were not among the retrieved sources"
        )
    return warnings


async def run_pipeline(
    session_id: str,
    query: str,
    embedder,
    queue: asyncio.Queue,
):
    start_time = time.time()

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(ResearchSession).where(ResearchSession.id == session_id))
        session = result.scalar_one_or_none()
        if not session:
            return
        session.status = "running"
        await db.commit()

    try:
        planner = PlannerAgent()
        sub_queries = await planner.run(query, queue)

        async with AsyncSessionLocal() as db:
            result = await db.execute(select(ResearchSession).where(ResearchSession.id == session_id))
            session = result.scalar_one_or_none()
            if session:
                session.sub_queries = sub_queries
                await db.commit()

        researcher = ResearcherAgent()
        tasks = [researcher.run(sq, queue) for sq in sub_queries]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        all_passages = []
        for r in raw_results:
            if isinstance(r, Exception):
                continue
            all_passages.extend(r)

        # Deduplicate by URL — prevents duplicate passages from inflating results
        seen_urls = set()
        deduped = []
        for p in all_passages:
            url = p.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            deduped.append(p)
        all_passages = deduped

        sources_searched = len(all_passages)

        async with AsyncSessionLocal() as db:
            result = await db.execute(select(ResearchSession).where(ResearchSession.id == session_id))
            session = result.scalar_one_or_none()
            if session:
                session.sources_raw = all_passages
                session.sources_searched = sources_searched
                await db.commit()

        extractor = ExtractorAgent(embedder)
        top_passages = await extractor.run(query, all_passages, queue)

        critic = CriticAgent()
        critique = await critic.run(query, top_passages, queue)

        synthesizer = SynthesizerAgent()
        report = await synthesizer.run(query, top_passages, critique, queue)

        warnings = collect_warnings(
            planner=planner,
            sources_searched=sources_searched,
            top_passages=top_passages,
            critic=critic,
            synthesizer=synthesizer,
        )
        status = STATUS_DEGRADED if warnings else STATUS_COMPLETED

        elapsed_ms = int((time.time() - start_time) * 1000)
        report["session_id"] = session_id
        report["query"] = query
        report["warnings"] = warnings

        done_event = AgentEvent(
            agent="Orchestrator",
            type="DONE",
            message="Research complete" if not warnings else f"Research complete with {len(warnings)} warning(s)",
            data={
                "session_id": session_id,
                "status": status,
                "warnings": warnings,
                "total_time_ms": elapsed_ms,
                "sources_searched": sources_searched,
                "sources_used": len(top_passages),
            },
        )
        await queue.put(done_event)

        async with AsyncSessionLocal() as db:
            result = await db.execute(select(ResearchSession).where(ResearchSession.id == session_id))
            session = result.scalar_one_or_none()
            if session:
                session.status = status
                session.report = report
                session.agent_events = _history(queue) or [asdict(done_event)]
                session.completed_at = datetime.utcnow()
                session.total_time_ms = elapsed_ms
                session.sources_used = len(top_passages)
                await db.commit()

    except Exception as e:
        error_event = AgentEvent(
            agent="Orchestrator",
            type="ERROR",
            message=str(e),
        )
        await queue.put(error_event)
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(ResearchSession).where(ResearchSession.id == session_id))
            session = result.scalar_one_or_none()
            if session:
                session.status = "failed"
                session.error_message = str(e)
                session.agent_events = _history(queue) or [asdict(error_event)]
                session.completed_at = datetime.utcnow()
                await db.commit()


async def run_followup_pipeline(
    session_id: str,
    followup_question: str,
    embedder,
    queue: asyncio.Queue,
):
    start_time = time.time()

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(ResearchSession).where(ResearchSession.id == session_id))
        session = result.scalar_one_or_none()
        if not session or not session.sources_raw:
            error_event = AgentEvent(
                agent="Orchestrator",
                type="ERROR",
                message="Session not found or no sources available for follow-up",
            )
            await queue.put(error_event)
            return

        query = session.query
        all_passages = session.sources_raw or []

    try:
        extractor = ExtractorAgent(embedder)
        top_passages = await extractor.run(followup_question, all_passages, queue)

        critic = CriticAgent()
        critique = await critic.run(followup_question, top_passages, queue)

        synthesizer = SynthesizerAgent()
        report = await synthesizer.run(
            query, top_passages, critique, queue, followup_context=followup_question
        )
    except Exception as e:
        await queue.put(AgentEvent(agent="Orchestrator", type="ERROR", message=str(e)))
        return

    warnings = collect_warnings(
        planner=None,
        sources_searched=len(all_passages),
        top_passages=top_passages,
        critic=critic,
        synthesizer=synthesizer,
    )
    status = STATUS_DEGRADED if warnings else STATUS_COMPLETED

    elapsed_ms = int((time.time() - start_time) * 1000)
    report["session_id"] = session_id
    report["query"] = query
    report["followup_question"] = followup_question
    report["warnings"] = warnings

    done_event = AgentEvent(
        agent="Orchestrator",
        type="DONE",
        message="Follow-up research complete" if not warnings
        else f"Follow-up research complete with {len(warnings)} warning(s)",
        data={"session_id": session_id, "status": status, "warnings": warnings, "total_time_ms": elapsed_ms},
    )
    await queue.put(done_event)

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(ResearchSession).where(ResearchSession.id == session_id))
        session = result.scalar_one_or_none()
        if session:
            session.status = status
            session.report = report
            session.agent_events = _history(queue) or [asdict(done_event)]
            session.completed_at = datetime.utcnow()
            session.total_time_ms = elapsed_ms
            session.sources_used = len(top_passages)
            await db.commit()
