from fastapi import APIRouter, Depends
from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models.session import ResearchSession
from app.limits import require_api_key

router = APIRouter()


# Lists every caller's questions, so it needs the API key when one is set.
@router.get("/sessions", dependencies=[Depends(require_api_key)])
async def list_sessions():
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ResearchSession).order_by(ResearchSession.created_at.desc()).limit(50)
        )
        sessions = result.scalars().all()

    return {
        "sessions": [
            {
                "session_id": s.id,
                "query": s.query,
                "status": s.status,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                "total_time_ms": s.total_time_ms,
                "sources_searched": s.sources_searched or 0,
                "sources_used": s.sources_used or 0,
            }
            for s in sessions
        ],
        "total": len(sessions),
    }
