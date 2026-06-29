from __future__ import annotations

from fastapi import APIRouter, HTTPException

from web.backend.schemas.demo import DemoCase, DemoCaseSummary
from web.backend.services.results_loader import get_case, list_case_summaries

router = APIRouter(prefix="/api", tags=["cases"])


@router.get("/cases", response_model=list[DemoCaseSummary])
def list_cases() -> list[DemoCaseSummary]:
    return list_case_summaries()


@router.get("/cases/{task_id}", response_model=DemoCase)
def read_case(task_id: str) -> DemoCase:
    case = get_case(task_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"unknown task_id: {task_id}")
    return case
