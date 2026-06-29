from __future__ import annotations

from fastapi import APIRouter, HTTPException

from web.backend.schemas.demo import DemoRunResponse, DryRunRequest
from web.backend.services.demo_runner import run_dry_demo

router = APIRouter(prefix="/api/demo", tags=["demo"])


@router.post("/dry-run", response_model=DemoRunResponse)
def dry_run(request: DryRunRequest) -> DemoRunResponse:
    result = run_dry_demo(request.task_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"unknown task_id: {request.task_id}")
    return result
