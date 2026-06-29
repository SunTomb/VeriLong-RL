from __future__ import annotations

from fastapi import APIRouter

from web.backend.schemas.demo import SummaryResponse
from web.backend.services.results_loader import load_summary

router = APIRouter(prefix="/api", tags=["summary"])


@router.get("/summary", response_model=SummaryResponse)
def get_summary() -> SummaryResponse:
    return load_summary()
