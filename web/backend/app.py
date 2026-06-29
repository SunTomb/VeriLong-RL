from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.backend.api.cases import router as cases_router
from web.backend.api.demo import router as demo_router
from web.backend.api.summary import router as summary_router

app = FastAPI(
    title="VeriLong-RL Demo API",
    description="Offline-first API for the VeriLong-RL homepage and interactive demo.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(summary_router)
app.include_router(cases_router)
app.include_router(demo_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
