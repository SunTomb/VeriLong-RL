from fastapi.testclient import TestClient

from web.backend.app import app


def test_summary_labels_smoke_baselines() -> None:
    client = TestClient(app)

    response = client.get("/api/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["project"] == "VeriLong-RL"
    assert payload["status"]["phase1_pilot"] == "completed"
    assert payload["status"]["phase2"] == "design_only"
    assert {family["id"] for family in payload["task_families"]} == {
        "anti_distractor_retrieval",
        "multi_hop_reasoning",
        "temporal_update",
    }
    assert payload["smoke_summaries"]
    assert all("Smoke baseline only" in summary["note"] for summary in payload["smoke_summaries"])
