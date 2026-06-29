from fastapi.testclient import TestClient

from web.backend.app import app


def test_dry_run_scores_with_existing_reward_pipeline() -> None:
    client = TestClient(app)
    task_id = client.get("/api/cases").json()[0]["task_id"]

    response = client.post("/api/demo/dry-run", json={"task_id": task_id})

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == task_id
    assert payload["source"] == "dry_run"
    assert payload["model"] == "dry_run_oracle_stub"
    assert payload["output_text"].startswith("Evidence:")
    assert "Steps:" in payload["output_text"]
    assert "Answer:" in payload["output_text"]
    assert payload["parsed_output"]["format_valid"] is True
    assert payload["metric_breakdown"]["reward_total"] > 0
    assert "reward_components" in payload["metric_breakdown"]


def test_dry_run_unknown_task_returns_404() -> None:
    client = TestClient(app)

    response = client.post("/api/demo/dry-run", json={"task_id": "not-a-task"})

    assert response.status_code == 404
