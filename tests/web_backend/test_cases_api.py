from fastapi.testclient import TestClient

from web.backend.app import app


def test_cases_list_and_detail_include_prompt_preview() -> None:
    client = TestClient(app)

    list_response = client.get("/api/cases")

    assert list_response.status_code == 200
    cases = list_response.json()
    assert len(cases) > 0
    first = cases[0]
    assert first["task_id"].startswith("vlr_pilot_")
    assert first["reward_total"] is not None

    detail_response = client.get(f"/api/cases/{first['task_id']}")

    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["task_id"] == first["task_id"]
    assert "Question:" in detail["prompt_preview"]
    assert "Documents:" in detail["prompt_preview"]
    assert detail["gold_answer"]
    assert detail["documents"]


def test_unknown_case_returns_404() -> None:
    client = TestClient(app)

    response = client.get("/api/cases/not-a-task")

    assert response.status_code == 404
