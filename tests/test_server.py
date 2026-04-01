from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_reset_endpoint_returns_observation() -> None:
    response = client.post("/reset", json={"task_id": "easy_password_reset_triage"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["observation"]["task_id"] == "easy_password_reset_triage"


def test_step_endpoint_returns_reward_payload() -> None:
    client.post("/reset", json={"task_id": "easy_password_reset_triage"})
    response = client.post(
        "/step",
        json={
            "action_type": "classify_ticket",
            "ticket_id": "TCK-1001",
            "category": "account_access",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "reward" in payload
    assert payload["done"] is False
