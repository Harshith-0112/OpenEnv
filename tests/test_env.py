from env import CustomerSupportEnv
from models import Action, ActionType, Priority, TicketCategory


def test_easy_task_completes_with_expected_flow() -> None:
    env = CustomerSupportEnv()
    observation = env.reset("easy_password_reset_triage")
    assert observation.task_id == "easy_password_reset_triage"

    observation, reward, done, _ = env.step(
        Action(
            action_type=ActionType.CLASSIFY_TICKET,
            ticket_id="TCK-1001",
            category=TicketCategory.ACCOUNT_ACCESS,
        )
    )
    assert reward.score > 0
    assert not done

    observation, reward, done, _ = env.step(
        Action(
            action_type=ActionType.SET_PRIORITY,
            ticket_id="TCK-1001",
            priority=Priority.HIGH,
        )
    )
    assert reward.score > 0
    assert not done

    observation, reward, done, info = env.step(
        Action(
            action_type=ActionType.RESPOND,
            ticket_id="TCK-1001",
            response_text="I can help restore access because your account is locked and we will restore access.",
        )
    )
    assert not done

    observation, reward, done, info = env.step(
        Action(
            action_type=ActionType.RESOLVE,
            ticket_id="TCK-1001",
            response_text="Use the reset link again after 15 minutes and the account should unlock.",
            resolution_code="unlock_after_15_minutes_reset_link",
        )
    )
    assert done
    assert info["deterministic_score"] >= 0.9
    assert env.state()["status"] == "resolved"


def test_invalid_action_is_penalized() -> None:
    env = CustomerSupportEnv()
    env.reset("easy_password_reset_triage")
    _, reward, done, _ = env.step(
        Action(
            action_type=ActionType.RESOLVE,
            ticket_id="TCK-1001",
            resolution_code="closed",
        )
    )
    assert reward.score < 0
    assert not done
