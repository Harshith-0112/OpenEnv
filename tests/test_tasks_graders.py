from graders import grade_easy_task, grade_hard_task, grade_medium_task
from tasks import build_tasks


def test_tasks_cover_three_difficulty_levels() -> None:
    tasks = build_tasks()
    assert set(tasks.keys()) == {
        "easy_password_reset_triage",
        "medium_refund_resolution",
        "hard_enterprise_incident",
        "medium_security_alert",
    }


def test_grader_scores_are_bounded() -> None:
    easy_score = grade_easy_task(
        {
            "category": "account_access",
            "priority": "high",
            "status": "resolved",
            "resolution_code": "account_unlock_guidance",
            "completed_milestones": ["classified", "prioritized", "responded", "resolved"],
            "last_agent_message": "Your account is locked, I can help restore access, and you should use the reset link after 15 minutes.",
            "steps_taken": 4,
        }
    )
    medium_score = grade_medium_task(
        {
            "category": "billing",
            "priority": "medium",
            "status": "resolved",
            "resolution_code": "refund_timeline_confirmed",
            "completed_milestones": ["classified", "prioritized", "responded", "resolved"],
            "last_agent_message": "The refund for order ORD-88421 should appear in 3-5 business days.",
            "steps_taken": 4,
        }
    )
    hard_score = grade_hard_task(
        {
            "category": "technical",
            "priority": "critical",
            "status": "resolved",
            "resolution_code": "sev1_incident_escalated",
            "completed_milestones": ["classified", "prioritized", "info_requested", "customer_replied", "responded", "escalated"],
            "last_agent_message": (
                "We recognize the impact, a workaround is available, the status page is active, "
                "and there is no evidence of data integrity issues."
            ),
            "steps_taken": 6,
        }
    )
    for score in (easy_score, medium_score, hard_score):
        assert 0.0 <= score <= 1.0
