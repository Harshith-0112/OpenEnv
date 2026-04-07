from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from tasks import build_tasks


TASKS = build_tasks()


def grade_task(task_id: str, state: Dict[str, Any]) -> float:
    task = TASKS[task_id]
    score = 0.0

    score += 0.2 if state.get("category") == task.true_issue_type.value else 0.0
    score += 0.1 if state.get("priority") == task.correct_priority.value else 0.0
    score += 0.25 * _partial_outcome_score(task, state)
    score += 0.2 * _milestone_coverage(task.required_resolution_steps, state.get("completed_milestones", []))
    score += 0.15 * _response_quality(task.response_requirements, state.get("last_agent_message", ""))
    score += 0.1 * _efficiency_score(state.get("steps_taken", 0), task.max_steps)
    score += 0.05 if _reasoning_depth_bonus(state) else 0.0
    score -= 0.12 if state.get("hallucinated_resolution") else 0.0
    score -= 0.12 if state.get("unnecessary_escalation") else 0.0
    score -= 0.1 if state.get("premature_resolution") else 0.0
    score -= min(state.get("invalid_actions", 0), 3) * 0.03
    score -= 0.08 if _looks_like_single_strategy(state) else 0.0
    score -= 0.06 if _keyword_spam_penalty(state.get("last_agent_message", "")) else 0.0

    return round(max(0.001, min(score, 0.999)), 4)


def grade_easy_task(state: Dict[str, Any]) -> float:
    return grade_task("easy_password_reset_triage", state)


def grade_medium_task(state: Dict[str, Any]) -> float:
    return grade_task("medium_refund_resolution", state)


def grade_hard_task(state: Dict[str, Any]) -> float:
    return grade_task("hard_enterprise_incident", state)


def grade_all_tasks(states: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, float]:
    return {task_id: grade_task(task_id, state) for task_id, state in states}


def _partial_outcome_score(task: Any, state: Dict[str, Any]) -> float:
    correct_status = (
        (task.preferred_final_action == "resolve" and state.get("status") == "resolved")
        or (task.preferred_final_action == "escalate" and state.get("status") == "escalated")
    )
    correct_code = state.get("resolution_code") == task.success_resolution_code
    if correct_status and correct_code:
        return 1.0
    if correct_status:
        return 0.6
    if correct_code:
        return 0.4
    return 0.0


def _milestone_coverage(expected_steps: Iterable[str], actual_steps: Iterable[str]) -> float:
    expected = list(expected_steps)
    if not expected:
        return 1.0
    actual = set(actual_steps)
    matches = sum(1 for item in expected if item in actual)
    return matches / len(expected)


def _response_quality(required_phrases: Iterable[str], reply: str) -> float:
    required = list(required_phrases)
    if not required:
        return 1.0
    normalized = reply.lower()
    matches = sum(1 for phrase in required if phrase.lower() in normalized)
    base_score = matches / len(required)
    if len(normalized.split()) < 9:
        return max(0.0, base_score - 0.2)
    return base_score


def _efficiency_score(steps_taken: int, max_steps: int) -> float:
    if max_steps <= 0:
        return 0.0
    ratio = steps_taken / max_steps
    if ratio <= 0.5:
        return 1.0
    if ratio <= 1.0:
        return 1.0 - (ratio - 0.5)
    return max(0.0, 0.5 - (ratio - 1.0))


def _looks_like_single_strategy(state: Dict[str, Any]) -> bool:
    action_counts = state.get("action_counts", {})
    steps_taken = max(state.get("steps_taken", 0), 1)
    if not action_counts or steps_taken < 4:
        return False
    return max(action_counts.values()) / steps_taken >= 0.8


def _keyword_spam_penalty(reply: str) -> bool:
    lowered = reply.lower()
    repeated_terms = ["refund", "status page", "workaround", "impact"]
    return len(lowered.split()) < 8 or any(lowered.count(term) >= 3 for term in repeated_terms)


def _reasoning_depth_bonus(state: Dict[str, Any]) -> bool:
    return len(state.get("completed_milestones", [])) >= 3
