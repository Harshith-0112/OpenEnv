from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

from graders import grade_task
from models import (
    Action,
    ActionType,
    CustomerMood,
    HistoryEntry,
    Message,
    Observation,
    Reward,
    TaskDefinition,
    WorkflowStatus,
)
from tasks import build_tasks


class CustomerSupportEnv:
    def __init__(self) -> None:
        self.tasks: Dict[str, TaskDefinition] = build_tasks()
        self._task_order: List[str] = list(self.tasks.keys())
        self._task_pointer: int = 0
        self._active_task: Optional[TaskDefinition] = None
        self._history: List[HistoryEntry] = []
        self._completed_milestones: Set[str] = set()
        self._seen_actions: Set[str] = set()
        self._steps_taken: int = 0
        self._invalid_actions: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._mistakes: List[str] = []
        self._internal_state: Dict[str, Any] = {}

    def reset(self, task_id: Optional[str] = None) -> Observation:
        selected_task_id = task_id or self._next_task_id()
        self._active_task = self.tasks[selected_task_id]
        self._history = []
        self._completed_milestones = set()
        self._seen_actions = set()
        self._steps_taken = 0
        self._invalid_actions = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._mistakes = []
        self._internal_state = {
            "task_id": self._active_task.task_id,
            "status": WorkflowStatus.OPEN.value,
            "category": None,
            "priority": None,
            "last_agent_action": None,
            "last_agent_message": "",
            "resolution_code": None,
            "conversation": deepcopy(self._active_task.ticket.messages),
            "customer_mood": self._active_task.customer_persona.value,
            "known_info": {},
            "info_requested": False,
            "customer_replied": False,
            "hallucinated_resolution": False,
            "unnecessary_escalation": False,
            "premature_resolution": False,
            "last_debug_note": "Ticket opened and awaiting agent triage.",
            "progress_notes": [],
            "sla_remaining": self._active_task.sla_deadline,
            "action_counts": {},
        }
        return self._build_observation()

    def step(self, action: Action | Dict[str, Any]) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._active_task is None:
            raise RuntimeError("Environment must be reset before stepping.")
        if self._done:
            reward = Reward(
                score=-0.1,
                components={"invalid_action": -0.1},
                cumulative_score=self._cumulative_reward,
                explanation="Episode already completed. Reset the environment to start a new task.",
            )
            return self._build_observation(), reward, True, self._info("completed")

        parsed_action = action if isinstance(action, Action) else Action.model_validate(action)
        self._steps_taken += 1
        self._internal_state["sla_remaining"] = max(0, self._active_task.sla_deadline - (self._steps_taken * 10))

        components: Dict[str, float] = {"step_pressure": -0.05}
        explanations: List[str] = ["SLA pressure applied for taking a step."]
        milestone: Optional[str] = None

        if parsed_action.ticket_id != self._active_task.ticket.ticket_id:
            self._invalid_actions += 1
            self._mistakes.append("wrong_ticket")
            components["invalid_action"] = -0.1
            explanations.append("Action referenced a different ticket.")
        else:
            action_key = json.dumps(parsed_action.model_dump(exclude_none=True), sort_keys=True)
            action_name = parsed_action.action_type.value
            self._internal_state["action_counts"][action_name] = self._internal_state["action_counts"].get(action_name, 0) + 1
            if action_key in self._seen_actions:
                components["repeat_penalty"] = -0.1
                self._mistakes.append("repeated_action")
                explanations.append("Repeated action was penalized.")
            else:
                self._seen_actions.add(action_key)

            action_reward, milestone, note = self._apply_action(parsed_action)
            if action_reward != 0.0:
                components["trajectory"] = round(components.get("trajectory", 0.0) + action_reward, 4)
            if note:
                explanations.append(note)
                self._internal_state["last_debug_note"] = note

        if self._steps_taken >= self._active_task.max_steps and not self._done:
            self._done = True
            self._mistakes.append("sla_breached")
            explanations.append("Maximum step budget reached before clean completion.")

        if self._done:
            final_score = grade_task(self._active_task.task_id, self.state())
            components["final_alignment"] = round((final_score - 0.5) * 0.2, 4)
            explanations.append(f"Deterministic final score: {final_score:.2f}.")

        reward_value = round(sum(components.values()), 4)
        self._cumulative_reward = round(self._cumulative_reward + reward_value, 4)
        reward = Reward(
            score=reward_value,
            components=components,
            cumulative_score=self._cumulative_reward,
            explanation=" ".join(explanations),
        )

        self._internal_state["last_agent_action"] = parsed_action.model_dump(exclude_none=True)
        self._history.append(
            HistoryEntry(
                step=self._steps_taken,
                action=parsed_action,
                reward_delta=reward.score,
                milestone=milestone,
                status_after=WorkflowStatus(self._internal_state["status"]),
                customer_mood_after=CustomerMood(self._internal_state["customer_mood"]),
            )
        )
        return self._build_observation(), reward, self._done, self._info(milestone or "updated")

    def state(self) -> Dict[str, Any]:
        if self._active_task is None:
            return {}
        return {
            "task_id": self._active_task.task_id,
            "status": self._internal_state["status"],
            "category": self._internal_state["category"],
            "priority": self._internal_state["priority"],
            "last_agent_message": self._internal_state["last_agent_message"],
            "resolution_code": self._internal_state["resolution_code"],
            "known_info": deepcopy(self._internal_state["known_info"]),
            "steps_taken": self._steps_taken,
            "completed_milestones": sorted(self._completed_milestones),
            "invalid_actions": self._invalid_actions,
            "mistakes": list(self._mistakes),
            "done": self._done,
            "customer_mood": self._internal_state["customer_mood"],
            "sla_remaining": self._internal_state["sla_remaining"],
            "hallucinated_resolution": self._internal_state["hallucinated_resolution"],
            "unnecessary_escalation": self._internal_state["unnecessary_escalation"],
            "premature_resolution": self._internal_state["premature_resolution"],
            "action_counts": deepcopy(self._internal_state["action_counts"]),
            "true_issue_type": self._active_task.true_issue_type.value,
            "required_resolution_steps": list(self._active_task.required_resolution_steps),
            "correct_priority": self._active_task.correct_priority.value,
            "sla_deadline": self._active_task.sla_deadline,
            "customer_persona": self._active_task.customer_persona.value,
        }

    def _build_observation(self) -> Observation:
        assert self._active_task is not None
        ticket = deepcopy(self._active_task.ticket)
        ticket.current_status = WorkflowStatus(self._internal_state["status"])
        ticket.messages = deepcopy(self._internal_state["conversation"])
        relevant_data = {
            "knowledge_base": [article.model_dump() for article in self._active_task.knowledge_base],
            "workflow_hint": self._public_workflow_hint(),
            "known_customer_facts": deepcopy(self._internal_state["known_info"]),
            "sla_bucket": self._sla_bucket(),
        }
        last_action = self._internal_state["last_agent_action"]
        return Observation(
            task_id=self._active_task.task_id,
            difficulty=self._active_task.difficulty,
            objective=self._active_task.objective,
            ticket=ticket,
            current_status=WorkflowStatus(self._internal_state["status"]),
            conversation_history=deepcopy(self._internal_state["conversation"]),
            history=self._history,
            relevant_data=relevant_data,
            available_actions=self._active_task.allowed_actions,
            completed_milestones=sorted(self._completed_milestones),
            last_agent_action=Action.model_validate(last_action) if last_action else None,
            customer_mood=CustomerMood(self._internal_state["customer_mood"]),
            steps_taken=self._steps_taken,
            max_steps=self._active_task.max_steps,
        )

    def _apply_action(self, action: Action) -> Tuple[float, Optional[str], str]:
        assert self._active_task is not None

        valid, note = self._validate_sequence(action)
        if not valid:
            self._invalid_actions += 1
            self._mistakes.append("invalid_sequence")
            self._adjust_mood(negative=True)
            return -0.1, None, note

        if action.action_type == ActionType.CLASSIFY_TICKET:
            return self._handle_classification(action)
        if action.action_type == ActionType.SET_PRIORITY:
            return self._handle_priority(action)
        if action.action_type == ActionType.REQUEST_INFO:
            return self._handle_request_info(action)
        if action.action_type == ActionType.RESPOND:
            return self._handle_respond(action)
        if action.action_type == ActionType.RESOLVE:
            return self._handle_resolve(action)
        if action.action_type == ActionType.ESCALATE:
            return self._handle_escalate(action)
        self._invalid_actions += 1
        self._mistakes.append("unknown_action")
        return -0.1, None, "Unknown action type."

    def _validate_sequence(self, action: Action) -> Tuple[bool, str]:
        status = self._internal_state["status"]
        classified = "classified" in self._completed_milestones

        if action.action_type in {ActionType.RESOLVE, ActionType.ESCALATE} and not classified:
            return False, "Cannot close or escalate before classification."
        if action.action_type == ActionType.SET_PRIORITY and not classified:
            return False, "Priority can only be set after classification."
        if action.action_type == ActionType.REQUEST_INFO and status == WorkflowStatus.RESOLVED.value:
            return False, "Cannot request information after resolution."
        if action.action_type == ActionType.ESCALATE:
            text = f"{action.response_text or ''} {action.notes or ''}".strip()
            if not text:
                return False, "Escalation requires a justification or customer-facing explanation."
        return True, "Action accepted."

    def _handle_classification(self, action: Action) -> Tuple[float, Optional[str], str]:
        if action.category and action.category == self._active_task.true_issue_type:
            self._internal_state["category"] = action.category.value
            self._internal_state["status"] = WorkflowStatus.IN_PROGRESS.value
            milestone = self._record_milestone("classified")
            return 0.2, milestone, "Correct classification advanced the ticket to in progress."
        self._adjust_mood(negative=True)
        self._mistakes.append("bad_classification")
        return -0.08, None, "Incorrect classification misdirected the workflow."

    def _handle_priority(self, action: Action) -> Tuple[float, Optional[str], str]:
        if action.priority and action.priority == self._active_task.correct_priority:
            self._internal_state["priority"] = action.priority.value
            milestone = self._record_milestone("prioritized")
            return 0.1, milestone, "Priority matched hidden urgency and SLA expectations."
        self._adjust_mood(negative=True)
        self._mistakes.append("bad_priority")
        return -0.06, None, "Priority choice did not fit the real urgency."

    def _handle_request_info(self, action: Action) -> Tuple[float, Optional[str], str]:
        question = (action.question or action.response_text or "").lower()
        if not self._active_task.requires_customer_info:
            self._adjust_mood(negative=True)
            self._mistakes.append("unneeded_info_request")
            self._internal_state["status"] = WorkflowStatus.NEEDS_INFO.value
            return -0.04, None, "Unnecessary clarification slowed a solvable issue."

        asked_for_missing = any(key.replace("_", " ") in question or key in question for key in self._active_task.missing_info_keys)
        if not asked_for_missing:
            self._adjust_mood(negative=True)
            self._mistakes.append("weak_question")
            self._internal_state["status"] = WorkflowStatus.NEEDS_INFO.value
            self._append_agent_message(action.question or "Can you provide more details?", "2026-03-29T05:18:00Z")
            self._simulate_customer_reply(False)
            return 0.02, None, "Customer was asked for information, but the question was too vague."

        self._internal_state["status"] = WorkflowStatus.NEEDS_INFO.value
        self._internal_state["info_requested"] = True
        self._append_agent_message(action.question or "Please share the missing diagnostics.", "2026-03-29T05:18:00Z")
        self._simulate_customer_reply(True)
        milestone = self._record_milestone("info_requested")
        self._record_milestone("customer_replied")
        return 0.2, milestone, "Good clarification request produced the missing customer details."

    def _handle_respond(self, action: Action) -> Tuple[float, Optional[str], str]:
        text = (action.response_text or "").strip()
        if not text:
            self._mistakes.append("empty_response")
            self._adjust_mood(negative=True)
            return -0.06, None, "Response action requires a customer-facing message."

        self._internal_state["last_agent_message"] = text
        self._append_agent_message(text, "2026-03-29T05:20:00Z")

        requirement_hits = sum(
            1 for phrase in self._active_task.response_requirements if phrase.lower() in text.lower()
        )
        coverage = requirement_hits / max(len(self._active_task.response_requirements), 1)
        if self._looks_like_keyword_spam(text):
            self._mistakes.append("keyword_spam")
            self._adjust_mood(negative=True)
            return -0.05, None, "Response sounded repetitive and generic instead of tailored to the case."
        if coverage >= 0.75:
            milestone = self._record_milestone("responded")
            self._adjust_mood(positive=True)
            return round(0.05 + (0.15 * coverage), 4), milestone, "Response covered key customer-facing requirements."

        self._adjust_mood(negative=True)
        self._mistakes.append("weak_response")
        return round(0.03 * coverage, 4), None, "Response was incomplete and increased customer frustration."

    def _handle_resolve(self, action: Action) -> Tuple[float, Optional[str], str]:
        if self._active_task.requires_customer_info and not self._internal_state["customer_replied"]:
            self._mistakes.append("premature_resolution")
            self._internal_state["premature_resolution"] = True
            self._adjust_mood(negative=True)
            return -0.2, None, "Resolution was attempted before required customer details were collected."

        text = f"{action.response_text or ''} {action.resolution_code or ''}".lower()
        has_keywords = all(keyword.lower() in text for keyword in self._active_task.resolution_keywords)
        if self._active_task.preferred_final_action != "resolve":
            self._mistakes.append("wrong_closure")
            self._internal_state["hallucinated_resolution"] = True
            self._adjust_mood(negative=True)
            return -0.2, None, "This incident should be escalated, not resolved locally."

        if not has_keywords:
            self._mistakes.append("hallucinated_resolution")
            self._internal_state["hallucinated_resolution"] = True
            self._adjust_mood(negative=True)
            return -0.2, None, "Resolution was attempted without the required concrete remediation."

        self._internal_state["status"] = WorkflowStatus.RESOLVED.value
        self._internal_state["resolution_code"] = self._active_task.success_resolution_code
        milestone = self._record_milestone("resolved")
        self._done = True
        self._adjust_mood(positive=True)
        return 0.3, milestone, "Ticket was resolved with the required remediation details."

    def _handle_escalate(self, action: Action) -> Tuple[float, Optional[str], str]:
        text = f"{action.response_text or ''} {action.notes or ''}".lower()
        if self._active_task.preferred_final_action != "escalate":
            self._internal_state["unnecessary_escalation"] = True
            self._mistakes.append("unnecessary_escalation")
            self._adjust_mood(negative=True)
            self._internal_state["status"] = WorkflowStatus.ESCALATED.value
            self._internal_state["resolution_code"] = "unnecessary_escalation"
            self._done = True
            return -0.2, None, "Issue was escalated even though it could be resolved directly."

        if not self._internal_state["customer_replied"] and self._active_task.requires_customer_info:
            self._mistakes.append("premature_escalation")
            self._adjust_mood(negative=True)
            return -0.08, None, "Escalation happened before collecting the missing diagnostic details."

        keyword_hits = sum(1 for phrase in self._active_task.escalation_keywords if phrase.lower() in text)
        response_hits = sum(1 for phrase in self._active_task.response_requirements if phrase.lower() in text)
        justification_hits = sum(
            1 for phrase in self._active_task.escalation_justification_keywords if phrase.lower() in text
        )
        if keyword_hits == 0 and response_hits == 0 and justification_hits == 0:
            self._mistakes.append("thin_escalation")
            self._adjust_mood(negative=True)
            return -0.1, None, "Escalation note lacked ownership, impact, or customer guidance."

        self._internal_state["status"] = WorkflowStatus.ESCALATED.value
        self._internal_state["resolution_code"] = self._active_task.success_resolution_code
        milestone = self._record_milestone("escalated")
        self._done = True
        if self._internal_state["customer_mood"] == CustomerMood.ANGRY.value:
            self._adjust_mood(positive=True)
        return 0.2, milestone, "Incident was escalated with the correct severity, ownership, and customer context."

    def _simulate_customer_reply(self, high_quality_question: bool) -> None:
        assert self._active_task is not None
        if high_quality_question:
            prefix = (
                "Thanks for asking something specific. "
                if self._internal_state["customer_mood"] != CustomerMood.ANGRY.value
                else "We are under pressure, but here are the details you asked for. "
            )
            suffix = (
                "We are under real deadline pressure and need a concrete next step."
                if self._active_task.task_id == "hard_enterprise_incident"
                else "Please use this to move the case forward."
            )
            body = prefix + "; ".join(
                f"{key.replace('_', ' ')}: {value}"
                for key, value in self._active_task.customer_info_responses.items()
            ) + f". {suffix}"
            for key, value in self._active_task.customer_info_responses.items():
                self._internal_state["known_info"][key] = value
            self._internal_state["customer_replied"] = True
            self._internal_state["status"] = WorkflowStatus.IN_PROGRESS.value
            self._adjust_mood(positive=False)
        else:
            body = (
                "We already said it is affecting multiple offices and finance close is blocked. "
                "What exact detail do you still need from us right now?"
            )
            self._adjust_mood(negative=True)

        self._internal_state["conversation"].append(
            Message(sender="customer", body=body, timestamp="2026-03-29T05:19:00Z")
        )

    def _append_agent_message(self, text: str, timestamp: str) -> None:
        self._internal_state["conversation"].append(Message(sender="agent", body=text, timestamp=timestamp))

    def _looks_like_keyword_spam(self, text: str) -> bool:
        lowered = text.lower()
        words = [word.strip(".,!?") for word in lowered.split() if word.strip(".,!?")]
        if len(words) < 6:
            return False
        repeated_phrases = max(lowered.count("status page"), lowered.count("refund"), lowered.count("workaround"))
        unique_ratio = len(set(words)) / len(words)
        return repeated_phrases >= 3 or unique_ratio < 0.45

    def _adjust_mood(self, negative: bool = False, positive: bool = False) -> None:
        mood = CustomerMood(self._internal_state["customer_mood"])
        if negative:
            if mood == CustomerMood.CALM:
                mood = CustomerMood.FRUSTRATED
            elif mood == CustomerMood.FRUSTRATED:
                mood = CustomerMood.ANGRY
        elif positive:
            if mood == CustomerMood.ANGRY:
                mood = CustomerMood.FRUSTRATED
            elif mood == CustomerMood.FRUSTRATED:
                mood = CustomerMood.CALM
        self._internal_state["customer_mood"] = mood.value

    def _record_milestone(self, name: str) -> Optional[str]:
        if name in self._completed_milestones:
            return None
        self._completed_milestones.add(name)
        return name

    def _next_task_id(self) -> str:
        task_id = self._task_order[self._task_pointer % len(self._task_order)]
        self._task_pointer += 1
        return task_id

    def _public_workflow_hint(self) -> str:
        assert self._active_task is not None
        if self._active_task.requires_customer_info and not self._internal_state["customer_replied"]:
            return "This case may require clarification before a safe final decision."
        return "Work through triage, communication, and a justified final action."

    def _sla_bucket(self) -> str:
        remaining = self._internal_state["sla_remaining"]
        if remaining <= 30:
            return "critical"
        if remaining <= 90:
            return "warning"
        return "healthy"

    def _info(self, progress: str) -> Dict[str, Any]:
        assert self._active_task is not None
        return {
            "progress": progress,
            "mistakes": list(self._mistakes),
            "current_stage": self._internal_state["status"],
            "sla_remaining": self._internal_state["sla_remaining"],
            "notes": self._internal_state["last_debug_note"],
            "deterministic_score": grade_task(self._active_task.task_id, self.state()) if self._done else None,
        }
