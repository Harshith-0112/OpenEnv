from __future__ import annotations

import json
import os
from typing import Dict, Tuple

from openai import OpenAI

from env import CustomerSupportEnv
from graders import grade_task
from models import Action, ActionType, Observation, Priority, TicketCategory


class WeakBaselineAgent:
    def __init__(self) -> None:

        self.api_base_url = os.getenv("API_BASE_URL")
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.client = (
            OpenAI(base_url=self.api_base_url, api_key=self.api_key)
            if self.api_base_url and self.api_key
            else None
        )
        self.task_attempts: Dict[str, int] = {}

    def next_action(self, observation: Observation) -> Tuple[Action, str]:
        self.task_attempts[observation.task_id] = self.task_attempts.get(observation.task_id, 0) + 1

        # Primary deterministic workflow
        action, rationale = self._rule_based_action(observation)

        # Optional LLM polishing for customer-facing text only
        if (
            self.client is not None
            and hasattr(action, "response_text")
            and action.response_text
            and action.action_type in {
                ActionType.RESPOND,
                ActionType.RESOLVE,
                ActionType.ESCALATE,
            }
        ):
            try:
                action.response_text = self._rewrite_text(
                    original_text=action.response_text,
                    observation=observation,
                )
            except Exception:
                pass

        return action, rationale

    def _llm_action(self, observation: Observation) -> Tuple[Action | None, str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a capable but imperfect junior support agent. Return a single JSON object "
                            "containing one valid Action plus a short rationale string."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "objective": observation.objective,
                                "status": observation.current_status.value,
                                "customer_mood": observation.customer_mood.value,
                                "ticket": observation.ticket.model_dump(),
                                "conversation_history": [msg.model_dump() for msg in observation.conversation_history],
                                "available_actions": [item.value for item in observation.available_actions],
                                "workflow_hint": observation.relevant_data.get("workflow_hint"),
                                "known_customer_facts": observation.relevant_data.get("known_customer_facts"),
                                "instruction": (
                                    "Choose a plausible next action. Do not assume hidden facts. This agent should "
                                    "feel competent but imperfect."
                                ),
                            }
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            rationale = payload.pop("rationale", "LLM selected a plausible next action.")
            return Action.model_validate(payload), rationale
        except Exception:
            return None, "LLM baseline unavailable; using deterministic local rules."
        
    def _rewrite_text(self, original_text: str, observation: Observation) -> str:
        if self.client is None:
            return original_text

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.3,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are polishing a customer support message written by a junior agent. "
                            "Keep it concise, professional, empathetic, and clear. "
                            "Do not invent facts. Preserve the original meaning."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "ticket_subject": observation.ticket.subject,
                                "customer_mood": observation.customer_mood.value,
                                "original_text": original_text,
                            }
                        ),
                    },
                ],
            )

            rewritten = response.choices[0].message.content
            return rewritten.strip() if rewritten else original_text

        except Exception:
            return original_text

    def _rule_based_action(self, observation: Observation) -> Tuple[Action, str]:
        task_id = observation.task_id
        turn = self.task_attempts[task_id]
        completed = set(observation.completed_milestones)
        ticket_id = observation.ticket.ticket_id
        visible_text = self._visible_text(observation)
        category = self._infer_category(visible_text)
        current_priority = self._infer_priority(observation, category)

        if "classified" not in completed:
            return (
                Action(action_type=ActionType.CLASSIFY_TICKET, ticket_id=ticket_id, category=category),
                "I am classifying from the subject line and the latest customer wording.",
            )

        if "prioritized" not in completed:
            if turn == 2:
                return (
                    Action(action_type=ActionType.SET_PRIORITY, ticket_id=ticket_id, priority=current_priority),
                    "I am setting urgency from the latest message tone and visible business impact.",
                )

        if self._should_request_info(observation, category, completed):
            return (
                Action(
                    action_type=ActionType.REQUEST_INFO,
                    ticket_id=ticket_id,
                    question="Can you share the exact error code you are seeing?",
                ),
                "I know I need more detail, but I am only asking for the first missing clue that stands out.",
            )

        if self._should_resolve(observation, category, completed, turn):
            return self._resolve_action(ticket_id, category, observation)

        if self._should_escalate(observation, category, completed, turn):
            return (
                Action(
                    action_type=ActionType.ESCALATE,
                    ticket_id=ticket_id,
                    response_text=(
                        "We recognize the impact and are keeping the status page active while we coordinate next steps."
                    ),
                    notes="customer impact is broad and internal review may be needed",
                ),
                "This feels bigger than a normal case, but my escalation note is still a little junior-level.",
            )

        return self._respond_action(ticket_id, category, observation, turn)

    def _visible_text(self, observation: Observation) -> str:
        parts = [
            observation.ticket.subject,
            observation.ticket.description,
            " ".join(message.body for message in observation.conversation_history[-3:]),
        ]
        return " ".join(parts).lower()

    def _infer_category(self, visible_text: str) -> TicketCategory:
        if any(term in visible_text for term in ["refund", "charged", "card", "order"]):
            return TicketCategory.BILLING

        if any(
            term in visible_text
            for term in [
                "security",
                "suspicious login",
                "unknown login",
                "compromised",
                "breach",
                "unauthorized",
                "foreign login",
            ]
        ):
            return TicketCategory.SECURITY

        if any(term in visible_text for term in ["sso", "outage", "status page", "error code", "callback"]):
            return TicketCategory.TECHNICAL

        if any(term in visible_text for term in ["login", "locked", "password", "mfa"]):
            return TicketCategory.ACCOUNT_ACCESS

        return TicketCategory.TECHNICAL

    def _infer_priority(self, observation: Observation, category: TicketCategory) -> Priority:
        visible_text = self._visible_text(observation)
        mood = observation.customer_mood.value

        if category == TicketCategory.TECHNICAL and any(
            term in visible_text
            for term in [
                "hundreds",
                "blocked",
                "quarter-end",
                "all users",
                "multiple regions",
                "outage",
                "status page",
                "sso",
            ]
        ):
            return Priority.CRITICAL

        if category == TicketCategory.SECURITY:
            return Priority.HIGH

        if category == TicketCategory.BILLING and mood in {"frustrated", "angry"}:
            return Priority.HIGH

        if category == TicketCategory.ACCOUNT_ACCESS:
            return Priority.HIGH

        if category == TicketCategory.BILLING:
            return Priority.MEDIUM

        return Priority.MEDIUM

    def _should_request_info(
        self, observation: Observation, category: TicketCategory, completed: set[str]
    ) -> bool:
        visible_text = self._visible_text(observation)
        return (
            category == TicketCategory.TECHNICAL
            and "info_requested" not in completed
            and ("partial logs" in visible_text or "workflow_hint" in str(observation.relevant_data))
        )

    def _should_resolve(
    self, observation: Observation, category: TicketCategory, completed: set[str], turn: int
) -> bool:
        if category == TicketCategory.ACCOUNT_ACCESS and "responded" in completed:
            return True

        # Delay security resolution until after at least one follow-up turn
        if category == TicketCategory.SECURITY:
            if "responded" in completed and turn >= 5:
                return True
            return False

        if category == TicketCategory.BILLING and turn >= 4:
            return True

        return False

    def _should_escalate(
    self, observation: Observation, category: TicketCategory, completed: set[str], turn: int
) -> bool:
        if category != TicketCategory.TECHNICAL:
            return False
        if "info_requested" in completed and "responded" in completed and turn >= 9:
            return True
        return False

    def _respond_action(
    self, ticket_id: str, category: TicketCategory, observation: Observation, turn: int
) -> Tuple[Action, str]:
        if category == TicketCategory.ACCOUNT_ACCESS:
            return (
                Action(
                    action_type=ActionType.RESPOND,
                    ticket_id=ticket_id,
                    response_text="I can help restore access. Your account is locked and we should get you back in soon.",
                ),
                "A short reassurance should be enough because the issue looks straightforward.",
            )

        if category == TicketCategory.SECURITY:
            if turn >= 4:
                return (
                    Action(
                        action_type=ActionType.RESPOND,
                        ticket_id=ticket_id,
                        response_text=(
                            "We have started securing the account. "
                            "Please confirm once you reset your password and enabled MFA. "
                            "We can then revoke active sessions and finalize the recovery."
                        ),
                    ),
                    "I want confirmation that the customer completed the security steps before closing the case.",
                )

            return (
                Action(
                    action_type=ActionType.RESPOND,
                    ticket_id=ticket_id,
                    response_text=(
                        "We can help secure the account. Please reset your password immediately, "
                        "enable MFA, and we can revoke active sessions if needed."
                    ),
                ),
                "I recognize this as a possible account compromise and want to give immediate security guidance.",
            )

        if category == TicketCategory.BILLING:
            return (
                Action(
                    action_type=ActionType.RESPOND,
                    ticket_id=ticket_id,
                    response_text="I checked the refund and it should appear in 3-5 business days.",
                ),
                "I am giving the likely billing timeline, but I may be skipping some detail the customer wanted.",
            )

        if turn >= 7:
            return (
                Action(
                    action_type=ActionType.RESPOND,
                    ticket_id=ticket_id,
                    response_text="We are still checking the status page.",
                ),
                "I am running out of time and defaulting to a thin update instead of taking the stronger escalation step.",
            )

        if turn % 2 == 0:
            text = "We understand the impact. There may be a workaround and we are checking the status page."
            rationale = "I am trying to calm the customer with a partial incident update before I am fully confident."
        else:
            text = "We are checking the status page and reviewing the new diagnostic detail while looking for a workaround."
            rationale = "I am hesitating after the clarification turn instead of escalating immediately."

        return Action(action_type=ActionType.RESPOND, ticket_id=ticket_id, response_text=text), rationale

    def _resolve_action(
        self, ticket_id: str, category: TicketCategory, observation: Observation
    ) -> Tuple[Action, str]:
        if category == TicketCategory.ACCOUNT_ACCESS:
            return (
                Action(
                    action_type=ActionType.RESOLVE,
                    ticket_id=ticket_id,
                    response_text="Please use the reset link again after 15 minutes and your account should unlock.",
                    resolution_code="unlock_after_15_minutes_reset_link",
                ),
                "I have enough confidence to close this with standard unlock guidance.",
            )

        if category == TicketCategory.SECURITY:
            return (
                Action(
                    action_type=ActionType.RESOLVE,
                    ticket_id=ticket_id,
                    response_text=(
                        "We completed the password reset, revoke sessions procedure, and MFA protection update. "
                        "The suspicious login attempt was flagged and your account is now secure."
                    ),
                    resolution_code="security_lockdown_completed"
                ),
                "The required recovery and security actions are complete, so I can safely close the case.",
            )

        return (
            Action(
                action_type=ActionType.RESOLVE,
                ticket_id=ticket_id,
                response_text="The refund is in flight and should settle in 3-5 business days.",
                resolution_code="refund_timeline_confirmed_ord-88421",
            ),
            "I think this is solvable now, even though my customer update was a bit thin.",
        )


def run_all_tasks() -> None:
    env = CustomerSupportEnv()
    agent = WeakBaselineAgent()

    print("Running finalized customer support OpenEnv baseline inference")
    for task_id in env.tasks:
        observation = env.reset(task_id=task_id)
        done = False
        print(f"\nTask: {task_id}")
        while not done:
            action, rationale = agent.next_action(observation)
            observation, reward, done, info = env.step(action)
            print(
                f"  step={observation.steps_taken} stage={observation.current_status.value} "
                f"mood={observation.customer_mood.value} action={action.action_type.value} "
                f"reward={reward.score:.3f} cumulative={reward.cumulative_score:.3f}"
            )
            print(
                f"    rationale={rationale} progress={info['progress']} mistakes={info['mistakes']} "
                f"sla_remaining={info['sla_remaining']} notes={info['notes']}"
            )

        final_state = env.state()
        final_score = grade_task(task_id, final_state)
        print(f"  final_state={final_state}")
        print(f"  deterministic_score={final_score:.3f}")


if __name__ == "__main__":
    run_all_tasks()
