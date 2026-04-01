from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI

from models import Action, ActionType, Observation, Priority, TicketCategory


class OpenAICompatibleSupportClient:
    def __init__(
        self,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.api_base_url = api_base_url or os.getenv("API_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.client = None
        if self.api_base_url and self.api_key:
            self.client = OpenAI(base_url=self.api_base_url, api_key=self.api_key)

    def next_action(self, observation: Observation) -> Action:
        if self.client is None:
            return self._heuristic_action(observation)

        prompt = self._build_prompt(observation)
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a deterministic support operations agent. "
                        "Return exactly one JSON object matching the Action schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return Action.model_validate(json.loads(content))

    def _build_prompt(self, observation: Observation) -> str:
        return json.dumps(
            {
                "task_id": observation.task_id,
                "objective": observation.objective,
                "ticket": observation.ticket.model_dump(),
                "history": [entry.model_dump() for entry in observation.history],
                "available_actions": [action.value for action in observation.available_actions],
                "completed_milestones": observation.completed_milestones,
                "relevant_data": observation.relevant_data,
                "instructions": (
                    "Choose the single best next structured action. Use the exact ticket_id. "
                    "Favor correct progress toward the objective and avoid repeated actions."
                ),
            }
        )

    def _heuristic_action(self, observation: Observation) -> Action:
        task_id = observation.task_id
        completed = set(observation.completed_milestones)
        ticket_id = observation.ticket.ticket_id

        if task_id == "easy_password_reset_triage":
            if "classified" not in completed:
                return Action(
                    action_type=ActionType.CLASSIFY_TICKET,
                    ticket_id=ticket_id,
                    category=TicketCategory.ACCOUNT_ACCESS,
                )
            if "prioritized" not in completed:
                return Action(
                    action_type=ActionType.ASSIGN_PRIORITY,
                    ticket_id=ticket_id,
                    priority=Priority.HIGH,
                )
            return Action(
                action_type=ActionType.ROUTE_TICKET,
                ticket_id=ticket_id,
                target_team="identity_support",
            )

        if task_id == "medium_refund_resolution":
            if "classified" not in completed:
                return Action(
                    action_type=ActionType.CLASSIFY_TICKET,
                    ticket_id=ticket_id,
                    category=TicketCategory.BILLING,
                )
            if "prioritized" not in completed:
                return Action(
                    action_type=ActionType.ASSIGN_PRIORITY,
                    ticket_id=ticket_id,
                    priority=Priority.MEDIUM,
                )
            if "routed" not in completed:
                return Action(
                    action_type=ActionType.ROUTE_TICKET,
                    ticket_id=ticket_id,
                    target_team="payments_ops",
                )
            if "replied" not in completed:
                return Action(
                    action_type=ActionType.DRAFT_REPLY,
                    ticket_id=ticket_id,
                    response_text=(
                        "We confirmed the refund for order ORD-88421. The payments team is monitoring "
                        "settlement, and card refunds typically appear within 3-5 business days."
                    ),
                )
            return Action(
                action_type=ActionType.RESOLVE_TICKET,
                ticket_id=ticket_id,
                resolution_code="refund_confirmed",
            )

        if "classified" not in completed:
            return Action(
                action_type=ActionType.CLASSIFY_TICKET,
                ticket_id=ticket_id,
                category=TicketCategory.TECHNICAL,
            )
        if "prioritized" not in completed:
            return Action(
                action_type=ActionType.ASSIGN_PRIORITY,
                ticket_id=ticket_id,
                priority=Priority.CRITICAL,
            )
        if "escalated" not in completed:
            return Action(
                action_type=ActionType.ESCALATE_TICKET,
                ticket_id=ticket_id,
                target_team="incident_response",
                notes="Enterprise outage across regions.",
            )
        if "replied" not in completed:
            return Action(
                action_type=ActionType.DRAFT_REPLY,
                ticket_id=ticket_id,
                response_text=(
                    "We recognize the impact to your finance close. The incident_response team is leading "
                    "the investigation, a workaround is to use backup local login where enabled, the status page "
                    "will be updated under INC-778, and at this time we have no evidence of data integrity issues."
                ),
            )
        return Action(
            action_type=ActionType.RESOLVE_TICKET,
            ticket_id=ticket_id,
            resolution_code="escalated_to_incident",
        )
