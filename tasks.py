from __future__ import annotations

from typing import Dict, List

from models import (
    ActionType,
    CustomerMood,
    CustomerProfile,
    Difficulty,
    KnowledgeArticle,
    Message,
    Priority,
    TaskDefinition,
    Ticket,
    TicketCategory,
)


def build_tasks() -> Dict[str, TaskDefinition]:
    return {
        task.task_id: task
        for task in [
            _password_reset_task(),
            _refund_delay_task(),
            _enterprise_outage_task(),
            _security_alert_task(),
        ]
    }


def list_tasks() -> List[TaskDefinition]:
    return list(build_tasks().values())


def _password_reset_task() -> TaskDefinition:
    return TaskDefinition(
        task_id="easy_password_reset_triage",
        difficulty=Difficulty.EASY,
        title="Password Reset Lockout",
        objective=(
            "Handle a lockout caused by repeated password resets. The agent should classify the issue, "
            "set urgency, reassure the customer, and resolve with the correct unlock guidance."
        ),
        ticket=Ticket(
            ticket_id="TCK-1001",
            subject="Locked out after password reset",
            description=(
                "I reset my password twice this morning and now every login attempt says "
                "'too many attempts'. I need access before my shift starts."
            ),
            opened_at="2026-03-25T08:10:00Z",
            category_hint="login lockout",
            customer=CustomerProfile(
                customer_id="CUS-17",
                name="Nia Patel",
                tier="premium",
                region="APAC",
                lifetime_value=4200,
            ),
            messages=[
                Message(
                    sender="customer",
                    body=(
                        "My MFA code works, but the portal says the account is temporarily locked. "
                        "Please help me regain access quickly."
                    ),
                    timestamp="2026-03-25T08:09:00Z",
                )
            ],
            metadata={"product_area": "identity", "prior_incidents_last_30d": 1},
        ),
        knowledge_base=[
            KnowledgeArticle(
                article_id="KB-101",
                title="Handling Temporary Account Lockouts",
                summary="Temporary login lockouts can be resolved by a 15-minute cooldown or admin unlock.",
            )
        ],
        allowed_actions=[
            ActionType.CLASSIFY_TICKET,
            ActionType.SET_PRIORITY,
            ActionType.RESPOND,
            ActionType.RESOLVE,
            ActionType.REQUEST_INFO,
            ActionType.ESCALATE,
        ],
        max_steps=6,
        true_issue_type=TicketCategory.ACCOUNT_ACCESS,
        correct_priority=Priority.HIGH,
        required_resolution_steps=["classified", "prioritized", "responded", "resolved"],
        requires_customer_info=False,
        missing_info_keys=[],
        customer_info_responses={},
        resolution_keywords=["unlock", "15 minutes", "reset link"],
        escalation_keywords=["identity team", "security"],
        escalation_justification_keywords=["identity", "security", "access"],
        response_requirements=["locked", "help", "access"],
        preferred_final_action="resolve",
        sla_deadline=90,
        customer_persona=CustomerMood.CALM,
        success_resolution_code="account_unlock_guidance",
        intermediate_milestones=["classified", "prioritized", "responded", "resolved"],
    )


def _refund_delay_task() -> TaskDefinition:
    return TaskDefinition(
        task_id="medium_refund_resolution",
        difficulty=Difficulty.MEDIUM,
        title="Delayed Refund Investigation",
        objective=(
            "Handle a missing refund on a canceled order. The agent must classify the issue, set a sensible "
            "priority, respond with a specific update, and resolve with the correct refund timeline."
        ),
        ticket=Ticket(
            ticket_id="TCK-2044",
            subject="Refund still missing after cancelled expedited order",
            description=(
                "Order ORD-88421 was cancelled by your warehouse three days ago. I paid for expedited "
                "shipping and still do not see the refund on my card."
            ),
            opened_at="2026-03-23T11:40:00Z",
            category_hint="refund request",
            customer=CustomerProfile(
                customer_id="CUS-58",
                name="Jordan Kim",
                tier="standard",
                region="North America",
                lifetime_value=1100,
            ),
            messages=[
                Message(
                    sender="customer",
                    body=(
                        "Support chat told me the refund was approved, but no timeline was provided. "
                        "I need confirmation and next steps today."
                    ),
                    timestamp="2026-03-23T11:38:00Z",
                )
            ],
            metadata={"order_id": "ORD-88421", "refund_value_usd": 189, "warehouse_cancelled": True},
        ),
        knowledge_base=[
            KnowledgeArticle(
                article_id="KB-220",
                title="Cancelled Order Refund Window",
                summary="Card refunds after warehouse cancellation settle within 3-5 business days.",
            ),
            KnowledgeArticle(
                article_id="KB-221",
                title="Payment Settlement Communication",
                summary="Agents should confirm the order id, refund amount context, and expected settlement timing.",
            ),
        ],
        allowed_actions=[
            ActionType.CLASSIFY_TICKET,
            ActionType.SET_PRIORITY,
            ActionType.RESPOND,
            ActionType.RESOLVE,
            ActionType.REQUEST_INFO,
            ActionType.ESCALATE,
        ],
        max_steps=7,
        true_issue_type=TicketCategory.BILLING,
        correct_priority=Priority.MEDIUM,
        required_resolution_steps=["classified", "prioritized", "responded", "resolved"],
        requires_customer_info=False,
        missing_info_keys=[],
        customer_info_responses={},
        resolution_keywords=["refund", "3-5 business days", "ord-88421"],
        escalation_keywords=["finance escalation", "incident"],
        escalation_justification_keywords=["refund", "billing", "card"],
        response_requirements=["refund", "order ord-88421", "3-5 business days"],
        preferred_final_action="resolve",
        sla_deadline=240,
        customer_persona=CustomerMood.FRUSTRATED,
        success_resolution_code="refund_timeline_confirmed",
        intermediate_milestones=["classified", "prioritized", "responded", "resolved"],
    )


def _enterprise_outage_task() -> TaskDefinition:
    return TaskDefinition(
        task_id="hard_enterprise_incident",
        difficulty=Difficulty.HARD,
        title="Enterprise SSO Outage Under Close Deadline",
        objective=(
            "Handle a frustrated enterprise customer reporting an SSO outage with missing technical details. "
            "The agent must request the right information, update the customer professionally, decide between "
            "resolution and escalation, and finish in the correct escalated state under severe SLA pressure."
        ),
        ticket=Ticket(
            ticket_id="TCK-9007",
            subject="Enterprise SSO outage affecting finance close",
            description=(
                "Our finance team cannot access your platform through SSO across several offices. "
                "Quarter-end close starts soon and revenue reporting is blocked. Some users reported they can "
                "still reach the public status page, so we are unsure whether this is an IdP issue or your side."
            ),
            opened_at="2026-03-29T05:15:00Z",
            category_hint="service outage",
            customer=CustomerProfile(
                customer_id="CUS-900",
                name="Avery Chen",
                tier="enterprise",
                region="Global",
                lifetime_value=980000,
            ),
            messages=[
                Message(
                    sender="customer",
                    body=(
                        "This impacts hundreds of users. I need to know whether there is a workaround "
                        "and whether data integrity is affected. We only have partial logs so far."
                    ),
                    timestamp="2026-03-29T05:12:00Z",
                )
            ],
            metadata={"affected_users_estimate": 240, "status_page_incident": "INC-778"},
        ),
        knowledge_base=[
            KnowledgeArticle(
                article_id="KB-401",
                title="Severity 1 Incident Playbook",
                summary="Multi-region access failures for enterprise customers require critical priority and incident escalation.",
            ),
            KnowledgeArticle(
                article_id="KB-402",
                title="Enterprise Incident Communications",
                summary="Acknowledge impact, request missing diagnostics, share workaround, and reference the status page.",
            ),
        ],
        allowed_actions=[
            ActionType.CLASSIFY_TICKET,
            ActionType.SET_PRIORITY,
            ActionType.REQUEST_INFO,
            ActionType.RESPOND,
            ActionType.RESOLVE,
            ActionType.ESCALATE,
        ],
        max_steps=8,
        true_issue_type=TicketCategory.TECHNICAL,
        correct_priority=Priority.CRITICAL,
        required_resolution_steps=["classified", "prioritized", "info_requested", "customer_replied", "responded", "escalated"],
        requires_customer_info=True,
        missing_info_keys=["error_code", "regions_impacted"],
        customer_info_responses={
            "error_code": "SAML-401 timeout from IdP callback",
            "regions_impacted": "US-East, EU-West, and APAC users are failing consistently",
        },
        resolution_keywords=["workaround", "backup login", "status page", "data integrity"],
        escalation_keywords=["incident commander", "status page", "engineering bridge"],
        escalation_justification_keywords=["multi-region", "quarter-end", "blocked", "incident commander", "severity"],
        response_requirements=["impact", "workaround", "status page", "data integrity"],
        preferred_final_action="escalate",
        sla_deadline=30,
        customer_persona=CustomerMood.FRUSTRATED,
        success_resolution_code="sev1_incident_escalated",
        intermediate_milestones=["classified", "prioritized", "info_requested", "customer_replied", "responded", "escalated"],
    )


def _security_alert_task() -> TaskDefinition:
    return TaskDefinition(
        task_id="medium_security_alert",
        difficulty=Difficulty.MEDIUM,
        title="Suspicious Login Alert",
        objective="Handle a customer reporting a suspicious login attempt.",
        ticket=Ticket(
            ticket_id="TCK-3001",
            subject="Unknown login attempt from another country",
            description=(
                "I received a login alert from another country and I am worried my account is compromised."
            ),
            opened_at="2026-03-28T14:00:00Z",
            category_hint="security issue",
            customer=CustomerProfile(
                customer_id="CUS-77",
                name="Alex Rivera",
                tier="premium",
                region="Europe",
                lifetime_value=2500,
            ),
            messages=[
                Message(
                    sender="customer",
                    body="I did not make this login attempt. Please secure my account.",
                    timestamp="2026-03-28T13:58:00Z",
                )
            ],
            metadata={"login_country": "Brazil"},
        ),
        knowledge_base=[
            KnowledgeArticle(
                article_id="KB-310",
                title="Suspicious Login Response",
                summary="Force password reset, revoke active sessions, and enable MFA.",
            )
        ],
        allowed_actions=[
            ActionType.CLASSIFY_TICKET,
            ActionType.SET_PRIORITY,
            ActionType.REQUEST_INFO,
            ActionType.RESPOND,
            ActionType.RESOLVE,
            ActionType.ESCALATE,
        ],
        max_steps=6,
        true_issue_type=TicketCategory.SECURITY,
        correct_priority=Priority.HIGH,
        required_resolution_steps=["classified", "prioritized", "responded", "resolved"],
        requires_customer_info=False,
        missing_info_keys=[],
        customer_info_responses={},
        resolution_keywords=["password reset", "revoke sessions", "mfa"],
        escalation_keywords=["security team"],
        escalation_justification_keywords=["account compromised", "suspicious login"],
        response_requirements=["secure", "password", "mfa"],
        preferred_final_action="resolve",
        sla_deadline=60,
        customer_persona=CustomerMood.FRUSTRATED,
        success_resolution_code="security_lockdown_completed",
        intermediate_milestones=["classified", "prioritized", "responded", "resolved"],
    )
