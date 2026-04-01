from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketCategory(str, Enum):
    ACCOUNT_ACCESS = "account_access"
    BILLING = "billing"
    SHIPPING = "shipping"
    TECHNICAL = "technical"
    SECURITY = "security"


class WorkflowStatus(str, Enum):
    OPEN = "open"
    NEEDS_INFO = "needs_info"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class CustomerMood(str, Enum):
    CALM = "calm"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


class ActionType(str, Enum):
    CLASSIFY_TICKET = "classify_ticket"
    SET_PRIORITY = "set_priority"
    REQUEST_INFO = "request_info"
    RESPOND = "respond"
    RESOLVE = "resolve"
    ESCALATE = "escalate"


class CustomerProfile(BaseModel):
    customer_id: str
    name: str
    tier: Literal["standard", "premium", "enterprise"]
    region: str
    lifetime_value: int


class Message(BaseModel):
    sender: Literal["customer", "agent", "system"]
    body: str
    timestamp: str


class KnowledgeArticle(BaseModel):
    article_id: str
    title: str
    summary: str


class Ticket(BaseModel):
    ticket_id: str
    subject: str
    description: str
    opened_at: str
    current_status: WorkflowStatus = WorkflowStatus.OPEN
    category_hint: Optional[str] = None
    customer: CustomerProfile
    messages: List[Message]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    action_type: ActionType
    ticket_id: str
    category: Optional[TicketCategory] = None
    priority: Optional[Priority] = None
    response_text: Optional[str] = None
    resolution_code: Optional[str] = None
    question: Optional[str] = None
    notes: Optional[str] = None


class HistoryEntry(BaseModel):
    step: int
    action: Action
    reward_delta: float
    milestone: Optional[str] = None
    status_after: WorkflowStatus
    customer_mood_after: CustomerMood


class Observation(BaseModel):
    task_id: str
    difficulty: Difficulty
    objective: str
    ticket: Ticket
    current_status: WorkflowStatus
    conversation_history: List[Message]
    history: List[HistoryEntry]
    relevant_data: Dict[str, Any]
    available_actions: List[ActionType]
    completed_milestones: List[str]
    last_agent_action: Optional[Action] = None
    customer_mood: CustomerMood
    steps_taken: int
    max_steps: int


class Reward(BaseModel):
    score: float = Field(..., ge=-1.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    cumulative_score: float = 0.0
    explanation: str

    @field_validator("components")
    @classmethod
    def validate_components(cls, value: Dict[str, float]) -> Dict[str, float]:
        for component, score in value.items():
            if score < -1.0 or score > 1.0:
                raise ValueError(f"Reward component {component} must be between -1.0 and 1.0.")
        return value


class TaskDefinition(BaseModel):
    task_id: str
    difficulty: Difficulty
    title: str
    objective: str
    ticket: Ticket
    knowledge_base: List[KnowledgeArticle]
    allowed_actions: List[ActionType]
    max_steps: int
    true_issue_type: TicketCategory
    correct_priority: Priority
    required_resolution_steps: List[str]
    requires_customer_info: bool = False
    missing_info_keys: List[str] = Field(default_factory=list)
    customer_info_responses: Dict[str, str] = Field(default_factory=dict)
    resolution_keywords: List[str] = Field(default_factory=list)
    escalation_keywords: List[str] = Field(default_factory=list)
    escalation_justification_keywords: List[str] = Field(default_factory=list)
    response_requirements: List[str] = Field(default_factory=list)
    preferred_final_action: Literal["resolve", "escalate"]
    sla_deadline: int
    customer_persona: CustomerMood
    success_resolution_code: str
    intermediate_milestones: List[str] = Field(default_factory=list)
