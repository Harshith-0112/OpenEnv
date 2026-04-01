# Customer Support Ticket Resolution OpenEnv

This project is a realistic OpenEnv-compatible environment for multi-step customer support operations. It is designed to feel like handling a real customer under pressure rather than classifying a ticket once and stopping. Agents must operate with incomplete information, hidden ground truth, dynamic customer mood, strict action sequencing, and deterministic evaluation.

The current benchmark includes 4 deterministic tasks: one easy, two medium, and one hard.

## Why this environment is realistic

Real support work is rarely a one-shot decision. Agents must:

- infer the true issue from noisy customer language
- decide how urgent the case really is
- ask for missing information when it is unsafe to proceed
- maintain a professional conversation as the customer becomes more frustrated
- choose between solving directly and escalating to a higher-severity workflow
- work under SLA pressure where every extra turn has cost

This environment models those realities explicitly.

## What makes the environment stronger

The upgraded environment includes:

- hidden state not exposed in the observation
- multi-stage workflow states: `open`, `needs_info`, `in_progress`, `resolved`, `escalated`
- customer simulation for clarification turns
- mood transitions from `calm` to `frustrated` to `angry`
- action constraints that penalize invalid sequences
- non-binary reward shaping over the full trajectory
- advanced deterministic graders that score more than final correctness
- exploit-resistant grading that penalizes spammy or one-strategy behavior
- an intentionally imperfect baseline so strong agents still have headroom

## Hidden ground truth

Each task contains internal fields that are only visible through `state()` and grading, not through the observation:

- `true_issue_type`
- `required_resolution_steps`
- `correct_priority`
- `sla_deadline`
- `customer_persona`

This means an agent must reason from observable signals rather than being spoon-fed the answer.

## Workflow

The environment enforces a realistic staged workflow:

1. `open`
2. `needs_info`
3. `in_progress`
4. `resolved`
5. `escalated`

Important rules:

- the agent cannot resolve before classification
- priority cannot be set before classification
- escalation requires justification, not just a blind hand-off
- the hard task requires a clarification turn before safe escalation
- invalid action order is penalized
- customer replies are simulated deterministically by the environment

## Action space

Supported actions are structured rather than free-form:

```json
{
  "action_type": "request_info",
  "ticket_id": "TCK-9007",
  "question": "Can you share the exact error code and impacted regions?"
}
```

Available actions:

- `classify_ticket`
- `set_priority`
- `request_info`
- `respond`
- `resolve`
- `escalate`

## Observation space

Each observation includes:

- ticket info
- full visible conversation history
- current workflow status
- last agent action
- dynamic customer mood
- public knowledge base context
- currently known customer facts
- available actions

The observation does not reveal hidden ground truth directly.

## Reward shaping

Reward is trajectory-based and incremental.

Positive reward:

- `+0.2` correct classification
- `+0.1` correct priority
- `+0.2` good clarification question
- `+0.3` correct resolution
- `+0.2` correct escalation

Penalties:

- `-0.1` invalid action or invalid sequence
- `-0.1` repeated action
- `-0.05` per step due to SLA pressure
- `-0.2` hallucinated resolution
- `-0.2` unnecessary escalation
- `-0.2` premature resolution
- extra penalties for weak responses and exploit-like spam

This gives learning signal across the whole trajectory rather than only at the end.

## Tasks

### Easy: Password Reset Lockout

The issue is solvable directly. Good agents classify correctly, set urgency, reassure the customer, and resolve without unnecessary escalation.

### Medium: Delayed Refund Investigation

The issue is still resolvable, but customer communication matters more. The agent must explain the refund timeline clearly and avoid overreacting.

### Medium: Suspicious Login Alert

This task adds a security-oriented workflow. The agent must recognize account risk, set appropriate urgency, guide the customer through protection steps, and avoid closing the case before the recovery path is complete.

### Hard: Enterprise SSO Outage Under Close Deadline

This is the most realistic task:

- information is missing at the start
- the customer is already frustrated
- the agent must ask a strong clarification question
- the customer replies with new diagnostic facts
- the agent must decide escalation versus local resolution
- an incorrect local resolution is treated as hallucination
- tight SLA pressure punishes indecisive trajectories

The hard task is intentionally not solvable in one step.

## Grading

Each task is scored deterministically in `[0.0, 1.0]` using:

- classification accuracy
- priority correctness
- resolution or escalation correctness
- reasoning-step coverage
- response quality
- efficiency relative to the step budget

The graders are intentionally harder to game:

- always escalating is penalized
- always resolving is penalized
- repetitive keyword stuffing is penalized
- repeating the same action pattern too often is penalized

## Baseline

`inference.py` uses a weak baseline:

- optional OpenAI-compatible prompting when `API_BASE_URL`, `MODEL_NAME`, and `OPENAI_API_KEY` are set
- otherwise a deterministic but imperfect rule-based policy built from visible ticket text, recent conversation turns, and mood cues
- the baseline prints short rationales so you can see its reasoning attempts

Baseline philosophy:

- it classifies mostly from keywords in the visible ticket and recent conversation
- it overweights recent tone when setting priority, so frustrated customers can get over-prioritized
- it asks only one obvious clarification question rather than exhaustively gathering context
- it gives partial but plausible customer updates
- it sometimes hesitates before escalating a hard case
- it is deterministic, so results are reproducible run to run

Expected rough score bands for the primary benchmark:

- easy: `~0.8-1.0`
- medium: `~0.5-0.8`
- hard: `~0.3-0.5`

This gives the environment meaningful headroom for stronger agents.

## Run

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Run the baseline:

```bash
python inference.py
```

Run tests:

```bash
pytest
```

## API

The FastAPI server exposes:

- `POST /reset`
- `POST /step`
- `GET /state`

These follow the OpenEnv pattern:

- `reset() -> observation`
- `step(action) -> observation, reward, done, info`
- `state() -> internal state`

## Files

- [`env.py`](/Users/harshiththota/Downloads/Scaler/env.py)
- [`models.py`](/Users/harshiththota/Downloads/Scaler/models.py)
- [`tasks.py`](/Users/harshiththota/Downloads/Scaler/tasks.py)
- [`graders.py`](/Users/harshiththota/Downloads/Scaler/graders.py)
- [`inference.py`](/Users/harshiththota/Downloads/Scaler/inference.py)
- [`openenv.yaml`](/Users/harshiththota/Downloads/Scaler/openenv.yaml)
