# Customer Support Ticket Resolution OpenEnv

This project is a realistic OpenEnv-compatible environment for multi-step customer support operations. It is designed to feel like handling a real customer under pressure rather than classifying a ticket once and stopping. Agents must operate with incomplete information, hidden ground truth, dynamic customer mood, strict action sequencing, and deterministic evaluation.

The current benchmark includes 4 deterministic tasks: one easy, two medium, and one hard.

---

## Why this environment is realistic

Real support work is rarely a one-shot decision. Agents must:

* infer the true issue from noisy customer language
* decide how urgent the case really is
* ask for missing information when it is unsafe to proceed
* maintain a professional conversation as the customer becomes more frustrated
* choose between solving directly and escalating to a higher-severity workflow
* work under SLA pressure where every extra turn has cost

This environment models those realities explicitly.

---

## What makes the environment stronger

The upgraded environment includes:

* hidden state not exposed in the observation
* multi-stage workflow states: `open`, `needs_info`, `in_progress`, `resolved`, `escalated`
* customer simulation for clarification turns
* mood transitions from `calm` to `frustrated` to `angry`
* action constraints that penalize invalid sequences
* non-binary reward shaping over the full trajectory
* advanced deterministic graders that score more than final correctness
* exploit-resistant grading that penalizes spammy or one-strategy behavior
* an intentionally imperfect baseline so strong agents still have headroom

---

## Hidden ground truth

Each task contains internal fields that are only visible through `state()` and grading, not through the observation:

* `true_issue_type`
* `required_resolution_steps`
* `correct_priority`
* `sla_deadline`
* `customer_persona`

This means an agent must reason from observable signals rather than being spoon-fed the answer.

---

## Workflow

The environment enforces a realistic staged workflow:

1. `open`
2. `needs_info`
3. `in_progress`
4. `resolved`
5. `escalated`

Important rules:

* the agent cannot resolve before classification
* priority cannot be set before classification
* escalation requires justification, not just a blind hand-off
* the hard task requires a clarification turn before safe escalation
* invalid action order is penalized
* customer replies are simulated deterministically by the environment

---

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

* `classify_ticket`
* `set_priority`
* `request_info`
* `respond`
* `resolve`
* `escalate`

---

## Observation space

Each observation includes:

* ticket info
* full visible conversation history
* current workflow status
* last agent action
* dynamic customer mood
* public knowledge base context
* currently known customer facts
* available actions

The observation does not reveal hidden ground truth directly.

---

## Reward shaping

Reward is trajectory-based and incremental.

### Positive reward

* `+0.2` correct classification
* `+0.1` correct priority
* `+0.2` good clarification question
* `+0.3` correct resolution
* `+0.2` correct escalation

### Penalties

* `-0.1` invalid action or invalid sequence
* `-0.1` repeated action
* `-0.05` per step due to SLA pressure
* `-0.2` hallucinated resolution
* `-0.2` unnecessary escalation
* `-0.2` premature resolution
* extra penalties for weak responses and exploit-like spam

This gives learning signal across the whole trajectory rather than only at the end.

---

## Tasks

### Easy: Password Reset Lockout

Directly solvable. Requires correct classification, priority, reassurance, and resolution.

### Medium: Delayed Refund Investigation

Still solvable, but communication quality matters more.

### Medium: Suspicious Login Alert

Security-focused workflow with correct urgency and recovery steps.

### Hard: Enterprise SSO Outage Under Close Deadline

* missing information initially
* frustrated customer
* requires clarification
* customer provides diagnostic details
* must decide escalation vs resolution
* tight SLA pressure

This task is intentionally not solvable in one step.

---

## Grading

Each task is scored deterministically in `[0.0, 1.0]` using:

* classification accuracy
* priority correctness
* resolution or escalation correctness
* reasoning-step coverage
* response quality
* efficiency

Anti-exploit protections:

* always escalating is penalized
* always resolving is penalized
* repetitive behavior is penalized

---

## Baseline

`inference.py` uses a weak baseline:

* optional OpenAI-compatible prompting when API_BASE_URL and an API key (`OPENAI_API_KEY` or `HF_TOKEN`) are set
* otherwise a deterministic rule-based policy
* prints reasoning rationales

### Baseline behavior

* keyword-based classification
* imperfect priority estimation
* minimal clarification
* partial responses
* hesitation in hard cases

---

## Deterministic Baseline Results

Example run of:

```bash
python inference.py
```

Results:

* easy_password_reset_triage: **1.000**
* medium_refund_resolution: **0.743**
* hard_enterprise_incident: **0.514**
* medium_security_alert: **0.967**

These results are deterministic and reproducible.

---

## Environment Variables

Supports both OpenAI and Hugging Face.

### Option 1: OpenAI

```bash
export OPENAI_API_KEY=your_key
```

### Option 2: Hugging Face

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export HF_TOKEN=your_token
export MODEL_NAME=meta-llama/Llama-3-8B-Instruct
```

If no API key is provided, a deterministic baseline is used.

---

## Run

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Run baseline

```bash
python inference.py
```

### Run tests

```bash
pytest
```

---

## API

FastAPI endpoints:

* `POST /reset`
* `POST /step`
* `GET /state`

Follows OpenEnv interface:

* `reset() -> observation`
* `step(action) -> observation, reward, done, info`
* `state() -> internal state`

---

## Files

* [env.py](env.py)
* [models.py](models.py)
* [tasks.py](tasks.py)
* [graders.py](graders.py)
* [inference.py](inference.py)
* [openenv.yaml](openenv.yaml)

---

## Summary

This environment simulates realistic customer support workflows with:

* multi-step reasoning
* dynamic user behavior
* structured actions
* deterministic grading
* meaningful reward shaping

It is designed for evaluating real-world agent capabilities beyond simple classification tasks.
