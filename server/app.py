from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from env import CustomerSupportEnv
from models import Action


app = FastAPI(
    title="Customer Support Ticket Resolution OpenEnv",
    description="OpenEnv-compatible environment for realistic support operations workflows.",
    version="1.0.0",
)
ENV = CustomerSupportEnv()


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "message": "Customer support OpenEnv server is running."}


@app.post("/reset")
def reset(request: ResetRequest | None = None) -> Dict[str, Any]:
    observation = ENV.reset(task_id=request.task_id if request else None)
    return {"observation": observation.model_dump()}


@app.post("/step")
def step(action: Action) -> Dict[str, Any]:
    observation, reward, done, info = ENV.step(action)
    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    return {"state": ENV.state()}
