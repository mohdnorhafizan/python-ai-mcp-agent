
from typing import Any, TypedDict
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

import asyncio
from ai_backend import Skill, Supervisor

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)


class ChatReply(BaseModel):
    answer: str
    skill: str | None

@dataclass
class ChatSession:
    supervisor: Supervisor = field(default_factory=Supervisor)
    previous_response_id: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

class ChatState(TypedDict, total=False):
    user_message: str
    supervisor: Supervisor
    previous_response_id: str | None
    mcp_session: Any
    skill: Skill | None
    allowed_tools: list[dict[str, Any]]
    response_id: str
    answer: str
    progress: Any
    provisioning_status: str
    provisioning_validation: str
    provisioning_action: str
    provisioning_verification: str