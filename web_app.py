import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import APIError
from pydantic import BaseModel, Field

from ai_backend import Supervisor, client, log_response

logger = logging.getLogger("web_app")
STATIC_DIR = Path(__file__).parent / "static"
MODEL = "gpt-4.1-mini"


@dataclass
class ChatSession:
    supervisor: Supervisor = field(default_factory=Supervisor)
    previous_response_id: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)


class ChatReply(BaseModel):
    answer: str
    skill: str | None


sessions: dict[str, ChatSession] = {}
app = FastAPI(title="Device Support Chat")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_session(request: Request, response: Response) -> ChatSession:
    session_id = request.cookies.get("device_chat_session")
    if session_id is None or session_id not in sessions:
        session_id = str(uuid4())
        sessions[session_id] = ChatSession()
        response.set_cookie(
            key="device_chat_session",
            value=session_id,
            httponly=True,
            samesite="lax",
        )
        logger.info("Created browser chat session: %s", session_id)
    return sessions[session_id]


async def run_agent(session_state: ChatSession, user_message: str) -> ChatReply:
    skill = session_state.supervisor.select_skill(user_message)
    if skill is None:
        logger.info("Supervisor selected no skill; no MCP tools exposed")
        response = await client.responses.create(
            model=MODEL,
            input=user_message,
            instructions=session_state.supervisor.build_instructions(None),
            previous_response_id=session_state.previous_response_id,
            tools=[],
        )
        session_state.previous_response_id = response.id
        log_response(response, "Web conversational model")
        return ChatReply(answer=response.output_text, skill=None)

    logger.info("Supervisor selected skill=%s", skill.name)
    server_params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()
            resolution = await mcp_session.call_tool(
                "resolve_device_reference",
                arguments={"user_request": user_message},
            )
            logger.info("Supervisor device resolution result: %s", resolution.content)
            session_state.supervisor.record_tool_result(
                "resolve_device_reference",
                resolution.content,
            )
            tools_result = await mcp_session.list_tools()
            allowed_tools = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                }
                for tool in tools_result.tools
                if tool.name in skill.allowed_tools
            ]

            logger.info("Tools exposed to model: %s", [tool["name"] for tool in allowed_tools])
            started = time.perf_counter()
            response = await client.responses.create(
                model=MODEL,
                input=user_message,
                instructions=session_state.supervisor.build_instructions(skill),
                previous_response_id=session_state.previous_response_id,
                tools=allowed_tools,
            )
            logger.info("Initial model response completed in %.2fs", time.perf_counter() - started)
            log_response(response, "Web model")

            while tool_calls := [
                item for item in response.output if item.type == "function_call"
            ]:
                tool_outputs = []
                for tool_call in tool_calls:
                    arguments = json.loads(tool_call.arguments)
                    logger.info("Calling MCP tool=%s arguments=%s", tool_call.name, arguments)
                    result = await mcp_session.call_tool(tool_call.name, arguments=arguments)
                    session_state.supervisor.record_tool_result(tool_call.name, result.content)
                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call.call_id,
                            "output": str(result.content),
                        }
                    )

                response = await client.responses.create(
                    model=MODEL,
                    input=tool_outputs,
                    previous_response_id=response.id,
                    tools=allowed_tools,
                )
                log_response(response, "Web follow-up model")

    session_state.previous_response_id = response.id
    return ChatReply(answer=response.output_text, skill=skill.name)


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat", response_model=ChatReply)
async def chat(payload: ChatRequest, request: Request, response: Response) -> ChatReply:
    session_state = get_session(request, response)
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    async with session_state.lock:
        try:
            return await run_agent(session_state, message)
        except APIError as error:
            logger.exception("OpenAI API request failed")
            raise HTTPException(status_code=502, detail="The AI service request failed.") from error
        except Exception as error:
            logger.exception("Chat request failed")
            raise HTTPException(status_code=500, detail="The chat request failed.") from error


@app.post("/api/session/reset")
async def reset_session(request: Request, response: Response) -> dict[str, bool]:
    session_id = request.cookies.get("device_chat_session")
    if session_id:
        sessions.pop(session_id, None)
    response.delete_cookie("device_chat_session")
    return {"reset": True}