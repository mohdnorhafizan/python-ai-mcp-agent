import asyncio
import json
import logging
import sys
import os
from models import ChatSession, ChatReply, ChatRequest
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import APIError

from ai_backend import Supervisor
from workflow import chat_workflow

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger("web_app")
STATIC_DIR = Path(__file__).parent / "static"
MODEL = "gpt-4.1-mini"


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
    server_params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()
            tools_result = await mcp_session.list_tools()
            all_tools = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                }
                for tool in tools_result.tools
            ]
            result = await chat_workflow.ainvoke(
                {
                    "user_message": user_message,
                    "supervisor": session_state.supervisor,
                    "previous_response_id": session_state.previous_response_id,
                    "mcp_session": mcp_session,
                    "allowed_tools": all_tools,
                }
            )

    if response_id := result.get("response_id"):
        session_state.previous_response_id = response_id
    skill = result.get("skill")
    return ChatReply(
        answer=result["answer"],
        skill=skill.name if skill else None,
    )


async def stream_agent(session_state: ChatSession, user_message: str):
    progress: asyncio.Queue[str | None] = asyncio.Queue()
    server_params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])

    async def run_workflow() -> dict:
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as mcp_session:
                    await mcp_session.initialize()
                    tools_result = await mcp_session.list_tools()

                    logger.info(
                        "MCP server returned %d tools: %s",
                        len(tools_result.tools),
                        [tool.name for tool in tools_result.tools],
                    )
                    
                    all_tools = [
                        {
                            "type": "function",
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,
                        }
                        for tool in tools_result.tools
                    ]
                    return await chat_workflow.ainvoke(
                        {
                            "user_message": user_message,
                            "supervisor": session_state.supervisor,
                            "previous_response_id": session_state.previous_response_id,
                            "mcp_session": mcp_session,
                            "allowed_tools": all_tools,
                            "progress": progress,
                        }
                    )
        finally:
            await progress.put(None)

    workflow_task = asyncio.create_task(run_workflow())
    try:
        while (message := await progress.get()) is not None:
            yield f"event: progress\ndata: {json.dumps({'message': message})}\n\n"

        result = await workflow_task
        if response_id := result.get("response_id"):
            session_state.previous_response_id = response_id
        skill = result.get("skill")
        payload = {
            "answer": result["answer"],
            "skill": skill.name if skill else None,
        }
        yield f"event: answer\ndata: {json.dumps(payload)}\n\n"
    except Exception as error:
        logger.exception("Streaming chat request failed")
        if not workflow_task.done():
            workflow_task.cancel()
        yield f"event: error\ndata: {json.dumps({'message': 'The chat request failed.'})}\n\n"


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


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request, response: Response):
    session_state = get_session(request, response)
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    async def guarded_stream():
        async with session_state.lock:
            async for event in stream_agent(session_state, message):
                yield event

    stream_response = StreamingResponse(guarded_stream(), media_type="text/event-stream")
    session_id = request.cookies.get("device_chat_session")
    if session_id is None or session_id not in sessions:
        stream_response.set_cookie(
            key="device_chat_session",
            value=next(session_id for session_id, state in sessions.items() if state is session_state),
            httponly=True,
            samesite="lax",
        )
    return stream_response


@app.post("/api/session/reset")
async def reset_session(request: Request, response: Response) -> dict[str, bool]:
    session_id = request.cookies.get("device_chat_session")
    if session_id:
        sessions.pop(session_id, None)
    response.delete_cookie("device_chat_session")
    return {"reset": True}