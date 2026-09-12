import json
import logging
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from ai_backend import Skill, Supervisor, client, log_response

logger = logging.getLogger("workflow")
MODEL = "gpt-4.1-mini"


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


async def emit_progress(state: ChatState, message: str) -> None:
    progress = state.get("progress")
    if progress is not None:
        await progress.put(message)


async def select_skill(state: ChatState) -> dict[str, Any]:
    await emit_progress(state, "Selecting the appropriate device-support workflow")
    supervisor = state["supervisor"]
    skill = supervisor.select_skill(state["user_message"])
    # A short follow-up such as a serial number has no intent keywords. In
    # that case, continue the workflow remembered by the supervisor.
    if skill is None:
        skill = supervisor.pending_skill
    if skill is None:
        logger.info("Graph node=select_skill result=out_of_scope")
    else:
        logger.info("Graph node=select_skill result=%s", skill.name)
    return {"skill": skill, "supervisor": supervisor}


def route_after_skill(state: ChatState) -> Literal["resolve_device", "out_of_scope"]:
    supervisor = state["supervisor"]
    return "resolve_device" if state.get("skill") or supervisor.pending_skill else "out_of_scope"


async def out_of_scope(state: ChatState) -> dict[str, Any]:
    await emit_progress(state, "This request is outside device-support workflows")
    logger.info("Graph node=out_of_scope")
    return {"answer": state["supervisor"].out_of_scope_reply()}


async def resolve_device(state: ChatState) -> dict[str, Any]:
    logger.info("Graph node=resolve_device")
    await emit_progress(state, "Now resolving the device serial number")
    skill = state.get("skill") or state["supervisor"].pending_skill
    result = await state["mcp_session"].call_tool(
        "resolve_device_reference",
        arguments={"user_request": state["user_message"]},
    )
    state["supervisor"].record_tool_result("resolve_device_reference", result.content)
    return {"skill": skill}


def route_after_resolution(state: ChatState) -> Literal["ask_for_device", "execute_skill"]:
    if state["supervisor"].current_device_serial:
        return "execute_skill"
    return "ask_for_device"


async def ask_for_device(state: ChatState) -> dict[str, Any]:
    logger.info("Graph node=ask_for_device")
    await emit_progress(state, "The device serial number is required before checking")
    return {"answer": "Please provide the device serial number."}


async def execute_skill(state: ChatState) -> dict[str, Any]:
    skill = state["skill"]
    allowed_tools = [
        tool for tool in state["allowed_tools"] if tool["name"] in skill.allowed_tools
    ]
    logger.info(
        "Graph node=execute_skill skill=%s tools=%s",
        skill.name,
        [tool["name"] for tool in allowed_tools],
    )
    await emit_progress(state, f"Starting {skill.name.replace('_', ' ')} workflow")
    response = await client.responses.create(
        model=MODEL,
        input=state["user_message"],
        instructions=state["supervisor"].build_instructions(skill),
        previous_response_id=state.get("previous_response_id"),
        tools=allowed_tools,
    )
    log_response(response, "LangGraph model")

    while tool_calls := [item for item in response.output if item.type == "function_call"]:
        tool_outputs = []
        for tool_call in tool_calls:
            arguments = json.loads(tool_call.arguments)
            logger.info("Graph calling MCP tool=%s arguments=%s", tool_call.name, arguments)
            progress_messages = {
                "get_device": "Now checking the device details",
                "get_device_metrics": "Now getting the device performance metrics",
                "check_provisioning_status": "Now checking the device provisioning status",
            }
            await emit_progress(
                state,
                progress_messages.get(tool_call.name, f"Now running {tool_call.name}"),
            )
            result = await state["mcp_session"].call_tool(tool_call.name, arguments=arguments)
            state["supervisor"].record_tool_result(tool_call.name, result.content)
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
        log_response(response, "LangGraph follow-up model")

    await emit_progress(state, "The checks are complete; preparing the response")
    return {"answer": response.output_text, "response_id": response.id}


def build_workflow():
    graph = StateGraph(ChatState)
    graph.add_node("select_skill", select_skill)
    graph.add_node("out_of_scope", out_of_scope)
    graph.add_node("resolve_device", resolve_device)
    graph.add_node("ask_for_device", ask_for_device)
    graph.add_node("execute_skill", execute_skill)
    graph.add_edge(START, "select_skill")
    graph.add_conditional_edges("select_skill", route_after_skill)
    graph.add_conditional_edges("resolve_device", route_after_resolution)
    graph.add_edge("out_of_scope", END)
    graph.add_edge("ask_for_device", END)
    graph.add_edge("execute_skill", END)
    return graph.compile()


chat_workflow = build_workflow()