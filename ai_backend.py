import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

logger = logging.getLogger("web_app")

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


@dataclass(frozen=True)
class Skill:
    """A backend-defined capability boundary for an agent request."""

    name: str
    instructions: str
    allowed_tools: frozenset[str]


class WorkflowState(str, Enum):
    IDLE = "idle"
    AWAITING_DEVICE = "awaiting_device"
    EXECUTING = "executing"


class WorkflowNode(str, Enum):
    RECEIVE_MESSAGE = "receive_message"
    SELECT_SKILL = "select_skill"
    RESOLVE_DEVICE = "resolve_device"
    EXECUTE_SKILL = "execute_skill"
    RESPOND = "respond"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class WorkflowEdge:
    source: WorkflowNode
    target: WorkflowNode
    condition: str


WORKFLOW_EDGES = (
    WorkflowEdge(WorkflowNode.RECEIVE_MESSAGE, WorkflowNode.SELECT_SKILL, "always"),
    WorkflowEdge(WorkflowNode.SELECT_SKILL, WorkflowNode.OUT_OF_SCOPE, "no active skill"),
    WorkflowEdge(WorkflowNode.SELECT_SKILL, WorkflowNode.RESOLVE_DEVICE, "active skill"),
    WorkflowEdge(WorkflowNode.RESOLVE_DEVICE, WorkflowNode.RESPOND, "device is missing"),
    WorkflowEdge(WorkflowNode.RESOLVE_DEVICE, WorkflowNode.EXECUTE_SKILL, "device is resolved"),
    WorkflowEdge(WorkflowNode.EXECUTE_SKILL, WorkflowNode.RESPOND, "tool workflow completed"),
)


class IntentClassification(BaseModel):
    intent: Literal["device_provisioning", "historical_reply", "diagnostic", "none"]
    reasoning: str = Field(description="Brief explanation for the classification decision")


@dataclass
class Supervisor:
    """Coordinates skill selection and session context for a conversation."""

    current_device_serial: str | None = None
    pending_skill: Skill | None = None
    state: WorkflowState = WorkflowState.IDLE

    async def select_skill(self, user_request: str) -> Skill | None:
        logger.info("Supervisor node=%s", WorkflowNode.SELECT_SKILL.value)

        system_prompt = (
            "You are an intent classification supervisor for an ACS device support system.\n"
            "Classify the user's message into exactly one of these skills:\n"
            "- 'device_provisioning': Requests about device activation, configuration, provisioning status, or setup.\n"
            "- 'historical_reply': Requests for past device performance, metrics, latency history, or usage trends over time.\n"
            "- 'diagnostic': Requests troubleshooting slow performance, errors, offline devices, or technical issues.\n"
            "- 'none': Greetings (e.g., 'hi', 'hello'), general chit-chat, personal questions, or anything unrelated to device support.\n\n"
            f"Current Session Context:\n"
            f"- Active Pending Workflow: {self.pending_skill.name if self.pending_skill else 'None'}\n"
            f"- Active Device Serial: {self.current_device_serial or 'None'}\n\n"
            "Note: If there is an active pending workflow and the user provides a follow-up answer "
            "(such as a device serial number like 'halalfood' or 'ABC123'), classify it under that active pending workflow skill."
        )

        try:
            response = await client.beta.chat.completions.parse(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_request},
                ],
                response_format=IntentClassification,
            )
            parsed = response.choices[0].message.parsed
            intent = parsed.intent if parsed else "none"
            reasoning = parsed.reasoning if parsed else ""
            logger.info("LLM Intent Classification: intent=%s, reasoning='%s'", intent, reasoning)
        except Exception as error:
            logger.warning("LLM intent classification failed: %s; falling back to regex", error)
            normalized_request = user_request.lower()
            if re.search(r"\b(provision|provisioning|activate|activation)\b", normalized_request):
                intent = "device_provisioning"
            elif re.search(r"\b(history|historical|metric|metrics|performance)\b", normalized_request):
                intent = "historical_reply"
            elif re.search(r"\b(slow|offline|error|issue|problem|diagnose|diagnostic)\b", normalized_request):
                intent = "diagnostic"
            else:
                intent = self.pending_skill.name if self.pending_skill else "none"

        if intent == "none":
            self.pending_skill = None
            self.state = WorkflowState.IDLE
            return None

        self.pending_skill = SKILLS.get(intent)
        if self.pending_skill:
            self.state = (
                WorkflowState.EXECUTING
                if self.current_device_serial
                else WorkflowState.AWAITING_DEVICE
            )
            logger.info(
                "Supervisor workflow state=%s, skill=%s",
                self.state.value,
                self.pending_skill.name,
            )

        return self.pending_skill

    def build_instructions(self, skill: Skill | None) -> str:
        device_context = (
            f"The current device serial number is {self.current_device_serial}. "
            "Use it for follow-up requests unless the user names another device."
            if self.current_device_serial
            else "No device has been confirmed in this session yet."
        )
        if skill is None:
            return (
                "You are a device-support chat assistant. Reply conversationally, "
                "but do not call tools until the user clearly asks about device "
                "history, provisioning, or a device problem. "
                f"Supervisor session context: {device_context}"
            )
        pending_context = (
            "This workflow is awaiting a device serial number from an earlier "
            "request. Treat a short follow-up message as the device reference."
            if self.state is WorkflowState.AWAITING_DEVICE
            else "The required device reference has been resolved."
        )
        return (
            f"{skill.instructions}\n\nSupervisor session context: "
            f"{device_context} {pending_context}"
        )

    def out_of_scope_reply(self) -> str:
        logger.info("Supervisor node=%s", WorkflowNode.OUT_OF_SCOPE.value)
        return (
            "I can help with device performance history, provisioning status, "
            "and device diagnostics. Please ask a device-support question."
        )

    def record_tool_result(self, tool_name: str, result_content: object) -> None:
        if tool_name in {"get_device_metrics", "check_provisioning_status"}:
            self.pending_skill = None
            self.state = WorkflowState.IDLE
            logger.info("Supervisor completed the active workflow")
            return

        if tool_name != "resolve_device_reference":
            return

        payload = extract_tool_payload(result_content)
        serial_number = payload.get("serial_number")
        if payload.get("found") and serial_number:
            self.current_device_serial = str(serial_number)
            self.state = WorkflowState.EXECUTING
            logger.info(
                "Supervisor saved current device serial: %s; workflow state=%s",
                self.current_device_serial,
                self.state.value,
            )


def extract_tool_payload(result_content: object) -> dict[str, object]:
    """Extract a JSON object from MCP text content or a direct dictionary."""

    if isinstance(result_content, dict):
        return result_content

    if isinstance(result_content, list):
        for item in result_content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text:
                payload = extract_tool_payload(text)
                if payload:
                    return payload
        return {}

    if isinstance(result_content, str):
        try:
            payload = json.loads(result_content)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    return {}


SKILLS = {
    "historical_reply": Skill(
        name="historical_reply",
        instructions=(
            "You are the Historical Reply workflow. The supervisor has already "
            "resolved any device reference and supplied it in the session context. "
            "If no device serial is in session context, ask only for the serial number. "
            "Otherwise, call get_device_metrics with that serial number. When the request omits "
            "a duration, use 3 days. Tool calls are internal: do not tell the user "
            "to wait or that you will check later. Complete the tool workflow before "
            "giving a concise factual summary. Do not perform provisioning actions."
        ),
        allowed_tools=frozenset(
            {"get_device", "get_device_metrics"}
        ),
    ),
    "device_provisioning": Skill(
        name="device_provisioning",
        instructions=(
            "You are the Device Provisioning workflow. The supervisor has already "
            "resolved any device reference and supplied it in the session context. "
            "If no device serial is in session context, ask only for the serial number. "
            "Otherwise, call check_provisioning_status with the resolved serial number "
            "and report the outcome. Tool calls are internal: do not tell the user "
            "to wait or that you will check later. Do not "
            "provide historical metric analysis."
        ),
        allowed_tools=frozenset(
            {
                "get_device",
                "check_provisioning_status",
                "validate_activation",
                "execute_provisioning",
                "verify_provisioning",
            }
        ),
    ),
    "diagnostic": Skill(
        name="diagnostic",
        instructions=(
            "You are the Diagnostic workflow. Inspect the device and relevant "
            "historical metrics. The supervisor has already resolved any device "
            "reference and supplied it in the session context. If no device serial is "
            "in session context, ask only for the serial number. Otherwise, call get_device and "
            "get_device_metrics with its serial number. When the request omits a "
            "duration, use 3 days. Tool calls are internal: do not tell the user to "
            "wait or that you will check later. Complete the tool workflow, then "
            "identify observable issues and report the evidence. Do not claim to "
            "make device changes."
        ),
        allowed_tools=frozenset(
            {"get_device", "get_device_metrics"}
        ),
    ),
}


def log_response(response, stage: str) -> None:
    output_types = [item.type for item in response.output]
    logger.info(
        "%s response received: id=%s, output_items=%d, output_types=%s",
        stage,
        response.id,
        len(response.output),
        output_types,
    )


async def main():

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
    )
    logger.info("Starting AI backend")
    logger.info(
        "MCP server command configured: command=%s, args=%s",
        server_params.command,
        server_params.args,
    )

    async with stdio_client(server_params) as (read, write):
        logger.info("MCP stdio connection established")

        async with ClientSession(read, write) as session:

            logger.info("Initializing MCP client session")
            await session.initialize()
            logger.info("MCP client session initialized")

            logger.info("Requesting available tools from MCP server")
            tools_result = await session.list_tools()
            logger.info(
                "MCP server returned %d tools: %s",
                len(tools_result.tools),
                [tool.name for tool in tools_result.tools],
            )

            logger.info(f"Tool Names: {[tool.name for tool in tools_result.tools]}")
            logger.info(f"Tool Descriptions: {[tool.description for tool in tools_result.tools]}")
            logger.info(f"Tool Parameters: {[tool.input_schema for tool in tools_result.tools]}")

            tools = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                }
                for tool in tools_result.tools
            ]

            conversation_response_id = None
            supervisor = Supervisor()
            print("Device support chat started. Type 'exit' or 'quit' to end.")
            while True:
                user_question = input("\nYou: ").strip()
                if user_question.lower() in {"exit", "quit"}:
                    logger.info("User ended the chat session")
                    print("Chat ended.")
                    break
                if not user_question:
                    print("Please enter a device question.")
                    continue

                logger.info("Received user request: %s", user_question)

                skill = await supervisor.select_skill(user_question)
                if skill is None:
                    logger.info("Supervisor selected no skill; returning scoped reply")
                    print(f"AI: {supervisor.out_of_scope_reply()}")
                    continue

                if skill:
                    resolution = await session.call_tool(
                        "resolve_device_reference",
                        arguments={"user_request": user_question},
                    )
                    logger.info("Supervisor device resolution result: %s", resolution.content)
                    supervisor.record_tool_result(
                        "resolve_device_reference",
                        resolution.content,
                    )
                allowed_tools = [
                    tool for tool in tools if skill and tool["name"] in skill.allowed_tools
                ]
                if skill:
                    logger.info(
                        "Supervisor selected skill: name=%s, allowed_tools=%s",
                        skill.name,
                        sorted(skill.allowed_tools),
                    )
                    logger.info(
                        "Exposing %d tools to the model: %s",
                        len(allowed_tools),
                        [tool["name"] for tool in allowed_tools],
                    )
                model_call_started = time.perf_counter()
                logger.info("Sending user request to model: model=%s", "gpt-4.1-mini")
                response = await client.responses.create(
                    model="gpt-4o",
                    input=user_question,
                    instructions=supervisor.build_instructions(skill),
                    previous_response_id=conversation_response_id,
                    tools=allowed_tools,
                )
                logger.info(
                    "Model request completed in %.2f seconds",
                    time.perf_counter() - model_call_started,
                )
                log_response(response, "Model")

                tool_round = 0
                logger.info(f"Model response output: {response.output}")
                while True:
                    tool_calls = [
                        item
                        for item in response.output
                        if item.type == "function_call"
                    ]
                    if not tool_calls:
                        logger.info("No further tool calls requested by the model")
                        break

                    tool_round += 1
                    logger.info(
                        "Processing tool round %d with %d call(s)",
                        tool_round,
                        len(tool_calls),
                    )
                    tool_outputs = []
                    for tool_call in tool_calls:
                        arguments = json.loads(tool_call.arguments)
                        logger.info(
                            "Calling MCP tool: name=%s, call_id=%s, arguments=%s",
                            tool_call.name,
                            tool_call.call_id,
                            json.dumps(arguments),
                        )
                        tool_call_started = time.perf_counter()
                        result = await session.call_tool(
                            tool_call.name,
                            arguments=arguments,
                        )
                        logger.info(
                            "MCP tool completed in %.2f seconds: name=%s, result=%s",
                            time.perf_counter() - tool_call_started,
                            tool_call.name,
                            result.content,
                        )
                        supervisor.record_tool_result(tool_call.name, result.content)
                        tool_outputs.append(
                            {
                                "type": "function_call_output",
                                "call_id": tool_call.call_id,
                                "output": str(result.content),
                            }
                        )

                    model_call_started = time.perf_counter()
                    logger.info("Sending %d tool result(s) back to model", len(tool_outputs))
                    response = await client.responses.create(
                        model="gpt-4o",
                        input=tool_outputs,
                        previous_response_id=response.id,
                        tools=allowed_tools,
                    )
                    logger.info(
                        "Follow-up model request completed in %.2f seconds",
                        time.perf_counter() - model_call_started,
                    )
                    log_response(response, "Follow-up model")

                conversation_response_id = response.id
                logger.info("Final response: %s", response.output_text)
                print(f"AI: {response.output_text}")


if __name__ == "__main__":
    asyncio.run(main())