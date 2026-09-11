import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AsyncOpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai_backend")

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


@dataclass(frozen=True)
class Skill:
    """A backend-defined capability boundary for an agent request."""

    name: str
    instructions: str
    allowed_tools: frozenset[str]


SKILLS = {
    "historical_reply": Skill(
        name="historical_reply",
        instructions=(
            "You are the Historical Reply workflow. Identify the requested "
            "device, retrieve only the requested historical metrics, and give "
            "a concise factual summary. Do not perform provisioning actions."
        ),
        allowed_tools=frozenset({"get_device", "get_device_metrics"}),
    ),
    "device_provisioning": Skill(
        name="device_provisioning",
        instructions=(
            "You are the Device Provisioning workflow. Verify device identity "
            "and its provisioning status, then report the outcome. Do not "
            "provide historical metric analysis."
        ),
        allowed_tools=frozenset({"get_device", "check_provisioning_status"}),
    ),
    "diagnostic": Skill(
        name="diagnostic",
        instructions=(
            "You are the Diagnostic workflow. Inspect the device and relevant "
            "historical metrics, identify observable issues, and report the "
            "evidence. Do not claim to make device changes."
        ),
        allowed_tools=frozenset({"get_device", "get_device_metrics"}),
    ),
}


def select_skill(user_request: str) -> Skill:
    """Route a request into one predefined workflow before exposing MCP tools."""

    normalized_request = user_request.lower()

    if re.search(r"\b(provision|provisioning|activate|activation)\b", normalized_request):
        return SKILLS["device_provisioning"]
    if re.search(r"\b(history|historical|metric|metrics|performance)\b", normalized_request):
        return SKILLS["historical_reply"]
    return SKILLS["diagnostic"]


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

            tools = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                }
                for tool in tools_result.tools
            ]

            user_question = input("\nAsk a device question: ").strip()
            if not user_question:
                logger.warning("No user request provided; stopping the program")
                print("Please enter a device question.")
                return

            logger.info("Received user request: %s", user_question)
            skill = select_skill(user_question)
            logger.info(
                "Selected skill: name=%s, allowed_tools=%s",
                skill.name,
                sorted(skill.allowed_tools),
            )

            tools = [
                tool
                for tool in tools
                if tool["name"] in skill.allowed_tools
            ]
            logger.info(
                "Exposing %d tools to the model: %s",
                len(tools),
                [tool["name"] for tool in tools],
            )

            model_call_started = time.perf_counter()
            logger.info("Sending initial request to model: model=%s", "gpt-4.1-mini")
            response = await client.responses.create(
                model="gpt-4.1-mini",
                input=f"{skill.instructions}\n\nUser request: {user_question}",
                tools=tools,
            )
            logger.info(
                "Initial model request completed in %.2f seconds",
                time.perf_counter() - model_call_started,
            )
            log_response(response, "Initial model")

            tool_round = 0
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
                    model="gpt-4.1-mini",
                    input=[*response.output, *tool_outputs],
                    tools=tools,
                )
                logger.info(
                    "Follow-up model request completed in %.2f seconds",
                    time.perf_counter() - model_call_started,
                )
                log_response(response, "Follow-up model")

            logger.info("Final response: %s", response.output_text)
            print(response.output_text)


if __name__ == "__main__":
    asyncio.run(main())