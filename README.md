# Python AI MCP Agent

An interactive Python agent that routes device-support questions into predefined skills, calls the permitted Model Context Protocol (MCP) tools, and uses the OpenAI Responses API to produce an answer.

## Architecture

```text
Terminal question
       |
       v
AI backend
  - select predefined skill
  - expose only allowed MCP tools
  - call OpenAI Responses API
       |
       v
MCP client (stdio)
       |
       v
MCP server
  - get_device
  - get_device_metrics
  - check_provisioning_status
```

The backend owns the skill registry. A skill supplies instructions and an allowlist of tools, so the model operates inside the selected workflow boundary.

## Predefined Skills

| Skill | Selected for | Permitted tools |
| --- | --- | --- |
| `historical_reply` | Requests containing `history`, `historical`, `metric`, `metrics`, or `performance` | `get_device`, `get_device_metrics` |
| `device_provisioning` | Requests containing `provision`, `provisioning`, `activate`, or `activation` | `get_device`, `check_provisioning_status` |
| `diagnostic` | All other requests | `get_device`, `get_device_metrics` |

## Prerequisites

- Python 3.12 or later
- An OpenAI API key

## Setup

Create and activate a virtual environment in PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install mcp openai python-dotenv
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

## Run

With the virtual environment active:

```powershell
python .\ai_backend.py
```

The program starts `mcp_server.py` automatically using the same Python interpreter, then prompts for a question:

```text
Ask a device question: Why is device ABC123 performing slowly?
```

## Sample Questions

```text
Show me the historical metrics for device ABC123 for the last 3 days.
```

```text
Is device ABC123 provisioned?
```

```text
Why is device ABC123 performing slowly?
```

## MCP Tools

The MCP server currently returns mock device data:

| Tool | Inputs | Result |
| --- | --- | --- |
| `get_device` | `serial_number` | Device model and current status |
| `get_device_metrics` | `serial_number`, `days` | Historical latency metrics |
| `check_provisioning_status` | `serial_number` | Provisioning status |

## Logging

The backend logs the full workflow to the terminal: MCP server startup, discovered tools, selected skill, permitted tools, model response rounds, tool arguments and results, and request durations.

The default log level is `INFO`. Enable additional library diagnostics for the current PowerShell session with:

```powershell
$env:LOG_LEVEL="DEBUG"
python .\ai_backend.py
```

## Notes

- This project uses MCP 2.x and imports `MCPServer` from `mcp.server`.
- `mcp_server.py` communicates over stdio; do not add ordinary `print()` calls to that server because stdout is reserved for the MCP protocol.