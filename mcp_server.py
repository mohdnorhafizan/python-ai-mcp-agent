from mcp.server import MCPServer

mcp = MCPServer("ACS Tools")


@mcp.tool()
def get_device(serial_number: str) -> dict:
    """Get device information."""

    return {
        "serial_number": serial_number,
        "model": "Router X",
        "status": "online"
    }


@mcp.tool()
def get_device_metrics(serial_number: str, days: int) -> dict:
    """Get historical device metrics."""

    return {
        "serial_number": serial_number,
        "days": days,
        "metrics": [
            {"day": 1, "latency": 20},
            {"day": 2, "latency": 25},
            {"day": 3, "latency": 40}
        ]
    }


@mcp.tool()
def check_provisioning_status(serial_number: str) -> dict:
    """Check device provisioning status."""

    return {
        "serial_number": serial_number,
        "provisioned": True
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")