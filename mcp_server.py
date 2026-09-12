from mcp.server import MCPServer

mcp = MCPServer("ACS Tools")

DEVICES = {
    "abc123": {
        "serial_number": "ABC123",
        "model": "Router X",
        "status": "online",
        "provisioned": True,
    },
    "halalfood": {
        "serial_number": "halalfood",
        "model": "Router Y",
        "status": "online",
        "provisioned": True,
    },
}


def get_registered_device(serial_number: str) -> dict:
    device = DEVICES.get(serial_number.lower())
    if device is None:
        raise ValueError(f"Device '{serial_number}' was not found.")
    return device


@mcp.tool()
def resolve_device_reference(user_request: str) -> dict:
    """Find a registered device serial number mentioned anywhere in a user request."""

    request_words = set(user_request.lower().split())
    for inventory_key, device in DEVICES.items():
        if inventory_key in request_words:
            return {
                "found": True,
                "serial_number": device["serial_number"],
            }
    return {"found": False, "serial_number": None}


@mcp.tool()
def get_device(serial_number: str) -> dict:
    """Get device information."""

    device = get_registered_device(serial_number)
    return {
        "serial_number": device["serial_number"],
        "model": device["model"],
        "status": device["status"],
    }


@mcp.tool()
def get_device_metrics(serial_number: str, days: int) -> dict:
    """Get historical device metrics."""

    device = get_registered_device(serial_number)
    return {
        "serial_number": device["serial_number"],
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

    device = get_registered_device(serial_number)
    return {
        "serial_number": device["serial_number"],
        "provisioned": device["provisioned"],
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")