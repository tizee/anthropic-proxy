def request_hook(payload):
    """
    Filters out specified tools from the list of available tools in the request payload.
    """
    if "tools" in payload:
        filtered_tool_names = ["WebSearch", "WebFetch"]
        payload["tools"] = [
            tool
            for tool in payload["tools"]
            if tool.get("name") not in filtered_tool_names
        ]
    return payload


def response_hook(payload):
    """
    A simple response hook for testing purposes.
    """
    payload["hook_applied"] = True
    return payload
