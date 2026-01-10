import logging

logger = logging.getLogger(__name__)


def request_hook(payload):
    """
    Filters out specified tools from the list of available tools in the request payload.
    Handles both Claude format ({"name": "ToolName"}) and OpenAI format ({"function": {"name": "ToolName"}}).
    """
    if "tools" in payload:
        filtered_tool_names = ["WebSearch", "NotebookEdit", "NotebookRead"]
        original_tools = payload["tools"]

        def get_tool_name(tool):
            """Extract tool name from either Claude or OpenAI format."""
            # Claude format: {"name": "ToolName"}
            if "name" in tool:
                return tool["name"]
            # OpenAI format: {"function": {"name": "ToolName"}}
            elif "function" in tool and "name" in tool["function"]:
                return tool["function"]["name"]
            else:
                return None

        # Extract original tool names
        original_tool_names = [get_tool_name(tool) for tool in original_tools]

        # Filter out specified tools
        filtered_tools = [
            tool
            for tool in original_tools
            if get_tool_name(tool) not in filtered_tool_names
        ]

        # Debug logging
        removed_tools = [
            name for name in original_tool_names if name in filtered_tool_names
        ]
        remaining_tools = [get_tool_name(tool) for tool in filtered_tools]

        logger.debug(f"TOOL_FILTER: Original tools: {original_tool_names}")
        logger.debug(f"TOOL_FILTER: Removed tools: {removed_tools}")
        logger.debug(f"TOOL_FILTER: Remaining tools: {remaining_tools}")

        payload["tools"] = filtered_tools
    return payload


def response_hook(payload):
    """
    A simple response hook for testing purposes.
    """
    payload["hook_applied"] = True
    return payload
