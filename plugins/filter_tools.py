def filter_tools(tools):
    """
    Filters out specified tools from the list of available tools.
    """
    filtered_tool_names = ["WebSearch", "WebFetch"]
    return [tool for tool in tools if tool.get("name") not in filtered_tool_names]
