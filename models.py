"""
Pydantic models for Claude proxy API requests and responses.
This module contains only the model definitions without any server startup code.
"""

from pydantic import BaseModel, field_validator, ConfigDict
from typing import Dict, List, Optional, Union, Literal, Any
import re
import json
import uuid

class ToolChoiceAuto(BaseModel):
    type: Literal["auto"] = "auto"
    disable_parallel_tool_use: Optional[bool] = None

    def to_openai(self) -> Literal["auto"]:
        return "auto"


class ToolChoiceAny(BaseModel):
    type: Literal["any"] = "any"
    disable_parallel_tool_use: Optional[bool] = None

    def to_openai(self) -> Literal["required"]:
        return "required"


class ToolChoiceTool(BaseModel):
    type: Literal["tool"] = "tool"
    name: str
    disable_parallel_tool_use: Optional[bool] = None

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name}
        }


class ToolChoiceNone(BaseModel):
    type: Literal["none"] = "none"

    def to_openai(self) -> None:
        return None


# Union type for all tool choice options
ClaudeToolChoice = Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool, ToolChoiceNone]


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

    def to_openai(self) -> Dict[str, str]:
        """Convert Claude text block to OpenAI text format."""
        return {"type": "text", "text": self.text}


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

    def to_openai(self) -> Optional[Dict[str, Any]]:
        """
        Convert Claude image block to OpenAI image_url format.
        
        Claude format:
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "iVBORw0KGgoAAAANSUhEUgAAAB..."
            }
        }
        
        OpenAI format:
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAB..."
            }
        }
        """
        if (
            isinstance(self.source, dict)
            and self.source.get("type") == "base64"
            and "media_type" in self.source
            and "data" in self.source
        ):
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{self.source['media_type']};base64,{self.source['data']}"
                },
            }
        return None


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

    def to_openai(self) -> Dict[str, Any]:
        """
        Convert Claude tool_use to OpenAI tool_call format.

        Claude format:
        {
            "type": "tool_use",
            "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
            "name": "get_stock_price",
            "input": { "ticker": "^GSPC" }
        }

        OpenAI format:
        {
            "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "arguments": "{\"ticker\":\"^GSPC\"}"
            }
        }
        """
        import json

        try:
            arguments_str = json.dumps(self.input, ensure_ascii=False, separators=(',', ':'))
        except (TypeError, ValueError):
            arguments_str = "{}"

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": arguments_str
            }
        }

    @staticmethod
    def extract_tool_call_data(tool_call) -> tuple:
        """Extract tool call data from different formats."""
        if isinstance(tool_call, dict):
            tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
            function_data = tool_call.get("function", {})
            name = function_data.get("name", "")
            arguments_str = function_data.get("arguments", "{}")
        elif hasattr(tool_call, "id") and hasattr(tool_call, "function"):
            tool_id = tool_call.id
            name = tool_call.function.name
            arguments_str = tool_call.function.arguments
        else:
            return None, None, None

        return tool_id, name, arguments_str

    @staticmethod
    def parse_arguments(arguments_str: str) -> dict:
        """Parse tool arguments safely."""
        try:
            arguments_dict = json.loads(arguments_str)
            if not isinstance(arguments_dict, dict):
                arguments_dict = {"input": arguments_dict}
            return arguments_dict
        except json.JSONDecodeError:
            return {"raw_arguments": arguments_str}


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

    def process_content(self) -> str:
        """
        Process Claude tool_result content into a string format.

        Claude supports various content formats:
        - Simple string: "259.75 USD"
        - List with text blocks: [{"type": "text", "text": "result"}]
        - Complex nested structures
        """
        import json

        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            # Handle list content by extracting all text
            content_parts = []
            for item in self.content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        content_parts.append(item["text"])
                    elif "text" in item:
                        content_parts.append(item["text"])
                    else:
                        # Fallback: serialize non-text items
                        content_parts.append(json.dumps(item))
                else:
                    content_parts.append(str(item))
            return "\n".join(content_parts) if content_parts else ""
        elif isinstance(self.content, dict):
            # Handle single dict content
            if self.content.get("type") == "text" and "text" in self.content:
                return self.content["text"]
            else:
                return json.dumps(self.content)
        else:
            # Fallback: serialize anything else
            return json.dumps(self.content) if self.content is not None else ""

    def to_openai(self) -> Dict[str, Any]:
        """
        Convert Claude tool_result to OpenAI tool role message format.

        Claude format:
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
            "content": "259.75 USD"
        }

        OpenAI format:
        {
            "role": "tool",
            "tool_call_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
            "content": "259.75 USD"
        }
        """
        return {
            "role": "tool",
            "tool_call_id": self.tool_use_id,
            "content": self.process_content()
        }

    @staticmethod
    def parse_content(content) -> str:
        """Parse and normalize tool result content into a string format."""
        if content is None:
            return "No content provided"

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            result_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    result_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    result_parts.append(item)
                elif isinstance(item, dict):
                    if "text" in item:
                        result_parts.append(item.get("text", ""))
                    else:
                        try:
                            result_parts.append(json.dumps(item))
                        except:
                            result_parts.append(str(item))
            return "\n".join(result_parts).strip()

        if isinstance(content, dict):
            if content.get("type") == "text":
                return content.get("text", "")
            try:
                return json.dumps(content)
            except:
                return str(content)

        try:
            return str(content)
        except:
            return "Unparseable content"


class ContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str
    signature: Optional[str] = None

    def to_openai(self) -> None:
        """Thinking blocks should be filtered out for OpenAI format."""
        return None


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
                ContentBlockThinking
            ]
        ],
    ]

    def extract_tool_calls(self) -> List[Dict[str, Any]]:
        """
        Extract tool calls from Claude message content for OpenAI format.
        Returns a list of OpenAI tool_call objects.
        """
        tool_calls = []

        if isinstance(self.content, list):
            for block in self.content:
                if isinstance(block, ContentBlockToolUse):
                    tool_calls.append(block.to_openai())
                elif isinstance(block, dict) and block.get("type") == "tool_use":
                    # Fallback for dict format
                    tool_use = ContentBlockToolUse.model_validate(block)
                    tool_calls.append(tool_use.to_openai())

        return tool_calls

    def extract_tool_results(self) -> List[Dict[str, Any]]:
        """
        Extract tool result messages from Claude message content for OpenAI format.
        Returns a list of OpenAI tool role message objects.
        """
        tool_messages = []

        if isinstance(self.content, list):
            for block in self.content:
                if isinstance(block, ContentBlockToolResult):
                    tool_messages.append(block.to_openai())
                elif isinstance(block, dict) and block.get("type") == "tool_result":
                    # Fallback for dict format
                    tool_result = ContentBlockToolResult.model_validate(block)
                    tool_messages.append(tool_result.to_openai())

        return tool_messages

    def extract_text_content(self) -> str:
        """
        Extract only text content from message, completely ignoring tool-related blocks.
        Returns a string with just the actual text content, no tool placeholders.
        """
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            content_parts = []

            for block in self.content:
                if isinstance(block, ContentBlockText):
                    content_parts.append(block.text)
                elif isinstance(block, dict):
                    if block.get("type") == "text" and "text" in block:
                        content_parts.append(block["text"])
                # Completely ignore tool_use and tool_result blocks

            content_text = "".join(content_parts).strip()
            return content_text if content_text else ""

        return ""

    def process_interrupted_content(self) -> str:
        """Process Claude Code interrupted messages."""
        if not isinstance(self.content, str):
            return ""
        
        content = self.content
        if content.startswith("[Request interrupted by user for tool use]"):
            # Split the interrupted message
            interrupted_prefix = "[Request interrupted by user for tool use]"
            remaining_content = content[len(interrupted_prefix):].strip()
            return remaining_content if remaining_content else content
        
        return content

    def to_openai_messages(self) -> List[Dict[str, Any]]:
        """
        Convert Claude user message to OpenAI message format.
        Handles complex logic including tool_result splitting, content block ordering, etc.
        Returns a list of OpenAI messages (can be multiple due to tool_result splitting).
        """
        if self.role != "user":
            raise ValueError("This method is only for user messages")
        
        openai_messages = []
        
        # Handle simple string content
        if isinstance(self.content, str):
            processed_content = self.process_interrupted_content()
            openai_messages.append({"role": "user", "content": processed_content})
            return openai_messages
        
        # Process content blocks in order, maintaining structure
        current_text_parts = []
        content_parts = []
        
        for block in self.content:
            block_type = block.type if hasattr(block, "type") else block.get("type")
            
            if block_type == "text":
                text_content = (
                    block.text if hasattr(block, "text") else block.get("text", "")
                )
                current_text_parts.append(text_content)
                
            elif block_type in ["image", "thinking"]:
                # Process any accumulated text first
                if current_text_parts:
                    text_content = "".join(current_text_parts)
                    if text_content:
                        content_parts.append({"type": "text", "text": text_content})
                    current_text_parts.clear()
                
                # Convert and add non-text block using utility function
                openai_content = convert_content_block_to_openai(block)
                if openai_content:  # None for thinking blocks
                    content_parts.append(openai_content)
                    
            elif block_type == "tool_result":
                # Process any remaining text first
                if current_text_parts:
                    text_content = "".join(current_text_parts)
                    if text_content:
                        content_parts.append({"type": "text", "text": text_content})
                    current_text_parts.clear()
                
                # CRITICAL: Split user message when tool_result is encountered
                if content_parts:
                    if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                        openai_messages.append(
                            {"role": "user", "content": content_parts[0]["text"]}
                        )
                    else:
                        openai_messages.append(
                            {"role": "user", "content": content_parts}
                        )
                    content_parts.clear()

                # Add tool result immediately to maintain chronological order
                tool_message = convert_content_block_to_openai(block)
                if tool_message:
                    openai_messages.append(tool_message)

        # Process any remaining content
        if current_text_parts:
            text_content = "".join(current_text_parts)
            if text_content:
                content_parts.append({"type": "text", "text": text_content})

        if content_parts:
            if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                openai_messages.append(
                    {"role": "user", "content": content_parts[0]["text"]}
                )
            else:
                openai_messages.append(
                    {"role": "user", "content": content_parts}
                )

        # Tool messages are now added immediately when encountered to maintain order
        
        return openai_messages

    def to_openai_assistant_message(self) -> Optional[Dict[str, Any]]:
        """
        Convert Claude assistant message to OpenAI assistant message format.
        Returns None if the message has no content or tool calls.
        """
        if self.role != "assistant":
            raise ValueError("This method is only for assistant messages")
        
        # Extract tool calls and text content using existing methods
        tool_calls = self.extract_tool_calls()
        text_content = self.extract_text_content()

        assistant_msg = {"role": "assistant"}

        # Handle content for assistant messages
        if text_content:
            assistant_msg["content"] = text_content
        else:
            assistant_msg["content"] = None

        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        # Only return message if it has actual content or tool calls
        if assistant_msg.get("content") or assistant_msg.get("tool_calls"):
            return assistant_msg
        
        return None

    @staticmethod
    def clean_tool_markers(content: str) -> str:
        """Clean up tool call markers and malformed content (DeepSeek-R1 style)."""
        if not content:
            return content

        # Remove specific tool call markers
        content = re.sub(r"<｜tool▁call▁end｜>", "", content)
        content = re.sub(r"<\|tool_call_end\|>", "", content)

        # Remove DeepSeek-R1 function call blocks completely
        patterns_to_remove = [
            # Pattern 1: function FunctionName\n```json\n{...}\n```
            r"function\s+\w+\s*\n\s*```json\s*\n\s*{[\s\S]*?}\s*\n\s*```",
            # Pattern 2: function FunctionName ```json\n{...}\n```
            r"function\s+\w+\s*```json\s*\n\s*{[\s\S]*?}\s*\n\s*```",
            # Pattern 3: function FunctionName\n```json{...}```
            r"function\s+\w+\s*\n?\s*```json\s*{.*?}\s*```",
            # Pattern 4: More flexible - function name followed by any JSON in code blocks
            r"function\s+\w+[\s\S]*?```(?:json)?\s*{[\s\S]*?}\s*```",
        ]

        for pattern in patterns_to_remove:
            content = re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE)

        # Remove standalone function names that appear by themselves
        content = re.sub(r"^\s*function\s+\w+\s*$", "", content, flags=re.MULTILINE)

        # Clean up excessive whitespace
        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)
        content = re.sub(r"^\s*\n+", "", content)  # Remove leading newlines
        content = content.strip()

        return content

    @staticmethod
    def parse_tool_calls_from_content(content: str) -> List[Dict[str, Any]]:
        """Extract tool calls from malformed content (DeepSeek-R1 style)."""
        tool_calls = []

        # DeepSeek-R1 specific patterns
        patterns = [
            # Pattern 1: function FunctionName\n```json\n{...}\n```
            r"function\s+(\w+)\s*\n\s*```json\s*\n\s*({[\s\S]*?})\s*\n\s*```",
            # Pattern 2: function FunctionName ```json\n{...}\n```
            r"function\s+(\w+)\s*```json\s*\n\s*({[\s\S]*?})\s*\n\s*```",
            # Pattern 3: function FunctionName\n```json{...}```
            r"function\s+(\w+)\s*\n?\s*```json\s*({.*?})\s*```",
            # Pattern 4: More flexible - function name followed by any JSON in code blocks
            r"function\s+(\w+)[\s\S]*?```(?:json)?\s*({[\s\S]*?})\s*```",
        ]

        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    function_name = match[0]
                    json_str = match[1].strip()

                    # Clean up the JSON string
                    json_str = re.sub(r"\n\s*", " ", json_str)  # Remove extra whitespace
                    json_str = re.sub(r",\s*}", "}", json_str)  # Remove trailing commas
                    json_str = re.sub(
                        r",\s*]", "]", json_str
                    )  # Remove trailing commas in arrays

                    try:
                        arguments = json.loads(json_str)
                        tool_call_id = f"toolu_{uuid.uuid4().hex[:24]}"
                        tool_calls.append(
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": json.dumps(arguments)
                                    if isinstance(arguments, dict)
                                    else json_str,
                                },
                            }
                        )
                    except json.JSONDecodeError:
                        # Try to create a tool call with raw arguments
                        tool_call_id = f"toolu_{uuid.uuid4().hex[:24]}"
                        tool_calls.append(
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": json.dumps({"raw_input": json_str}),
                                },
                            }
                        )

                # If we found matches with this pattern, stop trying others
                if tool_calls:
                    break

        return tool_calls


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfigEnabled(BaseModel):
    type: Literal["enabled"] = "enabled"
    budget_tokens: Optional[int] = None


class ThinkingConfigDisabled(BaseModel):
    type: Literal["disabled"] = "disabled"


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ClaudeToolChoice] = None
    thinking: Optional[Union[ThinkingConfigEnabled, ThinkingConfigDisabled]] = None

    @field_validator("thinking")
    def validate_thinking_field(cls, v):
        if isinstance(v, dict):
            if v.get("enabled") is True:
                return ThinkingConfigEnabled(
                    type="enabled", budget_tokens=v.get("budget_tokens")
                )
            elif v.get("enabled") is False:
                return ThinkingConfigDisabled(type="disabled")
            elif v.get("type") == "enabled":
                return ThinkingConfigEnabled(
                    type="enabled", budget_tokens=v.get("budget_tokens")
                )
            elif v.get("type") == "disabled":
                return ThinkingConfigDisabled(type="disabled")
        return v

    def extract_system_content(self) -> str:
        """Extract system content from various formats."""
        if not self.system:
            return ""
        
        if isinstance(self.system, str):
            return self.system
        elif isinstance(self.system, list):
            system_content = ""
            for block in self.system:
                if hasattr(block, "text"):
                    system_content += block.text
                elif isinstance(block, dict) and "text" in block:
                    system_content += block["text"]
            return system_content
        return ""


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[Union[ThinkingConfigEnabled, ThinkingConfigDisabled, dict]] = None
    tool_choice: Optional[Dict[str, Any]] = None

    @field_validator("thinking")
    def validate_thinking_field(cls, v):
        if isinstance(v, dict):
            if v.get("enabled") is True:
                return ThinkingConfigEnabled(
                    type="enabled", budget_tokens=v.get("budget_tokens")
                )
            elif v.get("enabled") is False:
                return ThinkingConfigDisabled(type="disabled")
            elif v.get("type") == "enabled":
                return ThinkingConfigEnabled(
                    type="enabled", budget_tokens=v.get("budget_tokens")
                )
            elif v.get("type") == "disabled":
                return ThinkingConfigDisabled(type="disabled")
        return v


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse, ContentBlockThinking]]
    type: Literal["message"] = "message"
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "error"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


# Utility function for converting content blocks to OpenAI format
def convert_content_block_to_openai(block: Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult, ContentBlockThinking, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Convert Claude content block to OpenAI format, handling both Pydantic models and dict formats.
    
    Args:
        block: Content block in either Pydantic model or dict format
    
    Returns:
        OpenAI-formatted content or None if block should be filtered out
    """
    # Handle Pydantic model instances
    if isinstance(block, (ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult, ContentBlockThinking)):
        return block.to_openai()
    
    # Handle dict format (legacy/raw API)
    elif isinstance(block, dict):
        block_type = block.get("type")
        
        if block_type == "text":
            return {"type": "text", "text": block.get("text", "")}
        
        elif block_type == "image":
            source = block.get("source", {})
            if (
                isinstance(source, dict)
                and source.get("type") == "base64"
                and "media_type" in source
                and "data" in source
            ):
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{source['media_type']};base64,{source['data']}"
                    },
                }
            return None
        
        elif block_type == "tool_use":
            import json
            tool_input = block.get("input", {})
            try:
                arguments_str = json.dumps(tool_input, ensure_ascii=False, separators=(',', ':'))
            except (TypeError, ValueError):
                arguments_str = "{}"
            
            return {
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": arguments_str
                }
            }
        
        elif block_type == "tool_result":
            # For tool_result in dict format, create a temporary ContentBlockToolResult to reuse logic
            temp_block = ContentBlockToolResult(
                type="tool_result",
                tool_use_id=block.get("tool_use_id", ""),
                content=block.get("content", "")
            )
            return temp_block.to_openai()
        
        elif block_type == "thinking":
            # Filter out thinking blocks
            return None
    
    # Unknown format
    return None
