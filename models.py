"""
Pydantic models for Claude proxy API requests and responses.
This module contains only the model definitions without any server startup code.
"""

from pydantic import BaseModel, field_validator
from typing import Dict, List, Optional, Union, Literal, Any

class ClaudeToolChoice(BaseModel):
    type: Literal["auto", "any", "tool", "none"] = "auto"
    function_name: str | None = None

    def to_openai(self) -> Union[Literal["auto", "required"], dict, None]:
        if self.type == "any":
            return "required"
        if self.type == "function" and self.function_name:
            return {
                "type": "function",
                "function": {"name": self.function_name}
            }
        if self.type == "auto":
            return "auto"
        if self.type == "none":
            return None


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


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


class ContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str
    signature: Optional[str] = None


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
