"""
Pydantic models for Claude proxy API requests and responses.
This module contains only the model definitions without any server startup code.
"""

from pydantic import BaseModel, field_validator
from typing import Dict, List, Optional, Union, Literal, Any


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


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


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
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[Union[ThinkingConfigEnabled, ThinkingConfigDisabled]] = None
    original_model: Optional[str] = None  # Will store the original model name

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
    original_model: Optional[str] = None  # Will store the original model name

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
