from google.genai import types as genai_types


def build_function_call_part(
    *,
    name: str = "get_weather",
    args: dict | None = None,
    call_id: str | None = "call-1",
) -> genai_types.Part:
    call = genai_types.FunctionCall(
        name=name,
        args=args or {},
        id=call_id,
    )
    return genai_types.Part(functionCall=call)


def build_response_with_parts(
    *,
    parts: list[genai_types.Part],
    finish_reason: genai_types.FinishReason = genai_types.FinishReason.STOP,
) -> dict:
    content = genai_types.Content(parts=parts)
    candidate = genai_types.Candidate(content=content, finishReason=finish_reason)
    response = genai_types.GenerateContentResponse(candidates=[candidate])
    return response.model_dump(exclude_none=True)
