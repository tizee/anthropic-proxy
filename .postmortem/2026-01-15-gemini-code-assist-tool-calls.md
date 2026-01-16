# Postmortem: Gemini Code Assist tool calls not executed

## Severity
High

## Root Cause
Gemini Code Assist returns tool calls in multiple encodings:
- Proper `functionCall` parts (structured)
- `<tool_code>` JSON embedded inside text parts (unstructured)

Our streaming converter only handled `functionCall` parts, so when Code Assist used `<tool_code>` in text, no tool events were emitted. This made the client appear to “stop early” with only text output.

## Causing Commits
- N/A (issue surfaced during ongoing refactor + provider behavior mismatch, not tied to a single commit)

## Fix Implementation
- Added a `<tool_code>` streaming parser that stitches text across chunks, extracts JSON payloads, and emits Anthropic `tool_use` events.
- Implemented tool name inference from payload keys (e.g., `todos` → `TodoWrite`, `file_path`+`content` → `Write`, `command` → `Bash)`, with schema-matching fallback.
- Centralized Gemini response normalization in `gemini_types.py` (snake_case/camelCase keys, functionCall normalization) to avoid scattered fixes.
- Preserved thought signatures by caching signatures from functionCall parts and attaching them to subsequent tool_use blocks.
- Added tests in `tests/gemini` for tool_code parsing, streaming tool conversion, thought signature caching, and normalization paths.

## Key Takeaways
- Code Assist can emit tool calls inside text `<tool_code>` blocks, not just structured `functionCall`.
- Streaming conversion must handle provider-specific variations (snake_case vs camelCase, tool payloads in text).
- Thought signatures are required for tool continuity on Gemini 3; caching must consider signatures embedded in functionCall parts.

## Prevention
- Maintain a dedicated Gemini normalization layer (`gemini_types.py`) used by both SDK transport and streaming conversion.
- Add regression fixtures from real Code Assist streams covering:
  - functionCall parts
  - `<tool_code>` payloads split across chunks
  - snake_case keys (finish_reason, usage_metadata)
- Keep a per-provider compatibility checklist (tool calls + thought signatures + schema constraints) updated with doc references.

## Detection & Evidence
- server.log showed repeated `parts=1 function_calls=0 finishReason=None` and `<tool_code>` in raw chunks.
- Tool execution never triggered despite tool payloads appearing in text.

## Impact
- Tool calls were silently dropped for Code Assist responses using `<tool_code>`, leading to incomplete workflows (no TodoWrite/Write execution).

## Resolution Timeline (high level)
- Observed tool call failures and missing thought signatures (400 errors).
- Added thought signature handling for functionCall parts.
- Discovered `<tool_code>` payloads in text streams and implemented parser.
- Added tests and verified tool calls now emit `tool_use` correctly.
