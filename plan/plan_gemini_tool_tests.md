# Plan: Gemini Tool Mapping Tests

Goal: Add automated tests (no manual server.log tail) to validate Gemini tool mapping + streaming tool calls using SDK types, and reorganize tests under `tests/gemini`.

Phases:
1) Inventory current tests + Gemini tool paths; capture gaps.
2) Define `tests/gemini` layout and fixtures (SDK types + raw chunks).
3) Implement mock builders + unit tests for tool-call variants using SDK types.
4) Reorganize/move tests + update imports.
5) Run targeted tests + document results.

Unresolved questions:
- Which tool-call variants are mandatory beyond functionCall/functionCalls/function_call?
