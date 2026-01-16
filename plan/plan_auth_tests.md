# Plan: auth_tests

1) Audit current auth classes for Gemini/Antigravity (login, refresh, project resolution) and identify stable seams to mock. (done)
2) Design tests mirroring Codex auth coverage: valid token path, refresh path, refresh failure, login bootstraps local server. (done)
3) Implement tests for GeminiAuth and AntigravityAuth using auth_provider patch points; run targeted tests. (done)

Unresolved questions:
- None.

Notes:
- Login tests mock browser open to avoid launching real auth pages.
