# Agent Train Code Review

## Status
Reviewed and fixed

## Date
2026-06-22

## Scope
This review covers the new Agent Train implementation on the `agent` branch:

- `mikazuki/agent_train/*`
- `mikazuki/app/agent_train_api.py`
- `mikazuki/app/application.py`
- `frontend/dist/agent-train.html`
- `frontend/dist/index.html`
- `tests/test_agent_train_*.py`
- LangGraph and LangChain dependency additions in `pyproject.toml` / `uv.lock`

## Review Summary
The feature now has a reasonable first production slice: browser chat, session persistence, LangGraph orchestration, explicit approval before side-effect tools, and integration with existing Tagger / Differential LoRA / Anima training endpoints.

The initial implementation was functionally close, but several edge cases could lead to stale UI state, inconsistent API errors, or poor recovery after a failed tool call. These were fixed in this pass and covered by regression tests.

## Findings Fixed

### 1. Stale plan and pending action after a non-action chat turn
Severity: Required

Problem: `/api/agent-train/chat` only updated `plan` and `pending_action` when the graph returned truthy values. If a later user message produced no plan, the previous pending tool action stayed visible and could still be approved.

Fix:
- `chat` now always writes `result.get("plan")`.
- `chat` now always writes `result.get("pending_action")`.
- `AgentSessionStore.set_pending_action()` accepts `None` to clear pending actions.

Regression test:
- `test_chat_endpoint_clears_stale_plan_and_pending_action`

### 2. Invalid session ids returned inconsistent errors
Severity: Required

Problem: `get_session` caught invalid ids, but `chat` and `status` did not catch `ValueError`. A malformed id such as `../bad` could produce an unstructured server error path.

Fix:
- `chat` catches `ValueError` during session lookup.
- `status` catches `ValueError`.

Regression tests:
- `test_chat_endpoint_rejects_invalid_session_id`
- `test_status_endpoint_rejects_invalid_session_id`

### 3. Failed tool responses consumed the pending action
Severity: Required

Problem: If an existing tool endpoint returned `APIResponseFail`, `approve_pending_action()` still treated that response as execution completion, cleared the pending action, and appended an "executed" message.

Fix:
- Failed tool responses are stored in `last_tool_result`.
- The pending action remains available for correction/retry.
- The assistant message now records that execution failed.

Regression test:
- `test_failed_tool_response_keeps_pending_action`

### 4. Frontend pending-only render could crash
Severity: Required

Problem: `renderPlan()` accessed `plan.title` and `plan.type` even when only `pending_action` existed. This is possible after partial session recovery or future API changes.

Fix:
- `renderPlan()` now uses `const activePlan = plan || {}`.
- Session restore failure also clears the in-memory `session_id`.

Regression test:
- `test_agent_train_page_contains_required_api_wiring` now checks the pending-only guard.

### 5. Runtime state directory was tied only to current working directory
Severity: Consideration addressed

Problem: session and checkpoint files defaulted to `Path.cwd()`. That is fine for the current launcher, but fragile if the app is started from another directory.

Fix:
- Added `AGENT_TRAIN_STATE_DIR` override.
- Default behavior remains `.cache/agent_train` under current working directory for compatibility.

## Verification
Commands run:

```bash
python -m unittest discover -s tests -p 'test_agent_train*.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_agent_train*.py' -v
python -m py_compile mikazuki/agent_train/__init__.py mikazuki/agent_train/graph.py mikazuki/agent_train/sessions.py mikazuki/agent_train/tools.py mikazuki/app/agent_train_api.py
```

Result:
- Agent Train tests pass: 16 tests.
- Syntax compilation passes.

Observed warnings:
- `pkg_resources` deprecation from existing `mikazuki/launch_utils.py`.
- `websockets.WebSocketClientProtocol` deprecation from existing `mikazuki/app/proxy.py`.

These warnings are pre-existing compatibility warnings and were not caused by this review pass.

## Remaining Risks

### Agent planning is still heuristic-first
The graph currently uses deterministic keyword planning and optionally lets a configured OpenAI-compatible chat model rewrite the response. It does not yet use structured LLM tool planning. This is acceptable for the first safe slice because tool execution remains approval-gated, but future work should add structured output validation before relying on model-generated tool payloads.

### Tool payload validation is delegated to existing endpoints
Agent Train passes approved JSON to existing Tagger / training APIs. This keeps boundaries simple, but the agent layer does not yet provide a schema-specific editor for each tool. User-facing validation can be improved by adding typed request models per tool kind.

### Static frontend is manually edited
`frontend/dist/index.html` is generated-style output and was patched directly to add the Agent Train entry. A source-level VuePress/sidebar integration would be cleaner if the project has a repeatable frontend build path available.

## Files Changed In This Review Pass

- `mikazuki/agent_train/sessions.py`
- `mikazuki/agent_train/graph.py`
- `mikazuki/app/agent_train_api.py`
- `frontend/dist/agent-train.html`
- `tests/test_agent_train_api.py`
- `tests/test_agent_train_static.py`
- `docs/agent-train-code-review.md`
