# Agent Train Skills

## Overview
Agent Train uses project-local runtime skills. A skill describes one action the web agent can plan and, after user approval, execute through the existing backend APIs.

This is separate from Codex or `AGENTS.md` guidance skills. Agent Train skills run inside this application and must keep a strict execution boundary.

## Built-in Skills
The initial registry is defined in `mikazuki/agent_train/skills.py` and contains:

| Skill ID | Action | Side Effect |
| --- | --- | --- |
| `tagger_leaf_batch_caption` | Tag-Edit-Leaf batch captioning | `writes_files` |
| `differential_lora_train` | Differential LoRA training | `gpu_training` |
| `anima_lora_train` | Anima LoRA training | `gpu_training` |
| `anima_fast_train` | Anima Fast training | `gpu_training` |

## Skill Contract
Each skill is an `AgentSkill` with these fields:

- `id`: stable machine-readable identifier.
- `title`: user-facing label shown in the plan.
- `description`: concise explanation of what the skill does.
- `trigger_keywords`: deterministic keyword fallback for the current planner.
- `input_schema`: JSON-schema-like metadata for expected inputs.
- `default_payload`: function that builds the initial API payload from the parsed path.
- `plan_type`: existing plan type kept for frontend/API compatibility.
- `action_kind`: executor action used by `AgentToolExecutor`.
- `side_effect_level`: one of `read_only`, `writes_files`, or `gpu_training`.
- `summary`: short user-facing risk/behavior bullets.
- `assistant_message`: default reply after the plan is generated.

## Default Payload Policy
Training skills should return a complete, editable JSON payload rather than a minimal request. Most fields should be populated from the existing schema defaults so the user can inspect and adjust the same parameters they would normally see in the training UI.

Current payload sizes:

- `anima_lora_train`: full Anima LoRA defaults, including model paths, dataset, save, optimizer, network, preview, logging, caption, precision/cache, and seed fields.
- `anima_fast_train`: full Anima Fast defaults, including plugin/runtime, dataset cache, save, optimizer, preview, compile, and network fields.
- `differential_lora_train`: full Differential LoRA defaults, including pairing, model, LoRA, optimizer, auto-tagging, logging, sampling, and postprocess fields.

Optional values that the user usually needs to fill manually may remain as empty strings. Existing backend sanitization removes empty strings before writing TOML for normal training, while Differential and Anima Fast adapters ignore empty optional fields.

## Execution Boundary
Skills with `writes_files` or `gpu_training` must never execute directly from chat generation. They must create a `pending_action` and wait for `/api/agent-train/approve`.

Failed tool responses keep the pending action available so the user can edit the JSON payload and retry.

## Planner, Validation, and Decision Log
Agent Train has a structured planner boundary before it creates a plan:

- If the user configured an OpenAI-compatible model, the planner asks the model for a strict JSON decision containing `skill_id`, `slots`, `confidence`, `reason`, and `questions`.
- Invalid JSON, unknown skills, low confidence, model errors, or missing model config fall back to the deterministic keyword planner.
- The model is only allowed to plan. It never directly executes a tool.

Before any `pending_action` is exposed, the agent validates the generated payload:

- Missing required fields and incompatible training types are blocking `validation.errors`.
- Suspicious but sometimes valid runtime conditions, such as absolute paths that do not exist in the current container, are non-blocking `validation.warnings`.
- The frontend hides execution controls when validation errors exist.
- `/api/agent-train/approve` validates the final payload again, including user-edited JSON, before any executor is called.

Read-only inspection tools run during planning and never require approval:

- `inspect_dataset` checks whether a dataset path exists, whether it is a directory, and counts images/captions.
- `inspect_model_file` checks whether configured model files are accessible and records file size.
- Inspection results are returned as `plan.readonly_checks`; they are advisory and do not write files or start GPU work.

Each chat turn appends a compact `decision_log` entry to the session. The log records planner source, selected skill, extracted slots, missing slots, validation result, whether a pending action was created, and the reason/fallback reason. API keys are redacted before session data is written.

## Slot Collection and Workflow
Agent Train now collects required slots before exposing an executable action:

- Tagger requires an image folder path.
- Anima LoRA and Anima Fast require the training folder, base model, VAE, and Qwen3 paths.
- Differential LoRA requires `folder_a`, `folder_b`, base model, VAE, and Qwen3 paths.

If any required slot is missing, the session stores `slots`, `missing_slots`, and `workflow.status="collecting_slots"`. The frontend shows the missing slot list and hides execution buttons until the missing values are provided.

Multi-step requests use a workflow queue under `workflow.steps`. For example, "tag this folder then train Anima Fast" creates steps for Tagger and Anima Fast. The first step waits for approval; after the tagger succeeds, its status becomes `completed`, the next step becomes `collecting_slots` or `awaiting_confirmation`, and GPU training still requires a separate confirmation.

The legacy `workflow.current_skill_id`, `workflow.next_skill_id`, and `workflow.status` fields remain for frontend/API compatibility.

The frontend exposes a key-parameter form above the full JSON editor. Editing key fields updates the JSON payload, and "提交关键参数" sends those values back through `/api/agent-train/chat` so the backend slot state remains authoritative.

## Adding a Skill
1. Add an `AgentSkill` entry to `BUILTIN_SKILLS`.
2. Add or reuse an executor method in `AgentToolExecutor`.
3. Add required slot definitions to `SLOT_DEFINITIONS` when execution needs user-provided values.
4. Add validation rules in `validate_skill_payload` for required fields, training type consistency, and risky numeric values.
5. Ensure the skill's `action_kind` is unique.
6. Add tests for registry uniqueness, slot collection, planner matching, validation, decision log metadata, and pending action metadata.
7. If the frontend needs custom display, update `frontend/dist/agent-train.html`; otherwise the generic key-field + JSON editor is enough.

## Current Limitation
The planner can ask a configured chat model for structured JSON decisions, but the deterministic planner remains the reliability fallback. When a request contains an API key, LangGraph checkpointing is bypassed for that turn so secrets are not written to checkpoint SQLite. Dynamic external skill loading is not implemented; new runtime skills still require code changes so executor boundaries stay explicit and reviewable.
