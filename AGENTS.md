# Codex Project Guidance

This file gives Codex durable project instructions. It should improve automatic
skill selection without requiring the user to name a skill explicitly.

## Project Defaults

- Read relevant source, config, and tests before editing.
- Keep edits narrowly scoped to the user's request.
- Prefer existing project patterns over new abstractions.
- Do not overwrite, revert, delete, or reformat user changes unless explicitly asked.
- Do not delete datasets, model files, checkpoints, LoRA outputs, downloads, or generated training artifacts unless the user explicitly asks.
- Prefer the project Python environment at `.venv/bin/python` when running project scripts.
- Use `rg`/`rg --files` for search when available.
- Use `apply_patch` for manual file edits.

## Skill Routing

Use the installed Codex skills automatically when the task matches these intents:

- Use `debugging-and-error-recovery` when a command, training run, conversion script, build, or test fails.
- Use `test-driven-development` when fixing bugs, changing behavior, or modifying model/tool conversion logic.
- Use `incremental-implementation` when a change touches multiple files or needs more than one implementation step.
- Use `frontend-ui-engineering` for user-facing UI, layout, component, browser, or visual behavior changes.
- Use `browser-testing-with-devtools` when verifying browser behavior or debugging frontend runtime issues.
- Use `api-and-interface-design` when changing public APIs, schemas, request/response shapes, or module boundaries.
- Use `security-and-hardening` when handling paths, user input, subprocesses, auth, secrets, file uploads, or external integrations.
- Use `source-driven-development` when framework/library behavior must be checked against official documentation.
- Use `code-review-and-quality` for review requests or before considering a substantial change ready.
- Use `git-workflow-and-versioning` for code changes, commits, branches, diffs, or worktree questions.
- Use `documentation-and-adrs` when adding durable docs, architectural decisions, or public workflow guidance.
- Use `openai-docs` for Codex, OpenAI API, skills, plugins, MCP, hooks, or `AGENTS.md` questions.

When multiple skills apply, use the smallest useful set. Prefer this order:

1. Understand and design with the relevant design/source skill.
2. Implement in small increments.
3. Verify with tests or targeted runtime checks.
4. Review the resulting diff for regressions.

## Project-Specific Verification

- For LoRA, Differential LoRA, ComfyUI conversion, safetensors, or model utility changes, run a lightweight validation where practical: key detection, tensor shapes, CLI argument parsing, or a small synthetic safetensors test.
- For scripts under `tools/`, prefer direct CLI checks with `.venv/bin/python`.
- If `pytest` is unavailable in the environment, use a focused Python assertion script and state that full pytest execution was not available.
- For frontend changes, run the local dev server when required and verify the rendered page in a browser.

## Safety Boundaries

- Treat files in `dataset/`, `downloads/`, `models/`, and training output directories as user data.
- Ask before running long training jobs, installing dependencies, or performing destructive cleanup.
- Never expose secrets, tokens, API keys, or private dataset contents in final responses.
- If the repo state is dirty, work with the existing changes and mention unrelated dirty files only when relevant.
