# Coding Agent Demo

A local coding-agent demo with a bounded engineering workflow, a compact CLI, and a small web operator surface.

The project is intentionally constrained:
- single agent
- Python repositories only
- bounded write scope
- explicit workflow approvals
- persisted session summaries and event logs

The main execution loop is:

`clarify -> plan -> read -> proposal -> edit -> test -> review -> report`

This repo includes two preset demo tasks for the bundled `demo_repo`, and it also supports generic bounded edits for other local Python repositories when no preset handler matches.

## What It Shows

- repo-aware retrieval before planning and editing
- proposal generation separate from real file edits
- approval-scoped workflow tools for read, research, edit, test, and review
- compact terminal output by default, with verbose audit available when needed
- local web console for running tasks and inspecting sessions
- optional external web research that is recorded as evidence but does not replace local grounding
- persisted session artifacts under `runtime/sessions/<session_id>/`

## Key User Surfaces

CLI:
- `python -m coding_agent run --task "..."`
- `python -m coding_agent run --task "..." --repo "C:\path\to\repo"`
- `python -m coding_agent run --task "..." --web-research --research-query "..."`
- `python -m coding_agent report --session-limit 5 --open`
- `python -m coding_agent serve --open`

Compatibility wrapper:
- `python run_agent.py ...`

Web console:
- start with `python -m coding_agent serve --open`
- default URL is `http://127.0.0.1:8765/`
- page shows current run status, step timeline, event timeline, changed files, and research evidence

## Repository Layout

```text
.
|- demo_repo/               # bundled baseline repo used by preset demo tasks
|- src/coding_agent/        # core implementation
|- tests/                   # project test suite
|- run_agent.py             # wrapper entrypoint for local execution
|- pyproject.toml
`- README.md
```

## Requirements

- Python `3.11+`
- no mandatory third-party runtime dependency for the offline path
- `QWEN_API_KEY` only if you want live model-backed execution

## Quick Start

Run the tests:

```bash
python -m unittest discover -s tests -q
```

Reset the bundled demo repo to the seeded baseline:

```bash
python run_agent.py reset-demo
```

Run the preset Todo API task:

```bash
python run_agent.py run --task "Add priority sorting to the Todo API and fix the PATCH partial update bug."
```

Open the local operator surface:

```bash
python run_agent.py serve --open
```

Generate and open the static dashboard:

```bash
python run_agent.py report --session-limit 5 --open
```

## Optional Environment Variables

- `QWEN_API_KEY`: enable live Qwen calls
- `CODING_AGENT_MODEL`: defaults to `qwen-plus`
- `QWEN_API_BASE`: defaults to `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `CODING_AGENT_RUNTIME_DIR`: defaults to `runtime`
- `CODING_AGENT_DEMO_REPO`: defaults to `demo_repo`
- `CODING_AGENT_QWEN_TIMEOUT_SECONDS`: defaults to `45`
- `CODING_AGENT_QWEN_MAX_RETRIES`: defaults to `2`
- `CODING_AGENT_QWEN_RETRY_BACKOFF_SECONDS`: defaults to `1`
- `CODING_AGENT_WEB_RESEARCH_API_BASE`: external research endpoint
- `CODING_AGENT_WEB_RESEARCH_TIMEOUT_SECONDS`: external research timeout
- `CODING_AGENT_WEB_RESEARCH_MAX_RESULTS`: external research result cap
- `CODING_AGENT_WEB_RESEARCH_USER_AGENT`: external research user agent

## Runtime Artifacts

Each run writes persisted artifacts under `runtime/sessions/<session_id>/`:
- `summary.json`
- `events.jsonl`

`events.jsonl` is the real-time event stream used by the web console to show:
- active run status
- current step
- latest event message
- step timeline
- event timeline

## Positioning

This is a bounded local coding-agent demo with an operator surface. It is intentionally not an unrestricted general-purpose autonomous coding product.

## Note

这是我第一次上传代码，我想这是一个值得纪念的时刻，看看我们能在 AI 时代都留下些什么，共勉。
