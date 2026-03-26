from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coding_agent.config import AppConfig
from coding_agent.events import make_event
from coding_agent.render import TerminalRenderer
from coding_agent.storage import JsonlRunStore
from coding_agent.webapp import WebRunCoordinator, build_console_state, render_web_console_html


class _BlockingRunner:
    def __init__(
        self,
        started: threading.Event,
        release: threading.Event,
        calls: list[dict[str, object]],
    ) -> None:
        self.started = started
        self.release = release
        self.calls = calls

    def run(
        self,
        task_text: str,
        repo_path: str | None = None,
        max_fix: int = 1,
        session_name: str | None = None,
        session_id: str | None = None,
        enable_web_research: bool = False,
        research_query: str | None = None,
    ) -> None:
        _ = max_fix
        self.calls.append(
            {
                "task_text": task_text,
                "session_name": session_name,
                "repo_path": repo_path,
                "session_id": session_id,
                "enable_web_research": enable_web_research,
                "research_query": research_query,
            }
        )
        self.started.set()
        self.release.wait(timeout=1.0)


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="coding-agent-web-test-"))
        runtime_dir = self.temp_dir / "runtime"
        self.config = AppConfig(
            base_dir=self.temp_dir,
            runtime_dir=runtime_dir,
            sessions_dir=runtime_dir / "sessions",
            default_repo_dir=self.temp_dir / "demo_repo",
            qwen_model="qwen-plus",
            qwen_api_key=None,
            qwen_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            qwen_timeout_seconds=45.0,
            qwen_max_retries=2,
            qwen_retry_backoff_seconds=1.0,
        )
        self.config.ensure_directories()
        self.store = JsonlRunStore(self.config.sessions_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _write_summary(self, session_id: str = "sess_live") -> None:
        session_dir = self.config.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "session_id": session_id,
            "session_name": "web_console_demo",
            "status": "completed",
            "created_at": "2026-03-26T13:00:00Z",
            "task_handler": "todo_priority_patch",
            "request": {"user_text": "Add priority sorting to the Todo API and fix the PATCH partial update bug."},
            "fallback_steps": [],
            "tool_calls": [
                {"step_type": "research", "tool_name": "web_researcher", "status": "completed", "approval_mode": "auto_allow"},
                {"step_type": "read", "tool_name": "repo_reader", "status": "completed", "approval_mode": "auto_allow"},
                {"step_type": "review", "tool_name": "review_compiler", "status": "completed", "approval_mode": "auto_allow"},
            ],
            "approval_checks": [{"approved": True}] * 3,
            "retrieved_files": [
                {"path": "todo_api.py", "reasons": ["required file", "handler priority"]},
                {"path": "tests/test_todo_api.py", "reasons": ["required file", "regression coverage file"]},
            ],
            "changed_files": [
                {"path": "todo_api.py", "summary": "Implement priority-aware todo ordering.", "diff_excerpt": "+_PRIORITY_ORDER = {'high': 0}"},
            ],
            "test_results": [
                {"command": ["python", "-m", "unittest", "discover", "-s", "tests", "-q"], "exit_code": 0, "duration_ms": 111}
            ],
            "review_summary": "Behavior change: list_todos now sorts by priority. External research: used Python docs (https://docs.python.org/3/). Validation: python -m unittest discover -s tests -q passed in 111 ms.",
            "final_summary": "Run completed successfully.",
            "proposal_assessment": {"status": "accepted", "score": 100, "used_fallback": False},
            "research_enabled": True,
            "research_query": "python sort key reverse order",
            "research_summary": "External research captured 1 source(s) for 'python sort key reverse order': Python docs.",
            "research_sources": [
                {"title": "Python docs", "url": "https://docs.python.org/3/", "snippet": "sorted supports key functions."}
            ],
        }
        (session_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self.store.append_event(make_event(session_id, "step_started", {"step_id": "step_research", "step_type": "research", "input_summary": "python sort key reverse order"}))
        self.store.append_event(make_event(session_id, "step_completed", {"step_id": "step_research", "step_type": "research", "output_summary": "External research captured 1 source."}))
        self.store.append_event(make_event(session_id, "step_started", {"step_id": "step_review", "step_type": "review", "input_summary": "compile final review"}))
        self.store.append_event(make_event(session_id, "step_completed", {"step_id": "step_review", "step_type": "review", "output_summary": "Behavior change: list_todos now sorts by priority."}))

    def test_build_console_state_includes_selected_session_details_steps_and_research(self) -> None:
        self._write_summary()
        coordinator = WebRunCoordinator(self.config, self.store, runner_factory=lambda renderer: _BlockingRunner(threading.Event(), threading.Event(), []))

        state = build_console_state(self.config, self.store, coordinator, selected_session_id="sess_live", session_limit=5)

        self.assertEqual(state["session_metrics"]["total_sessions"], 1)
        self.assertFalse(state["run_status"]["running"])
        self.assertEqual(state["default_repo_path"], str(self.config.default_repo_dir))
        self.assertEqual(state["selected_session"]["session_id"], "sess_live")
        self.assertEqual(state["selected_session"]["validation"], "python -m unittest discover -s tests -q | passed | 111 ms")
        self.assertEqual(len(state["selected_session"]["changed_files"]), 1)
        self.assertEqual(len(state["selected_session"]["steps"]), 2)
        self.assertTrue(state["selected_session"]["research_enabled"])
        self.assertEqual(state["selected_session"]["research_query"], "python sort key reverse order")
        self.assertEqual(len(state["selected_session"]["research_sources"]), 1)
        self.assertEqual(len(state["selected_session"]["events"]), 4)

    def test_build_console_state_exposes_active_run_step_status(self) -> None:
        started = threading.Event()
        release = threading.Event()
        calls: list[dict[str, object]] = []

        def runner_factory(renderer: TerminalRenderer) -> _BlockingRunner:
            _ = renderer
            return _BlockingRunner(started, release, calls)

        coordinator = WebRunCoordinator(self.config, self.store, runner_factory=runner_factory)
        session_id = coordinator.start_run("demo task")
        self.assertTrue(started.wait(timeout=1.0))
        self.store.append_event(make_event(session_id, "step_started", {"step_id": "step_clarify", "step_type": "clarify", "input_summary": "demo task"}))

        state = build_console_state(self.config, self.store, coordinator, session_limit=5)

        self.assertTrue(state["run_status"]["running"])
        self.assertEqual(state["run_status"]["active_session_id"], session_id)
        self.assertEqual(state["run_status"]["current_step"], "clarify")
        self.assertIn("clarify started", state["run_status"]["last_event_message"])

        release.set()

    def test_coordinator_rejects_second_run_while_active_and_passes_repo_path_session_id_and_research_flags(self) -> None:
        started = threading.Event()
        release = threading.Event()
        calls: list[dict[str, object]] = []

        def runner_factory(renderer: TerminalRenderer) -> _BlockingRunner:
            _ = renderer
            return _BlockingRunner(started, release, calls)

        coordinator = WebRunCoordinator(self.config, self.store, runner_factory=runner_factory)

        session_id = coordinator.start_run(
            "demo task",
            session_name="web-demo",
            repo_path="E:/repo",
            enable_web_research=True,
            research_query="python partial update merge dict",
        )
        self.assertTrue(started.wait(timeout=1.0))

        with self.assertRaisesRegex(RuntimeError, "already in progress"):
            coordinator.start_run("another task")

        release.set()
        deadline = time.time() + 1.0
        while coordinator.status().running and time.time() < deadline:
            time.sleep(0.02)

        self.assertFalse(coordinator.status().running)
        self.assertEqual(calls[0]["task_text"], "demo task")
        self.assertEqual(calls[0]["session_name"], "web-demo")
        self.assertEqual(calls[0]["repo_path"], "E:/repo")
        self.assertEqual(calls[0]["session_id"], session_id)
        self.assertTrue(calls[0]["enable_web_research"])
        self.assertEqual(calls[0]["research_query"], "python partial update merge dict")

    def test_render_web_console_html_contains_api_hooks_repo_input_and_research_controls(self) -> None:
        html = render_web_console_html()
        self.assertIn("Coding Agent Web Console", html)
        self.assertIn("/api/state", html)
        self.assertIn("/api/run", html)
        self.assertIn("Run Task", html)
        self.assertIn("repoPathInput", html)
        self.assertIn("enableResearchInput", html)
        self.assertIn("researchQueryInput", html)
        self.assertIn("Step Timeline", html)
        self.assertIn("Client error:", html)


if __name__ == "__main__":
    unittest.main()
