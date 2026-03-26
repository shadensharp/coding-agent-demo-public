from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from urllib import error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coding_agent.config import AppConfig
from coding_agent.demo_repo_seed import (
    BASELINE_TEST_TODO_API,
    BASELINE_TEST_TODO_QUERIES,
    BASELINE_TODO_API,
    BASELINE_TODO_QUERIES,
    restore_demo_repo,
)
from coding_agent.llm import QwenClient
from coding_agent.render import TerminalRenderer
from coding_agent.repo_ops import RepoOps
from coding_agent.runner import Runner
from coding_agent.storage import JsonlRunStore
from coding_agent.models import ResearchSource
from coding_agent.task_handlers import DEFAULT_VALIDATION_COMMAND
from coding_agent.workflow import EDIT_PROPOSAL_TOOL, PRESET_EDIT_TOOL, READ_REPO_TOOL, RESEARCH_TOOL, REVIEW_TOOL, TEST_COMMAND_TOOL


class FakeLlmClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.system_prompts: list[str | None] = []

    def is_configured(self) -> bool:
        return True

    def complete(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.1) -> str:
        _ = temperature
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        if not self.responses:
            raise RuntimeError("No fake LLM responses left")
        return self.responses.pop(0)


class FakeResearchClient:
    def __init__(self, sources: list[ResearchSource]) -> None:
        self.sources = list(sources)
        self.queries: list[str] = []

    def search(self, query: str, max_results: int | None = None) -> list[ResearchSource]:
        _ = max_results
        self.queries.append(query)
        return list(self.sources)


class QwenClientTests(unittest.TestCase):
    def test_complete_requires_api_key(self) -> None:
        config = AppConfig(
            base_dir=PROJECT_ROOT,
            runtime_dir=PROJECT_ROOT / "runtime",
            sessions_dir=PROJECT_ROOT / "runtime" / "sessions",
            default_repo_dir=PROJECT_ROOT / "demo_repo",
            qwen_model="qwen-plus",
            qwen_api_key=None,
            qwen_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            qwen_timeout_seconds=45.0,
            qwen_max_retries=2,
            qwen_retry_backoff_seconds=1.0,
        )
        client = QwenClient(config)

        with self.assertRaisesRegex(RuntimeError, "Qwen API key"):
            client.complete("hello")

    def test_complete_retries_transient_errors_then_succeeds(self) -> None:
        config = AppConfig(
            base_dir=PROJECT_ROOT,
            runtime_dir=PROJECT_ROOT / "runtime",
            sessions_dir=PROJECT_ROOT / "runtime" / "sessions",
            default_repo_dir=PROJECT_ROOT / "demo_repo",
            qwen_model="qwen-plus",
            qwen_api_key="sk-test",
            qwen_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            qwen_timeout_seconds=30.0,
            qwen_max_retries=2,
            qwen_retry_backoff_seconds=0.5,
        )
        attempts: list[int] = []
        sleeps: list[float] = []

        def transport(req: object, timeout: float) -> str:
            _ = req
            _ = timeout
            attempts.append(1)
            if len(attempts) == 1:
                raise error.URLError("temporary network issue")
            return json.dumps({"choices": [{"message": {"content": "Recovered response"}}]})

        client = QwenClient(config, transport=transport, sleep_fn=sleeps.append)

        self.assertEqual(client.complete("hello"), "Recovered response")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(sleeps, [0.5])

    def test_complete_does_not_retry_non_retryable_http_error(self) -> None:
        config = AppConfig(
            base_dir=PROJECT_ROOT,
            runtime_dir=PROJECT_ROOT / "runtime",
            sessions_dir=PROJECT_ROOT / "runtime" / "sessions",
            default_repo_dir=PROJECT_ROOT / "demo_repo",
            qwen_model="qwen-plus",
            qwen_api_key="sk-test",
            qwen_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            qwen_timeout_seconds=30.0,
            qwen_max_retries=3,
            qwen_retry_backoff_seconds=0.5,
        )
        attempts: list[int] = []
        sleeps: list[float] = []

        def transport(req: object, timeout: float) -> str:
            _ = req
            _ = timeout
            attempts.append(1)
            raise error.HTTPError(
                url="https://example.com",
                code=400,
                msg="bad request",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"bad request"}'),
            )

        client = QwenClient(config, transport=transport, sleep_fn=sleeps.append)

        with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
            client.complete("hello")
        self.assertEqual(len(attempts), 1)
        self.assertEqual(sleeps, [])

class RendererTests(unittest.TestCase):
    def test_summary_includes_schema_candidate_assessment_audit_and_diff_preview(self) -> None:
        stream = io.StringIO()
        renderer = TerminalRenderer(stream)
        renderer.render_session_summary(
            {
                "session_id": "sess_test",
                "status": "completed",
                "created_at": "2026-03-23T04:20:00Z",
                "task_handler": "todo_priority_patch",
                "approval_checks": [
                    {
                        "tool_name": EDIT_PROPOSAL_TOOL,
                        "step_type": "proposal",
                        "approved": True,
                        "mode": "auto_allow",
                    }
                ],
                "tool_calls": [
                    {
                        "tool_name": EDIT_PROPOSAL_TOOL,
                        "step_type": "proposal",
                        "status": "completed",
                        "approval_mode": "auto_allow",
                    }
                ],
                "retrieved_files": [
                    {
                        "path": "todo_api.py",
                        "score": 120,
                        "reasons": ["required file", "handler priority"],
                    }
                ],
                "clarify_artifact": {
                    "implementation_target": "Implement priority-aware ordering.",
                    "relevant_files": ["todo_api.py", "tests/test_todo_api.py"],
                    "validation_command": DEFAULT_VALIDATION_COMMAND,
                },
                "plan_artifact": {
                    "steps": ["inspect", "update", "test", "review"],
                    "target_files": ["todo_api.py", "tests/test_todo_api.py"],
                    "validation_command": DEFAULT_VALIDATION_COMMAND,
                },
                "read_summary": "Read step inspected: todo_api.py functions=create_todo, list_todos, apply_patch_update.",
                "proposal_summary": "Prepare a bounded diff candidate for todo_api.py and tests/test_todo_api.py. Candidate edits: update todo_api.py | targets list_todos, apply_patch_update | add priority-aware ordering; update tests/test_todo_api.py | targets TodoApiTests.test_list_todos_sorts_by_priority_then_id | cover regression behavior. Validation: python -m unittest discover -s tests -q.",
                "proposal_candidate": {
                    "summary_text": "Prepare a bounded diff candidate for todo_api.py and tests/test_todo_api.py.",
                    "validation_command": DEFAULT_VALIDATION_COMMAND,
                    "edits": [
                        {
                            "path": "todo_api.py",
                            "change_type": "update",
                            "target_symbols": ["list_todos", "apply_patch_update"],
                            "intent": "add priority-aware ordering",
                        },
                        {
                            "path": "tests/test_todo_api.py",
                            "change_type": "update",
                            "target_symbols": ["TodoApiTests.test_list_todos_sorts_by_priority_then_id"],
                            "intent": "cover regression behavior",
                        },
                    ],
                },
                "proposal_assessment": {
                    "status": "accepted",
                    "score": 90,
                    "matched_targets": ["todo_api.py", "tests/test_todo_api.py"],
                    "extra_targets": [],
                    "used_fallback": True,
                },
                "clarify_summary": "Clarify summary here.",
                "plan_summary": "1) plan",
                "final_summary": "Final summary here.",
                "review_summary": "Review: behavior changed. Proposal assessment: accepted score=90 source=fallback matched=todo_api.py, tests/test_todo_api.py. Proposal candidate: update todo_api.py. Audit: workflow used edit_proposal_generator. Validation: python -m unittest discover -s tests -q passed in 50 ms.",
                "fallback_steps": ["clarify:client_unconfigured", "plan:client_unconfigured", "proposal:client_unconfigured"],
                "request": {"user_text": "demo task"},
                "changed_files": [
                    {
                        "path": "todo_api.py",
                        "summary": "Updated behavior.",
                        "diff_excerpt": "--- a/todo_api.py\n+++ b/todo_api.py\n@@\n+_PRIORITY_ORDER = {'high': 0}\n+def list_todos(...):\n",
                    }
                ],
                "test_results": [
                    {
                        "command": ["python", "-m", "unittest"],
                        "cwd": "demo_repo",
                        "exit_code": 0,
                        "duration_ms": 50,
                    }
                ],
            }
        )
        rendered = stream.getvalue()
        self.assertIn("Task handler: todo_priority_patch", rendered)
        self.assertIn("Approvals: 1 checks | denied=0", rendered)
        self.assertIn("Tool calls: 1", rendered)
        self.assertIn("Workflow: proposal->edit_proposal_generator[completed,auto_allow]", rendered)
        self.assertIn("Retrieved files: 1", rendered)
        self.assertIn("Clarify artifact: files=todo_api.py, tests/test_todo_api.py", rendered)
        self.assertIn("Plan artifact: steps=4 targets=todo_api.py, tests/test_todo_api.py", rendered)
        self.assertIn("Proposal summary: Prepare a bounded diff candidate", rendered)
        self.assertIn("Proposal candidate: edits=2 validation=python -m unittest discover -s tests -q", rendered)
        self.assertIn("update todo_api.py: symbols=list_todos, apply_patch_update | intent=add priority-aware ordering", rendered)
        self.assertIn("Proposal assessment: status=accepted score=90 fallback=True matched=todo_api.py, tests/test_todo_api.py extra=none", rendered)
        self.assertIn("Read: Read step inspected: todo_api.py functions=create_todo, list_todos, apply_patch_update.", rendered)
        self.assertIn("Fallbacks: clarify:client_unconfigured, plan:client_unconfigured, proposal:client_unconfigured", rendered)
        self.assertIn("Review: Review: behavior changed.", rendered)
        self.assertIn("+_PRIORITY_ORDER", rendered)
        self.assertIn("Latest test: exit=0", rendered)

    def test_run_summary_renders_compact_user_facing_result(self) -> None:
        stream = io.StringIO()
        renderer = TerminalRenderer(stream, event_verbosity="compact", summary_verbosity="compact")
        renderer.render_run_summary(
            {
                "session_id": "sess_test",
                "status": "completed",
                "task_handler": "todo_priority_patch",
                "request": {"user_text": "demo task"},
                "retrieved_files": [
                    {"path": "todo_api.py"},
                    {"path": "tests/test_todo_api.py"},
                ],
                "changed_files": [
                    {"path": "todo_api.py", "summary": "Updated behavior."},
                ],
                "test_results": [
                    {
                        "command": ["python", "-m", "unittest", "discover", "-s", "tests", "-q"],
                        "exit_code": 0,
                        "duration_ms": 50,
                    }
                ],
                "review_summary": "Behavior change: list_todos now sorts by priority. Proposal assessment: accepted score=100.",
                "fallback_steps": [],
            }
        )
        rendered = stream.getvalue()
        self.assertIn("Run Result", rendered)
        self.assertIn("Handler: todo_priority_patch", rendered)
        self.assertIn("Grounding: todo_api.py, tests/test_todo_api.py", rendered)
        self.assertIn("Validation: passed | python -m unittest discover -s tests -q | 50 ms", rendered)
        self.assertIn("Review: Behavior change: list_todos now sorts by priority.", rendered)
        self.assertIn("python -m coding_agent report --session-limit 5 --open", rendered)
    def test_eval_summary_renders_case_lines_and_metrics(self) -> None:
        stream = io.StringIO()
        renderer = TerminalRenderer(stream)
        renderer.render_eval_summary(
            {
                "mode": "offline",
                "total_cases": 2,
                "passed_cases": 2,
                "failed_cases": 0,
                "pass_rate": 100.0,
                "fallback_cases": 2,
                "avg_retrieved_files": 2.0,
                "required_file_hit_rate": 100.0,
                "avg_changed_files": 2.0,
                "avg_latest_test_duration_ms": 30,
                "avg_fallback_steps_per_case": 3.0,
                "avg_tool_calls": 5.0,
                "avg_approval_checks": 5.0,
                "approval_denials": 0,
                "proposal_accept_rate": 100.0,
                "avg_proposal_score": 90.0,
                "avg_proposal_edits": 2.0,
                "fallback_step_counts": {
                    "clarify:client_unconfigured": 2,
                    "plan:client_unconfigured": 2,
                    "proposal:client_unconfigured": 2,
                },
                "results": [
                    {
                        "handler_name": "todo_priority_patch",
                        "status": "completed",
                        "retrieved_files": 2,
                        "required_file_hits": 2,
                        "required_file_total": 2,
                        "changed_files": 2,
                        "latest_test_duration_ms": 30,
                        "latest_test_exit": 0,
                        "fallback_steps": ["clarify:client_unconfigured", "proposal:client_unconfigured"],
                        "review_summary": "Behavior change: list_todos now sorts by priority.",
                        "tool_calls": 5,
                        "approval_checks": 5,
                        "denied_tool_calls": 0,
                        "proposal_status": "accepted",
                        "proposal_score": 90,
                        "proposal_edit_candidates": 2,
                    }
                ],
            }
        )
        rendered = stream.getvalue()
        self.assertIn("Eval mode: offline", rendered)
        self.assertIn("pass_rate=100.0%", rendered)
        self.assertIn("avg_retrieved_files=2.0", rendered)
        self.assertIn("required_file_hit_rate=100.0%", rendered)
        self.assertIn("Workflow: avg_tool_calls=5.0 avg_approval_checks=5.0 approval_denials=0", rendered)
        self.assertIn("Proposal: accept_rate=100.0% avg_score=90.0 avg_edits=2.0", rendered)
        self.assertIn("Fallback steps: clarify:client_unconfigured=2, plan:client_unconfigured=2, proposal:client_unconfigured=2", rendered)
        self.assertIn("todo_priority_patch: status=completed", rendered)
        self.assertIn("required_hits=2/2", rendered)
        self.assertIn("proposal=accepted:90", rendered)
        self.assertIn("proposal_edits=2", rendered)
        self.assertIn("tools=5 approvals=5 denied=0", rendered)
        self.assertIn("review=Behavior change: list_todos now sorts by priority.", rendered)


class RunnerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="coding-agent-test-"))
        self.repo_dir = self.temp_dir / "demo_repo"
        (self.repo_dir / "tests").mkdir(parents=True)
        (self.repo_dir / "README.md").write_text("# Demo Repo\n", encoding="utf-8")
        (self.repo_dir / "todo_api.py").write_text(BASELINE_TODO_API, encoding="utf-8")
        (self.repo_dir / "todo_queries.py").write_text(BASELINE_TODO_QUERIES, encoding="utf-8")
        (self.repo_dir / "math_utils.py").write_text(
            "from __future__ import annotations\n\n\ndef add(a: int, b: int) -> int:\n    return a + b\n\n\ndef subtract(a: int, b: int) -> int:\n    return a + b\n",
            encoding="utf-8",
        )
        (self.repo_dir / "tests" / "test_todo_api.py").write_text(BASELINE_TEST_TODO_API, encoding="utf-8")
        (self.repo_dir / "tests" / "test_todo_queries.py").write_text(BASELINE_TEST_TODO_QUERIES, encoding="utf-8")
        (self.repo_dir / "tests" / "test_math_utils.py").write_text(
            "from __future__ import annotations\n\nimport unittest\n\nfrom math_utils import add\n\n\nclass MathUtilsTests(unittest.TestCase):\n    def test_add(self) -> None:\n        self.assertEqual(add(2, 3), 5)\n\n\nif __name__ == \"__main__\":\n    unittest.main()\n",
            encoding="utf-8",
        )

        runtime_dir = self.temp_dir / "runtime"
        self.config = AppConfig(
            base_dir=self.temp_dir,
            runtime_dir=runtime_dir,
            sessions_dir=runtime_dir / "sessions",
            default_repo_dir=self.repo_dir,
            qwen_model="qwen-plus",
            qwen_api_key=None,
            qwen_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            qwen_timeout_seconds=45.0,
            qwen_max_retries=2,
            qwen_retry_backoff_seconds=1.0,
        )
        self.config.ensure_directories()
        self.store = JsonlRunStore(self.config.sessions_dir)
        self.renderer = TerminalRenderer(io.StringIO())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_runner_applies_todo_api_task_and_passes_tests(self) -> None:
        runner = Runner(config=self.config, store=self.store, renderer=self.renderer)

        session = runner.run(
            task_text="Add priority sorting to the Todo API and fix the PATCH partial update bug",
            repo_path=str(self.repo_dir),
        )

        self.assertEqual(session.status, "completed")
        self.assertEqual(session.task_handler, "todo_priority_patch")
        self.assertEqual([item.path for item in session.retrieved_files[:2]], ["tests/test_todo_api.py", "todo_api.py"])
        self.assertTrue(all(item.reasons for item in session.retrieved_files))
        self.assertEqual([item.path for item in session.context_summaries[:2]], ["tests/test_todo_api.py", "todo_api.py"])
        self.assertIsNotNone(session.clarify_artifact)
        self.assertEqual(session.clarify_artifact.relevant_files, ["todo_api.py", "tests/test_todo_api.py"])
        self.assertEqual(session.clarify_artifact.validation_command, DEFAULT_VALIDATION_COMMAND)
        self.assertIsNotNone(session.plan_artifact)
        self.assertEqual(session.plan_artifact.target_files, ["todo_api.py", "tests/test_todo_api.py"])
        self.assertEqual(len(session.plan_artifact.steps), 4)
        self.assertIn("TodoApiTests.test_create_todo_applies_defaults", session.read_summary or "")
        self.assertIn("create_todo, list_todos, apply_patch_update", session.read_summary or "")
        self.assertIsNotNone(session.proposal_candidate)
        self.assertEqual({edit.path for edit in session.proposal_candidate.edits}, {"todo_api.py", "tests/test_todo_api.py"})
        self.assertIn("list_todos", session.proposal_candidate.edits[0].target_symbols)
        self.assertIn("todo_api.py", session.proposal_summary or "")
        self.assertIn("tests/test_todo_api.py", session.proposal_summary or "")
        self.assertIn(DEFAULT_VALIDATION_COMMAND, session.proposal_summary or "")
        self.assertIsNotNone(session.proposal_assessment)
        self.assertEqual(session.proposal_assessment.status, "accepted")
        self.assertEqual(session.proposal_assessment.score, 90)
        self.assertEqual(set(session.proposal_assessment.matched_targets), {"todo_api.py", "tests/test_todo_api.py"})
        self.assertEqual(session.proposal_assessment.extra_targets, [])
        self.assertTrue(session.proposal_assessment.used_fallback)
        self.assertIsNotNone(session.clarify_summary)
        self.assertIsNotNone(session.plan_summary)
        self.assertIn("high -> medium -> low", session.review_summary or "")
        self.assertIn("Proposal assessment: accepted score=90 source=fallback", session.review_summary or "")
        self.assertIn("Proposal candidate:", session.review_summary or "")
        self.assertIn("Grounding: retrieved", session.review_summary or "")
        self.assertIn("Read evidence:", session.review_summary or "")
        self.assertIn("Evidence: changed todo_api.py, tests/test_todo_api.py.", session.review_summary or "")
        self.assertIn("Validation: python -m unittest discover -s tests -q passed in", session.review_summary or "")
        self.assertIn(
            "Audit: workflow used repo_reader, edit_proposal_generator, preset_file_editor, python_test_runner, review_compiler.",
            session.review_summary or "",
        )
        self.assertIn("Residual risk:", session.review_summary or "")
        self.assertEqual(session.test_results[-1].exit_code, 0)
        self.assertEqual(
            {change.path for change in session.changed_files},
            {"todo_api.py", "tests/test_todo_api.py"},
        )
        self.assertTrue(all(change.diff_excerpt for change in session.changed_files))
        self.assertEqual(
            [call.tool_name for call in session.tool_calls],
            [READ_REPO_TOOL, EDIT_PROPOSAL_TOOL, PRESET_EDIT_TOOL, TEST_COMMAND_TOOL, REVIEW_TOOL],
        )
        self.assertEqual(
            [call.step_type for call in session.tool_calls],
            ["read", "proposal", "edit", "test", "review"],
        )
        self.assertTrue(all(call.approved for call in session.tool_calls))
        self.assertEqual(len(session.approval_checks), 5)
        self.assertTrue(all(item.approved for item in session.approval_checks))

        todo_api_text = (self.repo_dir / "todo_api.py").read_text(encoding="utf-8")
        self.assertIn("_PRIORITY_ORDER", todo_api_text)
        self.assertIn("if field in patch", todo_api_text)

        summary = self.store.load_summary(session.session_id)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["task_handler"], "todo_priority_patch")
        self.assertIn("clarify_artifact", summary)
        self.assertIn("plan_artifact", summary)
        self.assertIn("proposal_candidate", summary)
        self.assertIn("proposal_assessment", summary)
        self.assertEqual(summary["clarify_artifact"]["relevant_files"], ["todo_api.py", "tests/test_todo_api.py"])
        self.assertEqual(summary["plan_artifact"]["target_files"], ["todo_api.py", "tests/test_todo_api.py"])
        self.assertEqual(len(summary["proposal_candidate"]["edits"]), 2)
        self.assertEqual(summary["proposal_candidate"]["edits"][0]["change_type"], "update")
        self.assertEqual(summary["proposal_assessment"]["status"], "accepted")
        self.assertEqual(summary["proposal_assessment"]["score"], 90)
        self.assertEqual(summary["proposal_assessment"]["extra_targets"], [])
        self.assertIn("approval_checks", summary)
        self.assertIn("tool_calls", summary)
        self.assertEqual(len(summary["tool_calls"]), 5)
        self.assertEqual(summary["tool_calls"][0]["tool_name"], READ_REPO_TOOL)
        self.assertEqual(summary["tool_calls"][1]["tool_name"], EDIT_PROPOSAL_TOOL)
        self.assertEqual(summary["approval_checks"][2]["tool_name"], PRESET_EDIT_TOOL)
        self.assertIn("Validation: python -m unittest discover -s tests -q passed in", str(summary["review_summary"]))
        self.assertEqual(len(summary["changed_files"]), 2)
        self.assertTrue(all(change["diff_excerpt"] for change in summary["changed_files"]))

        events = self.store.load_events(session.session_id)
        self.assertTrue(any(event["kind"] == "file_changed" for event in events))
        self.assertTrue(any(event["kind"] == "context_selected" for event in events))
        self.assertTrue(any(event["kind"] == "approval_checked" for event in events))
        self.assertTrue(any(event["kind"] == "proposal_assessed" for event in events))
        self.assertTrue(any(event["kind"] == "tool_started" for event in events))
        self.assertTrue(any(event["kind"] == "tool_completed" for event in events))

    def test_runner_applies_query_helper_task_and_passes_tests(self) -> None:
        runner = Runner(config=self.config, store=self.store, renderer=self.renderer)

        session = runner.run(
            task_text="Add completed-only filtering and case-insensitive search to the todo query helpers",
            repo_path=str(self.repo_dir),
        )

        self.assertEqual(session.status, "completed")
        self.assertEqual(session.task_handler, "todo_query_filters")
        self.assertEqual(session.proposal_assessment.status, "accepted")
        self.assertEqual(session.proposal_assessment.score, 90)
        self.assertEqual({edit.path for edit in session.proposal_candidate.edits}, {"todo_queries.py", "tests/test_todo_queries.py"})
        self.assertIn("todo_queries.py", session.proposal_summary or "")
        self.assertIn("case-insensitively", session.review_summary or "")
        self.assertIn("todo_queries.py", session.review_summary or "")
        self.assertIn("Proposal assessment: accepted score=90 source=fallback", session.review_summary or "")
        self.assertIn("Proposal candidate:", session.review_summary or "")
        self.assertIn("Read evidence:", session.review_summary or "")
        self.assertIn(
            "Audit: workflow used repo_reader, edit_proposal_generator, preset_file_editor, python_test_runner, review_compiler.",
            session.review_summary or "",
        )
        self.assertEqual(session.test_results[-1].exit_code, 0)
        self.assertEqual(
            {change.path for change in session.changed_files},
            {"todo_queries.py", "tests/test_todo_queries.py"},
        )

        todo_queries_text = (self.repo_dir / "todo_queries.py").read_text(encoding="utf-8")
        self.assertIn("query.strip().casefold()", todo_queries_text)
        self.assertIn("completed is None", todo_queries_text)

    def test_runner_generic_mode_can_edit_arbitrary_python_repo_files_with_model_output(self) -> None:
        fake_llm = FakeLlmClient(
            [
                json.dumps(
                    {
                        "implementation_target": "Update math_utils.py and tests/test_math_utils.py to correct subtract behavior and add regression coverage.",
                        "relevant_files": ["math_utils.py", "tests/test_math_utils.py"],
                        "validation_command": DEFAULT_VALIDATION_COMMAND,
                    }
                ),
                json.dumps(
                    {
                        "steps": [
                            "Inspect math_utils.py and tests/test_math_utils.py.",
                            "Update subtract in math_utils.py.",
                            "Add regression coverage in tests/test_math_utils.py.",
                            f"Run {DEFAULT_VALIDATION_COMMAND}.",
                        ],
                        "target_files": ["math_utils.py", "tests/test_math_utils.py"],
                        "validation_command": DEFAULT_VALIDATION_COMMAND,
                    }
                ),
                json.dumps(
                    {
                        "summary": "Prepare bounded edits for math_utils.py and tests/test_math_utils.py.",
                        "edits": [
                            {
                                "path": "math_utils.py",
                                "change_type": "update",
                                "target_symbols": ["subtract"],
                                "intent": "correct subtract so it performs subtraction instead of addition",
                            },
                            {
                                "path": "tests/test_math_utils.py",
                                "change_type": "update",
                                "target_symbols": ["MathUtilsTests.test_subtract"],
                                "intent": "add regression coverage for subtract",
                            },
                        ],
                        "validation_command": DEFAULT_VALIDATION_COMMAND,
                    }
                ),
                json.dumps(
                    {
                        "path": "math_utils.py",
                        "change_type": "update",
                        "summary": "Correct subtract to use subtraction.",
                        "content": "from __future__ import annotations\n\n\ndef add(a: int, b: int) -> int:\n    return a + b\n\n\ndef subtract(a: int, b: int) -> int:\n    return a - b\n",
                    }
                ),
                json.dumps(
                    {
                        "path": "tests/test_math_utils.py",
                        "change_type": "update",
                        "summary": "Add regression coverage for subtract.",
                        "content": "from __future__ import annotations\n\nimport unittest\n\nfrom math_utils import add, subtract\n\n\nclass MathUtilsTests(unittest.TestCase):\n    def test_add(self) -> None:\n        self.assertEqual(add(2, 3), 5)\n\n    def test_subtract(self) -> None:\n        self.assertEqual(subtract(5, 3), 2)\n\n\nif __name__ == \"__main__\":\n    unittest.main()\n",
                    }
                ),
            ]
        )
        runner = Runner(config=self.config, store=self.store, renderer=self.renderer, llm_client=fake_llm)

        session = runner.run(
            task_text="Fix subtract in math_utils.py and add regression coverage in the math utils tests",
            repo_path=str(self.repo_dir),
        )

        self.assertEqual(session.status, "completed")
        self.assertIsNone(session.task_handler)
        self.assertEqual(session.fallback_steps, [])
        self.assertEqual({edit.path for edit in session.proposal_candidate.edits}, {"math_utils.py", "tests/test_math_utils.py"})
        self.assertEqual(session.proposal_assessment.status, "accepted")
        self.assertEqual(session.proposal_assessment.score, 100)
        self.assertFalse(session.proposal_assessment.used_fallback)
        self.assertEqual({change.path for change in session.changed_files}, {"math_utils.py", "tests/test_math_utils.py"})
        self.assertEqual(session.test_results[-1].exit_code, 0)
        self.assertIn("math_utils.py", session.review_summary or "")
        self.assertIn("Proposal assessment: accepted score=100 source=model", session.review_summary or "")
        self.assertIn("Target file:", fake_llm.prompts[3])
        self.assertIn("math_utils.py", fake_llm.prompts[3])

        math_utils_text = (self.repo_dir / "math_utils.py").read_text(encoding="utf-8")
        self.assertIn("return a - b", math_utils_text)
        test_math_utils_text = (self.repo_dir / "tests" / "test_math_utils.py").read_text(encoding="utf-8")
        self.assertIn("test_subtract", test_math_utils_text)
        self.assertIn("subtract(5, 3)", test_math_utils_text)

    def test_runner_generic_mode_can_include_external_research_in_prompts_and_review(self) -> None:
        fake_llm = FakeLlmClient(
            [
                json.dumps(
                    {
                        "implementation_target": "Update math_utils.py and tests/test_math_utils.py to correct subtract behavior and add regression coverage.",
                        "relevant_files": ["math_utils.py", "tests/test_math_utils.py"],
                        "validation_command": DEFAULT_VALIDATION_COMMAND,
                    }
                ),
                json.dumps(
                    {
                        "steps": [
                            "Inspect math_utils.py and tests/test_math_utils.py.",
                            "Update subtract in math_utils.py.",
                            "Add regression coverage in tests/test_math_utils.py.",
                            f"Run {DEFAULT_VALIDATION_COMMAND}.",
                        ],
                        "target_files": ["math_utils.py", "tests/test_math_utils.py"],
                        "validation_command": DEFAULT_VALIDATION_COMMAND,
                    }
                ),
                json.dumps(
                    {
                        "summary": "Prepare bounded edits for math_utils.py and tests/test_math_utils.py.",
                        "edits": [
                            {
                                "path": "math_utils.py",
                                "change_type": "update",
                                "target_symbols": ["subtract"],
                                "intent": "correct subtract so it performs subtraction instead of addition",
                            },
                            {
                                "path": "tests/test_math_utils.py",
                                "change_type": "update",
                                "target_symbols": ["MathUtilsTests.test_subtract"],
                                "intent": "add regression coverage for subtract",
                            },
                        ],
                        "validation_command": DEFAULT_VALIDATION_COMMAND,
                    }
                ),
                json.dumps(
                    {
                        "path": "math_utils.py",
                        "change_type": "update",
                        "summary": "Correct subtract to use subtraction.",
                        "content": "from __future__ import annotations\n\n\ndef add(a: int, b: int) -> int:\n    return a + b\n\n\ndef subtract(a: int, b: int) -> int:\n    return a - b\n",
                    }
                ),
                json.dumps(
                    {
                        "path": "tests/test_math_utils.py",
                        "change_type": "update",
                        "summary": "Add regression coverage for subtract.",
                        "content": "from __future__ import annotations\n\nimport unittest\n\nfrom math_utils import add, subtract\n\n\nclass MathUtilsTests(unittest.TestCase):\n    def test_add(self) -> None:\n        self.assertEqual(add(2, 3), 5)\n\n    def test_subtract(self) -> None:\n        self.assertEqual(subtract(5, 3), 2)\n\n\nif __name__ == \"__main__\":\n    unittest.main()\n",
                    }
                ),
            ]
        )
        research_client = FakeResearchClient(
            [
                ResearchSource(
                    title="Python docs",
                    url="https://docs.python.org/3/library/functions.html#sorted",
                    snippet="sorted accepts a key function for custom ordering.",
                )
            ]
        )
        runner = Runner(
            config=self.config,
            store=self.store,
            renderer=self.renderer,
            llm_client=fake_llm,
            research_client=research_client,
        )

        session = runner.run(
            task_text="Fix subtract in math_utils.py and add regression coverage in the math utils tests",
            repo_path=str(self.repo_dir),
            enable_web_research=True,
            research_query="python sorted and arithmetic operator docs",
        )

        self.assertEqual(session.status, "completed")
        self.assertTrue(session.research_enabled)
        self.assertEqual(session.research_query, "python sorted and arithmetic operator docs")
        self.assertEqual(research_client.queries, ["python sorted and arithmetic operator docs"])
        self.assertEqual(len(session.research_sources), 1)
        self.assertIn("Python docs", session.research_summary or "")
        self.assertIn("External research:", session.review_summary or "")
        self.assertIn("https://docs.python.org/3/library/functions.html#sorted", session.review_summary or "")
        self.assertIn("External research:", fake_llm.prompts[0])
        self.assertIn("Python docs", fake_llm.prompts[0])
        self.assertEqual([call.tool_name for call in session.tool_calls][0], RESEARCH_TOOL)
        self.assertEqual(len(session.approval_checks), 6)

        summary = self.store.load_summary(session.session_id)
        self.assertTrue(summary["research_enabled"])
        self.assertEqual(summary["research_query"], "python sorted and arithmetic operator docs")
        self.assertEqual(len(summary["research_sources"]), 1)

    def test_runner_uses_grounded_structured_model_outputs_when_schema_matches(self) -> None:
        fake_llm = FakeLlmClient(
            [
                json.dumps(
                    {
                        "implementation_target": "Update todo_api.py and tests/test_todo_api.py to add priority-aware ordering and preserve unspecified fields during PATCH updates.",
                        "relevant_files": ["todo_api.py", "tests/test_todo_api.py"],
                        "validation_command": DEFAULT_VALIDATION_COMMAND,
                    }
                ),
                json.dumps(
                    {
                        "steps": [
                            "Inspect todo_api.py and tests/test_todo_api.py.",
                            "Update list_todos and apply_patch_update in todo_api.py.",
                            "Add regression coverage in tests/test_todo_api.py.",
                            f"Run {DEFAULT_VALIDATION_COMMAND}.",
                        ],
                        "target_files": ["todo_api.py", "tests/test_todo_api.py"],
                        "validation_command": DEFAULT_VALIDATION_COMMAND,
                    }
                ),
                json.dumps(
                    {
                        "summary": "Prepare a bounded diff candidate for todo_api.py and tests/test_todo_api.py.",
                        "edits": [
                            {
                                "path": "todo_api.py",
                                "change_type": "update",
                                "target_symbols": ["list_todos", "apply_patch_update"],
                                "intent": "add priority-aware ordering and preserve unspecified PATCH fields",
                            },
                            {
                                "path": "tests/test_todo_api.py",
                                "change_type": "update",
                                "target_symbols": [
                                    "TodoApiTests.test_list_todos_sorts_by_priority_then_id",
                                    "TodoApiTests.test_apply_patch_update_preserves_unspecified_fields",
                                ],
                                "intent": "cover the new ordering and PATCH regression behavior",
                            },
                        ],
                        "validation_command": DEFAULT_VALIDATION_COMMAND,
                    }
                ),
            ]
        )
        runner = Runner(config=self.config, store=self.store, renderer=self.renderer, llm_client=fake_llm)

        session = runner.run(
            task_text="Add priority sorting to the Todo API and fix the PATCH partial update bug",
            repo_path=str(self.repo_dir),
        )

        self.assertEqual(session.status, "completed")
        self.assertEqual(session.fallback_steps, [])
        self.assertEqual(session.proposal_assessment.status, "accepted")
        self.assertEqual(session.proposal_assessment.score, 100)
        self.assertFalse(session.proposal_assessment.used_fallback)
        self.assertEqual({edit.path for edit in session.proposal_candidate.edits}, {"todo_api.py", "tests/test_todo_api.py"})
        self.assertIn("Return only the raw JSON object for the proposal candidate", fake_llm.prompts[2])

    def test_runner_uses_grounded_fallback_when_llm_response_hallucinates(self) -> None:
        fake_llm = FakeLlmClient(
            [
                "Add priority-based sorting to the GET /todos endpoint in todo_api.py and verify with pytest tests/test_todo_api.py.",
                "1. Inspect the PATCH /todos/{id} handler. 2. Update the Pydantic model. 3. Use model_dump. 4. Run pytest tests/test_todo_api.py.",
                "- Files: update the FastAPI router and todo_api.py. - Changes: use Pydantic Todoupdate and pytest. - Validation: run pytest tests/test_todo_api.py.",
            ]
        )
        runner = Runner(config=self.config, store=self.store, renderer=self.renderer, llm_client=fake_llm)

        session = runner.run(
            task_text="Add priority sorting to the Todo API and fix the PATCH partial update bug",
            repo_path=str(self.repo_dir),
        )

        self.assertEqual(session.status, "completed")
        self.assertIn(DEFAULT_VALIDATION_COMMAND, session.clarify_summary or "")
        self.assertIn("todo_api.py", session.plan_summary or "")
        self.assertIn("tests/test_todo_api.py", session.proposal_summary or "")
        self.assertEqual(session.proposal_assessment.status, "accepted")
        self.assertEqual(session.proposal_assessment.score, 90)
        self.assertTrue(session.proposal_assessment.used_fallback)
        self.assertNotIn("pytest", session.clarify_summary or "")
        self.assertNotIn("pytest", session.plan_summary or "")
        self.assertNotIn("pytest", session.proposal_summary or "")
        self.assertIn("Retrieved files:", fake_llm.prompts[0])
        self.assertIn("Retrieved file summaries:", fake_llm.prompts[0])
        self.assertIn("functions=create_todo, list_todos, apply_patch_update", fake_llm.prompts[0])
        self.assertIn("Allowed edit scope:", fake_llm.prompts[2])
        self.assertIn("Read evidence:", fake_llm.prompts[2])
        self.assertEqual(
            session.fallback_steps,
            ["clarify:validator_rejected", "plan:validator_rejected", "proposal:validator_rejected"],
        )
        self.assertIn("grounding validator rejected the model output", session.review_summary or "")
        self.assertEqual(
            [call.tool_name for call in session.tool_calls],
            [READ_REPO_TOOL, EDIT_PROPOSAL_TOOL, PRESET_EDIT_TOOL, TEST_COMMAND_TOOL, REVIEW_TOOL],
        )

    def test_runner_tracks_task_handler_offline_fallbacks_and_approval_audit(self) -> None:
        runner = Runner(config=self.config, store=self.store, renderer=self.renderer)

        session = runner.run(
            task_text="Add completed-only filtering and case-insensitive search to the todo query helpers",
            repo_path=str(self.repo_dir),
        )

        self.assertEqual(session.task_handler, "todo_query_filters")
        self.assertEqual(
            session.fallback_steps,
            ["clarify:client_unconfigured", "plan:client_unconfigured", "proposal:client_unconfigured"],
        )
        self.assertEqual(session.proposal_assessment.status, "accepted")
        self.assertEqual(session.proposal_assessment.score, 90)
        self.assertEqual(len(session.proposal_candidate.edits), 2)
        self.assertIn("the client was not configured", session.review_summary or "")
        self.assertEqual(len(session.approval_checks), 5)
        self.assertTrue(all(item.approved for item in session.approval_checks))

        summary = self.store.load_summary(session.session_id)
        self.assertEqual(
            summary["fallback_steps"],
            ["clarify:client_unconfigured", "plan:client_unconfigured", "proposal:client_unconfigured"],
        )
        self.assertEqual(len(summary["approval_checks"]), 5)

    def test_runner_is_idempotent_after_repo_is_already_fixed(self) -> None:
        runner = Runner(config=self.config, store=self.store, renderer=self.renderer)
        runner.run(
            task_text="Add priority sorting to the Todo API and fix the PATCH partial update bug",
            repo_path=str(self.repo_dir),
        )

        second_session = runner.run(
            task_text="Add priority sorting to the Todo API and fix the PATCH partial update bug",
            repo_path=str(self.repo_dir),
        )

        self.assertEqual(second_session.status, "completed")
        self.assertEqual(second_session.test_results[-1].exit_code, 0)
        self.assertEqual(second_session.changed_files, [])
        self.assertEqual(second_session.proposal_assessment.status, "not_needed")
        self.assertEqual(second_session.proposal_assessment.score, 70)
        self.assertEqual(set(second_session.proposal_assessment.extra_targets), {"todo_api.py", "tests/test_todo_api.py"})
        self.assertIn("already satisfied", second_session.review_summary or "")
        self.assertIn("Evidence: no file edits were needed", second_session.review_summary or "")
        self.assertIn("Proposal assessment: not_needed score=70 source=fallback", second_session.review_summary or "")
        self.assertIn(
            "Audit: workflow used repo_reader, edit_proposal_generator, preset_file_editor, python_test_runner, review_compiler.",
            second_session.review_summary or "",
        )

    def test_restore_demo_repo_returns_repo_to_seed_state_after_query_task(self) -> None:
        runner = Runner(config=self.config, store=self.store, renderer=self.renderer)
        runner.run(
            task_text="Add completed-only filtering and case-insensitive search to the todo query helpers",
            repo_path=str(self.repo_dir),
        )

        changed_paths = restore_demo_repo(RepoOps(self.repo_dir))

        self.assertEqual(set(changed_paths), {"todo_queries.py", "tests/test_todo_queries.py"})
        self.assertEqual((self.repo_dir / "todo_queries.py").read_text(encoding="utf-8"), BASELINE_TODO_QUERIES)
        self.assertEqual(
            (self.repo_dir / "tests" / "test_todo_queries.py").read_text(encoding="utf-8"),
            BASELINE_TEST_TODO_QUERIES,
        )


if __name__ == "__main__":
    unittest.main()




