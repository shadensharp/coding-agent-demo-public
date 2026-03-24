from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coding_agent.config import AppConfig
from coding_agent.reporting import build_dashboard_report, render_dashboard_html, render_dashboard_markdown, write_dashboard_report
from coding_agent.storage import JsonlRunStore


class DashboardReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="coding-agent-report-test-"))
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
        self._write_summary(
            session_id="sess_live",
            session_name="live_smoke_schema_prompt_host",
            created_at="2026-03-24T00:48:04Z",
            fallback_steps=[],
            proposal_status="accepted",
            proposal_score=100,
            proposal_used_fallback=False,
            duration_ms=110,
        )
        self._write_summary(
            session_id="sess_offline",
            session_name="offline_smoke",
            created_at="2026-03-23T23:00:00Z",
            fallback_steps=[
                "clarify:client_unconfigured",
                "plan:client_unconfigured",
                "proposal:client_unconfigured",
            ],
            proposal_status="accepted",
            proposal_score=90,
            proposal_used_fallback=True,
            duration_ms=150,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _write_summary(
        self,
        session_id: str,
        session_name: str,
        created_at: str,
        fallback_steps: list[str],
        proposal_status: str,
        proposal_score: int,
        proposal_used_fallback: bool,
        duration_ms: int,
    ) -> None:
        session_dir = self.config.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "session_id": session_id,
            "session_name": session_name,
            "status": "completed",
            "created_at": created_at,
            "task_handler": "todo_priority_patch",
            "request": {
                "user_text": "Add priority sorting to the Todo API and fix the PATCH partial update bug.",
            },
            "fallback_steps": list(fallback_steps),
            "tool_calls": [{"tool_name": "repo_reader"}] * 5,
            "approval_checks": [{"approved": True}] * 5,
            "retrieved_files": [{"path": "todo_api.py"}, {"path": "tests/test_todo_api.py"}],
            "changed_files": [{"path": "todo_api.py"}, {"path": "tests/test_todo_api.py"}],
            "test_results": [{"duration_ms": duration_ms}],
            "proposal_assessment": {
                "status": proposal_status,
                "score": proposal_score,
                "used_fallback": proposal_used_fallback,
            },
        }
        (session_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_build_dashboard_report_aggregates_recent_sessions_and_eval_snapshot(self) -> None:
        report = build_dashboard_report(self.config, self.store, session_limit=10)

        self.assertEqual(report["session_metrics"]["total_sessions"], 2)
        self.assertEqual(report["session_metrics"]["completed_sessions"], 2)
        self.assertEqual(report["session_metrics"]["fallback_sessions"], 1)
        self.assertEqual(report["session_metrics"]["model_backed_sessions"], 1)
        self.assertEqual(report["session_metrics"]["avg_tool_calls"], 5.0)
        self.assertEqual(report["session_metrics"]["avg_proposal_score"], 95.0)
        self.assertEqual(report["latest_model_session"]["session_id"], "sess_live")
        self.assertEqual(report["latest_session"]["session_id"], "sess_live")
        self.assertEqual(report["eval_summary"]["mode"], "offline")
        self.assertEqual(report["eval_summary"]["pass_rate"], 100.0)

        markdown = render_dashboard_markdown(report)
        self.assertIn("# Coding Agent Demo Dashboard", markdown)
        self.assertIn("## Latest Model-Backed Session", markdown)
        self.assertIn("sess_live", markdown)
        self.assertIn("proposal_accept_rate: 100.0%", markdown)
        self.assertIn("## Eval Snapshot", markdown)
        self.assertIn("proposal_accept_rate: 100.0%", markdown)
        self.assertIn("## Recent Session Table", markdown)
        self.assertIn("live_smoke_schema_prompt_host", markdown)
        self.assertIn("## Eval Case Table", markdown)

        html = render_dashboard_html(report)
        self.assertIn("<title>Coding Agent Demo Dashboard</title>", html)
        self.assertIn("static report shell", html)
        self.assertIn("live_smoke_schema_prompt_host", html)
        self.assertIn("Eval Cases", html)

    def test_write_dashboard_report_writes_markdown_and_json_files(self) -> None:
        report = write_dashboard_report(self.config, self.store, session_limit=10)

        markdown_path = Path(report["markdown_path"])
        json_path = Path(report["json_path"])
        html_path = Path(report["html_path"])
        self.assertTrue(markdown_path.exists())
        self.assertTrue(json_path.exists())
        self.assertTrue(html_path.exists())

        markdown = markdown_path.read_text(encoding="utf-8")
        self.assertIn("Recent Session Table", markdown)
        self.assertIn("sess_offline", markdown)

        html = html_path.read_text(encoding="utf-8")
        self.assertIn("Coding Agent Demo Dashboard", html)
        self.assertIn("Latest Session Snapshot", html)
        self.assertIn("sess_offline", html)

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["session_metrics"]["total_sessions"], 2)
        self.assertEqual(payload["latest_model_session"]["session_id"], "sess_live")
        self.assertEqual(payload["eval_summary"]["total_cases"], 2)
        self.assertTrue(payload["html_path"].endswith("dashboard.html"))


if __name__ == "__main__":
    unittest.main()


