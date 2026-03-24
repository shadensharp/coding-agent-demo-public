from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coding_agent.config import AppConfig
from coding_agent.evals import run_preset_eval


class EvalHarnessTests(unittest.TestCase):
    def test_run_preset_eval_completes_all_registered_tasks_offline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="coding-agent-eval-test-") as temp_dir_raw:
            temp_root = Path(temp_dir_raw)
            config = AppConfig(
                base_dir=temp_root,
                runtime_dir=temp_root / "runtime",
                sessions_dir=temp_root / "runtime" / "sessions",
                default_repo_dir=temp_root / "demo_repo",
                qwen_model="qwen-plus",
                qwen_api_key=None,
                qwen_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                qwen_timeout_seconds=45.0,
                qwen_max_retries=2,
                qwen_retry_backoff_seconds=1.0,
            )
            config.ensure_directories()

            summary = run_preset_eval(config)

        self.assertEqual(summary["mode"], "offline")
        self.assertEqual(summary["total_cases"], 2)
        self.assertEqual(summary["passed_cases"], 2)
        self.assertEqual(summary["failed_cases"], 0)
        self.assertEqual(summary["pass_rate"], 100.0)
        self.assertEqual(summary["fallback_cases"], 2)
        self.assertEqual(summary["total_changed_files"], 4)
        self.assertEqual(summary["avg_changed_files"], 2.0)
        self.assertEqual(summary["avg_retrieved_files"], 2.0)
        self.assertEqual(summary["required_file_hit_rate"], 100.0)
        self.assertEqual(summary["avg_fallback_steps_per_case"], 3.0)
        self.assertEqual(summary["avg_tool_calls"], 5.0)
        self.assertEqual(summary["avg_approval_checks"], 5.0)
        self.assertEqual(summary["approval_denials"], 0)
        self.assertEqual(summary["proposal_accept_rate"], 100.0)
        self.assertEqual(summary["avg_proposal_score"], 90.0)
        self.assertEqual(summary["avg_proposal_edits"], 2.0)
        self.assertEqual(
            summary["fallback_step_counts"],
            {
                "clarify:client_unconfigured": 2,
                "plan:client_unconfigured": 2,
                "proposal:client_unconfigured": 2,
            },
        )
        results = summary["results"]
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["status"] == "completed" for result in results))
        self.assertTrue(all(result["latest_test_exit"] == 0 for result in results))
        self.assertTrue(all(result["changed_files"] == 2 for result in results))
        self.assertTrue(all(result["retrieved_files"] >= 2 for result in results))
        self.assertTrue(all(result["required_file_hits"] == result["required_file_total"] for result in results))
        self.assertTrue(all(result["review_summary"] for result in results))
        self.assertTrue(all("Proposal assessment:" in result["review_summary"] for result in results))
        self.assertTrue(all("Proposal candidate:" in result["review_summary"] for result in results))
        self.assertTrue(all("Grounding:" in result["review_summary"] for result in results))
        self.assertTrue(all("Validation:" in result["review_summary"] for result in results))
        self.assertTrue(all("Audit:" in result["review_summary"] for result in results))
        self.assertTrue(all(result["tool_calls"] == 5 for result in results))
        self.assertTrue(all(result["approval_checks"] == 5 for result in results))
        self.assertTrue(all(result["denied_tool_calls"] == 0 for result in results))
        self.assertTrue(all(result["proposal_status"] == "accepted" for result in results))
        self.assertTrue(all(result["proposal_score"] == 90 for result in results))
        self.assertTrue(all(result["proposal_edit_candidates"] == 2 for result in results))


if __name__ == "__main__":
    unittest.main()
