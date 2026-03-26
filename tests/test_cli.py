from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coding_agent.cli import _open_in_browser, build_parser


class CliTests(unittest.TestCase):
    def test_build_parser_accepts_run_report_and_serve_flags(self) -> None:
        parser = build_parser()

        run_args = parser.parse_args([
            "run",
            "--task",
            "demo task",
            "--open-report",
            "--report-session-limit",
            "7",
            "--verbose",
        ])
        self.assertTrue(run_args.open_report)
        self.assertEqual(run_args.report_session_limit, 7)
        self.assertTrue(run_args.verbose)

        report_args = parser.parse_args(["report", "--session-limit", "5", "--open"])
        self.assertEqual(report_args.session_limit, 5)
        self.assertTrue(report_args.open)

        serve_args = parser.parse_args(["serve", "--host", "127.0.0.1", "--port", "9001", "--session-limit", "8", "--open"])
        self.assertEqual(serve_args.host, "127.0.0.1")
        self.assertEqual(serve_args.port, 9001)
        self.assertEqual(serve_args.session_limit, 8)
        self.assertTrue(serve_args.open)

    def test_open_in_browser_uses_file_uri(self) -> None:
        captured: list[str] = []

        def fake_open(target: str) -> bool:
            captured.append(target)
            return True

        result = _open_in_browser(Path("runtime/reports/dashboard.html"), opener=fake_open)

        self.assertTrue(result)
        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0].startswith("file:///"))
        self.assertTrue(captured[0].endswith("dashboard.html"))


if __name__ == "__main__":
    unittest.main()
