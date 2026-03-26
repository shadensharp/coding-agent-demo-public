from __future__ import annotations

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
from coding_agent.research import WebResearchClient


class WebResearchClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="coding-agent-research-test-"))
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

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_search_parses_results_and_normalizes_duckduckgo_redirect_urls(self) -> None:
        html = '''
        <html><body>
          <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Ffunctions.html">Python sorted</a>
          <div class="result__snippet">Built-in <b>sorted</b> function documentation.</div>
          <a class="result__a" href="https://example.com/guide">Guide</a>
          <div class="result__snippet">Example guide text.</div>
        </body></html>
        '''
        client = WebResearchClient(self.config, transport=lambda req, timeout: html)

        results = client.search("python sorted key", max_results=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Python sorted")
        self.assertEqual(results[0].url, "https://docs.python.org/3/library/functions.html")
        self.assertIn("sorted function documentation", results[0].snippet)
        self.assertEqual(results[1].url, "https://example.com/guide")

    def test_search_returns_empty_for_blank_query(self) -> None:
        client = WebResearchClient(self.config, transport=lambda req, timeout: "")

        self.assertEqual(client.search("   "), [])

    def test_search_wraps_transport_failures(self) -> None:
        def failing_transport(req: object, timeout: float) -> str:
            _ = req
            _ = timeout
            raise error.URLError("offline")

        client = WebResearchClient(self.config, transport=failing_transport)

        with self.assertRaisesRegex(RuntimeError, "External research request failed"):
            client.search("python patch merge")


if __name__ == "__main__":
    unittest.main()
