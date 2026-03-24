from __future__ import annotations

import ast

from .models import ContextFileSummary, RetrievedFile
from .repo_ops import RepoOps


class RepoContextBuilder:
    def build(self, repo_ops: RepoOps, retrieved_files: list[RetrievedFile]) -> list[ContextFileSummary]:
        summaries: list[ContextFileSummary] = []
        for item in retrieved_files:
            summaries.append(self._build_one(repo_ops, item.path))
        return summaries

    def prompt_summaries(self, summaries: list[ContextFileSummary]) -> list[tuple[str, str]]:
        return [(summary.path, self._compact_summary(summary)) for summary in summaries]

    def read_findings(self, summaries: list[ContextFileSummary]) -> str:
        if not summaries:
            return "no retrieved file summaries were available"
        fragments = [f"{summary.path} {self._read_fragment(summary)}" for summary in summaries]
        return "; ".join(fragment for fragment in fragments if fragment).strip()

    def _build_one(self, repo_ops: RepoOps, path: str) -> ContextFileSummary:
        try:
            content = repo_ops.read_text(path)
        except OSError as exc:
            return ContextFileSummary(path=path, line_count=0, parse_error=str(exc))

        line_count = len(content.splitlines())
        if not path.endswith(".py"):
            return ContextFileSummary(path=path, line_count=line_count)

        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            return ContextFileSummary(path=path, line_count=line_count, parse_error=str(exc))

        function_names: list[str] = []
        class_names: list[str] = []
        test_names: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_names.append(node.name)
                if node.name.startswith("test_"):
                    test_names.append(node.name)
                continue
            if isinstance(node, ast.ClassDef):
                class_names.append(node.name)
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
                        test_names.append(f"{node.name}.{child.name}")

        return ContextFileSummary(
            path=path,
            line_count=line_count,
            function_names=function_names[:4],
            class_names=class_names[:3],
            test_names=test_names[:4],
        )

    def _compact_summary(self, summary: ContextFileSummary) -> str:
        if summary.parse_error:
            return f"lines={summary.line_count}; parse_error={summary.parse_error}"
        parts = [f"lines={summary.line_count}"]
        if summary.function_names:
            parts.append("functions=" + ", ".join(summary.function_names[:3]))
        if summary.class_names:
            parts.append("classes=" + ", ".join(summary.class_names[:2]))
        if summary.test_names:
            parts.append("tests=" + ", ".join(summary.test_names[:2]))
        return "; ".join(parts)

    def _read_fragment(self, summary: ContextFileSummary) -> str:
        if summary.parse_error:
            return f"parse_error={summary.parse_error}"
        if summary.test_names:
            return "tests=" + ", ".join(summary.test_names[:2])
        if summary.function_names:
            return "functions=" + ", ".join(summary.function_names[:3])
        if summary.class_names:
            return "classes=" + ", ".join(summary.class_names[:2])
        return f"lines={summary.line_count}"
