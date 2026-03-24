from __future__ import annotations

import difflib
import subprocess
from pathlib import Path
from time import perf_counter

from .models import CommandResult, FileChange


class RepoOps:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def ensure_repo_exists(self) -> None:
        if not self.repo_root.exists():
            raise FileNotFoundError(f"Repository path does not exist: {self.repo_root}")
        if not self.repo_root.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {self.repo_root}")

    def _resolve_under_root(self, relative_path: str) -> Path:
        candidate = (self.repo_root / relative_path).resolve()
        if not candidate.is_relative_to(self.repo_root):
            raise ValueError(f"Path escapes repository root: {relative_path}")
        return candidate

    def list_files(self, limit: int | None = 20) -> list[str]:
        skip_parts = {".git", "__pycache__", ".pytest_cache"}
        files: list[str] = []
        for path in self.repo_root.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(self.repo_root)
            if any(part in skip_parts for part in relative_path.parts):
                continue
            files.append(relative_path.as_posix())
        files.sort()
        if limit is None:
            return files
        return files[:limit]

    def describe_repo(self, limit: int = 8) -> str:
        visible_files = self.list_files(limit=limit)
        if not visible_files:
            return "Repository is empty."
        return "Visible files: " + ", ".join(visible_files)

    def read_text(self, relative_path: str) -> str:
        target = self._resolve_under_root(relative_path)
        return target.read_text(encoding="utf-8")

    def write_text(self, relative_path: str, content: str) -> None:
        target = self._resolve_under_root(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def apply_text_change(self, relative_path: str, content: str, summary: str) -> FileChange | None:
        target = self._resolve_under_root(relative_path)
        existed = target.exists()
        before = target.read_text(encoding="utf-8") if existed else ""
        if before == content:
            return None
        self.write_text(relative_path, content)
        return FileChange(
            path=relative_path,
            change_type="modified" if existed else "created",
            summary=summary,
            diff_excerpt=self._build_diff_excerpt(relative_path, before, content),
        )

    def run_command(self, command: list[str], cwd: Path | None = None) -> CommandResult:
        workdir = (cwd or self.repo_root).resolve()
        started = perf_counter()
        completed = subprocess.run(
            command,
            cwd=workdir,
            check=False,
            capture_output=True,
            text=True,
        )
        duration_ms = int((perf_counter() - started) * 1000)
        return CommandResult(
            command=command,
            cwd=str(workdir),
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=duration_ms,
        )

    def _build_diff_excerpt(self, relative_path: str, before: str, after: str) -> str | None:
        diff_lines = list(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
                n=2,
                lineterm="",
            )
        )
        if not diff_lines:
            return None
        excerpt = diff_lines[:40]
        if len(diff_lines) > 40:
            excerpt.append("... diff truncated ...")
        return "\n".join(excerpt)
