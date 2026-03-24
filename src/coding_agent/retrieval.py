from __future__ import annotations

from dataclasses import dataclass
import re

from .models import RetrievedFile
from .repo_ops import RepoOps
from .task_handlers import TaskHandler

_TOKEN_RE = re.compile(r"[A-Za-z_]{3,}")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "todo",
    "todos",
    "python",
    "repo",
    "repository",
    "task",
    "helper",
    "helpers",
    "tests",
    "test",
    "case",
    "cases",
    "bug",
    "fix",
    "add",
    "update",
}


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    path: str
    score: int
    reasons: tuple[str, ...]


class RepoRetriever:
    def build(
        self,
        repo_ops: RepoOps,
        task_text: str,
        handler: TaskHandler | None,
        limit: int = 3,
    ) -> list[RetrievedFile]:
        all_files = repo_ops.list_files(limit=None)
        if not all_files:
            return []

        query_tokens = self._query_tokens(task_text, handler)
        candidates: list[RetrievalCandidate] = []
        for path in all_files:
            candidate = self._score_path(repo_ops, path, query_tokens, handler)
            if candidate.score > 0:
                candidates.append(candidate)

        if not candidates:
            fallback_paths = self._fallback_paths(all_files, handler, limit)
            return [RetrievedFile(path=path, score=1, reasons=["default repo fallback"]) for path in fallback_paths]

        candidates.sort(key=lambda item: (-item.score, item.path))
        selected = candidates[:limit]
        return [RetrievedFile(path=item.path, score=item.score, reasons=list(item.reasons)) for item in selected]

    def _query_tokens(self, task_text: str, handler: TaskHandler | None) -> set[str]:
        tokens = self._extract_tokens(task_text)
        if handler is not None:
            for keyword in handler.keywords:
                tokens.update(self._extract_tokens(keyword))
            for path in handler.prompt_file_candidates:
                tokens.update(self._extract_tokens(path))
        return tokens

    def _score_path(
        self,
        repo_ops: RepoOps,
        path: str,
        query_tokens: set[str],
        handler: TaskHandler | None,
    ) -> RetrievalCandidate:
        score = 0
        reasons: list[str] = []

        if handler is not None and path in handler.required_files:
            score += 80
            reasons.append("required file")
        if handler is not None and path in handler.prompt_file_candidates:
            score += 40
            reasons.append("handler priority")

        path_tokens = self._extract_tokens(path)
        path_hits = sorted(query_tokens & path_tokens)
        if path_hits:
            score += 10 * min(3, len(path_hits))
            reasons.append(f"path match: {', '.join(path_hits[:3])}")

        try:
            content = repo_ops.read_text(path)
        except OSError:
            content = ""
        content_tokens = self._extract_tokens(self._truncate_content(content))
        content_hits = sorted(query_tokens & content_tokens)
        if content_hits:
            score += 4 * min(4, len(content_hits))
            reasons.append(f"content match: {', '.join(content_hits[:4])}")

        if path.startswith("tests/"):
            score += 2
            reasons.append("regression coverage file")

        return RetrievalCandidate(path=path, score=score, reasons=tuple(reasons))

    def _fallback_paths(self, all_files: list[str], handler: TaskHandler | None, limit: int) -> list[str]:
        if handler is not None:
            preferred = [path for path in handler.prompt_file_candidates if path in set(all_files)]
            if preferred:
                return preferred[:limit]
        return all_files[:limit]

    def _truncate_content(self, content: str, max_lines: int = 120, max_chars: int = 4000) -> str:
        lines = content.splitlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines]
        snippet = "\n".join(lines)
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars]
        return snippet

    def _extract_tokens(self, text: str) -> set[str]:
        base_tokens = [token.casefold() for token in _TOKEN_RE.findall(text)]
        tokens: set[str] = set()
        for token in base_tokens:
            parts = [part for part in token.split("_") if len(part) >= 3]
            tokens.add(token)
            tokens.update(parts)
        return {token for token in tokens if token not in _STOPWORDS}
