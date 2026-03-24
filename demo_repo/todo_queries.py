"""Query helpers for the coding-agent skeleton."""

from __future__ import annotations

from copy import deepcopy


def filter_todos(
    todos: list[dict[str, object]],
    completed: bool | None = None,
) -> list[dict[str, object]]:
    """Current implementation ignores the completed filter and always returns all items."""
    _ = completed
    return [deepcopy(item) for item in todos]


def search_todos(todos: list[dict[str, object]], query: str) -> list[dict[str, object]]:
    """Current implementation is intentionally case-sensitive and does not trim the query."""
    if not query:
        return [deepcopy(item) for item in todos]
    return [deepcopy(item) for item in todos if query in str(item.get("title", ""))]
