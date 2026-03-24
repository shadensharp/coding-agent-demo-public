"""Small demo repository used by the coding-agent skeleton."""

from __future__ import annotations

from copy import deepcopy


def create_todo(todo_id: int, title: str, completed: bool = False, priority: str = "medium") -> dict[str, object]:
    return {
        "id": todo_id,
        "title": title,
        "completed": completed,
        "priority": priority,
    }


def list_todos(todos: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return a deterministic listing for V1; priority-aware sorting is not implemented yet."""
    ordered = sorted(todos, key=lambda item: int(item["id"]))
    return [deepcopy(item) for item in ordered]


def apply_patch_update(todo: dict[str, object], patch: dict[str, object]) -> dict[str, object]:
    """Current implementation is intentionally naive and will be a later fix target."""
    return {
        "id": todo.get("id"),
        "title": patch.get("title"),
        "completed": patch.get("completed"),
        "priority": patch.get("priority"),
    }
