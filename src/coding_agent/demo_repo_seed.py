from __future__ import annotations

import textwrap

from .repo_ops import RepoOps

BASELINE_TODO_API = textwrap.dedent(
    '''
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
    '''
).strip() + "\n"

BASELINE_TEST_TODO_API = textwrap.dedent(
    '''
    from __future__ import annotations

    import unittest

    from todo_api import apply_patch_update, create_todo, list_todos


    class TodoApiTests(unittest.TestCase):
        def test_create_todo_applies_defaults(self) -> None:
            todo = create_todo(1, "Write docs")
            self.assertEqual(todo["title"], "Write docs")
            self.assertEqual(todo["completed"], False)
            self.assertEqual(todo["priority"], "medium")

        def test_list_todos_returns_stable_id_order(self) -> None:
            todos = [
                create_todo(3, "Third", priority="low"),
                create_todo(1, "First", priority="high"),
                create_todo(2, "Second", priority="medium"),
            ]
            listed = list_todos(todos)
            self.assertEqual([item["id"] for item in listed], [1, 2, 3])

        def test_apply_patch_update_handles_full_update(self) -> None:
            original = create_todo(10, "Initial", completed=False, priority="low")
            updated = apply_patch_update(
                original,
                {"title": "Updated", "completed": True, "priority": "high"},
            )
            self.assertEqual(
                updated,
                {"id": 10, "title": "Updated", "completed": True, "priority": "high"},
            )


    if __name__ == "__main__":
        unittest.main()
    '''
).strip() + "\n"

BASELINE_TODO_QUERIES = textwrap.dedent(
    '''
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
    '''
).strip() + "\n"

BASELINE_TEST_TODO_QUERIES = textwrap.dedent(
    '''
    from __future__ import annotations

    import unittest

    from todo_queries import filter_todos, search_todos


    class TodoQueryTests(unittest.TestCase):
        def test_filter_todos_returns_all_items_when_filter_is_none(self) -> None:
            todos = [
                {"id": 1, "title": "Write docs", "completed": False},
                {"id": 2, "title": "Ship demo", "completed": True},
            ]
            filtered = filter_todos(todos)
            self.assertEqual([item["id"] for item in filtered], [1, 2])

        def test_search_todos_matches_exact_case_today(self) -> None:
            todos = [
                {"id": 1, "title": "Write Docs", "completed": False},
                {"id": 2, "title": "ship demo", "completed": True},
            ]
            matched = search_todos(todos, "Docs")
            self.assertEqual([item["id"] for item in matched], [1])


    if __name__ == "__main__":
        unittest.main()
    '''
).strip() + "\n"

BASELINE_FILES: dict[str, str] = {
    "todo_api.py": BASELINE_TODO_API,
    "todo_queries.py": BASELINE_TODO_QUERIES,
    "tests/test_todo_api.py": BASELINE_TEST_TODO_API,
    "tests/test_todo_queries.py": BASELINE_TEST_TODO_QUERIES,
}


def restore_demo_repo(repo_ops: RepoOps) -> list[str]:
    repo_ops.ensure_repo_exists()
    changed_paths: list[str] = []
    for path, content in BASELINE_FILES.items():
        change = repo_ops.apply_text_change(
            path,
            content,
            "Restore demo repo to the seeded baseline state.",
        )
        if change is not None:
            changed_paths.append(path)
    return changed_paths
