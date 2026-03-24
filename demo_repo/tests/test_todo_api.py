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
