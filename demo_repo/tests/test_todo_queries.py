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
