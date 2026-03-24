from __future__ import annotations

from dataclasses import dataclass, field
import textwrap

DEFAULT_VALIDATION_COMMAND = "python -m unittest discover -s tests -q"
DEFAULT_TEST_COMMAND = ("python", "-m", "unittest", "discover", "-s", "tests", "-q")
DEFAULT_GROUNDING_BLOCKLIST = (
    "fastapi",
    "pydantic",
    "pytest",
    "model_dump",
    "todoupdate",
    "get /todos",
    "patch /todos",
)

TARGET_TODO_API = textwrap.dedent(
    '''
    """Small demo repository used by the coding-agent skeleton."""

    from __future__ import annotations

    from copy import deepcopy

    _PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


    def create_todo(todo_id: int, title: str, completed: bool = False, priority: str = "medium") -> dict[str, object]:
        return {
            "id": todo_id,
            "title": title,
            "completed": completed,
            "priority": priority,
        }


    def list_todos(todos: list[dict[str, object]]) -> list[dict[str, object]]:
        """Return todos ordered by priority first and id second without mutating input."""
        ordered = sorted(
            todos,
            key=lambda item: (
                _PRIORITY_ORDER.get(str(item.get("priority", "medium")), len(_PRIORITY_ORDER)),
                int(item["id"]),
            ),
        )
        return [deepcopy(item) for item in ordered]


    def apply_patch_update(todo: dict[str, object], patch: dict[str, object]) -> dict[str, object]:
        """Merge a partial PATCH payload into an existing todo."""
        updated = deepcopy(todo)
        for field in ("title", "completed", "priority"):
            if field in patch:
                updated[field] = patch[field]
        return updated
    '''
).strip() + "\n"

TARGET_TEST_TODO_API = textwrap.dedent(
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

        def test_list_todos_sorts_by_priority_then_id(self) -> None:
            todos = [
                create_todo(5, "Low task", priority="low"),
                create_todo(2, "Medium task", priority="medium"),
                create_todo(7, "High task", priority="high"),
                create_todo(1, "Medium earlier id", priority="medium"),
            ]
            listed = list_todos(todos)
            self.assertEqual([item["id"] for item in listed], [7, 1, 2, 5])

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

        def test_apply_patch_update_preserves_unspecified_fields(self) -> None:
            original = create_todo(11, "Keep title", completed=False, priority="medium")
            updated = apply_patch_update(original, {"completed": True})
            self.assertEqual(
                updated,
                {"id": 11, "title": "Keep title", "completed": True, "priority": "medium"},
            )
            self.assertEqual(
                original,
                {"id": 11, "title": "Keep title", "completed": False, "priority": "medium"},
            )


    if __name__ == "__main__":
        unittest.main()
    '''
).strip() + "\n"

TARGET_TODO_QUERIES = textwrap.dedent(
    '''
    """Query helpers for the coding-agent skeleton."""

    from __future__ import annotations

    from copy import deepcopy


    def filter_todos(
        todos: list[dict[str, object]],
        completed: bool | None = None,
    ) -> list[dict[str, object]]:
        """Return todos filtered by completion state when requested."""
        if completed is None:
            return [deepcopy(item) for item in todos]
        return [deepcopy(item) for item in todos if bool(item.get("completed")) is completed]


    def search_todos(todos: list[dict[str, object]], query: str) -> list[dict[str, object]]:
        """Return todos whose title contains the query, ignoring case and surrounding whitespace."""
        normalized_query = query.strip().casefold()
        if not normalized_query:
            return [deepcopy(item) for item in todos]
        return [
            deepcopy(item)
            for item in todos
            if normalized_query in str(item.get("title", "")).casefold()
        ]
    '''
).strip() + "\n"

TARGET_TEST_TODO_QUERIES = textwrap.dedent(
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

        def test_filter_todos_can_return_only_completed_items(self) -> None:
            todos = [
                {"id": 1, "title": "Write docs", "completed": False},
                {"id": 2, "title": "Ship demo", "completed": True},
                {"id": 3, "title": "Review PR", "completed": True},
            ]
            filtered = filter_todos(todos, completed=True)
            self.assertEqual([item["id"] for item in filtered], [2, 3])

        def test_filter_todos_can_return_only_incomplete_items(self) -> None:
            todos = [
                {"id": 1, "title": "Write docs", "completed": False},
                {"id": 2, "title": "Ship demo", "completed": True},
                {"id": 3, "title": "Review PR", "completed": False},
            ]
            filtered = filter_todos(todos, completed=False)
            self.assertEqual([item["id"] for item in filtered], [1, 3])

        def test_search_todos_is_case_insensitive_and_trims_whitespace(self) -> None:
            todos = [
                {"id": 1, "title": "Write Docs", "completed": False},
                {"id": 2, "title": "ship demo", "completed": True},
                {"id": 3, "title": "Review PR", "completed": False},
            ]
            matched = search_todos(todos, "  DOcs ")
            self.assertEqual([item["id"] for item in matched], [1])

        def test_query_helpers_do_not_mutate_input(self) -> None:
            todos = [{"id": 1, "title": "Write Docs", "completed": False}]
            filtered = filter_todos(todos, completed=False)
            filtered[0]["title"] = "Changed"
            searched = search_todos(todos, "write")
            searched[0]["title"] = "Changed again"
            self.assertEqual(todos[0]["title"], "Write Docs")


    if __name__ == "__main__":
        unittest.main()
    '''
).strip() + "\n"


@dataclass(frozen=True, slots=True)
class PlannedChange:
    path: str
    content: str
    summary: str


@dataclass(frozen=True, slots=True)
class ProposalEditSeed:
    path: str
    change_type: str
    target_symbols: tuple[str, ...]
    intent: str


@dataclass(frozen=True, slots=True)
class TaskHandler:
    name: str
    description: str
    sample_task_text: str
    keywords: tuple[str, ...]
    required_files: tuple[str, ...]
    prompt_file_candidates: tuple[str, ...]
    planned_changes: tuple[PlannedChange, ...]
    offline_plan_steps: tuple[str, ...]
    proposal_edit_seeds: tuple[ProposalEditSeed, ...]
    clarify_required_markers: tuple[str, ...]
    plan_required_markers: tuple[str, ...]
    proposal_required_markers: tuple[str, ...]
    offline_clarify: str
    offline_plan: str
    offline_proposal: str
    review_if_changed: str
    review_if_unchanged: str
    review_risk_note: str
    test_command: tuple[str, ...] = DEFAULT_TEST_COMMAND
    validation_command: str = DEFAULT_VALIDATION_COMMAND
    grounding_blocklist: tuple[str, ...] = DEFAULT_GROUNDING_BLOCKLIST

    def matches(self, task_text: str, visible_files: set[str]) -> bool:
        if not set(self.required_files).issubset(visible_files):
            return False
        normalized = task_text.casefold()
        return any(keyword in normalized for keyword in self.keywords)

    def prompt_files(self, visible_files: set[str]) -> list[str]:
        return [path for path in self.prompt_file_candidates if path in visible_files]

    def validate_clarify_response(self, text: str) -> bool:
        return self._is_grounded(text, self.clarify_required_markers)

    def validate_plan_response(self, text: str) -> bool:
        return self._is_grounded(text, self.plan_required_markers)

    def validate_proposal_response(self, text: str) -> bool:
        return self._is_grounded(text, self.proposal_required_markers)

    def _is_grounded(self, text: str, required_markers: tuple[str, ...]) -> bool:
        normalized = text.casefold()
        if any(marker in normalized for marker in self.grounding_blocklist):
            return False
        return all(marker.casefold() in normalized for marker in required_markers)


class TaskHandlerRegistry:
    def __init__(self, handlers: tuple[TaskHandler, ...]) -> None:
        self._handlers = handlers

    def all(self) -> tuple[TaskHandler, ...]:
        return self._handlers

    def match(self, task_text: str, visible_files: list[str]) -> TaskHandler | None:
        visible_set = set(visible_files)
        for handler in self._handlers:
            if handler.matches(task_text, visible_set):
                return handler
        return None



def _build_default_handlers() -> tuple[TaskHandler, ...]:
    todo_priority_patch = TaskHandler(
        name="todo_priority_patch",
        description="Fix priority-aware ordering and PATCH partial-update behavior in the todo API.",
        sample_task_text="Add priority sorting to the Todo API and fix the PATCH partial update bug.",
        keywords=(
            "priority sorting",
            "priority-aware",
            "priority order",
            "priority",
            "patch partial update",
            "partial update bug",
            "patch merge",
            "部分更新",
            "局部更新",
            "优先级",
        ),
        required_files=("todo_api.py", "tests/test_todo_api.py"),
        prompt_file_candidates=("todo_api.py", "tests/test_todo_api.py"),
        planned_changes=(
            PlannedChange(
                path="todo_api.py",
                content=TARGET_TODO_API,
                summary="Implement priority-aware todo ordering and correct PATCH merge semantics.",
            ),
            PlannedChange(
                path="tests/test_todo_api.py",
                content=TARGET_TEST_TODO_API,
                summary="Add regression coverage for priority sorting and PATCH partial updates.",
            ),
        ),
        offline_plan_steps=(
            "Inspect todo_api.py and tests/test_todo_api.py.",
            "Implement priority sorting and PATCH merge behavior in todo_api.py.",
            "Add regression coverage for both behaviors in tests/test_todo_api.py.",
            f"Run {DEFAULT_VALIDATION_COMMAND} and repair once if needed.",
        ),
        proposal_edit_seeds=(
            ProposalEditSeed(
                path="todo_api.py",
                change_type="update",
                target_symbols=("list_todos", "apply_patch_update"),
                intent="add priority-aware ordering and preserve unspecified fields during PATCH updates",
            ),
            ProposalEditSeed(
                path="tests/test_todo_api.py",
                change_type="update",
                target_symbols=(
                    "TodoApiTests.test_list_todos_sorts_by_priority_then_id",
                    "TodoApiTests.test_apply_patch_update_preserves_unspecified_fields",
                ),
                intent="cover priority sorting and PATCH partial-update regression behavior",
            ),
        ),
        clarify_required_markers=("todo_api.py", DEFAULT_VALIDATION_COMMAND),
        plan_required_markers=("todo_api.py", "tests/test_todo_api.py", DEFAULT_VALIDATION_COMMAND),
        proposal_required_markers=("todo_api.py", "tests/test_todo_api.py", DEFAULT_VALIDATION_COMMAND),
        offline_clarify=(
            "Implement priority-aware ordering and correct PATCH merge semantics in the Python demo repo. "
            "Inspect todo_api.py and tests/test_todo_api.py, update both behavior and regression coverage, "
            f"then validate with {DEFAULT_VALIDATION_COMMAND}."
        ),
        offline_plan=(
            "Inspect todo_api.py and tests/test_todo_api.py, update runtime behavior and regression coverage, "
            f"then validate with {DEFAULT_VALIDATION_COMMAND}."
        ),
        offline_proposal=(
            "Prepare a bounded diff candidate for todo_api.py and tests/test_todo_api.py covering priority-aware ordering, "
            "PATCH partial-update preservation, and matching regression coverage."
        ),
        review_if_changed=(
            "Behavior change: list_todos now sorts todos by priority high -> medium -> low and then by id; "
            "apply_patch_update now preserves unspecified fields during PATCH updates."
        ),
        review_if_unchanged=(
            "No file changes were needed; the repo already satisfied priority sorting high -> medium -> low "
            "and PATCH partial-merge behavior."
        ),
        review_risk_note=(
            "unknown priority values still fall behind known priorities because the demo keeps the domain intentionally small."
        ),
    )
    todo_query_filters = TaskHandler(
        name="todo_query_filters",
        description="Add completion filtering and case-insensitive search helpers for todos.",
        sample_task_text="Add completed-only filtering and case-insensitive search to the todo query helpers.",
        keywords=(
            "completed-only filtering",
            "completed filter",
            "filter completed",
            "case-insensitive search",
            "query helper",
            "搜索",
            "筛选",
            "已完成",
        ),
        required_files=("todo_queries.py", "tests/test_todo_queries.py"),
        prompt_file_candidates=("todo_queries.py", "tests/test_todo_queries.py"),
        planned_changes=(
            PlannedChange(
                path="todo_queries.py",
                content=TARGET_TODO_QUERIES,
                summary="Implement completion filtering and case-insensitive title search helpers.",
            ),
            PlannedChange(
                path="tests/test_todo_queries.py",
                content=TARGET_TEST_TODO_QUERIES,
                summary="Add regression coverage for completed filtering and trimmed case-insensitive search.",
            ),
        ),
        offline_plan_steps=(
            "Inspect todo_queries.py and tests/test_todo_queries.py.",
            "Implement completed filtering and trimmed case-insensitive search in todo_queries.py.",
            "Add regression coverage for both helpers and input immutability in tests/test_todo_queries.py.",
            f"Run {DEFAULT_VALIDATION_COMMAND} and repair once if needed.",
        ),
        proposal_edit_seeds=(
            ProposalEditSeed(
                path="todo_queries.py",
                change_type="update",
                target_symbols=("filter_todos", "search_todos"),
                intent="add completed-state filtering and trimmed case-insensitive title search",
            ),
            ProposalEditSeed(
                path="tests/test_todo_queries.py",
                change_type="update",
                target_symbols=(
                    "TodoQueryTests.test_filter_todos_can_return_only_completed_items",
                    "TodoQueryTests.test_search_todos_is_case_insensitive_and_trims_whitespace",
                ),
                intent="cover filtering, trimmed search behavior, and input immutability",
            ),
        ),
        clarify_required_markers=("todo_queries.py", DEFAULT_VALIDATION_COMMAND),
        plan_required_markers=("todo_queries.py", "tests/test_todo_queries.py", DEFAULT_VALIDATION_COMMAND),
        proposal_required_markers=("todo_queries.py", "tests/test_todo_queries.py", DEFAULT_VALIDATION_COMMAND),
        offline_clarify=(
            "Implement completed filtering and case-insensitive trimmed search in the Python demo repo. "
            "Inspect todo_queries.py and tests/test_todo_queries.py, update behavior and regression coverage, "
            f"then validate with {DEFAULT_VALIDATION_COMMAND}."
        ),
        offline_plan=(
            "Inspect todo_queries.py and tests/test_todo_queries.py, update helper behavior and regression coverage, "
            f"then validate with {DEFAULT_VALIDATION_COMMAND}."
        ),
        offline_proposal=(
            "Prepare a bounded diff candidate for todo_queries.py and tests/test_todo_queries.py covering completed filtering, "
            "trimmed case-insensitive search, and matching regression coverage."
        ),
        review_if_changed=(
            "Behavior change: filter_todos now honors completed=True/False when requested, and search_todos now trims "
            "the query and matches titles case-insensitively."
        ),
        review_if_unchanged=(
            "No file changes were needed; the repo already satisfied completed filtering and case-insensitive search behavior."
        ),
        review_risk_note="search still only checks the title field and does not rank or score matches.",
    )
    return (todo_priority_patch, todo_query_filters)


DEFAULT_TASK_HANDLER_REGISTRY = TaskHandlerRegistry(_build_default_handlers())

