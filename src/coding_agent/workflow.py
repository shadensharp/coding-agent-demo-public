from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from .events import new_id, utc_now
from .models import ApprovalCheck, Session, ToolCall
from .task_handlers import DEFAULT_TEST_COMMAND, TaskHandler

READ_REPO_TOOL = "repo_reader"
EDIT_PROPOSAL_TOOL = "edit_proposal_generator"
PRESET_EDIT_TOOL = "preset_file_editor"
TEST_COMMAND_TOOL = "python_test_runner"
REVIEW_TOOL = "review_compiler"
RESEARCH_TOOL = "web_researcher"

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ToolRequest:
    step_type: str
    tool_name: str
    purpose: str
    input_summary: str | None = None
    requested_targets: tuple[str, ...] = ()
    requested_command: tuple[str, ...] = ()


class ApprovalPolicy:
    def evaluate(self, request: ToolRequest, handler: TaskHandler | None) -> ApprovalCheck:
        if request.tool_name in {READ_REPO_TOOL, EDIT_PROPOSAL_TOOL, REVIEW_TOOL, RESEARCH_TOOL}:
            return ApprovalCheck(
                tool_name=request.tool_name,
                step_type=request.step_type,
                approved=True,
                mode="auto_allow",
                reason="read-only workflow tool",
                requested_targets=list(request.requested_targets),
                requested_command=list(request.requested_command),
            )

        if request.tool_name == PRESET_EDIT_TOOL:
            requested_targets = set(request.requested_targets)
            if not requested_targets:
                return ApprovalCheck(
                    tool_name=request.tool_name,
                    step_type=request.step_type,
                    approved=True,
                    mode="auto_allow",
                    reason="no-op write scope because no bounded edit targets were requested",
                    requested_targets=list(request.requested_targets),
                    requested_command=list(request.requested_command),
                )

            if handler is not None:
                allowed_targets = {change.path for change in handler.planned_changes}
                if requested_targets.issubset(allowed_targets):
                    return ApprovalCheck(
                        tool_name=request.tool_name,
                        step_type=request.step_type,
                        approved=True,
                        mode="auto_allow",
                        reason=f"write scope constrained to preset handler '{handler.name}'",
                        requested_targets=list(request.requested_targets),
                        requested_command=list(request.requested_command),
                    )
                return ApprovalCheck(
                    tool_name=request.tool_name,
                    step_type=request.step_type,
                    approved=False,
                    mode="auto_deny",
                    reason="requested write scope exceeds the preset handler allowlist",
                    requested_targets=list(request.requested_targets),
                    requested_command=list(request.requested_command),
                )

            if self._is_generic_python_scope(requested_targets):
                return ApprovalCheck(
                    tool_name=request.tool_name,
                    step_type=request.step_type,
                    approved=True,
                    mode="auto_allow",
                    reason="write scope constrained to bounded Python files selected by clarify/plan/proposal",
                    requested_targets=list(request.requested_targets),
                    requested_command=list(request.requested_command),
                )

            return ApprovalCheck(
                tool_name=request.tool_name,
                step_type=request.step_type,
                approved=False,
                mode="auto_deny",
                reason="generic write scope must stay within a small Python-only target set",
                requested_targets=list(request.requested_targets),
                requested_command=list(request.requested_command),
            )

        if request.tool_name == TEST_COMMAND_TOOL:
            allowed_command = tuple(handler.test_command if handler else DEFAULT_TEST_COMMAND)
            if tuple(request.requested_command) == allowed_command:
                return ApprovalCheck(
                    tool_name=request.tool_name,
                    step_type=request.step_type,
                    approved=True,
                    mode="auto_allow",
                    reason="command matches the allowlisted test command",
                    requested_targets=list(request.requested_targets),
                    requested_command=list(request.requested_command),
                )
            return ApprovalCheck(
                tool_name=request.tool_name,
                step_type=request.step_type,
                approved=False,
                mode="auto_deny",
                reason="command is outside the allowlisted test command",
                requested_targets=list(request.requested_targets),
                requested_command=list(request.requested_command),
            )

        return ApprovalCheck(
            tool_name=request.tool_name,
            step_type=request.step_type,
            approved=False,
            mode="auto_deny",
            reason="unknown workflow tool",
            requested_targets=list(request.requested_targets),
            requested_command=list(request.requested_command),
        )

    def _is_generic_python_scope(self, requested_targets: set[str]) -> bool:
        if not requested_targets or len(requested_targets) > 4:
            return False
        for target in requested_targets:
            normalized = str(target).strip().replace("\\", "/")
            if not normalized or normalized.startswith("/"):
                return False
            if ".." in normalized.split("/"):
                return False
            if not normalized.endswith(".py"):
                return False
        return True


class ToolExecutor:
    def __init__(self, approval_policy: ApprovalPolicy | None = None) -> None:
        self.approval_policy = approval_policy or ApprovalPolicy()

    def execute(
        self,
        session: Session,
        request: ToolRequest,
        handler: TaskHandler | None,
        action: Callable[[], tuple[T, str]],
        emit: Callable[[str, dict[str, object]], None],
        save_summary: Callable[[Session], None],
    ) -> T:
        approval = self.approval_policy.evaluate(request, handler)
        session.approval_checks.append(approval)
        save_summary(session)
        emit("approval_checked", approval.to_dict())

        tool_call = ToolCall(
            tool_call_id=new_id("tool"),
            tool_name=request.tool_name,
            step_type=request.step_type,
            purpose=request.purpose,
            status="running",
            approval_mode=approval.mode,
            approved=approval.approved,
            started_at=utc_now(),
            input_summary=request.input_summary,
            requested_targets=list(request.requested_targets),
            requested_command=list(request.requested_command),
        )
        session.tool_calls.append(tool_call)
        save_summary(session)
        emit(
            "tool_started",
            {
                "tool_call_id": tool_call.tool_call_id,
                "tool_name": tool_call.tool_name,
                "step_type": tool_call.step_type,
                "purpose": tool_call.purpose,
                "approval_mode": tool_call.approval_mode,
            },
        )

        if not approval.approved:
            message = approval.reason
            tool_call.status = "failed"
            tool_call.finished_at = utc_now()
            tool_call.error_message = message
            save_summary(session)
            emit(
                "tool_failed",
                {
                    "tool_call_id": tool_call.tool_call_id,
                    "tool_name": tool_call.tool_name,
                    "step_type": tool_call.step_type,
                    "error_message": message,
                },
            )
            raise PermissionError(message)

        try:
            result, output_summary = action()
        except Exception as exc:
            tool_call.status = "failed"
            tool_call.finished_at = utc_now()
            tool_call.error_message = str(exc)
            save_summary(session)
            emit(
                "tool_failed",
                {
                    "tool_call_id": tool_call.tool_call_id,
                    "tool_name": tool_call.tool_name,
                    "step_type": tool_call.step_type,
                    "error_message": tool_call.error_message,
                },
            )
            raise

        tool_call.status = "completed"
        tool_call.finished_at = utc_now()
        tool_call.output_summary = output_summary
        save_summary(session)
        emit(
            "tool_completed",
            {
                "tool_call_id": tool_call.tool_call_id,
                "tool_name": tool_call.tool_name,
                "step_type": tool_call.step_type,
                "output_summary": output_summary,
            },
        )
        return result
