from __future__ import annotations

from dataclasses import dataclass, field


def _clean_text(text: object) -> str:
    return " ".join(str(text).strip().split())


def _sentence(text: object) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


@dataclass(slots=True)
class TaskRequest:
    request_id: str
    user_text: str
    repo_path: str
    created_at: str
    constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "user_text": self.user_text,
            "repo_path": self.repo_path,
            "created_at": self.created_at,
            "constraints": list(self.constraints),
        }


@dataclass(slots=True)
class CommandResult:
    command: list[str]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int

    def to_dict(self) -> dict[str, object]:
        return {
            "command": list(self.command),
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
        }


@dataclass(slots=True)
class FileChange:
    path: str
    change_type: str
    summary: str
    diff_excerpt: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "change_type": self.change_type,
            "summary": self.summary,
            "diff_excerpt": self.diff_excerpt,
        }


@dataclass(slots=True)
class RetrievedFile:
    path: str
    score: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "score": self.score,
            "reasons": list(self.reasons),
        }


@dataclass(slots=True)
class ResearchSource:
    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
        }


@dataclass(slots=True)
class ContextFileSummary:
    path: str
    line_count: int
    function_names: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    test_names: list[str] = field(default_factory=list)
    parse_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line_count": self.line_count,
            "function_names": list(self.function_names),
            "class_names": list(self.class_names),
            "test_names": list(self.test_names),
            "parse_error": self.parse_error,
        }


@dataclass(slots=True)
class ClarifyArtifact:
    implementation_target: str
    relevant_files: list[str] = field(default_factory=list)
    validation_command: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "implementation_target": self.implementation_target,
            "relevant_files": list(self.relevant_files),
            "validation_command": self.validation_command,
        }

    def summary(self) -> str:
        parts: list[str] = []
        if self.implementation_target:
            parts.append(_sentence(self.implementation_target))
        if self.relevant_files:
            parts.append(_sentence("Relevant files: " + ", ".join(self.relevant_files)))
        if self.validation_command:
            parts.append(_sentence(f"Validation: {self.validation_command}"))
        return " ".join(parts).strip()


@dataclass(slots=True)
class PlanArtifact:
    steps: list[str] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    validation_command: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": list(self.steps),
            "target_files": list(self.target_files),
            "validation_command": self.validation_command,
        }

    def summary(self) -> str:
        parts: list[str] = []
        ordered_steps = [
            f"{index + 1}) {_clean_text(step).rstrip('.')}"
            for index, step in enumerate(self.steps)
            if _clean_text(step)
        ]
        if ordered_steps:
            parts.append(_sentence(" ".join(ordered_steps)))
        if self.target_files:
            parts.append(_sentence("Targets: " + ", ".join(self.target_files)))
        if self.validation_command:
            parts.append(_sentence(f"Validation: {self.validation_command}"))
        return " ".join(parts).strip()


@dataclass(slots=True)
class ProposalEditCandidate:
    path: str
    change_type: str
    target_symbols: list[str] = field(default_factory=list)
    intent: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "change_type": self.change_type,
            "target_symbols": list(self.target_symbols),
            "intent": self.intent,
        }

    def summary(self) -> str:
        parts = [f"{self.change_type} {self.path}"]
        if self.target_symbols:
            parts.append("targets " + ", ".join(self.target_symbols))
        if self.intent:
            parts.append(_clean_text(self.intent))
        return " | ".join(part for part in parts if part)


@dataclass(slots=True)
class ProposalCandidate:
    summary_text: str
    edits: list[ProposalEditCandidate] = field(default_factory=list)
    validation_command: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "summary_text": self.summary_text,
            "edits": [item.to_dict() for item in self.edits],
            "validation_command": self.validation_command,
        }

    def summary(self) -> str:
        parts: list[str] = []
        if self.summary_text:
            parts.append(_sentence(self.summary_text))
        edit_summaries = [item.summary() for item in self.edits if item.summary()]
        if edit_summaries:
            parts.append(_sentence("Candidate edits: " + "; ".join(edit_summaries)))
        if self.validation_command:
            parts.append(_sentence(f"Validation: {self.validation_command}"))
        return " ".join(parts).strip()


@dataclass(slots=True)
class ProposalAssessment:
    status: str
    score: int
    matched_targets: list[str] = field(default_factory=list)
    missing_targets: list[str] = field(default_factory=list)
    extra_targets: list[str] = field(default_factory=list)
    has_validation_command: bool = False
    used_fallback: bool = False
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "score": self.score,
            "matched_targets": list(self.matched_targets),
            "missing_targets": list(self.missing_targets),
            "extra_targets": list(self.extra_targets),
            "has_validation_command": self.has_validation_command,
            "used_fallback": self.used_fallback,
            "note": self.note,
        }


@dataclass(slots=True)
class ApprovalCheck:
    tool_name: str
    step_type: str
    approved: bool
    mode: str
    reason: str
    requested_targets: list[str] = field(default_factory=list)
    requested_command: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "tool_name": self.tool_name,
            "step_type": self.step_type,
            "approved": self.approved,
            "mode": self.mode,
            "reason": self.reason,
            "requested_targets": list(self.requested_targets),
            "requested_command": list(self.requested_command),
        }


@dataclass(slots=True)
class ToolCall:
    tool_call_id: str
    tool_name: str
    step_type: str
    purpose: str
    status: str
    approval_mode: str
    approved: bool
    started_at: str | None = None
    finished_at: str | None = None
    input_summary: str | None = None
    output_summary: str | None = None
    error_message: str | None = None
    requested_targets: list[str] = field(default_factory=list)
    requested_command: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "step_type": self.step_type,
            "purpose": self.purpose,
            "status": self.status,
            "approval_mode": self.approval_mode,
            "approved": self.approved,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "error_message": self.error_message,
            "requested_targets": list(self.requested_targets),
            "requested_command": list(self.requested_command),
        }


@dataclass(slots=True)
class Step:
    step_id: str
    session_id: str
    type: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    input_summary: str | None = None
    output_summary: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "session_id": self.session_id,
            "type": self.type,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "error_message": self.error_message,
        }


@dataclass(slots=True)
class Session:
    session_id: str
    request: TaskRequest
    status: str
    created_at: str
    session_name: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    task_handler: str | None = None
    research_enabled: bool = False
    research_query: str | None = None
    research_summary: str | None = None
    research_sources: list[ResearchSource] = field(default_factory=list)
    clarify_artifact: ClarifyArtifact | None = None
    clarify_summary: str | None = None
    plan_artifact: PlanArtifact | None = None
    plan_summary: str | None = None
    read_summary: str | None = None
    proposal_candidate: ProposalCandidate | None = None
    proposal_summary: str | None = None
    proposal_assessment: ProposalAssessment | None = None
    final_summary: str | None = None
    review_summary: str | None = None
    fallback_steps: list[str] = field(default_factory=list)
    retrieved_files: list[RetrievedFile] = field(default_factory=list)
    context_summaries: list[ContextFileSummary] = field(default_factory=list)
    approval_checks: list[ApprovalCheck] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    changed_files: list[FileChange] = field(default_factory=list)
    test_results: list[CommandResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "task_handler": self.task_handler,
            "research_enabled": self.research_enabled,
            "research_query": self.research_query,
            "research_summary": self.research_summary,
            "research_sources": [item.to_dict() for item in self.research_sources],
            "clarify_artifact": self.clarify_artifact.to_dict() if self.clarify_artifact else None,
            "clarify_summary": self.clarify_summary,
            "plan_artifact": self.plan_artifact.to_dict() if self.plan_artifact else None,
            "plan_summary": self.plan_summary,
            "read_summary": self.read_summary,
            "proposal_candidate": self.proposal_candidate.to_dict() if self.proposal_candidate else None,
            "proposal_summary": self.proposal_summary,
            "proposal_assessment": self.proposal_assessment.to_dict() if self.proposal_assessment else None,
            "final_summary": self.final_summary,
            "review_summary": self.review_summary,
            "fallback_steps": list(self.fallback_steps),
            "retrieved_files": [item.to_dict() for item in self.retrieved_files],
            "context_summaries": [item.to_dict() for item in self.context_summaries],
            "approval_checks": [item.to_dict() for item in self.approval_checks],
            "tool_calls": [item.to_dict() for item in self.tool_calls],
            "task_title": self.request.user_text[:80],
            "request": self.request.to_dict(),
            "changed_files": [change.to_dict() for change in self.changed_files],
            "test_results": [result.to_dict() for result in self.test_results],
        }
