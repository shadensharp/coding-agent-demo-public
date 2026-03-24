from __future__ import annotations

from collections import Counter
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .demo_repo_seed import restore_demo_repo
from .render import TerminalRenderer
from .repo_ops import RepoOps
from .runner import Runner
from .storage import JsonlRunStore
from .task_handlers import DEFAULT_TASK_HANDLER_REGISTRY, TaskHandlerRegistry


@dataclass(slots=True)
class EvalCaseResult:
    handler_name: str
    task_text: str
    status: str
    retrieved_files: int
    required_file_hits: int
    required_file_total: int
    changed_files: int
    latest_test_exit: int | None
    latest_test_duration_ms: int | None
    fallback_steps: list[str]
    review_summary: str
    session_id: str
    tool_calls: int
    approval_checks: int
    denied_tool_calls: int
    proposal_status: str
    proposal_score: int
    proposal_edit_candidates: int

    def to_dict(self) -> dict[str, object]:
        return {
            "handler_name": self.handler_name,
            "task_text": self.task_text,
            "status": self.status,
            "retrieved_files": self.retrieved_files,
            "required_file_hits": self.required_file_hits,
            "required_file_total": self.required_file_total,
            "changed_files": self.changed_files,
            "latest_test_exit": self.latest_test_exit,
            "latest_test_duration_ms": self.latest_test_duration_ms,
            "fallback_steps": list(self.fallback_steps),
            "review_summary": self.review_summary,
            "session_id": self.session_id,
            "tool_calls": self.tool_calls,
            "approval_checks": self.approval_checks,
            "denied_tool_calls": self.denied_tool_calls,
            "proposal_status": self.proposal_status,
            "proposal_score": self.proposal_score,
            "proposal_edit_candidates": self.proposal_edit_candidates,
        }



def _clone_config(base_config: AppConfig, base_dir: Path, repo_dir: Path, enable_live_llm: bool) -> AppConfig:
    return AppConfig(
        base_dir=base_dir,
        runtime_dir=base_dir / "runtime",
        sessions_dir=base_dir / "runtime" / "sessions",
        default_repo_dir=repo_dir,
        qwen_model=base_config.qwen_model,
        qwen_api_key=base_config.qwen_api_key if enable_live_llm else None,
        qwen_api_base=base_config.qwen_api_base,
        qwen_timeout_seconds=base_config.qwen_timeout_seconds,
        qwen_max_retries=base_config.qwen_max_retries,
        qwen_retry_backoff_seconds=base_config.qwen_retry_backoff_seconds,
    )



def run_preset_eval(
    config: AppConfig,
    task_registry: TaskHandlerRegistry | None = None,
    enable_live_llm: bool = False,
) -> dict[str, object]:
    registry = task_registry or DEFAULT_TASK_HANDLER_REGISTRY
    case_results: list[EvalCaseResult] = []

    with tempfile.TemporaryDirectory(prefix="coding-agent-eval-") as temp_dir_raw:
        temp_root = Path(temp_dir_raw)
        for index, handler in enumerate(registry.all(), start=1):
            case_dir = temp_root / f"case_{index}_{handler.name}"
            repo_dir = case_dir / "demo_repo"
            repo_dir.mkdir(parents=True, exist_ok=True)
            restore_demo_repo(RepoOps(repo_dir))

            case_config = _clone_config(config, case_dir, repo_dir, enable_live_llm=enable_live_llm)
            case_config.ensure_directories()
            runner = Runner(
                config=case_config,
                store=JsonlRunStore(case_config.sessions_dir),
                renderer=TerminalRenderer(io.StringIO()),
                task_registry=registry,
            )
            session = runner.run(
                task_text=handler.sample_task_text,
                repo_path=str(repo_dir),
                session_name=f"eval:{handler.name}",
            )
            latest_test = session.test_results[-1] if session.test_results else None
            retrieved_paths = {item.path for item in session.retrieved_files}
            required_hits = sum(1 for path_name in handler.required_files if path_name in retrieved_paths)
            denied_tool_calls = sum(1 for item in session.approval_checks if not item.approved)
            proposal_assessment = session.proposal_assessment
            proposal_candidate = session.proposal_candidate
            case_results.append(
                EvalCaseResult(
                    handler_name=handler.name,
                    task_text=handler.sample_task_text,
                    status=session.status,
                    retrieved_files=len(session.retrieved_files),
                    required_file_hits=required_hits,
                    required_file_total=len(handler.required_files),
                    changed_files=len(session.changed_files),
                    latest_test_exit=latest_test.exit_code if latest_test else None,
                    latest_test_duration_ms=latest_test.duration_ms if latest_test else None,
                    fallback_steps=list(session.fallback_steps),
                    review_summary=session.review_summary or "",
                    session_id=session.session_id,
                    tool_calls=len(session.tool_calls),
                    approval_checks=len(session.approval_checks),
                    denied_tool_calls=denied_tool_calls,
                    proposal_status=proposal_assessment.status if proposal_assessment else "unavailable",
                    proposal_score=proposal_assessment.score if proposal_assessment else 0,
                    proposal_edit_candidates=len(proposal_candidate.edits) if proposal_candidate else 0,
                )
            )

    total_cases = len(case_results)
    passed_cases = sum(1 for result in case_results if result.status == "completed")
    failed_cases = total_cases - passed_cases
    fallback_cases = sum(1 for result in case_results if result.fallback_steps)
    total_changed_files = sum(result.changed_files for result in case_results)
    total_retrieved_files = sum(result.retrieved_files for result in case_results)
    total_required_file_hits = sum(result.required_file_hits for result in case_results)
    total_required_file_targets = sum(result.required_file_total for result in case_results)
    total_fallback_steps = sum(len(result.fallback_steps) for result in case_results)
    total_tool_calls = sum(result.tool_calls for result in case_results)
    total_approval_checks = sum(result.approval_checks for result in case_results)
    total_proposal_score = sum(result.proposal_score for result in case_results)
    total_proposal_edits = sum(result.proposal_edit_candidates for result in case_results)
    proposal_accepted_cases = sum(1 for result in case_results if result.proposal_status == "accepted")
    approval_denials = sum(result.denied_tool_calls for result in case_results)
    durations = [result.latest_test_duration_ms for result in case_results if result.latest_test_duration_ms is not None]
    avg_latest_test_duration_ms = int(sum(durations) / len(durations)) if durations else 0
    avg_changed_files = round(total_changed_files / total_cases, 2) if total_cases else 0.0
    avg_retrieved_files = round(total_retrieved_files / total_cases, 2) if total_cases else 0.0
    avg_fallback_steps_per_case = round(total_fallback_steps / total_cases, 2) if total_cases else 0.0
    avg_tool_calls = round(total_tool_calls / total_cases, 2) if total_cases else 0.0
    avg_approval_checks = round(total_approval_checks / total_cases, 2) if total_cases else 0.0
    avg_proposal_score = round(total_proposal_score / total_cases, 1) if total_cases else 0.0
    avg_proposal_edits = round(total_proposal_edits / total_cases, 2) if total_cases else 0.0
    pass_rate = round((passed_cases / total_cases) * 100, 1) if total_cases else 0.0
    proposal_accept_rate = round((proposal_accepted_cases / total_cases) * 100, 1) if total_cases else 0.0
    required_file_hit_rate = (
        round((total_required_file_hits / total_required_file_targets) * 100, 1)
        if total_required_file_targets
        else 0.0
    )
    fallback_step_counts = Counter(step for result in case_results for step in result.fallback_steps)

    return {
        "mode": "live_llm" if enable_live_llm else "offline",
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate": pass_rate,
        "fallback_cases": fallback_cases,
        "total_changed_files": total_changed_files,
        "avg_changed_files": avg_changed_files,
        "avg_retrieved_files": avg_retrieved_files,
        "required_file_hit_rate": required_file_hit_rate,
        "total_fallback_steps": total_fallback_steps,
        "avg_fallback_steps_per_case": avg_fallback_steps_per_case,
        "avg_latest_test_duration_ms": avg_latest_test_duration_ms,
        "avg_tool_calls": avg_tool_calls,
        "avg_approval_checks": avg_approval_checks,
        "approval_denials": approval_denials,
        "proposal_accept_rate": proposal_accept_rate,
        "avg_proposal_score": avg_proposal_score,
        "avg_proposal_edits": avg_proposal_edits,
        "fallback_step_counts": dict(sorted(fallback_step_counts.items())),
        "results": [result.to_dict() for result in case_results],
    }
