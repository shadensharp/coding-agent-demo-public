from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path

from .config import AppConfig
from .events import make_event, new_id, utc_now
from .llm import QwenClient
from .models import (
    ClarifyArtifact,
    CommandResult,
    ContextFileSummary,
    PlanArtifact,
    ProposalAssessment,
    ProposalCandidate,
    ProposalEditCandidate,
    ResearchSource,
    RetrievedFile,
    Session,
    Step,
    TaskRequest,
)
from .prompts import (
    CLARIFY_PROMPT,
    EDIT_PROMPT,
    PLAN_PROMPT,
    PROPOSAL_PROMPT,
    SYSTEM_PROMPT,
    build_clarify_prompt,
    build_edit_prompt,
    build_plan_prompt,
    build_proposal_prompt,
)
from .render import TerminalRenderer
from .research import WebResearchClient
from .repo_ops import RepoOps
from .repo_summary import RepoContextBuilder
from .retrieval import RepoRetriever
from .storage import RunStore
from .task_handlers import (
    DEFAULT_TASK_HANDLER_REGISTRY,
    DEFAULT_TEST_COMMAND,
    DEFAULT_VALIDATION_COMMAND,
    TaskHandler,
    TaskHandlerRegistry,
)
from .workflow import (
    EDIT_PROPOSAL_TOOL,
    PRESET_EDIT_TOOL,
    READ_REPO_TOOL,
    RESEARCH_TOOL,
    REVIEW_TOOL,
    TEST_COMMAND_TOOL,
    ToolExecutor,
    ToolRequest,
)

@dataclass(slots=True)
class GeneratedFileEdit:
    path: str
    change_type: str
    summary: str
    content: str | None = None


class Runner:
    def __init__(
        self,
        config: AppConfig,
        store: RunStore,
        renderer: TerminalRenderer,
        llm_client: QwenClient | None = None,
        task_registry: TaskHandlerRegistry | None = None,
        retriever: RepoRetriever | None = None,
        context_builder: RepoContextBuilder | None = None,
        tool_executor: ToolExecutor | None = None,
        research_client: WebResearchClient | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.renderer = renderer
        self.llm_client = llm_client or QwenClient(config)
        self.task_registry = task_registry or DEFAULT_TASK_HANDLER_REGISTRY
        self.retriever = retriever or RepoRetriever()
        self.context_builder = context_builder or RepoContextBuilder()
        self.tool_executor = tool_executor or ToolExecutor()
        self.research_client = research_client or WebResearchClient(config)

    def run(
        self,
        task_text: str,
        repo_path: str | None = None,
        max_fix: int = 1,
        session_name: str | None = None,
        session_id: str | None = None,
        enable_web_research: bool = False,
        research_query: str | None = None,
    ) -> Session:
        repo_root = Path(repo_path).resolve() if repo_path else self.config.default_repo_dir
        repo_ops = RepoOps(repo_root)
        repo_ops.ensure_repo_exists()
        matched_handler = self.task_registry.match(task_text, repo_ops.list_files(limit=None))
        max_fix = max(0, max_fix)

        created_at = utc_now()
        constraints = [
            "Python repository only",
            "Single agent",
            "One-shot CLI flow",
            f"Auto-fix limit: {max_fix}",
        ]
        normalized_research_query = self._normalize_text(research_query or task_text) if enable_web_research else ""
        if enable_web_research:
            constraints.append("Optional external research enabled")
        request = TaskRequest(
            request_id=new_id("req"),
            user_text=task_text,
            repo_path=str(repo_root),
            created_at=created_at,
            constraints=constraints,
        )
        session = Session(
            session_id=session_id or new_id("sess"),
            session_name=session_name,
            request=request,
            status="running",
            created_at=created_at,
            started_at=created_at,
            task_handler=matched_handler.name if matched_handler else None,
            research_enabled=enable_web_research,
            research_query=normalized_research_query or None,
        )
        retrieval_limit = 2 if matched_handler is not None else 3
        session.retrieved_files = self.retriever.build(
            repo_ops,
            task_text,
            matched_handler,
            limit=retrieval_limit,
        )
        session.context_summaries = self.context_builder.build(repo_ops, session.retrieved_files)
        self.store.save_summary(session)

        try:
            self._emit(
                session.session_id,
                "session_started",
                {"task": task_text, "repo": str(repo_root), "task_handler": session.task_handler or "generic"},
            )
            self._emit(
                session.session_id,
                "context_selected",
                {
                    "files": [
                        {
                            "path": item.path,
                            "score": item.score,
                            "reasons": list(item.reasons),
                            "summary": self._context_summary_text(session.context_summaries, item.path),
                        }
                        for item in session.retrieved_files
                    ]
                },
            )

            if session.research_enabled and session.research_query:
                session.research_summary = self._run_step(
                    session,
                    "research",
                    session.research_query,
                    lambda: self._execute_tool(
                        session,
                        step_type="research",
                        tool_name=RESEARCH_TOOL,
                        purpose="gather optional external sources that may inform the implementation",
                        input_summary=session.research_query,
                        handler=matched_handler,
                        action=lambda: self._research_tool(session),
                    ),
                )
                self.store.save_summary(session)

            session.clarify_summary = self._run_step(
                session,
                "clarify",
                task_text,
                lambda: self._clarify_request(session, task_text, repo_ops, request.constraints, matched_handler),
            )
            session.plan_summary = self._run_step(
                session,
                "plan",
                session.clarify_summary,
                lambda: self._build_plan(
                    session,
                    task_text,
                    session.clarify_summary or "",
                    repo_ops,
                    request.constraints,
                    matched_handler,
                ),
            )
            self.store.save_summary(session)
            session.read_summary = self._run_step(
                session,
                "read",
                session.plan_summary,
                lambda: self._execute_tool(
                    session,
                    step_type="read",
                    tool_name=READ_REPO_TOOL,
                    purpose="inspect retrieved repo context",
                    input_summary=session.plan_summary,
                    handler=matched_handler,
                    action=lambda: self._inspect_repo_tool(session),
                ),
            )
            self.store.save_summary(session)
            session.proposal_summary = self._run_step(
                session,
                "proposal",
                session.read_summary,
                lambda: self._execute_tool(
                    session,
                    step_type="proposal",
                    tool_name=EDIT_PROPOSAL_TOOL,
                    purpose="generate a proposal-only edit summary inside the allowed scope",
                    input_summary=session.read_summary,
                    requested_targets=self._planned_targets(session, matched_handler),
                    handler=matched_handler,
                    action=lambda: self._proposal_tool(session, task_text, repo_ops, request.constraints, matched_handler),
                ),
            )
            self.store.save_summary(session)
            self._run_step(
                session,
                "edit",
                session.proposal_summary,
                lambda: self._execute_tool(
                    session,
                    step_type="edit",
                    tool_name=PRESET_EDIT_TOOL,
                    purpose="apply the matched preset change set",
                    input_summary=session.proposal_summary,
                    requested_targets=self._planned_targets(session, matched_handler),
                    handler=matched_handler,
                    action=lambda: self._edit_repo_tool(session, repo_ops, matched_handler),
                ),
            )
            tests_ok, test_summary = self._run_test_step(session, repo_ops, matched_handler)

            fix_attempts = 0
            while not tests_ok and fix_attempts < max_fix:
                fix_attempts += 1
                self._run_step(
                    session,
                    "fix",
                    test_summary,
                    lambda: self._execute_tool(
                        session,
                        step_type="fix",
                        tool_name=PRESET_EDIT_TOOL,
                        purpose="apply one scoped repair pass",
                        input_summary=test_summary,
                        requested_targets=self._planned_targets(session, matched_handler),
                        handler=matched_handler,
                        action=lambda: self._fix_repo_tool(session, repo_ops, matched_handler, test_summary),
                    ),
                )
                tests_ok, test_summary = self._run_test_step(
                    session,
                    repo_ops,
                    matched_handler,
                    attempt=fix_attempts + 1,
                )

            session.proposal_assessment = self._assess_proposal(session, matched_handler)
            self.store.save_summary(session)
            self._emit(session.session_id, "proposal_assessed", session.proposal_assessment.to_dict())

            session.review_summary = self._run_step(
                session,
                "review",
                test_summary,
                lambda: self._execute_tool(
                    session,
                    step_type="review",
                    tool_name=REVIEW_TOOL,
                    purpose="compile behavior, proposal, grounding, validation, and audit evidence",
                    input_summary=test_summary,
                    handler=matched_handler,
                    action=lambda: self._review_tool(session, tests_ok, matched_handler),
                ),
            )
            session.status = "completed" if tests_ok else "failed"
            session.finished_at = utc_now()
            session.final_summary = self._final_summary(session, tests_ok)
            self.store.save_summary(session)
            self._emit(
                session.session_id,
                "session_completed",
                {
                    "status": session.status,
                    "final_summary": session.final_summary,
                    "session_id": session.session_id,
                },
            )
            return session
        except Exception as exc:
            session.status = "failed"
            session.finished_at = utc_now()
            session.final_summary = f"Run aborted before completion: {exc}"
            self.store.save_summary(session)
            self._emit(
                session.session_id,
                "session_completed",
                {
                    "status": session.status,
                    "final_summary": session.final_summary,
                    "session_id": session.session_id,
                },
            )
            raise

    def _emit(self, session_id: str, kind: str, payload: dict[str, object]) -> None:
        event = make_event(session_id, kind, payload)
        self.store.append_event(event)
        self.renderer.render_event(event)

    def _run_step(
        self,
        session: Session,
        step_type: str,
        input_summary: str | None,
        action: Callable[[], str],
    ) -> str:
        step = Step(
            step_id=new_id("step"),
            session_id=session.session_id,
            type=step_type,
            status="running",
            started_at=utc_now(),
            input_summary=input_summary,
        )
        self._emit(
            session.session_id,
            "step_started",
            {"step_id": step.step_id, "step_type": step.type, "input_summary": step.input_summary or ""},
        )

        try:
            output_summary = action()
        except Exception as exc:
            step.status = "failed"
            step.finished_at = utc_now()
            step.error_message = str(exc)
            self._emit(
                session.session_id,
                "step_failed",
                {
                    "step_id": step.step_id,
                    "step_type": step.type,
                    "error_message": step.error_message,
                },
            )
            raise

        step.status = "completed"
        step.finished_at = utc_now()
        step.output_summary = output_summary
        self._emit(
            session.session_id,
            "step_completed",
            {
                "step_id": step.step_id,
                "step_type": step.type,
                "output_summary": output_summary,
            },
        )
        return output_summary

    def _run_test_step(
        self,
        session: Session,
        repo_ops: RepoOps,
        handler: TaskHandler | None,
        attempt: int = 1,
    ) -> tuple[bool, str]:
        command = list(handler.test_command if handler else DEFAULT_TEST_COMMAND)
        command_text = " ".join(command)
        input_summary = command_text
        if attempt > 1:
            input_summary = f"{input_summary} (attempt {attempt})"

        step = Step(
            step_id=new_id("step"),
            session_id=session.session_id,
            type="test",
            status="running",
            started_at=utc_now(),
            input_summary=input_summary,
        )
        self._emit(
            session.session_id,
            "step_started",
            {"step_id": step.step_id, "step_type": step.type, "input_summary": step.input_summary},
        )

        result = self._execute_tool(
            session,
            step_type="test",
            tool_name=TEST_COMMAND_TOOL,
            purpose="run the allowlisted regression command",
            input_summary=input_summary,
            requested_command=tuple(command),
            handler=handler,
            action=lambda: self._run_test_command_tool(session, repo_ops, command),
        )
        if result.exit_code == 0:
            summary = f"Tests passed using {command_text}."
            if attempt > 1:
                summary = f"Tests passed on retry {attempt} using {command_text}."
            self._emit(
                session.session_id,
                "step_completed",
                {
                    "step_id": step.step_id,
                    "step_type": step.type,
                    "output_summary": summary,
                },
            )
            return True, summary

        summary = f"Tests failed with exit code {result.exit_code}. {self._failure_excerpt(result)}"
        self._emit(
            session.session_id,
            "step_failed",
            {
                "step_id": step.step_id,
                "step_type": step.type,
                "error_message": summary,
            },
        )
        return False, summary

    def _clarify_request(
        self,
        session: Session,
        task_text: str,
        repo_ops: RepoOps,
        constraints: list[str],
        handler: TaskHandler | None,
    ) -> str:
        visible_files = repo_ops.list_files(limit=8)
        all_visible_files = repo_ops.list_files(limit=None)
        retrieved_summary = self._prompt_retrieved_files(session.retrieved_files)
        file_summaries = self._prompt_context_summaries(session.context_summaries)
        file_snippets = self._prompt_file_snippets(repo_ops, session.retrieved_files)
        validation_command = handler.validation_command if handler else DEFAULT_VALIDATION_COMMAND
        prompt = build_clarify_prompt(
            task_text,
            repo_ops.repo_root.name,
            constraints,
            visible_files,
            retrieved_summary,
            file_summaries,
            file_snippets,
            validation_command,
            external_research=self._prompt_research_sources(session.research_sources),
        )
        validator = handler.validate_clarify_response if handler else None
        artifact = self._complete_structured_or_fallback(
            prompt,
            f"{SYSTEM_PROMPT}\n{CLARIFY_PROMPT}",
            fallback=self._offline_clarify_artifact(session, handler, validation_command),
            parser=lambda response: self._parse_clarify_artifact(response, all_visible_files, validation_command),
            validator=validator,
            session=session,
            step_type="clarify",
        )
        session.clarify_artifact = artifact
        return artifact.summary()

    def _build_plan(
        self,
        session: Session,
        task_text: str,
        clarify_summary: str,
        repo_ops: RepoOps,
        constraints: list[str],
        handler: TaskHandler | None,
    ) -> str:
        visible_files = repo_ops.list_files(limit=8)
        all_visible_files = repo_ops.list_files(limit=None)
        retrieved_summary = self._prompt_retrieved_files(session.retrieved_files)
        file_summaries = self._prompt_context_summaries(session.context_summaries)
        file_snippets = self._prompt_file_snippets(repo_ops, session.retrieved_files)
        validation_command = handler.validation_command if handler else DEFAULT_VALIDATION_COMMAND
        prompt = build_plan_prompt(
            task_text,
            clarify_summary,
            constraints,
            visible_files,
            retrieved_summary,
            file_summaries,
            file_snippets,
            validation_command,
            external_research=self._prompt_research_sources(session.research_sources),
        )
        validator = handler.validate_plan_response if handler else None
        artifact = self._complete_structured_or_fallback(
            prompt,
            f"{SYSTEM_PROMPT}\n{PLAN_PROMPT}",
            fallback=self._offline_plan_artifact(session, handler, validation_command),
            parser=lambda response: self._parse_plan_artifact(response, all_visible_files, validation_command),
            validator=validator,
            session=session,
            step_type="plan",
        )
        session.plan_artifact = artifact
        return artifact.summary()

    def _build_proposal(
        self,
        session: Session,
        task_text: str,
        repo_ops: RepoOps,
        constraints: list[str],
        handler: TaskHandler | None,
    ) -> str:
        retrieved_summary = self._prompt_retrieved_files(session.retrieved_files)
        file_summaries = self._prompt_context_summaries(session.context_summaries)
        file_snippets = self._prompt_file_snippets(repo_ops, session.retrieved_files)
        allowed_edit_scope = list(self._planned_targets(session, handler))
        validation_command = handler.validation_command if handler else DEFAULT_VALIDATION_COMMAND
        prompt = build_proposal_prompt(
            task_text,
            session.clarify_summary or "",
            session.plan_summary or "",
            session.read_summary or "",
            constraints,
            retrieved_summary,
            file_summaries,
            file_snippets,
            validation_command,
            allowed_edit_scope,
            external_research=self._prompt_research_sources(session.research_sources),
        )
        validator = handler.validate_proposal_response if handler else None
        candidate = self._complete_structured_or_fallback(
            prompt,
            f"{SYSTEM_PROMPT}\n{PROPOSAL_PROMPT}",
            fallback=self._offline_proposal_candidate(session, handler, validation_command),
            parser=lambda response: self._parse_proposal_candidate(response, allowed_edit_scope, validation_command),
            validator=validator,
            session=session,
            step_type="proposal",
        )
        session.proposal_candidate = candidate
        return candidate.summary()

    def _normalize_text(self, value: object) -> str:
        return " ".join(str(value).strip().split())

    def _parse_json_payload(self, response: str) -> dict[str, object]:
        payload_text = response.strip()
        if payload_text.startswith("```"):
            lines = payload_text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            payload_text = "\n".join(lines).strip()
        start = payload_text.find("{")
        end = payload_text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("response did not contain a JSON object")
        payload = json.loads(payload_text[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("response JSON was not an object")
        return payload

    def _normalize_string_list(self, value: object, allowed: set[str] | None = None) -> list[str]:
        if not isinstance(value, list):
            return []
        items: list[str] = []
        seen: set[str] = set()
        for raw_item in value:
            item = self._normalize_text(raw_item)
            if not item or item in seen:
                continue
            if allowed is not None and item not in allowed:
                continue
            seen.add(item)
            items.append(item)
        return items

    def _parse_clarify_artifact(
        self,
        response: str,
        visible_files: list[str],
        expected_validation_command: str,
    ) -> ClarifyArtifact:
        payload = self._parse_json_payload(response)
        implementation_target = self._normalize_text(payload.get("implementation_target", ""))
        relevant_files = self._normalize_string_list(payload.get("relevant_files", []), allowed=set(visible_files))
        validation_command = self._normalize_text(payload.get("validation_command", ""))
        if not implementation_target:
            raise ValueError("clarify artifact missing implementation_target")
        if not relevant_files:
            raise ValueError("clarify artifact missing relevant_files")
        if validation_command != expected_validation_command:
            raise ValueError("clarify artifact validation command mismatch")
        return ClarifyArtifact(
            implementation_target=implementation_target,
            relevant_files=relevant_files,
            validation_command=validation_command,
        )

    def _parse_plan_artifact(
        self,
        response: str,
        visible_files: list[str],
        expected_validation_command: str,
    ) -> PlanArtifact:
        payload = self._parse_json_payload(response)
        steps = self._normalize_string_list(payload.get("steps", []))
        target_files = self._normalize_string_list(payload.get("target_files", []), allowed=set(visible_files))
        validation_command = self._normalize_text(payload.get("validation_command", ""))
        if len(steps) < 3:
            raise ValueError("plan artifact must include at least three steps")
        if not target_files:
            raise ValueError("plan artifact missing target_files")
        if validation_command != expected_validation_command:
            raise ValueError("plan artifact validation command mismatch")
        return PlanArtifact(
            steps=steps,
            target_files=target_files,
            validation_command=validation_command,
        )

    def _parse_proposal_candidate(
        self,
        response: str,
        allowed_edit_scope: list[str],
        expected_validation_command: str,
    ) -> ProposalCandidate:
        payload = self._parse_json_payload(response)
        summary_text = self._normalize_text(payload.get("summary", payload.get("summary_text", "")))
        validation_command = self._normalize_text(payload.get("validation_command", ""))
        allowed_targets = set(allowed_edit_scope)
        raw_edits = payload.get("edits", [])
        if not isinstance(raw_edits, list):
            raise ValueError("proposal candidate edits must be a list")

        edits: list[ProposalEditCandidate] = []
        seen_paths: set[str] = set()
        for raw_edit in raw_edits:
            if not isinstance(raw_edit, dict):
                continue
            path = self._normalize_text(raw_edit.get("path", ""))
            if not path or path in seen_paths:
                continue
            if path not in allowed_targets:
                raise ValueError(f"proposal candidate path outside allowed scope: {path}")
            change_type = self._normalize_text(raw_edit.get("change_type", "update")).lower() or "update"
            if change_type not in {"update", "create", "delete"}:
                raise ValueError(f"unsupported proposal change_type: {change_type}")
            target_symbols = self._normalize_string_list(raw_edit.get("target_symbols", []))
            intent = self._normalize_text(raw_edit.get("intent", ""))
            if not intent:
                intent = summary_text
            edits.append(
                ProposalEditCandidate(
                    path=path,
                    change_type=change_type,
                    target_symbols=target_symbols,
                    intent=intent,
                )
            )
            seen_paths.add(path)

        if allowed_targets and not edits:
            raise ValueError("proposal candidate missing edits inside the allowed scope")
        if not summary_text and edits:
            summary_text = "Prepare bounded edits for " + ", ".join(edit.path for edit in edits)
        if not summary_text:
            raise ValueError("proposal candidate missing summary")
        if validation_command != expected_validation_command:
            raise ValueError("proposal candidate validation command mismatch")
        return ProposalCandidate(
            summary_text=summary_text,
            edits=edits,
            validation_command=validation_command,
        )

    def _offline_clarify_artifact(
        self,
        session: Session,
        handler: TaskHandler | None,
        validation_command: str,
    ) -> ClarifyArtifact:
        relevant_files = list(handler.required_files) if handler else list(self._generic_target_paths(session)[:3])
        return ClarifyArtifact(
            implementation_target=handler.offline_clarify if handler else self._offline_clarify(),
            relevant_files=relevant_files,
            validation_command=validation_command,
        )

    def _offline_plan_artifact(
        self,
        session: Session,
        handler: TaskHandler | None,
        validation_command: str,
    ) -> PlanArtifact:
        target_files = list(handler.required_files) if handler else list(self._generic_target_paths(session)[:3])
        steps = list(handler.offline_plan_steps) if handler else [
            "Inspect the retrieved source and test files.",
            "Prepare focused edits inside the bounded repository scope.",
            "Run the allowlisted regression command.",
            "Summarize the change set and any remaining risks.",
        ]
        return PlanArtifact(steps=steps, target_files=target_files, validation_command=validation_command)

    def _offline_proposal_candidate(
        self,
        session: Session,
        handler: TaskHandler | None,
        validation_command: str,
    ) -> ProposalCandidate:
        if handler is None:
            generic_targets = list(self._generic_target_paths(session))
            edits = [
                ProposalEditCandidate(
                    path=path,
                    change_type="update",
                    target_symbols=[],
                    intent="update this bounded Python file to satisfy the current task while preserving unrelated behavior",
                )
                for path in generic_targets
            ]
            summary_text = self._offline_proposal()
            if generic_targets:
                summary_text = f"Prepare bounded edits for {', '.join(generic_targets)}."
            return ProposalCandidate(
                summary_text=summary_text,
                edits=edits,
                validation_command=validation_command,
            )
        edits = [
            ProposalEditCandidate(
                path=seed.path,
                change_type=seed.change_type,
                target_symbols=list(seed.target_symbols),
                intent=seed.intent,
            )
            for seed in handler.proposal_edit_seeds
        ]
        return ProposalCandidate(
            summary_text=handler.offline_proposal,
            edits=edits,
            validation_command=validation_command,
        )

    def _research_tool(self, session: Session) -> tuple[str, str]:
        query = self._normalize_text(session.research_query or session.request.user_text)
        if not query:
            summary = "No external research query was provided."
            session.research_summary = summary
            self.store.save_summary(session)
            return summary, summary

        try:
            sources = self.research_client.search(query, max_results=self.config.web_research_max_results)
        except RuntimeError as exc:
            session.research_sources = []
            summary = f"External research unavailable: {exc}"
            session.research_summary = summary
            self.store.save_summary(session)
            return summary, summary

        session.research_sources = list(sources)
        summary = self._research_summary_text(query, session.research_sources)
        session.research_summary = summary
        self.store.save_summary(session)
        return summary, summary

    def _research_summary_text(self, query: str, sources: list[ResearchSource]) -> str:
        if not sources:
            return f"External research returned no sources for query '{query}'."
        source_titles = ", ".join(source.title for source in sources[:3])
        return f"External research captured {len(sources)} source(s) for '{query}': {source_titles}."

    def _prompt_research_sources(self, research_sources: list[ResearchSource]) -> list[tuple[str, str, str]]:
        prompt_items: list[tuple[str, str, str]] = []
        for source in research_sources[:4]:
            prompt_items.append((source.title, source.url, self._truncate_for_prompt(source.snippet, max_lines=3, max_chars=280)))
        return prompt_items

    def _generic_target_paths(self, session: Session) -> tuple[str, ...]:
        ordered_paths: list[str] = []
        for candidate_path in (
            *(session.plan_artifact.target_files if session.plan_artifact else []),
            *(session.clarify_artifact.relevant_files if session.clarify_artifact else []),
            *(item.path for item in session.retrieved_files[:3]),
        ):
            normalized = self._normalize_text(candidate_path)
            if not normalized or normalized in ordered_paths:
                continue
            ordered_paths.append(normalized)
        return tuple(ordered_paths[:4])

    def _parse_generated_file_edit(
        self,
        response: str,
        expected_path: str,
        expected_change_type: str,
        default_summary: str,
    ) -> GeneratedFileEdit:
        payload = self._parse_json_payload(response)
        path = self._normalize_text(payload.get("path", ""))
        change_type = self._normalize_text(payload.get("change_type", expected_change_type)).lower() or expected_change_type
        summary = self._normalize_text(payload.get("summary", default_summary)) or default_summary
        if path != expected_path:
            raise ValueError("generated edit path mismatch")
        if change_type != expected_change_type:
            raise ValueError("generated edit change_type mismatch")
        if change_type not in {"update", "create"}:
            raise ValueError("generated edit must be update or create")
        content = payload.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("generated edit missing content")
        if not content.endswith("\n"):
            content += "\n"
        return GeneratedFileEdit(
            path=path,
            change_type=change_type,
            summary=summary,
            content=content,
        )

    def _inspect_repo(self, session: Session) -> str:
        findings = self.context_builder.read_findings(session.context_summaries)
        return f"Read step inspected: {findings}."

    def _inspect_repo_tool(self, session: Session) -> tuple[str, str]:
        summary = self._inspect_repo(session)
        return summary, summary

    def _proposal_tool(
        self,
        session: Session,
        task_text: str,
        repo_ops: RepoOps,
        constraints: list[str],
        handler: TaskHandler | None,
    ) -> tuple[str, str]:
        summary = self._build_proposal(session, task_text, repo_ops, constraints, handler)
        return summary, summary

    def _generate_generic_file_edit(
        self,
        session: Session,
        repo_ops: RepoOps,
        edit: ProposalEditCandidate,
        failure_summary: str | None = None,
    ) -> GeneratedFileEdit | None:
        if edit.change_type not in {"update", "create"}:
            raise ValueError(f"unsupported generic change_type: {edit.change_type}")

        try:
            current_content = repo_ops.read_text(edit.path)
        except OSError:
            current_content = ""

        prompt = build_edit_prompt(
            session.request.user_text,
            session.clarify_summary or "",
            session.plan_summary or "",
            session.proposal_summary or "",
            session.proposal_candidate.validation_command if session.proposal_candidate and session.proposal_candidate.validation_command else DEFAULT_VALIDATION_COMMAND,
            edit.path,
            edit.change_type,
            list(edit.target_symbols),
            edit.intent,
            self._context_summary_text(session.context_summaries, edit.path),
            self._truncate_for_prompt(current_content, max_lines=220, max_chars=12000),
            [(path, snippet) for path, snippet in self._prompt_file_snippets(repo_ops, session.retrieved_files) if path != edit.path],
            failure_summary=failure_summary,
            external_research=self._prompt_research_sources(session.research_sources),
        )
        generated = self._complete_structured_or_fallback(
            prompt,
            f"{SYSTEM_PROMPT}\n{EDIT_PROMPT}",
            fallback=None,
            parser=lambda response: self._parse_generated_file_edit(response, edit.path, edit.change_type, edit.intent or f"update {edit.path}"),
            session=session,
            step_type="edit",
        )
        if isinstance(generated, GeneratedFileEdit):
            return generated
        return None

    def _apply_generic_candidate_changes(
        self,
        session: Session,
        repo_ops: RepoOps,
        empty_message: str,
        success_prefix: str,
        unchanged_message: str,
        invalid_message: str,
        failure_summary: str | None = None,
    ) -> str:
        candidate = session.proposal_candidate
        if candidate is None or not candidate.edits:
            return empty_message

        generated_edits: list[GeneratedFileEdit] = []
        for edit in candidate.edits:
            generated = self._generate_generic_file_edit(session, repo_ops, edit, failure_summary=failure_summary)
            if generated is None:
                return invalid_message
            generated_edits.append(generated)

        changed_paths: list[str] = []
        for generated in generated_edits:
            change = repo_ops.apply_text_change(
                generated.path,
                generated.content or "",
                generated.summary,
            )
            if change is None:
                continue
            session.changed_files.append(change)
            self.store.save_summary(session)
            self._emit(
                session.session_id,
                "file_changed",
                {
                    "path": change.path,
                    "summary": change.summary,
                    "change_type": change.change_type,
                },
            )
            changed_paths.append(change.path)

        if changed_paths:
            return f"{success_prefix}: {', '.join(changed_paths)}."
        return unchanged_message

    def _edit_repo(self, session: Session, repo_ops: RepoOps, handler: TaskHandler | None) -> str:
        if handler is not None:
            return self._apply_handler_changes(
                session,
                repo_ops,
                handler,
                empty_message="No supported automated edit strategy matched the current repo and task.",
                success_prefix="Applied preset task changes",
                unchanged_message="Repository already matches the selected preset task; no file edits were needed.",
            )
        return self._apply_generic_candidate_changes(
            session,
            repo_ops,
            empty_message="No bounded generic edit candidate was available for the current task.",
            success_prefix="Applied model-backed bounded changes",
            unchanged_message="Model-backed bounded edits matched the current repo snapshot; no file changes were needed.",
            invalid_message="Model-backed edit generation did not yield a valid bounded change set.",
        )

    def _edit_repo_tool(
        self,
        session: Session,
        repo_ops: RepoOps,
        handler: TaskHandler | None,
    ) -> tuple[str, str]:
        summary = self._edit_repo(session, repo_ops, handler)
        return summary, summary

    def _fix_repo(
        self,
        session: Session,
        repo_ops: RepoOps,
        handler: TaskHandler | None,
        failure_summary: str,
    ) -> str:
        if handler is not None:
            return self._apply_handler_changes(
                session,
                repo_ops,
                handler,
                empty_message="No supported automated fix strategy matched the current repo and task.",
                success_prefix="Applied one repair pass",
                unchanged_message="No additional fix was needed because the repo already matches the selected preset task.",
            )
        return self._apply_generic_candidate_changes(
            session,
            repo_ops,
            empty_message="No bounded repair candidate was available for the current task.",
            success_prefix="Applied one repair pass",
            unchanged_message="No additional repair changes were needed after the latest test failure.",
            invalid_message="Model-backed repair generation did not yield a valid bounded change set.",
            failure_summary=failure_summary,
        )

    def _fix_repo_tool(
        self,
        session: Session,
        repo_ops: RepoOps,
        handler: TaskHandler | None,
        failure_summary: str,
    ) -> tuple[str, str]:
        summary = self._fix_repo(session, repo_ops, handler, failure_summary)
        return summary, summary

    def _review_tool(self, session: Session, tests_ok: bool, handler: TaskHandler | None) -> tuple[str, str]:
        summary = self._review(session, tests_ok, handler)
        return summary, summary

    def _apply_handler_changes(
        self,
        session: Session,
        repo_ops: RepoOps,
        handler: TaskHandler | None,
        empty_message: str,
        success_prefix: str,
        unchanged_message: str,
    ) -> str:
        if handler is None:
            return empty_message

        changed_paths: list[str] = []
        for planned_change in handler.planned_changes:
            change = repo_ops.apply_text_change(
                planned_change.path,
                planned_change.content,
                planned_change.summary,
            )
            if change is None:
                continue
            session.changed_files.append(change)
            self.store.save_summary(session)
            self._emit(
                session.session_id,
                "file_changed",
                {
                    "path": change.path,
                    "summary": change.summary,
                    "change_type": change.change_type,
                },
            )
            changed_paths.append(change.path)

        if changed_paths:
            return f"{success_prefix}: {', '.join(changed_paths)}."
        return unchanged_message

    def _run_test_command_tool(
        self,
        session: Session,
        repo_ops: RepoOps,
        command: list[str],
    ) -> tuple[CommandResult, str]:
        self._emit(session.session_id, "command_started", {"command": command, "cwd": str(repo_ops.repo_root)})
        result = repo_ops.run_command(command)
        session.test_results.append(result)
        self.store.save_summary(session)
        self._emit(session.session_id, "command_completed", result.to_dict())
        summary = f"Ran {' '.join(command)} with exit={result.exit_code} in {result.duration_ms} ms."
        return result, summary

    def _planned_targets(self, session: Session, handler: TaskHandler | None) -> tuple[str, ...]:
        if handler is not None:
            return tuple(change.path for change in handler.planned_changes)
        return self._generic_target_paths(session)

    def _execute_tool(
        self,
        session: Session,
        step_type: str,
        tool_name: str,
        purpose: str,
        input_summary: str | None,
        handler: TaskHandler | None,
        action: Callable[[], tuple[object, str]],
        requested_targets: tuple[str, ...] = (),
        requested_command: tuple[str, ...] = (),
    ) -> object:
        request = ToolRequest(
            step_type=step_type,
            tool_name=tool_name,
            purpose=purpose,
            input_summary=input_summary,
            requested_targets=requested_targets,
            requested_command=requested_command,
        )
        return self.tool_executor.execute(
            session=session,
            request=request,
            handler=handler,
            action=action,
            emit=lambda kind, payload: self._emit(session.session_id, kind, payload),
            save_summary=self.store.save_summary,
        )

    def _prompt_retrieved_files(self, retrieved_files: list[RetrievedFile]) -> list[tuple[str, str]]:
        summary: list[tuple[str, str]] = []
        for item in retrieved_files:
            reason_text = "; ".join(item.reasons) if item.reasons else "retrieved context"
            summary.append((item.path, reason_text))
        return summary

    def _prompt_context_summaries(self, context_summaries: list[ContextFileSummary]) -> list[tuple[str, str]]:
        return self.context_builder.prompt_summaries(context_summaries)

    def _prompt_file_snippets(self, repo_ops: RepoOps, retrieved_files: list[RetrievedFile]) -> list[tuple[str, str]]:
        if retrieved_files:
            target_files = [item.path for item in retrieved_files[:3]]
        else:
            target_files = list(repo_ops.list_files(limit=2))

        snippets: list[tuple[str, str]] = []
        for path in target_files:
            try:
                content = repo_ops.read_text(path)
            except OSError:
                continue
            snippets.append((path, self._truncate_for_prompt(content)))
        return snippets

    def _context_summary_text(self, context_summaries: list[ContextFileSummary], path: str) -> str:
        for summary in context_summaries:
            if summary.path == path:
                prompt_items = self.context_builder.prompt_summaries([summary])
                if prompt_items:
                    return prompt_items[0][1]
        return ""

    def _truncate_for_prompt(self, content: str, max_lines: int = 80, max_chars: int = 2200) -> str:
        lines = content.splitlines()
        truncated = False
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            truncated = True
        snippet = "\n".join(lines)
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip()
            truncated = True
        if truncated:
            snippet += "\n... snippet truncated ..."
        return snippet

    def _complete_structured_or_fallback(
        self,
        prompt: str,
        system_prompt: str,
        fallback: object,
        parser: Callable[[str], object],
        validator: Callable[[str], bool] | None = None,
        session: Session | None = None,
        step_type: str | None = None,
    ) -> object:
        if not self.llm_client.is_configured():
            self._record_fallback(session, step_type, "client_unconfigured")
            return fallback
        try:
            response = self.llm_client.complete(prompt, system_prompt=system_prompt, temperature=0.1)
        except RuntimeError:
            self._record_fallback(session, step_type, "client_error")
            return fallback
        if validator and not validator(response):
            self._record_fallback(session, step_type, "validator_rejected")
            return fallback
        try:
            return parser(response)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            self._record_fallback(session, step_type, "schema_parse_failed")
            return fallback

    def _record_fallback(self, session: Session | None, step_type: str | None, reason: str) -> None:
        if session is None or step_type is None:
            return
        session.fallback_steps.append(f"{step_type}:{reason}")

    def _offline_clarify(self) -> str:
        return (
            "Interpret the request inside the current Python repo, inspect the most relevant source and test files, "
            "apply focused edits, and validate with the repo test command."
        )

    def _offline_plan(self) -> str:
        return (
            "1) inspect the relevant repo files, 2) prepare focused edits, 3) run the repo tests, "
            "4) summarize the current change set and any remaining risks."
        )

    def _offline_proposal(self) -> str:
        return (
            "- Files: no supported preset edit scope matched the current task.\n"
            "- Changes: inspect the retrieved repo files and define a bounded write set before editing.\n"
            f"- Validation: run {DEFAULT_VALIDATION_COMMAND}."
        )

    def _failure_excerpt(self, result: object) -> str:
        stderr = getattr(result, "stderr", "")
        stdout = getattr(result, "stdout", "")
        lines: list[str] = []
        for stream in (stderr, stdout):
            for line in str(stream).splitlines():
                line = line.strip()
                if line:
                    lines.append(line)
                if len(lines) == 3:
                    break
            if len(lines) == 3:
                break
        if not lines:
            return "No stdout/stderr captured."
        return "Excerpt: " + " | ".join(lines)
    def _review(self, session: Session, tests_ok: bool, handler: TaskHandler | None) -> str:
        base_summary = self._review_base_summary(session, tests_ok, handler)
        evidence_parts: list[str] = []

        proposal_assessment_summary = self._proposal_assessment_review_summary(session)
        if proposal_assessment_summary:
            evidence_parts.append(proposal_assessment_summary)

        proposal_summary = self._proposal_review_summary(session)
        if proposal_summary:
            evidence_parts.append(proposal_summary)

        research_summary = self._research_review_summary(session)
        if research_summary:
            evidence_parts.append(research_summary)

        retrieval_summary = self._retrieval_review_summary(session)
        if retrieval_summary:
            evidence_parts.append(retrieval_summary)

        read_evidence = self._read_evidence_summary(session)
        if read_evidence:
            evidence_parts.append(read_evidence)

        changed_files_summary = self._changed_files_review_summary(session)
        if changed_files_summary:
            evidence_parts.append(changed_files_summary)

        validation_summary = self._validation_review_summary(session)
        if validation_summary:
            evidence_parts.append(validation_summary)

        audit_summary = self._audit_review_summary(session)
        if audit_summary:
            evidence_parts.append(audit_summary)

        fallback_summary = self._fallback_review_summary(session)
        if fallback_summary:
            evidence_parts.append(fallback_summary)

        if handler and handler.review_risk_note:
            evidence_parts.append(f"Residual risk: {handler.review_risk_note}")

        return " ".join([base_summary, *evidence_parts]).strip()

    def _review_base_summary(self, session: Session, tests_ok: bool, handler: TaskHandler | None) -> str:
        if handler and tests_ok:
            if session.changed_files:
                return handler.review_if_changed
            return handler.review_if_unchanged

        changed_paths = ", ".join(change.path for change in session.changed_files)
        if tests_ok and changed_paths:
            return f"Current change set updates {changed_paths}; the latest regression run passed."
        if tests_ok:
            return "No file changes were needed; the latest regression run still passed."
        if changed_paths:
            return f"Tests still fail after the allowed auto-fix attempts. Current change set touches {changed_paths}."
        return "Tests failed before any automated edit or repair resolved them."

    def _proposal_assessment_review_summary(self, session: Session) -> str:
        assessment = session.proposal_assessment
        if assessment is None:
            return ""
        source = "fallback" if assessment.used_fallback else "model"
        matched_text = "none"
        if assessment.matched_targets:
            matched_text = ", ".join(assessment.matched_targets)
        extra_text = ""
        if assessment.extra_targets:
            extra_text = f" extra={', '.join(assessment.extra_targets)}"
        note_text = f"; note={assessment.note}" if assessment.note else ""
        return (
            f"Proposal assessment: {assessment.status} score={assessment.score} source={source} "
            f"matched={matched_text}{extra_text}{note_text}."
        )

    def _proposal_review_summary(self, session: Session) -> str:
        candidate = session.proposal_candidate
        if candidate is None:
            if not session.proposal_summary:
                return ""
            return "Proposal candidate: " + " ".join(session.proposal_summary.strip().split())
        return "Proposal candidate: " + " ".join(candidate.summary().strip().split())

    def _research_review_summary(self, session: Session) -> str:
        if not session.research_enabled:
            return ""
        if not session.research_sources:
            if session.research_summary:
                return f"External research: {session.research_summary}"
            return "External research: enabled but no external sources were captured."
        parts = [f"{source.title} ({source.url})" for source in session.research_sources[:3]]
        if session.research_summary and session.research_summary.startswith("External research returned no sources"):
            return f"External research: {session.research_summary}"
        return "External research: used " + ", ".join(parts) + "."

    def _retrieval_review_summary(self, session: Session) -> str:
        if not session.retrieved_files:
            return "Grounding: no repo-aware retrieval context was selected."
        parts: list[str] = []
        for item in session.retrieved_files[:3]:
            reason_text = "; ".join(item.reasons[:2]) if item.reasons else "retrieved context"
            parts.append(f"{item.path} ({reason_text})")
        return "Grounding: retrieved " + ", ".join(parts) + "."

    def _read_evidence_summary(self, session: Session) -> str:
        if not session.read_summary:
            return ""
        if session.read_summary.startswith("Read step inspected: "):
            return "Read evidence: " + session.read_summary.removeprefix("Read step inspected: ")
        return "Read evidence: " + session.read_summary

    def _changed_files_review_summary(self, session: Session) -> str:
        if not session.changed_files:
            return "Evidence: no file edits were needed on the current repo snapshot."
        changed_paths = ", ".join(change.path for change in session.changed_files)
        return f"Evidence: changed {changed_paths}."

    def _validation_review_summary(self, session: Session) -> str:
        if not session.test_results:
            return ""
        latest_result = session.test_results[-1]
        command_text = " ".join(latest_result.command)
        status_text = "passed" if latest_result.exit_code == 0 else f"failed with exit code {latest_result.exit_code}"
        return f"Validation: {command_text} {status_text} in {latest_result.duration_ms} ms."

    def _audit_review_summary(self, session: Session) -> str:
        if not session.tool_calls:
            return ""
        tool_names = [call.tool_name for call in session.tool_calls]
        ordered_unique_tools = list(dict.fromkeys(tool_names))
        denied_count = sum(1 for item in session.approval_checks if not item.approved)
        base = f"Audit: workflow used {', '.join(ordered_unique_tools)}."
        if denied_count:
            return f"{base} Approval denials={denied_count}."
        return f"{base} All recorded tool invocations passed the scoped approval policy."

    def _fallback_review_summary(self, session: Session) -> str:
        if not session.fallback_steps:
            return "LLM note: clarify/plan/proposal used model output without recorded fallback."

        step_names = sorted({item.split(":", maxsplit=1)[0] for item in session.fallback_steps})
        reasons = [item.split(":", maxsplit=1)[1] for item in session.fallback_steps if ":" in item]
        reason_counts = Counter(reasons)
        reason_texts = [
            f"{self._fallback_reason_label(reason)} x{count}" if count > 1 else self._fallback_reason_label(reason)
            for reason, count in sorted(reason_counts.items())
        ]
        step_text = "/".join(step_names)
        return f"LLM note: {step_text} used deterministic fallback because {', '.join(reason_texts)}."

    def _assess_proposal(self, session: Session, handler: TaskHandler | None) -> ProposalAssessment:
        candidate = session.proposal_candidate
        if candidate is None:
            return ProposalAssessment(
                status="unavailable",
                score=0,
                has_validation_command=False,
                used_fallback=self._proposal_fallback_used(session),
                note="no proposal candidate was recorded",
            )

        changed_targets = sorted({change.path for change in session.changed_files})
        candidate_targets: list[str] = []
        for edit in candidate.edits:
            if edit.path not in candidate_targets:
                candidate_targets.append(edit.path)
        expected_validation_command = handler.validation_command if handler else DEFAULT_VALIDATION_COMMAND
        has_validation_command = candidate.validation_command == expected_validation_command
        used_fallback = self._proposal_fallback_used(session)

        if not changed_targets:
            score = 70 if has_validation_command else 50
            note = "repository already matched the preset task before editing" if handler else "no generic file edits were applied on the current repo snapshot"
            return ProposalAssessment(
                status="not_needed",
                score=score,
                matched_targets=[],
                missing_targets=[],
                extra_targets=list(candidate_targets),
                has_validation_command=has_validation_command,
                used_fallback=used_fallback,
                note=note,
            )

        matched_targets = [path for path in changed_targets if path in candidate_targets]
        missing_targets = [path for path in changed_targets if path not in candidate_targets]
        extra_targets = [path for path in candidate_targets if path not in changed_targets]
        file_ratio = len(matched_targets) / len(changed_targets)
        precision_penalty = min(len(extra_targets) * 10, 20)
        score = int(round((70 * file_ratio) + (20 if has_validation_command else 0) + (10 if not used_fallback else 0) - precision_penalty))
        score = max(0, min(100, score))

        if file_ratio == 1.0 and has_validation_command and not extra_targets:
            status = "accepted"
            note = "proposal candidate covered all executed targets and the validation command"
        elif file_ratio > 0 or has_validation_command:
            status = "partial"
            note = "proposal candidate only partially aligned with the executed change set"
        else:
            status = "rejected"
            note = "proposal candidate missed the executed targets and validation command"

        if extra_targets:
            note += f"; extra candidate targets: {', '.join(extra_targets)}"

        return ProposalAssessment(
            status=status,
            score=score,
            matched_targets=matched_targets,
            missing_targets=missing_targets,
            extra_targets=extra_targets,
            has_validation_command=has_validation_command,
            used_fallback=used_fallback,
            note=note,
        )

    def _proposal_fallback_used(self, session: Session) -> bool:
        return any(item.startswith("proposal:") for item in session.fallback_steps)

    def _fallback_reason_label(self, reason: str) -> str:
        labels = {
            "client_unconfigured": "the client was not configured",
            "client_error": "the client call failed",
            "validator_rejected": "the grounding validator rejected the model output",
            "schema_parse_failed": "the model output did not satisfy the structured schema",
        }
        return labels.get(reason, reason)

    def _final_summary(self, session: Session, tests_ok: bool) -> str:
        if tests_ok:
            if session.changed_files:
                if session.task_handler:
                    return (
                        f"Run completed successfully for preset task '{session.task_handler}'. "
                        "The edit loop applied real file changes and tests passed."
                    )
                return "Run completed successfully. The edit loop applied real file changes and tests passed."
            return "Run completed successfully without needing file changes; tests passed."
        return "Run completed with failing tests after the allowed auto-fix attempts."



























