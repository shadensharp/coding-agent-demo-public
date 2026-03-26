from __future__ import annotations

from pathlib import Path
import sys
from typing import TextIO

from .events import Event



def _short_time(timestamp: str) -> str:
    if "T" in timestamp:
        return timestamp.split("T", maxsplit=1)[1][:8]
    return timestamp[:8]



def _one_line(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split())



def _command_text(command: object) -> str:
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return ""



def _diff_preview(diff_excerpt: str | None, limit: int = 3) -> list[str]:
    if not diff_excerpt:
        return []
    preview: list[str] = []
    for line in diff_excerpt.splitlines():
        stripped = line.strip()
        if not stripped or stripped in {"+", "-"}:
            continue
        if line.startswith(("---", "+++", "@@")):
            continue
        preview.append(line)
        if len(preview) >= limit:
            break
    return preview



def _list_text(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    return ", ".join(str(item) for item in value)



def _repo_name(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return Path(text).name or text



def _compact_review_text(value: object) -> str:
    text = _one_line(str(value or ""))
    if not text:
        return ""

    markers = [
        "Proposal assessment:",
        "Grounding:",
        "Read evidence:",
        "Evidence:",
        "Validation:",
        "Audit:",
        "LLM note:",
        "Residual risk:",
    ]
    end = len(text)
    for marker in markers:
        index = text.find(marker)
        if index != -1:
            end = min(end, index)
    if end != len(text):
        return text[:end].rstrip(" .") + "."
    return text


class TerminalRenderer:
    def __init__(
        self,
        stream: TextIO | None = None,
        event_verbosity: str = "verbose",
        summary_verbosity: str = "verbose",
    ) -> None:
        self.stream = stream or sys.stdout
        self.event_verbosity = event_verbosity
        self.summary_verbosity = summary_verbosity

    def write(self, text: str = "") -> None:
        print(text, file=self.stream)

    def render_event(self, event: Event | dict[str, object]) -> None:
        payload_event = event.to_dict() if isinstance(event, Event) else event
        if self.event_verbosity == "compact":
            self._render_event_compact(payload_event)
            return

        timestamp = _short_time(str(payload_event.get("timestamp", "")))
        kind = str(payload_event.get("kind", "unknown"))
        payload = payload_event.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        if kind == "session_started":
            handler = payload.get("task_handler", "")
            handler_suffix = f" | handler={handler}" if handler else ""
            self.write(
                f"[{timestamp}] session_started: {payload.get('task', '')} | repo={payload.get('repo', '')}{handler_suffix}"
            )
            return
        if kind == "context_selected":
            files = payload.get("files", [])
            if isinstance(files, list) and files:
                paths = [str(item.get("path", "")) for item in files if isinstance(item, dict)]
                self.write(f"[{timestamp}] context_selected: {', '.join(path for path in paths if path)}")
                return
            self.write(f"[{timestamp}] context_selected: none")
            return
        if kind == "step_started":
            self.write(f"[{timestamp}] step_started: {payload.get('step_type', '')}")
            return
        if kind == "step_completed":
            summary = _one_line(str(payload.get("output_summary", "")))
            self.write(f"[{timestamp}] step_completed: {payload.get('step_type', '')} | {summary}")
            return
        if kind == "step_failed":
            self.write(
                f"[{timestamp}] step_failed: {payload.get('step_type', '')} | {payload.get('error_message', '')}"
            )
            return
        if kind == "approval_checked":
            self.write(
                f"[{timestamp}] approval_checked: {payload.get('step_type', '')} -> {payload.get('tool_name', '')} | "
                f"approved={payload.get('approved', '')} mode={payload.get('mode', '')}"
            )
            return
        if kind == "proposal_assessed":
            self.write(
                f"[{timestamp}] proposal_assessed: status={payload.get('status', '')} score={payload.get('score', '')} "
                f"| fallback={payload.get('used_fallback', '')}"
            )
            return
        if kind == "tool_started":
            self.write(
                f"[{timestamp}] tool_started: {payload.get('step_type', '')} -> {payload.get('tool_name', '')} "
                f"| approval={payload.get('approval_mode', '')}"
            )
            return
        if kind == "tool_completed":
            summary = _one_line(str(payload.get("output_summary", "")))
            self.write(
                f"[{timestamp}] tool_completed: {payload.get('step_type', '')} -> {payload.get('tool_name', '')} | {summary}"
            )
            return
        if kind == "tool_failed":
            self.write(
                f"[{timestamp}] tool_failed: {payload.get('step_type', '')} -> {payload.get('tool_name', '')} | {payload.get('error_message', '')}"
            )
            return
        if kind == "command_started":
            self.write(f"[{timestamp}] command_started: {_command_text(payload.get('command', []))}")
            return
        if kind == "command_completed":
            self.write(
                f"[{timestamp}] command_completed: exit={payload.get('exit_code', '')} duration_ms={payload.get('duration_ms', '')}"
            )
            return
        if kind == "file_changed":
            self.write(
                f"[{timestamp}] file_changed: {payload.get('path', '')} | {payload.get('summary', '')}"
            )
            return
        if kind == "session_completed":
            self.write(
                f"[{timestamp}] session_completed: status={payload.get('status', '')} | {payload.get('final_summary', '')}"
            )
            return

        self.write(f"[{timestamp}] {kind}: {payload}")

    def _render_event_compact(self, payload_event: dict[str, object]) -> None:
        timestamp = _short_time(str(payload_event.get("timestamp", "")))
        kind = str(payload_event.get("kind", "unknown"))
        payload = payload_event.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        if kind == "session_started":
            handler = str(payload.get("task_handler", "") or "generic")
            self.write(
                f"[{timestamp}] run started | handler={handler} | repo={_repo_name(payload.get('repo', ''))}"
            )
            return
        if kind == "context_selected":
            files = payload.get("files", [])
            if isinstance(files, list) and files:
                paths = [str(item.get("path", "")) for item in files if isinstance(item, dict) and item.get("path")]
                self.write(f"[{timestamp}] grounding selected | {', '.join(paths)}")
            else:
                self.write(f"[{timestamp}] grounding selected | none")
            return
        if kind == "step_started":
            step_type = str(payload.get("step_type", "") or "step")
            if step_type != "test":
                self.write(f"[{timestamp}] {step_type}...")
            return
        if kind == "step_completed":
            step_type = str(payload.get("step_type", "") or "step")
            if step_type == "test":
                return
            labels = {
                "clarify": "clarify ready",
                "plan": "plan ready",
                "read": "read ready",
                "proposal": "proposal ready",
                "edit": "edit applied",
                "fix": "fix applied",
                "review": "review ready",
            }
            self.write(f"[{timestamp}] {labels.get(step_type, step_type + ' done')}")
            return
        if kind == "step_failed":
            self.write(f"[{timestamp}] {payload.get('step_type', '')} failed | {payload.get('error_message', '')}")
            return
        if kind == "command_started":
            command = _command_text(payload.get("command", []))
            if command:
                self.write(f"[{timestamp}] validation running | {command}")
            return
        if kind == "command_completed":
            exit_code = str(payload.get("exit_code", ""))
            duration_ms = payload.get("duration_ms", "")
            status = "passed" if exit_code == "0" else f"failed (exit={exit_code})"
            self.write(f"[{timestamp}] validation {status} | {duration_ms} ms")
            return
        if kind == "session_completed":
            self.write(f"[{timestamp}] run completed | status={payload.get('status', '')}")
            return

    def render_run_summary(self, summary: dict[str, object]) -> None:
        request = summary.get("request", {})
        task = request.get("user_text", "") if isinstance(request, dict) else ""

        self.write("Run Result")
        self.write(f"Session: {summary.get('session_id', '')}")
        self.write(f"Status: {summary.get('status', '')}")

        task_handler = _one_line(str(summary.get("task_handler", "")))
        if task_handler:
            self.write(f"Handler: {task_handler}")
        if task:
            self.write(f"Task: {task}")

        retrieved_files = summary.get("retrieved_files", [])
        if isinstance(retrieved_files, list) and retrieved_files:
            retrieved_paths = [
                str(item.get("path", ""))
                for item in retrieved_files
                if isinstance(item, dict) and item.get("path")
            ]
            self.write("Grounding: " + ", ".join(retrieved_paths))

        changed_files = summary.get("changed_files", [])
        self.write("Changed files:")
        if isinstance(changed_files, list) and changed_files:
            for change in changed_files:
                if not isinstance(change, dict):
                    continue
                self.write(f"  - {change.get('path', '')}: {change.get('summary', '')}")
        else:
            self.write("  - none")

        test_results = summary.get("test_results", [])
        if isinstance(test_results, list) and test_results:
            latest = test_results[-1]
            if isinstance(latest, dict):
                command_text = _command_text(latest.get("command", []))
                if not command_text:
                    command_text = str(latest.get("command", ""))
                validation_status = "passed" if latest.get("exit_code", 1) == 0 else f"failed (exit={latest.get('exit_code', '')})"
                self.write(
                    "Validation: "
                    f"{validation_status} | {command_text} | {latest.get('duration_ms', '')} ms"
                )

        review_text = _compact_review_text(summary.get("review_summary", ""))
        if not review_text:
            review_text = _one_line(str(summary.get("final_summary", "")))
        if review_text:
            self.write(f"Review: {review_text}")

        fallback_steps = summary.get("fallback_steps", [])
        if isinstance(fallback_steps, list) and fallback_steps:
            self.write("Model note: fallback used in " + ", ".join(str(item) for item in fallback_steps))
        else:
            self.write("Model note: model-backed clarify/plan/proposal without recorded fallback.")

        session_id = str(summary.get("session_id", "") or "")
        if session_id:
            self.write("Next commands:")
            self.write(f"  - python -m coding_agent show {session_id}")
            self.write(f"  - python -m coding_agent replay {session_id}")
            self.write("  - python -m coding_agent report --session-limit 5 --open")

    def render_session_summary(self, summary: dict[str, object]) -> None:
        request = summary.get("request", {})
        task = request.get("user_text", "") if isinstance(request, dict) else ""
        self.write(f"Session: {summary.get('session_id', '')}")
        self.write(f"Status: {summary.get('status', '')}")
        self.write(f"Task: {task}")
        self.write(f"Created: {summary.get('created_at', '')}")

        task_handler = _one_line(str(summary.get("task_handler", "")))
        if task_handler:
            self.write(f"Task handler: {task_handler}")

        approval_checks = summary.get("approval_checks", [])
        if isinstance(approval_checks, list) and approval_checks:
            denied = sum(1 for item in approval_checks if isinstance(item, dict) and not bool(item.get("approved", False)))
            self.write(f"Approvals: {len(approval_checks)} checks | denied={denied}")

        tool_calls = summary.get("tool_calls", [])
        if isinstance(tool_calls, list) and tool_calls:
            workflow_parts: list[str] = []
            self.write(f"Tool calls: {len(tool_calls)}")
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                step_type = str(call.get("step_type", ""))
                tool_name = str(call.get("tool_name", ""))
                status = str(call.get("status", ""))
                approval_mode = str(call.get("approval_mode", ""))
                workflow_parts.append(f"{step_type}->{tool_name}[{status},{approval_mode}]")
                self.write(f"  - {step_type} -> {tool_name}: {status} | approval={approval_mode}")
            if workflow_parts:
                self.write("Workflow: " + " -> ".join(workflow_parts))

        retrieved_files = summary.get("retrieved_files", [])
        if isinstance(retrieved_files, list) and retrieved_files:
            self.write(f"Retrieved files: {len(retrieved_files)}")
            for item in retrieved_files:
                if not isinstance(item, dict):
                    continue
                reasons = item.get("reasons", [])
                reason_text = ""
                if isinstance(reasons, list) and reasons:
                    reason_text = " | reasons=" + "; ".join(str(reason) for reason in reasons)
                self.write(f"  - {item.get('path', '')}: score={item.get('score', '')}{reason_text}")

        clarify_artifact = summary.get("clarify_artifact")
        if isinstance(clarify_artifact, dict) and clarify_artifact:
            self.write(
                "Clarify artifact: "
                f"files={_list_text(clarify_artifact.get('relevant_files', []))} "
                f"validation={clarify_artifact.get('validation_command', '')}"
            )

        plan_artifact = summary.get("plan_artifact")
        if isinstance(plan_artifact, dict) and plan_artifact:
            steps = plan_artifact.get("steps", [])
            step_count = len(steps) if isinstance(steps, list) else 0
            self.write(
                "Plan artifact: "
                f"steps={step_count} targets={_list_text(plan_artifact.get('target_files', []))} "
                f"validation={plan_artifact.get('validation_command', '')}"
            )

        read_summary = _one_line(str(summary.get("read_summary", "")))
        if read_summary:
            self.write(f"Read: {read_summary}")

        proposal_summary = _one_line(str(summary.get("proposal_summary", "")))
        if proposal_summary:
            self.write(f"Proposal summary: {proposal_summary}")

        proposal_candidate = summary.get("proposal_candidate")
        if isinstance(proposal_candidate, dict) and proposal_candidate:
            edits = proposal_candidate.get("edits", [])
            edit_count = len(edits) if isinstance(edits, list) else 0
            self.write(
                "Proposal candidate: "
                f"edits={edit_count} validation={proposal_candidate.get('validation_command', '')}"
            )
            if isinstance(edits, list):
                for edit in edits:
                    if not isinstance(edit, dict):
                        continue
                    self.write(
                        f"  - {edit.get('change_type', '')} {edit.get('path', '')}: "
                        f"symbols={_list_text(edit.get('target_symbols', []))} | intent={edit.get('intent', '')}"
                    )

        proposal_assessment = summary.get("proposal_assessment")
        if isinstance(proposal_assessment, dict) and proposal_assessment:
            self.write(
                "Proposal assessment: "
                f"status={proposal_assessment.get('status', '')} "
                f"score={proposal_assessment.get('score', '')} "
                f"fallback={proposal_assessment.get('used_fallback', '')} "
                f"matched={_list_text(proposal_assessment.get('matched_targets', []))} "
                f"extra={_list_text(proposal_assessment.get('extra_targets', []))}"
            )

        clarify_summary = _one_line(str(summary.get("clarify_summary", "")))
        if clarify_summary:
            self.write(f"Clarify: {clarify_summary}")

        plan_summary = _one_line(str(summary.get("plan_summary", "")))
        if plan_summary:
            self.write(f"Plan: {plan_summary}")

        final_summary = _one_line(str(summary.get("final_summary", "")))
        if final_summary:
            self.write(f"Final: {final_summary}")

        review_summary = _one_line(str(summary.get("review_summary", "")))
        if review_summary:
            self.write(f"Review: {review_summary}")

        fallback_steps = summary.get("fallback_steps", [])
        if isinstance(fallback_steps, list) and fallback_steps:
            self.write("Fallbacks: " + ", ".join(str(item) for item in fallback_steps))

        changed_files = summary.get("changed_files", [])
        if isinstance(changed_files, list):
            self.write(f"Changed files: {len(changed_files)}")
            for change in changed_files:
                if not isinstance(change, dict):
                    continue
                self.write(f"  - {change.get('path', '')}: {change.get('summary', '')}")
                for preview_line in _diff_preview(change.get("diff_excerpt", None)):
                    self.write(f"      {preview_line}")

        test_results = summary.get("test_results", [])
        if isinstance(test_results, list) and test_results:
            latest = test_results[-1]
            if isinstance(latest, dict):
                self.write(
                    "Latest test: "
                    f"exit={latest.get('exit_code', '')} "
                    f"duration_ms={latest.get('duration_ms', '')} "
                    f"| command={_command_text(latest.get('command', []))}"
                )
                self.write(f"Latest test cwd: {latest.get('cwd', '')}")

    def render_session_list(self, summaries: list[dict[str, object]]) -> None:
        if not summaries:
            self.write("No sessions found.")
            return

        for summary in summaries:
            task_title = _one_line(str(summary.get("task_title", "")))
            self.write(
                f"{summary.get('session_id', '')} | {summary.get('status', '')} | {summary.get('created_at', '')} | {task_title}"
            )

    def render_eval_summary(self, summary: dict[str, object]) -> None:
        self.write(f"Eval mode: {summary.get('mode', '')}")
        self.write(
            "Cases: "
            f"total={summary.get('total_cases', 0)} "
            f"passed={summary.get('passed_cases', 0)} "
            f"failed={summary.get('failed_cases', 0)} "
            f"pass_rate={summary.get('pass_rate', 0.0)}% "
            f"fallback_cases={summary.get('fallback_cases', 0)}"
        )
        self.write(
            "Quality: "
            f"avg_retrieved_files={summary.get('avg_retrieved_files', 0.0)} "
            f"required_file_hit_rate={summary.get('required_file_hit_rate', 0.0)}% "
            f"avg_changed_files={summary.get('avg_changed_files', 0.0)} "
            f"avg_latest_test_duration_ms={summary.get('avg_latest_test_duration_ms', 0)} "
            f"avg_fallback_steps={summary.get('avg_fallback_steps_per_case', 0.0)}"
        )
        self.write(
            "Workflow: "
            f"avg_tool_calls={summary.get('avg_tool_calls', 0.0)} "
            f"avg_approval_checks={summary.get('avg_approval_checks', 0.0)} "
            f"approval_denials={summary.get('approval_denials', 0)}"
        )
        self.write(
            "Proposal: "
            f"accept_rate={summary.get('proposal_accept_rate', 0.0)}% "
            f"avg_score={summary.get('avg_proposal_score', 0.0)} "
            f"avg_edits={summary.get('avg_proposal_edits', 0.0)}"
        )

        fallback_step_counts = summary.get("fallback_step_counts", {})
        if isinstance(fallback_step_counts, dict) and fallback_step_counts:
            parts = [f"{name}={count}" for name, count in fallback_step_counts.items()]
            self.write("Fallback steps: " + ", ".join(parts))

        results = summary.get("results", [])
        if not isinstance(results, list):
            return
        for result in results:
            if not isinstance(result, dict):
                continue
            fallback_steps = result.get("fallback_steps", [])
            fallback_text = "none"
            if isinstance(fallback_steps, list) and fallback_steps:
                fallback_text = ", ".join(str(item) for item in fallback_steps)
            review_excerpt = _one_line(str(result.get("review_summary", "")))
            self.write(
                f"  - {result.get('handler_name', '')}: status={result.get('status', '')} "
                f"retrieved={result.get('retrieved_files', 0)} "
                f"required_hits={result.get('required_file_hits', 0)}/{result.get('required_file_total', 0)} "
                f"changed_files={result.get('changed_files', 0)} duration_ms={result.get('latest_test_duration_ms', '')} "
                f"latest_test_exit={result.get('latest_test_exit', '')} fallbacks={fallback_text} "
                f"proposal={result.get('proposal_status', '')}:{result.get('proposal_score', '')} "
                f"proposal_edits={result.get('proposal_edit_candidates', 0)} "
                f"tools={result.get('tool_calls', 0)} approvals={result.get('approval_checks', 0)} denied={result.get('denied_tool_calls', 0)}"
            )
            if review_excerpt:
                self.write(f"      review={review_excerpt}")
