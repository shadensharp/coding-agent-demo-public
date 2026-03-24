from __future__ import annotations

from collections import Counter
from html import escape
import json
from pathlib import Path

from .config import AppConfig
from .evals import run_preset_eval
from .events import utc_now
from .storage import RunStore


def _one_line(text: object) -> str:
    return " ".join(str(text).strip().split())


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    return []


def _html_text(value: object) -> str:
    return escape(_one_line(value))


def _markdown_cell(value: object) -> str:
    return _one_line(value).replace("|", "\\|")


def _resolve_output_path(base_dir: Path, raw_path: Path | None, default_relative: str) -> Path:
    candidate = raw_path if raw_path is not None else Path(default_relative)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _status_tone(value: object) -> str:
    normalized = _one_line(value).casefold()
    if normalized in {"completed", "accepted"}:
        return "good"
    if normalized in {"partial", "unavailable"}:
        return "warn"
    if normalized in {"failed", "rejected"}:
        return "bad"
    return "neutral"


def _session_row(summary: dict[str, object]) -> dict[str, object]:
    request = summary.get("request", {})
    request_payload = request if isinstance(request, dict) else {}
    fallback_steps = [str(item) for item in _safe_list(summary.get("fallback_steps", []))]
    tool_calls = _safe_list(summary.get("tool_calls", []))
    approval_checks = _safe_list(summary.get("approval_checks", []))
    denied_approvals = sum(
        1 for item in approval_checks if isinstance(item, dict) and not bool(item.get("approved", False))
    )
    test_results = _safe_list(summary.get("test_results", []))
    latest_test = test_results[-1] if test_results and isinstance(test_results[-1], dict) else {}
    proposal_assessment = summary.get("proposal_assessment", {})
    proposal_payload = proposal_assessment if isinstance(proposal_assessment, dict) else {}

    return {
        "session_id": str(summary.get("session_id", "")),
        "session_name": _one_line(summary.get("session_name", "")),
        "created_at": str(summary.get("created_at", "")),
        "status": str(summary.get("status", "")),
        "task_handler": str(summary.get("task_handler", "")),
        "task_text": _one_line(request_payload.get("user_text", "")),
        "fallback_steps": fallback_steps,
        "tool_calls": len(tool_calls),
        "approval_checks": len(approval_checks),
        "approval_denials": denied_approvals,
        "retrieved_files": len(_safe_list(summary.get("retrieved_files", []))),
        "changed_files": len(_safe_list(summary.get("changed_files", []))),
        "latest_test_duration_ms": _safe_int(latest_test.get("duration_ms", 0)),
        "proposal_status": str(proposal_payload.get("status", "unavailable")),
        "proposal_score": _safe_int(proposal_payload.get("score", 0)),
        "proposal_used_fallback": bool(proposal_payload.get("used_fallback", False)),
    }


def _aggregate_session_rows(session_rows: list[dict[str, object]]) -> dict[str, object]:
    total_sessions = len(session_rows)
    completed_sessions = sum(1 for row in session_rows if row.get("status") == "completed")
    failed_sessions = total_sessions - completed_sessions
    fallback_sessions = sum(1 for row in session_rows if row.get("fallback_steps"))
    model_backed_sessions = sum(
        1
        for row in session_rows
        if row.get("status") == "completed"
        and not row.get("fallback_steps")
        and not bool(row.get("proposal_used_fallback", False))
    )
    approval_denials = sum(_safe_int(row.get("approval_denials", 0)) for row in session_rows)
    total_tool_calls = sum(_safe_int(row.get("tool_calls", 0)) for row in session_rows)
    total_approval_checks = sum(_safe_int(row.get("approval_checks", 0)) for row in session_rows)
    total_retrieved_files = sum(_safe_int(row.get("retrieved_files", 0)) for row in session_rows)
    total_changed_files = sum(_safe_int(row.get("changed_files", 0)) for row in session_rows)
    total_proposal_score = sum(_safe_int(row.get("proposal_score", 0)) for row in session_rows)
    accepted_sessions = sum(1 for row in session_rows if row.get("proposal_status") == "accepted")
    durations = [_safe_int(row.get("latest_test_duration_ms", 0)) for row in session_rows if row.get("latest_test_duration_ms")]
    fallback_step_counts = Counter(
        step for row in session_rows for step in _safe_list(row.get("fallback_steps", [])) if str(step)
    )

    return {
        "total_sessions": total_sessions,
        "completed_sessions": completed_sessions,
        "failed_sessions": failed_sessions,
        "fallback_sessions": fallback_sessions,
        "model_backed_sessions": model_backed_sessions,
        "approval_denials": approval_denials,
        "avg_tool_calls": round(total_tool_calls / total_sessions, 2) if total_sessions else 0.0,
        "avg_approval_checks": round(total_approval_checks / total_sessions, 2) if total_sessions else 0.0,
        "avg_retrieved_files": round(total_retrieved_files / total_sessions, 2) if total_sessions else 0.0,
        "avg_changed_files": round(total_changed_files / total_sessions, 2) if total_sessions else 0.0,
        "avg_latest_test_duration_ms": int(sum(durations) / len(durations)) if durations else 0,
        "proposal_accept_rate": round((accepted_sessions / total_sessions) * 100, 1) if total_sessions else 0.0,
        "avg_proposal_score": round(total_proposal_score / total_sessions, 1) if total_sessions else 0.0,
        "fallback_step_counts": dict(sorted(fallback_step_counts.items())),
    }


def build_dashboard_report(
    config: AppConfig,
    store: RunStore,
    session_limit: int = 10,
    enable_live_llm_eval: bool = False,
) -> dict[str, object]:
    recent_summaries = store.list_sessions(limit=session_limit)
    recent_sessions = [_session_row(summary) for summary in recent_summaries]
    latest_session = recent_sessions[0] if recent_sessions else None
    latest_model_session = next(
        (
            row
            for row in recent_sessions
            if row.get("status") == "completed"
            and not row.get("fallback_steps")
            and not bool(row.get("proposal_used_fallback", False))
        ),
        None,
    )

    return {
        "generated_at": utc_now(),
        "session_limit": session_limit,
        "sessions_dir": str(config.sessions_dir),
        "session_metrics": _aggregate_session_rows(recent_sessions),
        "recent_sessions": recent_sessions,
        "latest_session": latest_session,
        "latest_model_session": latest_model_session,
        "eval_summary": run_preset_eval(config, enable_live_llm=enable_live_llm_eval),
    }


def render_dashboard_markdown(report: dict[str, object]) -> str:
    session_metrics = report.get("session_metrics", {})
    session_payload = session_metrics if isinstance(session_metrics, dict) else {}
    eval_summary = report.get("eval_summary", {})
    eval_payload = eval_summary if isinstance(eval_summary, dict) else {}
    recent_sessions = report.get("recent_sessions", [])
    recent_rows = recent_sessions if isinstance(recent_sessions, list) else []
    latest_model_session = report.get("latest_model_session", None)
    latest_model_payload = latest_model_session if isinstance(latest_model_session, dict) else None
    latest_session = report.get("latest_session", None)
    latest_session_payload = latest_session if isinstance(latest_session, dict) else None

    lines = [
        "# Coding Agent Demo Dashboard",
        "",
        f"Generated: {report.get('generated_at', '')}",
        f"Sessions dir: {report.get('sessions_dir', '')}",
        f"Session window: latest {report.get('session_limit', 0)} persisted sessions",
        "",
        "## Recent Sessions",
        "",
        f"- total_sessions: {session_payload.get('total_sessions', 0)}",
        f"- completed_sessions: {session_payload.get('completed_sessions', 0)}",
        f"- failed_sessions: {session_payload.get('failed_sessions', 0)}",
        f"- fallback_sessions: {session_payload.get('fallback_sessions', 0)}",
        f"- model_backed_sessions: {session_payload.get('model_backed_sessions', 0)}",
        f"- avg_retrieved_files: {session_payload.get('avg_retrieved_files', 0.0)}",
        f"- avg_changed_files: {session_payload.get('avg_changed_files', 0.0)}",
        f"- avg_latest_test_duration_ms: {session_payload.get('avg_latest_test_duration_ms', 0)}",
        f"- avg_tool_calls: {session_payload.get('avg_tool_calls', 0.0)}",
        f"- avg_approval_checks: {session_payload.get('avg_approval_checks', 0.0)}",
        f"- approval_denials: {session_payload.get('approval_denials', 0)}",
        f"- proposal_accept_rate: {session_payload.get('proposal_accept_rate', 0.0)}%",
        f"- avg_proposal_score: {session_payload.get('avg_proposal_score', 0.0)}",
        "",
    ]

    if latest_model_payload:
        lines.extend(
            [
                "## Latest Model-Backed Session",
                "",
                f"- session_id: {latest_model_payload.get('session_id', '')}",
                f"- session_name: {latest_model_payload.get('session_name', '') or 'none'}",
                f"- created_at: {latest_model_payload.get('created_at', '')}",
                f"- task_handler: {latest_model_payload.get('task_handler', '')}",
                f"- proposal: {latest_model_payload.get('proposal_status', '')}:{latest_model_payload.get('proposal_score', 0)}",
                f"- fallback_steps: {len(_safe_list(latest_model_payload.get('fallback_steps', [])))}",
                "",
            ]
        )
    elif latest_session_payload:
        lines.extend(
            [
                "## Latest Model-Backed Session",
                "",
                "- none in the current session window",
                f"- latest_session_id: {latest_session_payload.get('session_id', '')}",
                f"- latest_session_fallbacks: {len(_safe_list(latest_session_payload.get('fallback_steps', [])))}",
                "",
            ]
        )

    lines.extend(
        [
            "## Eval Snapshot",
            "",
            f"- mode: {eval_payload.get('mode', '')}",
            f"- pass_rate: {eval_payload.get('pass_rate', 0.0)}%",
            f"- required_file_hit_rate: {eval_payload.get('required_file_hit_rate', 0.0)}%",
            f"- avg_retrieved_files: {eval_payload.get('avg_retrieved_files', 0.0)}",
            f"- avg_changed_files: {eval_payload.get('avg_changed_files', 0.0)}",
            f"- avg_latest_test_duration_ms: {eval_payload.get('avg_latest_test_duration_ms', 0)}",
            f"- avg_tool_calls: {eval_payload.get('avg_tool_calls', 0.0)}",
            f"- avg_approval_checks: {eval_payload.get('avg_approval_checks', 0.0)}",
            f"- approval_denials: {eval_payload.get('approval_denials', 0)}",
            f"- proposal_accept_rate: {eval_payload.get('proposal_accept_rate', 0.0)}%",
            f"- avg_proposal_score: {eval_payload.get('avg_proposal_score', 0.0)}",
            f"- avg_proposal_edits: {eval_payload.get('avg_proposal_edits', 0.0)}",
            "",
        ]
    )

    if recent_rows:
        lines.extend(
            [
                "## Recent Session Table",
                "",
                "| Session | Name | Handler | Status | Fallbacks | Proposal | Tools | Approvals |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in recent_rows:
            if not isinstance(row, dict):
                continue
            session_name = row.get("session_name", "") or "none"
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(row.get("session_id", "")),
                        _markdown_cell(session_name),
                        _markdown_cell(row.get("task_handler", "")),
                        _markdown_cell(row.get("status", "")),
                        _markdown_cell(len(_safe_list(row.get("fallback_steps", [])))),
                        _markdown_cell(f"{row.get('proposal_status', '')}:{row.get('proposal_score', 0)}"),
                        _markdown_cell(row.get("tool_calls", 0)),
                        _markdown_cell(
                            f"{row.get('approval_checks', 0)} checks / {row.get('approval_denials', 0)} denied"
                        ),
                    ]
                )
                + " |"
            )
        lines.append("")

    fallback_step_counts = session_payload.get("fallback_step_counts", {})
    if isinstance(fallback_step_counts, dict) and fallback_step_counts:
        lines.extend(
            [
                "## Session Fallback Breakdown",
                "",
                "| Fallback Step | Count |",
                "| --- | --- |",
            ]
        )
        for name, count in fallback_step_counts.items():
            lines.append(f"| {_markdown_cell(name)} | {_markdown_cell(count)} |")
        lines.append("")

    eval_results = eval_payload.get("results", [])
    if isinstance(eval_results, list) and eval_results:
        lines.extend(
            [
                "## Eval Case Table",
                "",
                "| Handler | Status | Retrieved | Required Hits | Proposal | Tools | Approvals |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for result in eval_results:
            if not isinstance(result, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(result.get("handler_name", "")),
                        _markdown_cell(result.get("status", "")),
                        _markdown_cell(result.get("retrieved_files", 0)),
                        _markdown_cell(
                            f"{result.get('required_file_hits', 0)}/{result.get('required_file_total', 0)}"
                        ),
                        _markdown_cell(f"{result.get('proposal_status', '')}:{result.get('proposal_score', 0)}"),
                        _markdown_cell(result.get("tool_calls", 0)),
                        _markdown_cell(
                            f"{result.get('approval_checks', 0)} checks / {result.get('denied_tool_calls', 0)} denied"
                        ),
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_dashboard_html(report: dict[str, object]) -> str:
    session_payload = report.get("session_metrics", {}) if isinstance(report.get("session_metrics", {}), dict) else {}
    eval_payload = report.get("eval_summary", {}) if isinstance(report.get("eval_summary", {}), dict) else {}
    recent_rows = report.get("recent_sessions", []) if isinstance(report.get("recent_sessions", []), list) else []
    latest_model_payload = report.get("latest_model_session") if isinstance(report.get("latest_model_session"), dict) else None
    latest_session_payload = report.get("latest_session") if isinstance(report.get("latest_session"), dict) else None
    hero_session = latest_model_payload or latest_session_payload or {}
    hero_title = "Latest model-backed session" if latest_model_payload else "Latest persisted session"
    fallback_payload = session_payload.get("fallback_step_counts", {}) if isinstance(session_payload.get("fallback_step_counts", {}), dict) else {}
    eval_rows = eval_payload.get("results", []) if isinstance(eval_payload.get("results", []), list) else []

    cards = [
        ("Completed sessions", session_payload.get("completed_sessions", 0), f"of {session_payload.get('total_sessions', 0)} recent runs", "good"),
        ("Model-backed sessions", session_payload.get("model_backed_sessions", 0), "no fallback in current window", "accent"),
        ("Proposal accept rate", f"{session_payload.get('proposal_accept_rate', 0.0)}%", "recent persisted sessions", "neutral"),
        ("Eval pass rate", f"{eval_payload.get('pass_rate', 0.0)}%", eval_payload.get("mode", "offline"), "good"),
    ]
    card_html = "".join(
        "<article class=\"metric-card tone-{tone}\"><span class=\"metric-label\">{label}</span><strong class=\"metric-value\">{value}</strong><span class=\"metric-meta\">{meta}</span></article>".format(
            tone=tone,
            label=_html_text(label),
            value=_html_text(value),
            meta=_html_text(meta),
        )
        for label, value, meta, tone in cards
    )

    session_rows: list[str] = []
    for row in recent_rows:
        if not isinstance(row, dict):
            continue
        proposal_text = f"{row.get('proposal_status', '')}:{row.get('proposal_score', 0)}"
        session_rows.append(
            "<tr>"
            f"<td><code>{_html_text(row.get('session_id', ''))}</code><div class=\"cell-subtle\">{_html_text(row.get('session_name', '') or 'none')}</div></td>"
            f"<td>{_html_text(row.get('task_handler', ''))}<div class=\"cell-subtle\">{_html_text(row.get('task_text', ''))}</div></td>"
            f"<td><span class=\"badge tone-{_status_tone(row.get('status', ''))}\">{_html_text(row.get('status', ''))}</span></td>"
            f"<td>{len(_safe_list(row.get('fallback_steps', [])))}</td>"
            f"<td><span class=\"badge tone-{_status_tone(row.get('proposal_status', ''))}\">{_html_text(proposal_text)}</span></td>"
            f"<td>{_html_text(row.get('tool_calls', 0))}</td>"
            f"<td>{_html_text(row.get('approval_checks', 0))}<div class=\"cell-subtle\">denied={_html_text(row.get('approval_denials', 0))}</div></td>"
            "</tr>"
        )
    session_rows_html = "".join(session_rows)

    eval_case_rows: list[str] = []
    for row in eval_rows:
        if not isinstance(row, dict):
            continue
        required_hits = f"{row.get('required_file_hits', 0)}/{row.get('required_file_total', 0)}"
        proposal_text = f"{row.get('proposal_status', '')}:{row.get('proposal_score', 0)}"
        eval_case_rows.append(
            "<tr>"
            f"<td>{_html_text(row.get('handler_name', ''))}</td>"
            f"<td><span class=\"badge tone-{_status_tone(row.get('status', ''))}\">{_html_text(row.get('status', ''))}</span></td>"
            f"<td>{_html_text(row.get('retrieved_files', 0))}</td>"
            f"<td>{_html_text(required_hits)}</td>"
            f"<td><span class=\"badge tone-{_status_tone(row.get('proposal_status', ''))}\">{_html_text(proposal_text)}</span></td>"
            f"<td>{_html_text(row.get('tool_calls', 0))}</td>"
            f"<td>{_html_text(row.get('approval_checks', 0))}<div class=\"cell-subtle\">denied={_html_text(row.get('denied_tool_calls', 0))}</div></td>"
            "</tr>"
        )
    eval_rows_html = "".join(eval_case_rows)

    fallback_html = "".join(
        f"<li class=\"fallback-item\"><span>{_html_text(name)}</span><strong>{_html_text(count)}</strong></li>"
        for name, count in fallback_payload.items()
    ) or '<li class="fallback-item"><span>none</span><strong>0</strong></li>'

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Coding Agent Demo Dashboard</title>
  <style>
    :root {{ --bg:#f5eee4; --ink:#1f1b16; --muted:#65584b; --panel:rgba(255,250,244,.92); --line:rgba(83,66,48,.14); --accent:#b25231; --teal:#0f5b63; --good:#2f6b4f; --warn:#a86b11; --bad:#9e2a2b; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at top left, rgba(178,82,49,.12), transparent 28%), radial-gradient(circle at top right, rgba(15,91,99,.12), transparent 24%), linear-gradient(180deg, #f8f2ea 0%, var(--bg) 100%); font:16px/1.5 "Segoe UI Variable Text","Segoe UI",sans-serif; }}
    .shell {{ width:min(1180px, calc(100vw - 28px)); margin:24px auto 40px; }}
    .hero,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:24px; box-shadow:0 18px 45px rgba(74,56,36,.12); }}
    .hero {{ padding:26px; }}
    .panel {{ padding:22px; margin-top:22px; }}
    .hero-grid,.split {{ display:grid; gap:18px; grid-template-columns:1.4fr .9fr; }}
    .eyebrow {{ display:inline-block; font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); border:1px solid var(--line); border-radius:999px; padding:6px 12px; background:rgba(255,255,255,.58); }}
    h1,h2 {{ font-family:Georgia,"Times New Roman",serif; letter-spacing:-.02em; margin:0; }}
    h1 {{ font-size:clamp(2rem, 5vw, 3.4rem); margin-top:16px; max-width:10ch; }}
    h2 {{ font-size:clamp(1.35rem, 2.8vw, 1.9rem); margin-bottom:16px; }}
    p,.cell-subtle,.footnote,.hero-note {{ color:var(--muted); }}
    .hero-meta {{ border:1px solid var(--line); border-radius:18px; padding:16px; background:rgba(255,255,255,.64); display:grid; gap:10px; align-content:start; }}
    .hero-value {{ font-size:1.05rem; font-weight:700; color:var(--ink); }}
    .metrics {{ display:grid; gap:14px; grid-template-columns:repeat(4, minmax(0, 1fr)); }}
    .metric-card {{ min-height:132px; padding:16px; border-radius:18px; border:1px solid var(--line); display:flex; flex-direction:column; justify-content:space-between; background:rgba(255,255,255,.7); }}
    .metric-label {{ color:var(--muted); font-size:.92rem; }}
    .metric-value {{ font-size:clamp(1.7rem, 4vw, 2.5rem); line-height:1; }}
    .metric-meta {{ color:var(--muted); font-size:.9rem; }}
    .tone-good {{ background:linear-gradient(180deg, rgba(47,107,79,.12), rgba(255,250,244,.96)); }}
    .tone-accent {{ background:linear-gradient(180deg, rgba(15,91,99,.12), rgba(255,250,244,.96)); }}
    .tone-neutral {{ background:linear-gradient(180deg, rgba(178,82,49,.12), rgba(255,250,244,.96)); }}
    .snapshot {{ list-style:none; margin:0; padding:0; display:grid; gap:10px; }}
    .snapshot li,.fallback-item {{ display:flex; justify-content:space-between; gap:16px; padding:12px 0; border-bottom:1px solid var(--line); }}
    .snapshot li:last-child,.fallback-item:last-child {{ border-bottom:none; }}
    .snapshot-label {{ color:var(--muted); }}
    .snapshot-value {{ font-weight:700; text-align:right; }}
    .fallback-list {{ list-style:none; margin:0; padding:0; }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:16px; background:rgba(255,255,255,.56); }}
    table {{ width:100%; border-collapse:collapse; min-width:760px; }}
    th,td {{ text-align:left; vertical-align:top; padding:14px 16px; border-bottom:1px solid var(--line); }}
    th {{ color:var(--muted); font-size:12px; letter-spacing:.08em; text-transform:uppercase; background:rgba(255,255,255,.72); }}
    tr:last-child td {{ border-bottom:none; }}
    .badge {{ display:inline-flex; border-radius:999px; padding:4px 10px; font-size:12px; font-weight:700; border:1px solid transparent; }}
    .badge.tone-good {{ color:var(--good); background:rgba(47,107,79,.13); border-color:rgba(47,107,79,.15); }}
    .badge.tone-warn {{ color:var(--warn); background:rgba(168,107,17,.13); border-color:rgba(168,107,17,.15); }}
    .badge.tone-bad {{ color:var(--bad); background:rgba(158,42,43,.13); border-color:rgba(158,42,43,.15); }}
    .badge.tone-neutral {{ color:var(--teal); background:rgba(15,91,99,.13); border-color:rgba(15,91,99,.15); }}
    code {{ font-family:"Cascadia Code","Consolas",monospace; }}
    @media (max-width: 960px) {{ .hero-grid,.split,.metrics {{ grid-template-columns:1fr 1fr; }} .hero-grid,.split {{ grid-template-columns:1fr; }} }}
    @media (max-width: 640px) {{ .shell {{ width:min(100vw - 16px, 1180px); }} .hero,.panel {{ padding:18px; border-radius:20px; }} .metrics {{ grid-template-columns:1fr; }} h1 {{ max-width:none; }} }}
  </style>
</head>
<body>
  <main class=\"shell\">
    <section class=\"hero\">
      <span class=\"eyebrow\">static report shell</span>
      <div class=\"hero-grid\">
        <div>
          <h1>Coding Agent Demo Dashboard</h1>
          <p>Thin presentation wrapper over persisted sessions and the current eval snapshot. The agent loop stays unchanged; this page only repackages the existing evidence into a demo-ready artifact.</p>
        </div>
        <aside class=\"hero-meta\">
          <span class=\"hero-note\">{_html_text(hero_title)}</span>
          <div class=\"hero-value\">{_html_text(hero_session.get('session_id', 'none'))}</div>
          <div class=\"hero-note\">name={_html_text(hero_session.get('session_name', '') or 'none')}</div>
          <div class=\"hero-note\">generated={_html_text(report.get('generated_at', ''))}</div>
          <div class=\"hero-note\">window=latest {_html_text(report.get('session_limit', 0))} sessions</div>
        </aside>
      </div>
    </section>
    <section class=\"panel\">
      <h2>Scoreboard</h2>
      <div class=\"metrics\">{card_html}</div>
      <p class=\"footnote\">Sessions dir: <code>{_html_text(report.get('sessions_dir', ''))}</code></p>
    </section>
    <section class=\"panel\">
      <div class=\"split\">
        <div>
          <h2>Latest Session Snapshot</h2>
          <ul class=\"snapshot\">
            <li><span class=\"snapshot-label\">Handler</span><span class=\"snapshot-value\">{_html_text(hero_session.get('task_handler', ''))}</span></li>
            <li><span class=\"snapshot-label\">Proposal</span><span class=\"snapshot-value\"><span class=\"badge tone-{_status_tone(hero_session.get('proposal_status', ''))}\">{_html_text(f"{hero_session.get('proposal_status', '')}:{hero_session.get('proposal_score', 0)}")}</span></span></li>
            <li><span class=\"snapshot-label\">Fallback steps</span><span class=\"snapshot-value\">{len(_safe_list(hero_session.get('fallback_steps', [])))}</span></li>
            <li><span class=\"snapshot-label\">Avg tool calls</span><span class=\"snapshot-value\">{_html_text(session_payload.get('avg_tool_calls', 0.0))}</span></li>
          </ul>
        </div>
        <div>
          <h2>Fallback Breakdown</h2>
          <ul class=\"fallback-list\">{fallback_html}</ul>
        </div>
      </div>
    </section>
    <section class=\"panel\">
      <h2>Recent Sessions</h2>
      <div class=\"table-wrap\"><table><thead><tr><th>Session</th><th>Task</th><th>Status</th><th>Fallbacks</th><th>Proposal</th><th>Tools</th><th>Approvals</th></tr></thead><tbody>{session_rows_html}</tbody></table></div>
    </section>
    <section class=\"panel\">
      <h2>Eval Cases</h2>
      <div class=\"table-wrap\"><table><thead><tr><th>Handler</th><th>Status</th><th>Retrieved</th><th>Required Hits</th><th>Proposal</th><th>Tools</th><th>Approvals</th></tr></thead><tbody>{eval_rows_html}</tbody></table></div>
    </section>
  </main>
</body>
</html>
"""


def write_dashboard_report(
    config: AppConfig,
    store: RunStore,
    session_limit: int = 10,
    markdown_path: Path | None = None,
    json_path: Path | None = None,
    html_path: Path | None = None,
    enable_live_llm_eval: bool = False,
) -> dict[str, object]:
    report = build_dashboard_report(
        config,
        store,
        session_limit=session_limit,
        enable_live_llm_eval=enable_live_llm_eval,
    )
    markdown_target = _resolve_output_path(config.base_dir, markdown_path, "runtime/reports/dashboard.md")
    json_target = _resolve_output_path(config.base_dir, json_path, "runtime/reports/dashboard.json")
    html_target = _resolve_output_path(config.base_dir, html_path, "runtime/reports/dashboard.html")
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    html_target.parent.mkdir(parents=True, exist_ok=True)

    report["markdown_path"] = str(markdown_target)
    report["json_path"] = str(json_target)
    report["html_path"] = str(html_target)

    markdown_target.write_text(render_dashboard_markdown(report), encoding="utf-8")
    json_target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_target.write_text(render_dashboard_html(report), encoding="utf-8")
    return report
