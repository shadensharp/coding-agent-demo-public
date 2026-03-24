from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import load_config
from .demo_repo_seed import restore_demo_repo
from .evals import run_preset_eval
from .render import TerminalRenderer
from .repo_ops import RepoOps
from .reporting import write_dashboard_report
from .runner import Runner
from .storage import JsonlRunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coding-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a new coding-agent session")
    run_parser.add_argument("--task", required=True, help="Task description for the agent")
    run_parser.add_argument("--repo", help="Path to the repository to operate on")
    run_parser.add_argument("--max-fix", type=int, default=1, help="Maximum auto-fix attempts")
    run_parser.add_argument("--session-name", help="Optional human-readable session label")

    show_parser = subparsers.add_parser("show", help="Show one stored session summary")
    show_parser.add_argument("session_id", help="Session id to inspect")

    replay_parser = subparsers.add_parser("replay", help="Replay stored session events")
    replay_parser.add_argument("session_id", help="Session id to replay")

    list_parser = subparsers.add_parser("list", help="List recent sessions")
    list_parser.add_argument("--limit", type=int, default=10, help="Maximum sessions to show")

    reset_parser = subparsers.add_parser("reset-demo", help="Restore the demo repo to its seeded baseline state")
    reset_parser.add_argument("--repo", help="Path to the demo repository to reset")

    eval_parser = subparsers.add_parser("eval", help="Run preset task evals against fresh seeded demo repos")
    eval_parser.add_argument(
        "--live-llm",
        action="store_true",
        help="Use the configured Qwen client during evals instead of offline fallback mode",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Generate a static dashboard/report from recent sessions plus an eval snapshot",
    )
    report_parser.add_argument(
        "--session-limit",
        type=int,
        default=10,
        help="Maximum persisted sessions to include in the report",
    )
    report_parser.add_argument("--output", help="Path to write the Markdown report")
    report_parser.add_argument("--json-output", help="Path to write the JSON report")
    report_parser.add_argument("--html-output", help="Path to write the HTML report shell")
    report_parser.add_argument(
        "--live-llm-eval",
        action="store_true",
        help="Use the configured Qwen client during the eval snapshot instead of offline fallback mode",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config()
    store = JsonlRunStore(config.sessions_dir)
    renderer = TerminalRenderer()

    try:
        if args.command == "run":
            runner = Runner(config=config, store=store, renderer=renderer)
            session = runner.run(
                task_text=args.task,
                repo_path=args.repo,
                max_fix=args.max_fix,
                session_name=args.session_name,
            )
            renderer.write("")
            renderer.render_session_summary(store.load_summary(session.session_id))
            return 0 if session.status == "completed" else 1

        if args.command == "show":
            renderer.render_session_summary(store.load_summary(args.session_id))
            return 0

        if args.command == "replay":
            for event in store.load_events(args.session_id):
                renderer.render_event(event)
            return 0

        if args.command == "list":
            renderer.render_session_list(store.list_sessions(limit=args.limit))
            return 0

        if args.command == "reset-demo":
            repo_root = Path(args.repo).resolve() if args.repo else config.default_repo_dir
            repo_ops = RepoOps(repo_root)
            changed_paths = restore_demo_repo(repo_ops)
            if changed_paths:
                renderer.write(f"Reset demo repo at {repo_root}")
                for path in changed_paths:
                    renderer.write(f"  - restored {path}")
            else:
                renderer.write(f"Demo repo already matches the seeded baseline: {repo_root}")
            return 0

        if args.command == "eval":
            summary = run_preset_eval(config, enable_live_llm=args.live_llm)
            renderer.render_eval_summary(summary)
            return 0 if int(summary.get("failed_cases", 0)) == 0 else 1

        if args.command == "report":
            report = write_dashboard_report(
                config,
                store,
                session_limit=max(1, args.session_limit),
                markdown_path=Path(args.output) if args.output else None,
                json_path=Path(args.json_output) if args.json_output else None,
                html_path=Path(args.html_output) if args.html_output else None,
                enable_live_llm_eval=args.live_llm_eval,
            )
            renderer.write(f"Report markdown: {report.get('markdown_path', '')}")
            renderer.write(f"Report json: {report.get('json_path', '')}")
            renderer.write(f"Report html: {report.get('html_path', '')}")

            session_metrics = report.get("session_metrics", {})
            if isinstance(session_metrics, dict):
                renderer.write(
                    "Recent sessions: "
                    f"total={session_metrics.get('total_sessions', 0)} "
                    f"completed={session_metrics.get('completed_sessions', 0)} "
                    f"failed={session_metrics.get('failed_sessions', 0)} "
                    f"fallback={session_metrics.get('fallback_sessions', 0)} "
                    f"model_backed={session_metrics.get('model_backed_sessions', 0)}"
                )
                renderer.write(
                    "Recent workflow: "
                    f"avg_tool_calls={session_metrics.get('avg_tool_calls', 0.0)} "
                    f"avg_approval_checks={session_metrics.get('avg_approval_checks', 0.0)} "
                    f"approval_denials={session_metrics.get('approval_denials', 0)}"
                )
                renderer.write(
                    "Recent proposal: "
                    f"accept_rate={session_metrics.get('proposal_accept_rate', 0.0)}% "
                    f"avg_score={session_metrics.get('avg_proposal_score', 0.0)}"
                )

            eval_summary = report.get("eval_summary", {})
            if isinstance(eval_summary, dict):
                renderer.write(
                    "Eval snapshot: "
                    f"mode={eval_summary.get('mode', '')} "
                    f"pass_rate={eval_summary.get('pass_rate', 0.0)}% "
                    f"required_file_hit_rate={eval_summary.get('required_file_hit_rate', 0.0)}% "
                    f"proposal_accept_rate={eval_summary.get('proposal_accept_rate', 0.0)}%"
                )

            latest_model_session = report.get("latest_model_session")
            if isinstance(latest_model_session, dict):
                renderer.write(
                    "Latest model-backed session: "
                    f"{latest_model_session.get('session_id', '')} "
                    f"| name={latest_model_session.get('session_name', '') or 'none'} "
                    f"| proposal={latest_model_session.get('proposal_status', '')}:{latest_model_session.get('proposal_score', 0)}"
                )
            else:
                renderer.write("Latest model-backed session: none in the current session window")
            return 0

        parser.error(f"Unknown command: {args.command}")
        return 2
    except FileNotFoundError as exc:
        renderer.write(f"Error: {exc}")
        return 1
    except Exception as exc:
        renderer.write(f"Unexpected error: {exc}")
        return 1

