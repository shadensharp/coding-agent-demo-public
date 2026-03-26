from __future__ import annotations

from dataclasses import dataclass
import io
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import Callable
from urllib.parse import parse_qs, urlparse
import webbrowser

from .config import AppConfig
from .events import new_id, utc_now
from .render import TerminalRenderer
from .reporting import build_session_dashboard_snapshot
from .runner import Runner
from .storage import JsonlRunStore, RunStore


RunnerFactory = Callable[[TerminalRenderer], Runner]


@dataclass(slots=True)
class RunStatus:
    running: bool
    active_session_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "running": self.running,
            "active_session_id": self.active_session_id,
        }


class WebRunCoordinator:
    def __init__(
        self,
        config: AppConfig,
        store: RunStore,
        runner_factory: RunnerFactory | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.runner_factory = runner_factory or (lambda renderer: Runner(config=config, store=store, renderer=renderer))
        self._lock = threading.Lock()
        self._active_thread: threading.Thread | None = None
        self._active_session_id: str | None = None

    def start_run(
        self,
        task_text: str,
        session_name: str | None = None,
        repo_path: str | None = None,
        enable_web_research: bool = False,
        research_query: str | None = None,
    ) -> str:
        task_text = _one_line(task_text)
        session_name = _one_line(session_name)
        if not task_text:
            raise ValueError("Task text is required.")

        with self._lock:
            if self._active_thread is not None and self._active_thread.is_alive():
                raise RuntimeError("A run is already in progress. Wait for it to finish before starting another one.")
            session_id = new_id("sess")
            thread = threading.Thread(
                target=self._run_session,
                args=(session_id, task_text, session_name or None, repo_path, enable_web_research, research_query),
                daemon=True,
                name=f"coding-agent-web-{session_id}",
            )
            self._active_session_id = session_id
            self._active_thread = thread
            thread.start()
            return session_id

    def status(self) -> RunStatus:
        with self._lock:
            running = self._active_thread is not None and self._active_thread.is_alive()
            active_session_id = self._active_session_id if running else None
            if not running:
                self._active_thread = None
                self._active_session_id = None
            return RunStatus(running=running, active_session_id=active_session_id)

    def _run_session(
        self,
        session_id: str,
        task_text: str,
        session_name: str | None,
        repo_path: str | None,
        enable_web_research: bool,
        research_query: str | None,
    ) -> None:
        renderer = TerminalRenderer(stream=io.StringIO(), event_verbosity="compact", summary_verbosity="compact")
        try:
            runner = self.runner_factory(renderer)
            runner.run(
                task_text=task_text,
                repo_path=repo_path,
                session_name=session_name,
                session_id=session_id,
                enable_web_research=enable_web_research,
                research_query=research_query,
            )
        finally:
            with self._lock:
                if self._active_session_id == session_id:
                    self._active_thread = None
                    self._active_session_id = None


class CodingAgentWebServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        config: AppConfig,
        store: RunStore,
        coordinator: WebRunCoordinator,
        session_limit: int = 10,
    ) -> None:
        super().__init__(server_address, CodingAgentWebHandler)
        self.config = config
        self.store = store
        self.coordinator = coordinator
        self.session_limit = max(1, session_limit)


class CodingAgentWebHandler(BaseHTTPRequestHandler):
    server: CodingAgentWebServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(render_web_console_html())
            return
        if parsed.path == "/api/state":
            query = parse_qs(parsed.query)
            selected_session_id = _one_line(query.get("session_id", [""])[0]) or None
            limit_text = _one_line(query.get("limit", [str(self.server.session_limit)])[0])
            try:
                limit = max(1, int(limit_text))
            except ValueError:
                limit = self.server.session_limit
            payload = build_console_state(
                self.server.config,
                self.server.store,
                self.server.coordinator,
                selected_session_id=selected_session_id,
                session_limit=limit,
            )
            self._send_json(payload)
            return
        self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)
            return

        payload = self._read_json_body()
        if payload is None:
            self._send_json({"error": "Invalid JSON body."}, status=HTTPStatus.BAD_REQUEST)
            return

        task_text = _one_line(payload.get("task", ""))
        session_name = _one_line(payload.get("session_name", "")) or None
        repo_path = _one_line(payload.get("repo_path", "")) or None
        research_query = _one_line(payload.get("research_query", "")) or None
        enable_web_research = _coerce_bool(payload.get("enable_web_research", False))
        if not task_text:
            self._send_json({"error": "Task text is required."}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            session_id = self.server.coordinator.start_run(
                task_text=task_text,
                session_name=session_name,
                repo_path=repo_path,
                enable_web_research=enable_web_research,
                research_query=research_query,
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
            return

        self._send_json(
            {
                "accepted": True,
                "session_id": session_id,
                "message": "Run started.",
            },
            status=HTTPStatus.ACCEPTED,
        )

    def log_message(self, format: str, *args: object) -> None:
        _ = format
        _ = args

    def _read_json_body(self) -> dict[str, object] | None:
        content_length = self.headers.get("Content-Length", "0")
        try:
            length = max(0, int(content_length))
        except ValueError:
            return None
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)



def build_console_state(
    config: AppConfig,
    store: RunStore,
    coordinator: WebRunCoordinator,
    selected_session_id: str | None = None,
    session_limit: int = 10,
) -> dict[str, object]:
    snapshot = build_session_dashboard_snapshot(config, store, session_limit=max(1, session_limit))
    recent_sessions = snapshot.get("recent_sessions", [])

    run_status = coordinator.status().to_dict()
    active_session_id = str(run_status.get("active_session_id", "") or "")
    active_events: list[dict[str, object]] = []
    if active_session_id:
        try:
            active_events = store.load_events(active_session_id)
        except FileNotFoundError:
            active_events = []
        active_steps = _step_timeline_payload(active_events)
        if active_steps:
            current_step = next((item for item in reversed(active_steps) if item.get("status") == "running"), active_steps[-1])
            run_status["current_step"] = current_step.get("step_type", "")
            run_status["current_step_status"] = current_step.get("status", "")
            run_status["current_step_summary"] = current_step.get("summary", "")
        if active_events:
            run_status["last_event_message"] = _event_message(active_events[-1])

    selected_id = selected_session_id or active_session_id
    if not selected_id and isinstance(snapshot.get("latest_session"), dict):
        selected_id = str(snapshot["latest_session"].get("session_id", "") or "")

    selected_summary: dict[str, object] | None = None
    selected_events: list[dict[str, object]] = []
    if selected_id:
        try:
            selected_summary = store.load_summary(selected_id)
        except FileNotFoundError:
            selected_summary = None
        try:
            selected_events = store.load_events(selected_id)
        except FileNotFoundError:
            selected_events = []

    return {
        "generated_at": utc_now(),
        "session_metrics": snapshot.get("session_metrics", {}),
        "recent_sessions": recent_sessions,
        "latest_model_session": snapshot.get("latest_model_session", None),
        "run_status": run_status,
        "default_repo_path": str(config.default_repo_dir),
        "selected_session": _session_detail_payload(selected_summary, selected_events),
    }



def render_web_console_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Coding Agent Web Console</title>
  <style>
    :root { --bg:#f0e5d6; --ink:#171410; --muted:#675a4d; --line:rgba(79,63,47,.14); --panel:rgba(255,251,246,.92); --accent:#b94e2f; --teal:#145861; --good:#2f6b4f; --warn:#b37412; --bad:#9b2c2c; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:radial-gradient(circle at top left, rgba(185,78,47,.14), transparent 24%), radial-gradient(circle at top right, rgba(20,88,97,.12), transparent 24%), linear-gradient(180deg, #f8f2ea 0%, var(--bg) 100%); font:16px/1.5 "Segoe UI Variable Text","Segoe UI",sans-serif; }
    a { color:var(--teal); }
    code { font-family:"Cascadia Code","Consolas",monospace; }
    .shell { width:min(1320px, calc(100vw - 24px)); margin:20px auto 36px; display:grid; gap:18px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:24px; box-shadow:0 16px 40px rgba(68,53,37,.12); }
    .hero { padding:24px; display:grid; gap:18px; grid-template-columns:1.15fr .85fr; }
    .eyebrow { display:inline-flex; font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); border:1px solid var(--line); border-radius:999px; padding:6px 12px; background:rgba(255,255,255,.65); }
    h1,h2,h3 { font-family:Georgia,"Times New Roman",serif; letter-spacing:-.02em; margin:0; }
    h1 { font-size:clamp(2.1rem, 5vw, 3.6rem); margin-top:14px; max-width:12ch; }
    h2 { font-size:1.45rem; margin-bottom:14px; }
    h3 { font-size:1rem; margin-bottom:10px; }
    p,.muted,.meta { color:var(--muted); }
    .hero-right { display:grid; gap:12px; align-content:start; }
    .run-form,.status-card,.metrics,.list-panel,.detail-panel { padding:20px; }
    .metrics-grid,.content-grid,.detail-grid,.form-grid,.section-grid { display:grid; gap:16px; }
    .metrics-grid { grid-template-columns:repeat(4,minmax(0,1fr)); }
    .content-grid { grid-template-columns:360px 1fr; align-items:start; }
    .detail-grid,.section-grid { grid-template-columns:1fr 1fr; }
    .metric { border:1px solid var(--line); border-radius:18px; padding:16px; background:rgba(255,255,255,.66); }
    .metric-value { display:block; font-size:2rem; line-height:1; margin:8px 0 10px; }
    .metric-label,.metric-meta { color:var(--muted); font-size:.92rem; }
    textarea,input,button { font:inherit; }
    textarea,input { width:100%; border:1px solid var(--line); border-radius:14px; padding:12px 14px; background:rgba(255,255,255,.85); color:var(--ink); }
    textarea { min-height:110px; resize:vertical; }
    input:disabled { opacity:.65; background:rgba(240,232,221,.8); }
    .form-grid { grid-template-columns:1fr 1fr; margin-top:12px; }
    .full-span { grid-column:1 / -1; }
    .toggle-row { display:grid; gap:8px; margin-top:14px; padding:12px 14px; border:1px solid var(--line); border-radius:16px; background:rgba(255,255,255,.56); }
    .checkline { display:flex; align-items:center; gap:10px; font-weight:600; }
    .checkline input { width:auto; margin:0; }
    .actions { display:flex; gap:10px; margin-top:12px; }
    button { border:none; border-radius:999px; padding:11px 16px; cursor:pointer; }
    button:disabled { cursor:not-allowed; opacity:.7; }
    .primary { background:var(--accent); color:#fff7f1; }
    .secondary { background:rgba(255,255,255,.7); color:var(--ink); border:1px solid var(--line); }
    .session-list,.stack,.timeline,.change-list,.source-list,.step-list { list-style:none; padding:0; margin:0; display:grid; gap:10px; }
    .session-item,.fact-item,.timeline-item,.change-item,.source-item,.step-item { border:1px solid var(--line); border-radius:16px; padding:14px; background:rgba(255,255,255,.62); }
    .session-item { cursor:pointer; }
    .session-item.active { border-color:rgba(20,88,97,.32); box-shadow:0 0 0 2px rgba(20,88,97,.08) inset; }
    .session-task { color:var(--ink); font-weight:600; margin-top:6px; }
    .badge { display:inline-flex; border-radius:999px; padding:4px 10px; font-size:12px; font-weight:700; }
    .badge.good { background:rgba(47,107,79,.12); color:var(--good); }
    .badge.warn { background:rgba(179,116,18,.12); color:var(--warn); }
    .badge.bad { background:rgba(155,44,44,.12); color:var(--bad); }
    .badge.neutral { background:rgba(20,88,97,.12); color:var(--teal); }
    .detail-top,.item-top,.step-top { display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }
    .fact-list { list-style:none; padding:0; margin:0; display:grid; gap:0; }
    .fact-pair { display:flex; justify-content:space-between; gap:16px; padding:8px 0; border-bottom:1px solid var(--line); }
    .fact-pair:last-child { border-bottom:none; }
    .detail-stack { display:grid; gap:16px; }
    .code-preview { margin-top:10px; border-radius:12px; background:#191714; color:#f3e8d6; padding:10px 12px; overflow:auto; font:13px/1.45 "Cascadia Code","Consolas",monospace; }
    .empty { border:1px dashed var(--line); border-radius:18px; padding:18px; color:var(--muted); background:rgba(255,255,255,.5); }
    .error { color:var(--bad); min-height:1.3em; margin-top:8px; }
    .info { color:var(--teal); min-height:1.3em; margin-top:8px; }
    .small { font-size:.92rem; }
    @media (max-width: 1100px) { .hero,.content-grid,.detail-grid,.metrics-grid,.form-grid,.section-grid { grid-template-columns:1fr; } .full-span { grid-column:auto; } }
  </style>
</head>
<body>
  <main class="shell">
    <section class="panel hero">
      <div>
        <span class="eyebrow">interactive local console</span>
        <h1>Coding Agent Operator Surface</h1>
        <p>Run one bounded coding task, point it at a local Python repo, and inspect the result without reading raw terminal audit logs.</p>
        <p class="meta">This page is interactive only when opened through <code>python -m coding_agent serve</code>. The static report opened by <code>report --open</code> is read-only.</p>
      </div>
      <div class="hero-right">
        <section class="panel run-form">
          <h2>Start Run</h2>
          <label class="muted" for="taskInput">Task</label>
          <textarea id="taskInput">Add priority sorting to the Todo API and fix the PATCH partial update bug.</textarea>
          <div class="form-grid">
            <div>
              <label class="muted" for="sessionNameInput">Session name</label>
              <input id="sessionNameInput" placeholder="optional" />
            </div>
            <div>
              <label class="muted" for="repoPathInput">Repo path</label>
              <input id="repoPathInput" placeholder="optional; defaults to configured demo repo" />
            </div>
          </div>
          <div class="toggle-row">
            <label class="checkline" for="enableResearchInput">
              <input id="enableResearchInput" type="checkbox" />
              <span>Enable external research</span>
            </label>
            <div class="meta small">Use lightweight web search to gather cited context before clarify, plan, proposal, and edit.</div>
          </div>
          <div class="form-grid">
            <div class="full-span">
              <label class="muted" for="researchQueryInput">Research query</label>
              <input id="researchQueryInput" placeholder="optional; defaults to the task text when research is enabled" />
            </div>
          </div>
          <p id="repoPathHint" class="meta"></p>
          <div class="actions">
            <button id="runButton" class="primary" type="button">Run Task</button>
            <button id="refreshButton" class="secondary" type="button">Refresh</button>
          </div>
          <div id="formInfo" class="info"></div>
          <div id="formError" class="error"></div>
        </section>
        <section class="panel status-card">
          <h2>Run Status</h2>
          <div id="runStatus" class="empty">Loading...</div>
        </section>
      </div>
    </section>

    <section class="panel metrics">
      <h2>Session Scoreboard</h2>
      <div id="metricsGrid" class="metrics-grid"></div>
    </section>

    <section class="content-grid">
      <section class="panel list-panel">
        <h2>Recent Sessions</h2>
        <div id="sessionList" class="session-list"></div>
      </section>

      <section class="panel detail-panel">
        <div class="detail-top">
          <div>
            <h2>Selected Session</h2>
            <p id="selectedSessionMeta" class="meta"></p>
          </div>
          <div id="selectedSessionBadge"></div>
        </div>
        <div id="selectedSessionBody" class="empty">No session selected.</div>
      </section>
    </section>
  </main>

  <script>
    const state = { selectedSessionId: null, pendingRun: false, defaultRepoPath: '', sessionLimit: 10 };
    const HTML_ESCAPE_MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, (char) => HTML_ESCAPE_MAP[char] || char);
    }

    function setFormError(message) {
      document.getElementById('formError').textContent = message || '';
    }

    function setFormInfo(message) {
      document.getElementById('formInfo').textContent = message || '';
    }

    function toneFor(status) {
      const normalized = String(status || '').toLowerCase();
      if (normalized === 'completed' || normalized === 'accepted') return 'good';
      if (normalized === 'failed' || normalized === 'rejected') return 'bad';
      if (normalized === 'partial' || normalized === 'running') return 'warn';
      return 'neutral';
    }

    function metricCard(label, value, meta) {
      return `<article class="metric"><span class="metric-label">${escapeHtml(label)}</span><strong class="metric-value">${escapeHtml(value)}</strong><span class="metric-meta">${escapeHtml(meta)}</span></article>`;
    }

    function factPairs(rows) {
      return rows.map((row) => `<div class="fact-pair"><span>${escapeHtml(row.label)}</span><span>${row.html ? row.value : escapeHtml(row.value)}</span></div>`).join('');
    }

    function setRunButtonState(runStatus) {
      const running = Boolean(runStatus && runStatus.running);
      const disabled = state.pendingRun || running;
      const button = document.getElementById('runButton');
      button.disabled = disabled;
      button.textContent = disabled ? 'Run In Progress' : 'Run Task';
    }

    function syncResearchControls() {
      const enabled = document.getElementById('enableResearchInput').checked;
      const queryInput = document.getElementById('researchQueryInput');
      queryInput.disabled = !enabled;
      if (!enabled) {
        queryInput.value = '';
      }
    }

    function renderMetrics(metrics) {
      const grid = document.getElementById('metricsGrid');
      grid.innerHTML = [
        metricCard('Completed Sessions', metrics.completed_sessions ?? 0, `of ${metrics.total_sessions ?? 0} recent runs`),
        metricCard('Model-Backed Sessions', metrics.model_backed_sessions ?? 0, 'without recorded fallback'),
        metricCard('Proposal Accept Rate', `${metrics.proposal_accept_rate ?? 0}%`, 'recent persisted sessions'),
        metricCard('Avg Test Duration', `${metrics.avg_latest_test_duration_ms ?? 0} ms`, 'latest regression result'),
      ].join('');
    }

    function renderRunStatus(runStatus) {
      const container = document.getElementById('runStatus');
      setRunButtonState(runStatus || {});
      if (!runStatus || !runStatus.running) {
        container.className = 'empty';
        container.innerHTML = 'No active run.';
        return;
      }
      container.className = '';
      const rows = [
        { label: 'Current state', value: '<span class="badge warn">running</span>', html: true },
        { label: 'Tracking session', value: `<code>${escapeHtml(runStatus.active_session_id || '')}</code>`, html: true },
        { label: 'Current step', value: runStatus.current_step || 'waiting' },
        { label: 'Step summary', value: runStatus.current_step_summary || 'pending' },
        { label: 'Last update', value: runStatus.last_event_message || 'waiting for events' },
      ];
      container.innerHTML = `<div class="fact-item"><div class="fact-list">${factPairs(rows)}</div></div>`;
    }

    function renderSessionList(sessions) {
      const container = document.getElementById('sessionList');
      if (!sessions || !sessions.length) {
        container.innerHTML = '<div class="empty">No sessions found yet.</div>';
        return;
      }
      container.innerHTML = sessions.map((session) => {
        const active = state.selectedSessionId === session.session_id ? ' active' : '';
        const sessionName = session.session_name || 'none';
        const handler = session.task_handler || 'generic';
        return `
          <article class="session-item${active}" data-session-id="${escapeHtml(session.session_id)}">
            <div class="item-top">
              <code>${escapeHtml(session.session_id)}</code>
              <span class="badge ${toneFor(session.status)}">${escapeHtml(session.status)}</span>
            </div>
            <div class="session-task">${escapeHtml(session.task_text || '')}</div>
            <div class="meta">name=${escapeHtml(sessionName)} | handler=${escapeHtml(handler)}</div>
          </article>
        `;
      }).join('');
      container.querySelectorAll('[data-session-id]').forEach((item) => {
        item.addEventListener('click', () => {
          state.selectedSessionId = item.getAttribute('data-session-id');
          refreshState();
        });
      });
    }

    function renderResearch(session) {
      if (!session.research_enabled) {
        return '<div class="empty">External research was not enabled for this session.</div>';
      }
      const sourceItems = (session.research_sources || []).length
        ? `<ul class="source-list">${session.research_sources.map((source) => `
            <li class="source-item">
              <div><strong>${escapeHtml(source.title || '')}</strong></div>
              <div class="meta"><a href="${escapeHtml(source.url || '')}" target="_blank" rel="noreferrer">${escapeHtml(source.url || '')}</a></div>
              <div class="meta">${escapeHtml(source.snippet || 'no snippet')}</div>
            </li>
          `).join('')}</ul>`
        : '<div class="empty">No external sources were captured.</div>';
      return `
        <div class="stack">
          <div class="fact-item">
            <div class="fact-list">
              ${factPairs([
                { label: 'Query', value: session.research_query || 'used task text' },
                { label: 'Summary', value: session.research_summary || 'none' },
              ])}
            </div>
          </div>
          ${sourceItems}
        </div>
      `;
    }

    function renderChangedFiles(files) {
      if (!files || !files.length) {
        return '<div class="empty">No changed files were recorded for this session.</div>';
      }
      return `<ul class="change-list">${files.map((change) => `
        <li class="change-item">
          <div><code>${escapeHtml(change.path || '')}</code></div>
          <div class="meta">${escapeHtml(change.summary || '')}</div>
          ${(change.diff_preview || []).length ? `<pre class="code-preview"><code>${escapeHtml(change.diff_preview.join('\n'))}</code></pre>` : ''}
        </li>
      `).join('')}</ul>`;
    }

    function renderSteps(steps) {
      if (!steps || !steps.length) {
        return '<div class="empty">No step timeline was stored for this session.</div>';
      }
      return `<ul class="step-list">${steps.map((step) => `
        <li class="step-item">
          <div class="step-top">
            <strong>${escapeHtml(step.step_type || '')}</strong>
            <span class="badge ${toneFor(step.status)}">${escapeHtml(step.status || '')}</span>
          </div>
          <div class="meta">started=${escapeHtml(step.started_at || 'n/a')} | finished=${escapeHtml(step.finished_at || 'n/a')}</div>
          <div class="meta">${escapeHtml(step.summary || 'no summary')}</div>
        </li>
      `).join('')}</ul>`;
    }

    function renderEvents(events) {
      if (!events || !events.length) {
        return '<div class="empty">No event log was stored for this session.</div>';
      }
      return `<ul class="timeline">${events.map((event) => `
        <li class="timeline-item">
          <div class="item-top"><strong>${escapeHtml(event.kind || '')}</strong><span class="meta">${escapeHtml(event.timestamp || '')}</span></div>
          <div class="meta">${escapeHtml(event.message || '')}</div>
        </li>
      `).join('')}</ul>`;
    }

    function renderSelectedSession(session) {
      const meta = document.getElementById('selectedSessionMeta');
      const badge = document.getElementById('selectedSessionBadge');
      const body = document.getElementById('selectedSessionBody');
      if (!session) {
        meta.textContent = '';
        badge.innerHTML = '';
        body.className = 'empty';
        body.textContent = 'No session selected.';
        return;
      }
      meta.textContent = `${session.created_at || ''} | handler=${session.task_handler || 'generic'}`;
      badge.innerHTML = `<span class="badge ${toneFor(session.status)}">${escapeHtml(session.status || '')}</span>`;
      const fallbacks = (session.fallback_steps || []).join(', ') || 'none';
      const review = session.review_summary || session.final_summary || '';
      body.className = '';
      body.innerHTML = `
        <div class="detail-stack">
          <div class="section-grid">
            <section>
              <h3>Outcome</h3>
              <div class="fact-item">
                <div class="fact-list">
                  ${factPairs([
                    { label: 'Task', value: session.task_text || '' },
                    { label: 'Validation', value: session.validation || 'none' },
                    { label: 'Review', value: review },
                    { label: 'Fallbacks', value: fallbacks },
                  ])}
                </div>
              </div>
            </section>
            <section>
              <h3>Research</h3>
              ${renderResearch(session)}
            </section>
          </div>
          <div class="detail-grid">
            <section>
              <h3>Changed Files</h3>
              ${renderChangedFiles(session.changed_files || [])}
            </section>
            <section>
              <h3>Step Timeline</h3>
              ${renderSteps(session.steps || [])}
            </section>
          </div>
          <section>
            <h3>Event Timeline</h3>
            ${renderEvents(session.events || [])}
          </section>
        </div>
      `;
    }

    async function refreshState() {
      try {
        const params = new URLSearchParams({ limit: String(state.sessionLimit) });
        if (state.selectedSessionId) {
          params.set('session_id', state.selectedSessionId);
        }
        const response = await fetch(`/api/state?${params.toString()}`, { cache: 'no-store' });
        if (!response.ok) {
          throw new Error(`State request failed (${response.status})`);
        }
        const payload = await response.json();
        state.defaultRepoPath = payload.default_repo_path || '';
        document.getElementById('repoPathHint').textContent = state.defaultRepoPath ? `Default repo: ${state.defaultRepoPath}` : '';
        if (payload.run_status && payload.run_status.active_session_id) {
          state.selectedSessionId = payload.run_status.active_session_id;
        } else if (!state.selectedSessionId && payload.selected_session) {
          state.selectedSessionId = payload.selected_session.session_id;
        }
        if (state.pendingRun && payload.run_status && !payload.run_status.running) {
          state.pendingRun = false;
          setFormInfo('Run finished. Review the latest session details below.');
        }
        renderMetrics(payload.session_metrics || {});
        renderRunStatus(payload.run_status || {});
        renderSessionList(payload.recent_sessions || []);
        renderSelectedSession(payload.selected_session || null);
        setFormError('');
      } catch (error) {
        renderRunStatus({ running: false });
        setFormError(error && error.message ? error.message : 'Unable to refresh state.');
      }
    }

    async function submitRun() {
      const task = document.getElementById('taskInput').value.trim();
      const sessionName = document.getElementById('sessionNameInput').value.trim();
      const repoPath = document.getElementById('repoPathInput').value.trim();
      const enableResearch = document.getElementById('enableResearchInput').checked;
      const researchQuery = document.getElementById('researchQueryInput').value.trim();
      if (!task) {
        setFormError('Task text is required.');
        return;
      }
      state.pendingRun = true;
      setRunButtonState({ running: true });
      setFormError('');
      setFormInfo('Submitting run...');
      try {
        const response = await fetch('/api/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            task,
            session_name: sessionName,
            repo_path: repoPath,
            enable_web_research: enableResearch,
            research_query: researchQuery,
          }),
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || `Run failed to start (${response.status})`);
        }
        state.selectedSessionId = payload.session_id || state.selectedSessionId;
        setFormInfo(payload.message || 'Run started.');
        await refreshState();
      } catch (error) {
        state.pendingRun = false;
        setRunButtonState({ running: false });
        setFormError(error && error.message ? error.message : 'Run failed to start.');
      }
    }

    window.addEventListener('error', (event) => {
      if (event && event.message) {
        setFormError(`Client error: ${event.message}`);
      }
    });

    window.addEventListener('unhandledrejection', (event) => {
      const reason = event && event.reason ? String(event.reason) : 'Unknown async error';
      setFormError(`Client error: ${reason}`);
    });

    document.getElementById('enableResearchInput').addEventListener('change', syncResearchControls);
    document.getElementById('runButton').addEventListener('click', submitRun);
    document.getElementById('refreshButton').addEventListener('click', refreshState);
    syncResearchControls();
    refreshState();
    window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        refreshState();
      }
    }, 2000);
  </script>
</body>
</html>
"""


def serve_console(
    config: AppConfig,
    store: RunStore | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    session_limit: int = 10,
    open_browser: bool = False,
    browser_opener: Callable[[str], bool] | None = None,
    renderer: TerminalRenderer | None = None,
) -> int:
    store_obj = store or JsonlRunStore(config.sessions_dir)
    console_renderer = renderer or TerminalRenderer()
    coordinator = WebRunCoordinator(config, store_obj)
    server = CodingAgentWebServer((host, port), config=config, store=store_obj, coordinator=coordinator, session_limit=session_limit)
    actual_host, actual_port = server.server_address
    display_host = actual_host if actual_host not in {"0.0.0.0", "::"} else "127.0.0.1"
    url = f"http://{display_host}:{actual_port}/"
    console_renderer.write(f"Web console: {url}")
    console_renderer.write("Press Ctrl+C to stop.")
    if open_browser:
        opener = browser_opener or webbrowser.open_new_tab
        try:
            if opener(url):
                console_renderer.write("Opened web console in browser.")
            else:
                console_renderer.write("Open the web console URL manually in your browser.")
        except Exception:
            console_renderer.write("Open the web console URL manually in your browser.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console_renderer.write("Stopping web console...")
    finally:
        server.server_close()
    return 0



def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = _one_line(value).casefold()
    return normalized in {"1", "true", "yes", "on"}



def _one_line(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())



def _diff_preview(diff_excerpt: object, limit: int = 3) -> list[str]:
    if not isinstance(diff_excerpt, str):
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



def _task_text(summary: dict[str, object]) -> str:
    request = summary.get("request", {})
    if not isinstance(request, dict):
        return ""
    return _one_line(request.get("user_text", ""))



def _review_preview(value: object) -> str:
    text = _one_line(value)
    if not text:
        return ""
    markers = ["Proposal assessment:", "Grounding:", "External research:", "Read evidence:", "Evidence:", "Validation:", "Audit:", "LLM note:", "Residual risk:"]
    end = len(text)
    for marker in markers:
        index = text.find(marker)
        if index != -1:
            end = min(end, index)
    if end != len(text):
        return text[:end].rstrip(" .") + "."
    return text


def _validation_text(summary: dict[str, object]) -> str:
    test_results = summary.get("test_results", [])
    if not isinstance(test_results, list) or not test_results:
        return "none"
    latest = test_results[-1]
    if not isinstance(latest, dict):
        return "none"
    command = latest.get("command", [])
    if isinstance(command, list):
        command_text = " ".join(str(part) for part in command)
    else:
        command_text = _one_line(command)
    duration_text = latest.get("duration_ms", "")
    if "exit_code" in latest:
        status = "passed" if latest.get("exit_code", 1) == 0 else f"failed (exit={latest.get('exit_code', '')})"
        return f"{command_text} | {status} | {duration_text} ms" if command_text else f"{status} | {duration_text} ms"
    return f"{command_text} | duration={duration_text} ms" if command_text else f"duration={duration_text} ms"



def _event_message(event: dict[str, object]) -> str:
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}
    kind = str(event.get("kind", ""))
    if kind == "step_started":
        return f"{payload.get('step_type', '')} started"
    if kind == "step_completed":
        return _one_line(payload.get("output_summary", ""))
    if kind == "step_failed":
        return _one_line(payload.get("error_message", ""))
    if kind == "approval_checked":
        return f"approved={payload.get('approved', '')} | tool={payload.get('tool_name', '')} | mode={payload.get('mode', '')}"
    if kind == "tool_started":
        return f"{payload.get('step_type', '')} -> {payload.get('tool_name', '')}"
    if kind == "tool_completed":
        return _one_line(payload.get("output_summary", ""))
    if kind == "proposal_assessed":
        return f"status={payload.get('status', '')} score={payload.get('score', '')}"
    if kind == "command_started":
        command = payload.get("command", [])
        return " ".join(str(part) for part in command) if isinstance(command, list) else _one_line(command)
    if kind == "command_completed":
        return f"exit={payload.get('exit_code', '')} duration_ms={payload.get('duration_ms', '')}"
    if kind == "file_changed":
        return _one_line(payload.get("summary", ""))
    if kind == "session_completed":
        return _one_line(payload.get("final_summary", ""))
    if kind == "context_selected":
        files = payload.get("files", [])
        if isinstance(files, list):
            return ", ".join(str(item.get("path", "")) for item in files if isinstance(item, dict))
    return _one_line(json.dumps(payload, ensure_ascii=False))
def _step_timeline_payload(events: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered_steps: list[dict[str, object]] = []
    by_step_id: dict[str, dict[str, object]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        kind = str(event.get("kind", ""))
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        timestamp = str(event.get("timestamp", ""))
        step_id = _one_line(payload.get("step_id", ""))
        if kind == "step_started":
            entry = {
                "step_id": step_id,
                "step_type": _one_line(payload.get("step_type", "")),
                "status": "running",
                "started_at": timestamp,
                "finished_at": "",
                "summary": _one_line(payload.get("input_summary", "")),
            }
            ordered_steps.append(entry)
            if step_id:
                by_step_id[step_id] = entry
            continue
        if kind not in {"step_completed", "step_failed"}:
            continue
        entry = by_step_id.get(step_id)
        if entry is None:
            entry = {
                "step_id": step_id,
                "step_type": _one_line(payload.get("step_type", "")),
                "status": "running",
                "started_at": "",
                "finished_at": "",
                "summary": "",
            }
            ordered_steps.append(entry)
            if step_id:
                by_step_id[step_id] = entry
        entry["status"] = "completed" if kind == "step_completed" else "failed"
        entry["finished_at"] = timestamp
        entry["summary"] = _one_line(payload.get("output_summary", payload.get("error_message", "")))
    return ordered_steps[-12:]



def _session_detail_payload(summary: dict[str, object] | None, events: list[dict[str, object]]) -> dict[str, object] | None:
    if summary is None:
        return None
    research_sources = summary.get("research_sources", [])
    if not isinstance(research_sources, list):
        research_sources = []
    return {
        "session_id": summary.get("session_id", ""),
        "created_at": summary.get("created_at", ""),
        "status": summary.get("status", ""),
        "task_handler": summary.get("task_handler", ""),
        "task_text": _task_text(summary),
        "review_summary": _review_preview(summary.get("review_summary", "")),
        "final_summary": _one_line(summary.get("final_summary", "")),
        "validation": _validation_text(summary),
        "fallback_steps": [str(item) for item in summary.get("fallback_steps", [])] if isinstance(summary.get("fallback_steps", []), list) else [],
        "research_enabled": bool(summary.get("research_enabled", False)),
        "research_query": _one_line(summary.get("research_query", "")),
        "research_summary": _one_line(summary.get("research_summary", "")),
        "research_sources": [
            {
                "title": _one_line(item.get("title", "")),
                "url": _one_line(item.get("url", "")),
                "snippet": _one_line(item.get("snippet", "")),
            }
            for item in research_sources
            if isinstance(item, dict)
        ],
        "changed_files": [
            {
                "path": item.get("path", ""),
                "summary": item.get("summary", ""),
                "diff_preview": _diff_preview(item.get("diff_excerpt", None), limit=6),
            }
            for item in summary.get("changed_files", [])
            if isinstance(item, dict)
        ],
        "steps": _step_timeline_payload(events),
        "events": [
            {
                "timestamp": str(item.get("timestamp", "")),
                "kind": str(item.get("kind", "")),
                "message": _event_message(item),
            }
            for item in events[-40:]
            if isinstance(item, dict)
        ],
    }












