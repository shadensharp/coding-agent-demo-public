from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_WEB_RESEARCH_API_BASE = "https://html.duckduckgo.com/html/"
DEFAULT_WEB_RESEARCH_USER_AGENT = "coding-agent-demo/0.1 (+https://local)"


def _resolve_path(base_dir: Path, raw_value: str | None, default_value: str) -> Path:
    candidate = Path(raw_value) if raw_value else Path(default_value)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(slots=True)
class AppConfig:
    base_dir: Path
    runtime_dir: Path
    sessions_dir: Path
    default_repo_dir: Path
    qwen_model: str
    qwen_api_key: str | None
    qwen_api_base: str
    qwen_timeout_seconds: float
    qwen_max_retries: int
    qwen_retry_backoff_seconds: float
    web_research_api_base: str = DEFAULT_WEB_RESEARCH_API_BASE
    web_research_timeout_seconds: float = 20.0
    web_research_max_results: int = 5
    web_research_user_agent: str = DEFAULT_WEB_RESEARCH_USER_AGENT

    def ensure_directories(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)


def load_config(base_dir: Path | None = None) -> AppConfig:
    root = Path(base_dir or Path.cwd()).resolve()
    runtime_dir = _resolve_path(root, os.getenv("CODING_AGENT_RUNTIME_DIR"), "runtime")
    default_repo_dir = _resolve_path(root, os.getenv("CODING_AGENT_DEMO_REPO"), "demo_repo")
    config = AppConfig(
        base_dir=root,
        runtime_dir=runtime_dir,
        sessions_dir=runtime_dir / "sessions",
        default_repo_dir=default_repo_dir,
        qwen_model=os.getenv("CODING_AGENT_MODEL", "qwen-plus"),
        qwen_api_key=os.getenv("QWEN_API_KEY"),
        qwen_api_base=os.getenv(
            "QWEN_API_BASE",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        qwen_timeout_seconds=max(1.0, _env_float("CODING_AGENT_QWEN_TIMEOUT_SECONDS", 45.0)),
        qwen_max_retries=max(0, _env_int("CODING_AGENT_QWEN_MAX_RETRIES", 2)),
        qwen_retry_backoff_seconds=max(0.0, _env_float("CODING_AGENT_QWEN_RETRY_BACKOFF_SECONDS", 1.0)),
        web_research_api_base=os.getenv("CODING_AGENT_WEB_RESEARCH_API_BASE", DEFAULT_WEB_RESEARCH_API_BASE),
        web_research_timeout_seconds=max(1.0, _env_float("CODING_AGENT_WEB_RESEARCH_TIMEOUT_SECONDS", 20.0)),
        web_research_max_results=max(1, _env_int("CODING_AGENT_WEB_RESEARCH_MAX_RESULTS", 5)),
        web_research_user_agent=os.getenv("CODING_AGENT_WEB_RESEARCH_USER_AGENT", DEFAULT_WEB_RESEARCH_USER_AGENT),
    )
    config.ensure_directories()
    return config
