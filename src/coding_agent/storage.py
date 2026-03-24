from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .events import Event
from .models import Session


class RunStore(Protocol):
    def append_event(self, event: Event) -> None: ...
    def save_summary(self, session: Session) -> None: ...
    def load_summary(self, session_id: str) -> dict[str, object]: ...
    def load_events(self, session_id: str) -> list[dict[str, object]]: ...
    def list_sessions(self, limit: int = 20) -> list[dict[str, object]]: ...


class JsonlRunStore:
    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir.resolve()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        path = self.sessions_dir / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _events_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "events.jsonl"

    def _summary_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "summary.json"

    def append_event(self, event: Event) -> None:
        events_path = self._events_path(event.session_id)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def save_summary(self, session: Session) -> None:
        summary_path = self._summary_path(session.session_id)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(session.to_dict(), handle, ensure_ascii=False, indent=2)

    def load_summary(self, session_id: str) -> dict[str, object]:
        summary_path = self._summary_path(session_id)
        if not summary_path.exists():
            raise FileNotFoundError(f"Session summary not found: {session_id}")
        with summary_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_events(self, session_id: str) -> list[dict[str, object]]:
        events_path = self._events_path(session_id)
        if not events_path.exists():
            raise FileNotFoundError(f"Session events not found: {session_id}")
        events: list[dict[str, object]] = []
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
        return events

    def list_sessions(self, limit: int = 20) -> list[dict[str, object]]:
        summaries: list[dict[str, object]] = []
        for summary_path in self.sessions_dir.glob("*/summary.json"):
            with summary_path.open("r", encoding="utf-8") as handle:
                summaries.append(json.load(handle))
        summaries.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return summaries[:limit]
