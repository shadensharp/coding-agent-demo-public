from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(slots=True)
class Event:
    event_id: str
    session_id: str
    timestamp: str
    kind: str
    payload: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "payload": dict(self.payload),
        }


def make_event(session_id: str, kind: str, payload: dict[str, object]) -> Event:
    return Event(
        event_id=new_id("evt"),
        session_id=session_id,
        timestamp=utc_now(),
        kind=kind,
        payload=payload,
    )
