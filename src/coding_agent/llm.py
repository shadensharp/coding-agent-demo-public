from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import socket
import time
from urllib import error, parse, request

from .config import AppConfig

RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(slots=True)
class _RequestFailure(Exception):
    message: str
    retryable: bool

    def __str__(self) -> str:
        return self.message


@dataclass(slots=True)
class QwenClient:
    config: AppConfig
    transport: Callable[[request.Request, float], str] | None = None
    sleep_fn: Callable[[float], None] | None = None

    def is_configured(self) -> bool:
        return bool(self.config.qwen_api_key)

    def describe(self) -> str:
        if self.is_configured():
            return f"Qwen client configured for model {self.config.qwen_model}"
        return f"Qwen client pending API key for model {self.config.qwen_model}"

    def complete(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.2) -> str:
        if not self.is_configured():
            raise RuntimeError("Qwen API key is not configured. Set QWEN_API_KEY before calling the real model.")

        body = self._request_with_retry(prompt, system_prompt=system_prompt, temperature=temperature)
        return self._parse_completion_body(body)

    def _request_with_retry(self, prompt: str, system_prompt: str | None, temperature: float) -> str:
        last_error: _RequestFailure | None = None
        total_attempts = self.config.qwen_max_retries + 1
        for attempt in range(total_attempts):
            req = self._build_request(prompt, system_prompt=system_prompt, temperature=temperature)
            try:
                return self._send_request(req)
            except _RequestFailure as exc:
                last_error = exc
                if attempt >= self.config.qwen_max_retries or not exc.retryable:
                    raise RuntimeError(str(exc)) from exc
                self._sleep(self._retry_delay_seconds(attempt))

        if last_error is None:
            raise RuntimeError("Qwen request failed before any attempt completed.")
        raise RuntimeError(str(last_error)) from last_error

    def _build_request(self, prompt: str, system_prompt: str | None, temperature: float) -> request.Request:
        url = parse.urljoin(self.config.qwen_api_base.rstrip("/") + "/", "chat/completions")
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.config.qwen_model,
            "messages": messages,
            "temperature": temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        return request.Request(
            url=url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.config.qwen_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

    def _send_request(self, req: request.Request) -> str:
        try:
            if self.transport is not None:
                return self.transport(req, self.config.qwen_timeout_seconds)
            with request.urlopen(req, timeout=self.config.qwen_timeout_seconds) as response:
                return response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in RETRYABLE_HTTP_STATUS_CODES
            raise _RequestFailure(
                f"Qwen request failed with HTTP {exc.code}: {details}",
                retryable=retryable,
            ) from exc
        except error.URLError as exc:
            raise _RequestFailure(f"Qwen request failed: {exc.reason}", retryable=True) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise _RequestFailure("Qwen request timed out.", retryable=True) from exc

    def _retry_delay_seconds(self, attempt: int) -> float:
        return self.config.qwen_retry_backoff_seconds * (2 ** attempt)

    def _sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        if self.sleep_fn is not None:
            self.sleep_fn(seconds)
            return
        time.sleep(seconds)

    def _parse_completion_body(self, body: str) -> str:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Qwen response was not valid JSON: {body[:200]}") from exc

        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Qwen response did not include choices: {parsed}")

        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            raise RuntimeError(f"Qwen choice payload is malformed: {choices[0]}")

        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            text = "".join(parts)
        else:
            raise RuntimeError(f"Qwen message content is malformed: {message}")

        text = text.strip()
        if not text:
            raise RuntimeError("Qwen response content was empty.")
        return text
