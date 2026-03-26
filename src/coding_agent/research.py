from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from html import unescape
import re
import socket
from urllib import error, parse, request

from .config import AppConfig
from .models import ResearchSource

_RESULT_RE = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>(?P<tail>.*?)(?=<a[^>]*class="[^"]*result__a[^"]*"|\Z)',
    re.IGNORECASE | re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</(?:a|div|span)>',
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class WebResearchClient:
    config: AppConfig
    transport: Callable[[request.Request, float], str] | None = None

    def search(self, query: str, max_results: int | None = None) -> list[ResearchSource]:
        normalized_query = " ".join(str(query).strip().split())
        if not normalized_query:
            return []

        limit = max(1, min(max_results or self.config.web_research_max_results, 8))
        request_obj = self._build_request(normalized_query)
        html = self._send_request(request_obj)
        return self._parse_results(html, limit)

    def _build_request(self, query: str) -> request.Request:
        params = parse.urlencode({"q": query, "kl": "us-en"})
        base = self.config.web_research_api_base.rstrip("?")
        separator = "&" if "?" in base else "?"
        url = f"{base}{separator}{params}"
        return request.Request(
            url=url,
            headers={
                "User-Agent": self.config.web_research_user_agent,
                "Accept": "text/html,application/xhtml+xml",
            },
            method="GET",
        )

    def _send_request(self, req: request.Request) -> str:
        try:
            if self.transport is not None:
                return str(self.transport(req, self.config.web_research_timeout_seconds))
            with request.urlopen(req, timeout=self.config.web_research_timeout_seconds) as response:
                return response.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"External research request failed with HTTP {exc.code}: {details[:200]}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"External research request failed: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError("External research request timed out.") from exc

    def _parse_results(self, html: str, limit: int) -> list[ResearchSource]:
        results: list[ResearchSource] = []
        seen_urls: set[str] = set()
        for match in _RESULT_RE.finditer(html):
            url = self._normalize_url(match.group("href"))
            if not url or url in seen_urls:
                continue
            title = self._html_to_text(match.group("title"))
            snippet_match = _SNIPPET_RE.search(match.group("tail"))
            snippet = self._html_to_text(snippet_match.group("snippet")) if snippet_match else ""
            if not title:
                continue
            results.append(ResearchSource(title=title, url=url, snippet=snippet))
            seen_urls.add(url)
            if len(results) >= limit:
                break
        return results

    def _normalize_url(self, raw_url: str) -> str:
        href = unescape(raw_url).strip()
        if not href:
            return ""
        if href.startswith("//"):
            href = "https:" + href
        parsed = parse.urlparse(href)
        if "duckduckgo.com" in parsed.netloc:
            query = parse.parse_qs(parsed.query)
            uddg = query.get("uddg", [])
            if uddg:
                return parse.unquote(uddg[0])
        return href

    def _html_to_text(self, value: str) -> str:
        text = _TAG_RE.sub(" ", unescape(value))
        return _WHITESPACE_RE.sub(" ", text).strip()
