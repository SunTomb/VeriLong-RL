"""OpenAI-compatible API client for VeriLong-RL offline evaluation.

Many gateways (including the temporary Gemini-backed proxy used for pilot
testing) expose an OpenAI-compatible ``/v1/chat/completions`` endpoint. This
client targets that contract directly with ``urllib`` (no extra SDK dependency)
and returns the same :class:`ClaudeResponse` shape the runner already consumes,
so the rest of the pipeline (prompt construction, caching, output formatting,
scoring) stays provider-agnostic.

Configuration precedence for base URL / key:

- explicit constructor args, else
- ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` environment variables.

Sampling defaults: a low temperature is used for grounded, repeatable answers.
``-thinking`` model variants are supported; their hidden reasoning is not part
of ``message.content`` and is therefore never treated as the answer.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from experiments.eval_api.claude_client import ClaudeResponse


class OpenAICompatibleClient:
    """Minimal OpenAI Chat Completions client over an arbitrary base URL."""

    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 90.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        base = base_url or os.environ.get("OPENAI_BASE_URL") or ""
        self.base_url = base.rstrip("/")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

    # -- request construction -------------------------------------------------

    def build_payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

    # -- live call ------------------------------------------------------------

    def complete(self, system_prompt: str, user_prompt: str) -> ClaudeResponse:
        if not self.base_url:
            raise RuntimeError("base_url is not set (pass --base-url or set OPENAI_BASE_URL).")
        if not self._api_key:
            raise RuntimeError("api key is not set (pass --api-key or set OPENAI_API_KEY).")

        payload = json.dumps(self.build_payload(system_prompt, user_prompt)).encode("utf-8")
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return ClaudeResponse(
                    output_text=_extract_choice_text(body),
                    raw_metadata=_response_metadata(body),
                )
            except urllib.error.HTTPError as exc:  # pragma: no cover - network dependent
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise last_error from exc
            except urllib.error.URLError as exc:  # pragma: no cover - network dependent
                last_error = RuntimeError(f"connection error: {exc.reason}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise last_error from exc

        assert last_error is not None
        raise last_error


def _extract_choice_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    # Some gateways return content as a list of parts.
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "".join(parts).strip()
    return ""


def _response_metadata(body: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("id", "model"):
        if body.get(key) is not None:
            metadata[key] = body[key]
    choices = body.get("choices") or []
    if choices and choices[0].get("finish_reason") is not None:
        metadata["finish_reason"] = choices[0]["finish_reason"]
    if body.get("usage") is not None:
        metadata["usage"] = body["usage"]
    return metadata
