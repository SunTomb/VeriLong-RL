"""Claude API client for VeriLong-RL offline evaluation.

This module isolates all Claude-specific request construction so the rest of
the eval pipeline stays provider-agnostic. It follows the constraints recorded
in ``experiments/eval_api/README.md``:

- Default strong model: ``claude-opus-4-8``.
- Adaptive thinking via ``thinking={"type": "adaptive"}``.
- Never set ``budget_tokens`` on Opus 4.8/4.7 / Fable 5.
- Never set ``temperature``/``top_p``/``top_k`` on Opus 4.8/4.7 / Fable 5.
- Stream long-context / large ``max_tokens`` requests.
- Do not use assistant prefill to force output shape.

The Anthropic SDK is imported lazily so that the surrounding pipeline (prompt
construction, caching, output formatting, scoring) can be exercised with
``--dry-run`` without the SDK installed or an API key configured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Models that reject sampling params and budget_tokens. Kept as a set so the
# guardrails below stay declarative and easy to extend.
NO_SAMPLING_MODELS = {
    "claude-opus-4-8",
    "claude-opus-4-7",
    "fable-5",
}

# Stream when the requested completion size is large enough that a single
# blocking request risks a network/SDK timeout.
STREAM_MAX_TOKENS_THRESHOLD = 4096


@dataclass
class ClaudeResponse:
    """Normalized response returned to the runner."""

    output_text: str
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class ClaudeClient:
    """Thin wrapper over the Anthropic Messages API.

    Parameters
    ----------
    model:
        Target Claude model. Defaults to ``claude-opus-4-8``.
    max_tokens:
        Completion budget. Streaming is enabled automatically above
        :data:`STREAM_MAX_TOKENS_THRESHOLD`.
    api_key:
        Optional explicit key. Falls back to ``ANTHROPIC_API_KEY``.
    adaptive_thinking:
        Whether to send ``thinking={"type": "adaptive"}``.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        max_tokens: int = 2048,
        api_key: str | None = None,
        adaptive_thinking: bool = True,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.adaptive_thinking = adaptive_thinking
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Any | None = None

    # -- request construction -------------------------------------------------

    def build_request_kwargs(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Build the keyword arguments for ``messages.create``.

        Sampling parameters and ``budget_tokens`` are intentionally omitted for
        models in :data:`NO_SAMPLING_MODELS`. No assistant prefill message is
        added, so the model is never forced into an output shape.
        """

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        if self.adaptive_thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        # For models outside the no-sampling set we still avoid setting sampling
        # params here: the eval contract wants deterministic-by-default grounded
        # answers, and the README forbids them for the default models. We leave
        # the hook explicit so future non-Opus models can opt in deliberately.
        if self.model not in NO_SAMPLING_MODELS:
            # Deliberately left empty: do not inject temperature/top_p/top_k
            # unless a future model is explicitly added with vetted defaults.
            pass

        return kwargs

    def should_stream(self) -> bool:
        return self.max_tokens >= STREAM_MAX_TOKENS_THRESHOLD

    # -- live call ------------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; cannot make live API calls. "
                "Use --dry-run to exercise the pipeline without spending budget."
            )
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "The 'anthropic' package is not installed. Install it before a "
                "live run, or use --dry-run to exercise the pipeline offline."
            ) from exc
        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> ClaudeResponse:
        """Make a single live completion request.

        Streaming is used automatically for large ``max_tokens``. The text is
        flattened from all returned text content blocks; ``thinking`` blocks are
        not treated as the answer.
        """

        client = self._ensure_client()
        kwargs = self.build_request_kwargs(system_prompt, user_prompt)

        if self.should_stream():
            return self._complete_streaming(client, kwargs)
        return self._complete_blocking(client, kwargs)

    def _complete_blocking(self, client: Any, kwargs: dict[str, Any]) -> ClaudeResponse:
        message = client.messages.create(**kwargs)
        return ClaudeResponse(
            output_text=_extract_text(message),
            raw_metadata=_response_metadata(message),
        )

    def _complete_streaming(self, client: Any, kwargs: dict[str, Any]) -> ClaudeResponse:
        with client.messages.stream(**kwargs) as stream:
            for _ in stream.text_stream:
                # Drain the stream; the SDK accumulates the final message.
                pass
            message = stream.get_final_message()
        return ClaudeResponse(
            output_text=_extract_text(message),
            raw_metadata=_response_metadata(message),
        )


def _extract_text(message: Any) -> str:
    """Concatenate text blocks from a Messages API response object."""

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if not content:
        return ""

    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
        if block_type != "text":
            continue
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _response_metadata(message: Any) -> dict[str, Any]:
    """Best-effort extraction of non-content response metadata."""

    metadata: dict[str, Any] = {}
    for key in ("id", "model", "stop_reason"):
        value = getattr(message, key, None)
        if value is None and isinstance(message, dict):
            value = message.get(key)
        if value is not None:
            metadata[key] = value

    usage = getattr(message, "usage", None)
    if usage is not None:
        if hasattr(usage, "model_dump"):
            metadata["usage"] = usage.model_dump()
        elif isinstance(usage, dict):
            metadata["usage"] = usage
        else:
            metadata["usage"] = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            }
    return metadata
