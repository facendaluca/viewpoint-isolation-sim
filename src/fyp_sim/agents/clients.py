from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Sampling controls the client can transmit explicitly. None means "not sent",
# so the server's own default applies — recorded as such in effective_sampling()
# rather than silently depending on LM Studio GUI state.
_OPTIONAL_SAMPLING_FIELDS = (
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "stop",
)


@dataclass(slots=True)
class OpenAICompatClient:
    """
    Minimal OpenAI-compatible client for local servers (LM Studio / vLLM / etc.)

    Expects:
        POST {base_url}/chat/completions
        -> JSON with choices[0].message.context
    """

    base_url: str  # e.g. "http://localhost:1234/v1" or "http://[IP_ADDRESS]/v1"
    model: str
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    # Optional sampling parameters; only transmitted when set, so existing
    # configs keep their exact previous request bodies.
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: list[str] | None = None
    last_usage: dict[str, int] | None = field(init=False, default=None)
    # Introspection for verification and provenance. last_request_payload is
    # the exact JSON body of the most recent request (in memory only, never
    # logged wholesale); last_response_model is the model string the server
    # reported back, which can differ from the requested id if the server
    # remaps it.
    last_request_payload: dict[str, Any] | None = field(init=False, default=None)
    last_response_model: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("OpenAICompatClient.base_url must be a non-empty string")
        self.base_url = self.base_url.rstrip("/")

    def effective_sampling(self) -> dict[str, Any]:
        """The complete sampling configuration this client sends.

        Explicit values appear as-is; parameters left to the server are marked
        "server_default" so manifests state exactly what was and was not
        pinned, instead of implying the GUI defaults were controlled.
        """
        out: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens if self.max_tokens is not None else "server_default",
        }
        for name in _OPTIONAL_SAMPLING_FIELDS:
            value = getattr(self, name)
            out[name] = value if value is not None else "server_default"
        return out

    def complete(self, prompt: str, *, timeout_s: float, request_seed: int | None = None) -> str:
        self.last_usage = None
        self.last_response_model = None
        url = self.base_url + "/chat/completions"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = int(self.max_tokens)
        for name in _OPTIONAL_SAMPLING_FIELDS:
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        if request_seed is not None:
            payload["seed"] = int(request_seed)

        self.last_request_payload = payload
        data = json.dumps(payload).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = Request(url=url, data=data, headers=headers, method="POST")

        try:
            with urlopen(req, timeout=float(timeout_s)) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            raise RuntimeError(f"LLM HTTPError {e.code}: {body}".strip()) from e
        except URLError as e:
            # Includes connection refused, DNS failure, timeouts, etc.
            raise TimeoutError(str(e.reason)) from e

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"LLM response was not JSON: {e.msg}") from e

        response_model = parsed.get("model")
        if isinstance(response_model, str) and response_model:
            self.last_response_model = response_model

        usage = parsed.get("usage")
        if isinstance(usage, dict):
            parsed_usage: dict[str, int] = {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    parsed_usage[key] = value
            if parsed_usage:
                self.last_usage = parsed_usage

        try:
            return parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError("LLM response missing choices[0].message.content") from e
