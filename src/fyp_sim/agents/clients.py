from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(slots=True)
class OpenAICompactClient:
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

    def complete(self, prompt: str, *, timeout_s: float) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = int(self.max_tokens)

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

        try:
            return parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError("LLM response missing choices[0].message.content") from e
