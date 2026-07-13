from __future__ import annotations

import json

from fyp_sim.agents.clients import OpenAICompatClient


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False

    def read(self) -> bytes:
        return json.dumps(
            {
                "choices": [{"message": {"content": '{"action":"Watch","confidence":1}'}}],
                "usage": {"prompt_tokens": 123, "completion_tokens": 9, "total_tokens": 132},
            }
        ).encode()


def test_openai_compat_client_preserves_provider_usage(monkeypatch) -> None:
    monkeypatch.setattr("fyp_sim.agents.clients.urlopen", lambda request, timeout: _Response())
    client = OpenAICompatClient(base_url="http://localhost:1234/v1", model="fake")

    client.complete("prompt", timeout_s=1.0)

    assert client.last_usage == {
        "prompt_tokens": 123,
        "completion_tokens": 9,
        "total_tokens": 132,
    }
