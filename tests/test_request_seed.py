from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from fyp_sim.agents.clients import OpenAICompatClient
from fyp_sim.agents.deciders import LLMDecider
from fyp_sim.llm.request_seed import (
    REQUEST_SEED_SCHEMA_VERSION,
    RequestSeedMonitor,
    derive_request_seed,
)
from fyp_sim.models import User, UserPhenotype, Video
from fyp_sim.simulation.engine import run_simulation

_BASE_IDENTITY = {
    "experiment_seed": 7,
    "agent_id": "user",
    "step": 3,
    "call_role": "rerank_candidate",
    "draw_index": 553,
}


# ---------------------------------------------------------------------------
# Derivation function
# ---------------------------------------------------------------------------


def test_same_inputs_same_seed_across_processes_and_hash_randomisation() -> None:
    expected = derive_request_seed(**_BASE_IDENTITY)
    code = (
        "from fyp_sim.llm.request_seed import derive_request_seed;"
        f"print(derive_request_seed(**{_BASE_IDENTITY!r}))"
    )
    for hash_seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        assert int(out.stdout.strip()) == expected


def test_each_identity_field_changes_the_seed() -> None:
    base = derive_request_seed(**_BASE_IDENTITY)
    variants = [
        {**_BASE_IDENTITY, "experiment_seed": 8},
        {**_BASE_IDENTITY, "agent_id": "agent_b"},
        {**_BASE_IDENTITY, "step": 4},
        {**_BASE_IDENTITY, "call_role": "serve_decision"},
        {**_BASE_IDENTITY, "draw_index": 554},
        {**_BASE_IDENTITY, "attempt": 1},
        {**_BASE_IDENTITY, "stream": "arm:independent"},
    ]
    seeds = [derive_request_seed(**variant) for variant in variants]
    assert all(seed != base for seed in seeds)
    assert len(set(seeds)) == len(seeds)


def test_keyword_order_cannot_change_the_seed() -> None:
    forward = dict(_BASE_IDENTITY)
    reordered = dict(reversed(list(_BASE_IDENTITY.items())))
    assert derive_request_seed(**forward) == derive_request_seed(**reordered)


def test_matched_streams_share_and_independent_streams_differ() -> None:
    # Two comparison arms evaluating the same logical context: the arm label is
    # not an input, so both derive the identical seed (matched draw).
    arm_a = derive_request_seed(**_BASE_IDENTITY)
    arm_b = derive_request_seed(**_BASE_IDENTITY)
    assert arm_a == arm_b
    # A design that wants independent draws opts out with a named stream.
    independent = derive_request_seed(**_BASE_IDENTITY, stream="arm:llm_b")
    assert independent != arm_a


def test_transport_retry_reuses_and_repair_gets_its_own_seed() -> None:
    first = derive_request_seed(**_BASE_IDENTITY)
    retry = derive_request_seed(**_BASE_IDENTITY)  # same logical request, attempt unchanged
    assert retry == first
    repair = derive_request_seed(**{**_BASE_IDENTITY, "call_role": "repair"})
    assert repair != first


def test_seed_range_and_schema_version_sensitivity() -> None:
    rng = random.Random(0)
    for _ in range(200):
        seed = derive_request_seed(
            experiment_seed=rng.randrange(10_000),
            agent_id=f"agent_{rng.randrange(50)}",
            step=rng.randrange(500),
            call_role="serve_decision",
            draw_index=rng.randrange(100_000),
        )
        assert 0 <= seed <= 2**31 - 1
    bumped = derive_request_seed(**_BASE_IDENTITY, schema_version=REQUEST_SEED_SCHEMA_VERSION + 1)
    assert bumped != derive_request_seed(**_BASE_IDENTITY)


def test_invalid_call_role_fails_loudly() -> None:
    with pytest.raises(ValueError, match="call_role"):
        derive_request_seed(**{**_BASE_IDENTITY, "call_role": "banana"})


def test_monitor_flags_only_genuine_collisions(caplog) -> None:
    monitor = RequestSeedMonitor()
    assert monitor.check(123, **_BASE_IDENTITY) is False
    # The same logical request again (a transport retry) is not a collision.
    assert monitor.check(123, **_BASE_IDENTITY) is False
    assert monitor.collisions == 0
    with caplog.at_level("WARNING"):
        collided = monitor.check(123, **{**_BASE_IDENTITY, "draw_index": 999})
    assert collided is True
    assert monitor.collisions == 1
    assert any("request-seed collision" in message for message in caplog.messages)


# ---------------------------------------------------------------------------
# Decider wiring
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SeedAwareFakeClient:
    output: str = '{"action":"Watch","confidence":0.9}'
    received_seeds: list = field(default_factory=list)

    def complete(self, prompt: str, *, timeout_s: float, request_seed: int | None = None) -> str:  # noqa: ARG002
        self.received_seeds.append(request_seed)
        return self.output


@dataclass(slots=True)
class LegacyFakeClient:
    output: str = '{"action":"Watch","confidence":0.9}'
    calls: int = 0

    def complete(self, prompt: str, *, timeout_s: float) -> str:  # noqa: ARG002
        self.calls += 1
        return self.output


def _make_user_video() -> tuple[User, Video]:
    user = User(
        phenotype=UserPhenotype.WATCHER,
        viewpoint_score=0.2,
        interest_vector={"sports": 0.8},
        sentiment_threshold=-1.0,
    )
    return user, Video(553, "sports", 0.5, 0.0, 30, tags=("meme",))


def test_decider_derives_transmits_and_logs_the_seed() -> None:
    user, video = _make_user_video()
    client = SeedAwareFakeClient()
    decider = LLMDecider(prompt_id="decision_v4", client=client, timeout_s=1.0)

    decider.set_request_context(experiment_seed=7, agent_id="user", stream="decision", step=3)
    decider.set_request_context(call_role="rerank_candidate", draw_index=video.video_id)
    decider.decide_next_action(user, video)

    expected = derive_request_seed(**_BASE_IDENTITY)
    assert client.received_seeds == [expected]
    meta = decider.last_meta
    assert meta.request_seed == expected
    assert meta.request_seed_sent is True
    assert meta.call_role == "rerank_candidate"
    assert len(meta.prompt_sha256) == 64
    assert len(meta.response_sha256) == 64
    stats = decider.diagnostics_snapshot()
    assert stats["llm_seeded_request_count"] == 1
    assert stats["llm_seed_collision_count"] == 0

    # The call scope was consumed: a follow-up call without fresh call scope
    # must not inherit the previous identity.
    decider.decide_next_action(user, video)
    assert client.received_seeds[1] is None
    assert decider.last_meta.request_seed is None


def test_decider_keeps_working_with_legacy_clients() -> None:
    user, video = _make_user_video()
    client = LegacyFakeClient()
    decider = LLMDecider(prompt_id="decision_v4", client=client, timeout_s=1.0)
    decider.set_request_context(
        experiment_seed=7, agent_id="user", step=3, call_role="serve_decision", draw_index=553
    )

    decider.decide_next_action(user, video)

    assert client.calls == 1
    meta = decider.last_meta
    assert meta.request_seed is not None  # derived and logged
    assert meta.request_seed_sent is False  # but not transmitted
    assert decider.diagnostics_snapshot()["llm_seeded_request_count"] == 0


def test_decider_rejects_unknown_context_fields() -> None:
    decider = LLMDecider(client=SeedAwareFakeClient())
    with pytest.raises(ValueError, match="unknown request-context"):
        decider.set_request_context(arm="llm")


# ---------------------------------------------------------------------------
# Engine wiring: reruns regenerate identical per-call seeds
# ---------------------------------------------------------------------------


def _tiny_pool() -> list[Video]:
    return [
        Video(1, "sports", 0.0, 0.0, 20, tags=("meme",)),
        Video(2, "politics", 0.5, 0.0, 30, tags=()),
        Video(3, "sports", 1.0, 0.0, 40, tags=("viral",)),
    ]


def _run_once(experiment_seed: int) -> list[tuple[int, int, int | None]]:
    from fyp_sim.candidate_trace import CandidateTraceCollector

    user, _ = _make_user_video()
    client = SeedAwareFakeClient()
    decider = LLMDecider(prompt_id="decision_v4", client=client, timeout_s=1.0)
    decider.set_request_context(experiment_seed=experiment_seed, agent_id="user", stream="decision")
    collector = CandidateTraceCollector(seed=experiment_seed)
    run_simulation(
        user=user,
        video_pool=_tiny_pool(),
        steps=4,
        rng=random.Random(experiment_seed),
        top_k=2,
        rank_alpha=0.3,
        decider=decider,
        llm_rerank=True,
        candidate_trace=collector,
    )
    return [(row.t, row.video_id, row.request_seed) for row in collector.rows]


def test_rerun_regenerates_identical_request_seeds() -> None:
    first = _run_once(11)
    second = _run_once(11)
    assert first == second
    assert all(seed is not None for _, _, seed in first)
    # Every (step, video) pair carries its own seed value.
    assert len({seed for _, _, seed in first}) == len(first)
    # A different simulation seed produces a different stream.
    other = _run_once(12)
    assert {seed for _, _, seed in other}.isdisjoint({seed for _, _, seed in first})


# ---------------------------------------------------------------------------
# Request body: what actually reaches the server
# ---------------------------------------------------------------------------


class _CaptureHandler(BaseHTTPRequestHandler):
    captured: list[dict] = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        type(self).captured.append(json.loads(self.rfile.read(length)))
        body = json.dumps(
            {
                "model": "stub-model",
                "choices": [{"message": {"content": '{"action":"Watch","confidence":0.9}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: ANN002
        pass


@pytest.fixture()
def capture_server():
    _CaptureHandler.captured = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", _CaptureHandler.captured
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_request_body_contains_seed_and_pinned_sampling(capture_server) -> None:
    base_url, captured = capture_server
    client = OpenAICompatClient(
        base_url=base_url,
        model="stub-model",
        temperature=0.7,
        max_tokens=64,
        top_p=0.9,
        top_k=40,
    )
    client.complete("fixture prompt", timeout_s=5.0, request_seed=123456789)

    assert len(captured) == 1
    body = captured[0]
    assert body["seed"] == 123456789
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 64
    assert body["top_p"] == 0.9
    assert body["top_k"] == 40
    assert client.last_request_payload == body
    assert client.last_response_model == "stub-model"


def test_request_body_unchanged_for_existing_configs_except_seed(capture_server) -> None:
    # The frozen production configs pin only temperature and max_tokens; the
    # client must not invent any other sampling value for them.
    base_url, captured = capture_server
    repo_root = Path(__file__).resolve().parents[1]
    frozen = json.loads((repo_root / "configs" / "experiment_compare.json").read_text())
    llm_cfg = frozen["policy"]["llm"]

    client = OpenAICompatClient(
        base_url=base_url,
        model=llm_cfg["model"],
        temperature=float(llm_cfg["temperature"]),
        max_tokens=int(llm_cfg["max_tokens"]),
    )
    client.complete("fixture prompt", timeout_s=5.0)

    body = captured[0]
    assert set(body) == {"model", "messages", "temperature", "max_tokens"}
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 200
    sampling = client.effective_sampling()
    assert sampling["temperature"] == 0.0
    assert sampling["top_p"] == "server_default"
    assert sampling["top_k"] == "server_default"

    client.complete("fixture prompt", timeout_s=5.0, request_seed=42)
    assert set(captured[1]) == {"model", "messages", "temperature", "max_tokens", "seed"}
    assert captured[1]["seed"] == 42


def test_decider_to_server_end_to_end_seed(capture_server) -> None:
    base_url, captured = capture_server
    user, video = _make_user_video()
    client = OpenAICompatClient(base_url=base_url, model="stub-model")
    decider = LLMDecider(prompt_id="decision_v4", client=client, timeout_s=5.0)
    decider.set_request_context(
        experiment_seed=7, agent_id="user", step=3, call_role="rerank_candidate", draw_index=553
    )

    decider.decide_next_action(user, video)

    assert captured[0]["seed"] == derive_request_seed(**_BASE_IDENTITY)
    assert decider.last_meta.request_seed_sent is True
