"""Deterministic derivation of per-request sampling seeds for LLM calls.

The experiment keeps stochastic sampling (temperature and friends stay
whatever the config says), but every logical request gets a reproducibly
derived sampling seed. Rerunning the same configuration regenerates the same
seeds; different agents, simulation seeds, steps and call roles get different
seeds; matched contexts across comparison arms or sweep cells share a seed
because the arm/cell label is deliberately not part of the derivation.

Derivation: canonical JSON (sorted keys, compact separators, UTF-8) of the
structured fields below, SHA-256, first four digest bytes big-endian, masked
to 31 bits. The result sits in 0..2^31-1, the conservative non-negative range
accepted by OpenAI-compatible `seed` fields. Python's builtin hash() is never
involved, so PYTHONHASHSEED and process boundaries cannot change the stream.

Fields (schema_version 1):
    stream       comparison-group / stochastic-stream label. The default
                 "decision" stream is shared: two arms evaluating the same
                 logical context draw the same seed. A design that needs
                 independent draws opts out by naming a different stream.
    experiment_seed  the master simulation seed of the current run.
    agent_id     stable logical agent id ("user" for single-user configs).
    step         simulation timestep of the call.
    call_role    semantic role: "rerank_candidate", "serve_decision",
                 "probe", or "repair".
    draw_index   stable index distinguishing repeated calls of one role at
                 one step; the candidate's video_id for decision calls.
    attempt      0 for the request and any transport-level retry of it (a
                 retry re-sends the same logical request, so it reuses the
                 seed). A deliberate semantic repair call is a different
                 logical request and must use call_role="repair" instead of
                 bumping attempt.

Bumping REQUEST_SEED_SCHEMA_VERSION changes every derived seed, so it must
only happen with a documented methodology change; manifests record the
version for exactly that reason.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

REQUEST_SEED_SCHEMA_VERSION = 1

# OpenAI-compatible `seed` fields accept at least a non-negative int32.
_SEED_BITS = 31
_SEED_MASK = (1 << _SEED_BITS) - 1

VALID_CALL_ROLES = ("rerank_candidate", "serve_decision", "probe", "repair")


def derive_request_seed(
    *,
    experiment_seed: int,
    agent_id: str,
    step: int,
    call_role: str,
    draw_index: int,
    attempt: int = 0,
    stream: str = "decision",
    schema_version: int = REQUEST_SEED_SCHEMA_VERSION,
) -> int:
    """Derive the sampling seed for one logical LLM request.

    Deterministic across processes and machines; every field must be a stable
    experiment-level value (never timestamps, PIDs, paths or object ids).
    """
    if call_role not in VALID_CALL_ROLES:
        raise ValueError(f"call_role must be one of {VALID_CALL_ROLES}, got {call_role!r}")
    if not agent_id:
        raise ValueError("agent_id must be a non-empty string")
    payload = {
        "schema_version": int(schema_version),
        "stream": str(stream),
        "experiment_seed": int(experiment_seed),
        "agent_id": str(agent_id),
        "step": int(step),
        "call_role": str(call_role),
        "draw_index": int(draw_index),
        "attempt": int(attempt),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    return int.from_bytes(digest[:4], "big") & _SEED_MASK


@dataclass(slots=True)
class RequestSeedMonitor:
    """Detects distinct logical requests colliding on one derived seed.

    A 31-bit seed space makes rare birthday collisions possible (~0.013% for
    the 750 calls of one production compare seed-run). Two different prompts
    sharing a sampling seed is statistically harmless, so a collision warns
    loudly and is counted rather than aborting a multi-hour run — but it is
    never silent. Reusing the same seed for the *same* logical request (a
    transport retry) is correct and not a collision.
    """

    _seen: dict[int, tuple] = field(default_factory=dict)
    collisions: int = 0

    def check(
        self,
        seed: int,
        *,
        experiment_seed: int,
        agent_id: str,
        step: int,
        call_role: str,
        draw_index: int,
        attempt: int = 0,
        stream: str = "decision",
    ) -> bool:
        """Record one derivation; return True when a genuine collision occurred."""
        key = (
            int(experiment_seed),
            str(agent_id),
            int(step),
            str(call_role),
            int(draw_index),
            int(attempt),
            str(stream),
        )
        previous = self._seen.get(seed)
        if previous is None:
            self._seen[seed] = key
            return False
        if previous == key:
            return False
        self.collisions += 1
        logger.warning(
            "request-seed collision: seed %d derived for %s and %s; "
            "recording and continuing (see request_seed.py collision policy)",
            seed,
            previous,
            key,
        )
        return True

    def reset(self) -> None:
        self._seen.clear()
