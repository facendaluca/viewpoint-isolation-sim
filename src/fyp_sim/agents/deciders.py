from __future__ import annotations

import hashlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from fyp_sim.llm.decision_contract import DecisionValidationError, parse_decision_json
from fyp_sim.llm.prompting import render_decision_prompt
from fyp_sim.llm.request_seed import RequestSeedMonitor, derive_request_seed
from fyp_sim.models import User, UserAction, Video
from fyp_sim.policy import decide_action

logger = logging.getLogger(__name__)

# Which request-context keys live how long. Run-scoped keys are set once per
# (simulation seed, agent) by the runner; the step is refreshed by the engine
# every timestep; call-scoped keys identify one request and are consumed by it,
# so a later call can never silently inherit another call's identity.
_RUN_SCOPED_KEYS = ("experiment_seed", "agent_id", "stream")
_STEP_SCOPED_KEYS = ("step",)
_CALL_SCOPED_KEYS = ("call_role", "draw_index", "attempt")
_CONTEXT_KEYS = _RUN_SCOPED_KEYS + _STEP_SCOPED_KEYS + _CALL_SCOPED_KEYS


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class DecisionMeta:
    policy_mode: str
    prompt_id: str | None = None
    valid: bool = True
    fallback_reason: str = ""
    llm_action: str = ""
    llm_confidence: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    token_count_estimated: bool = False
    # Request-seed provenance: the derived sampling seed, whether the client
    # actually transmitted it, the semantic role of the call, and content
    # hashes that identify the exact request/response without logging them.
    request_seed: int | None = None
    request_seed_sent: bool = False
    call_role: str = ""
    prompt_sha256: str = ""
    response_sha256: str = ""


class ActionDecider(Protocol):
    """Single deision interface used by the simulation loop."""

    def decide_next_action(self, user: User, video: Video) -> UserAction: ...


class LLMClient(Protocol):
    """Provider-agnostic LLM client interface (implemented later.

    Clients may additionally accept a keyword-only `request_seed: int | None`;
    LLMDecider inspects the signature once and only passes the seed to clients
    that declare it, so older client doubles keep working unchanged.
    """

    def complete(self, prompt: str, *, timeout_s: float) -> str: ...


@dataclass(slots=True)
class HeuristicDecider:
    """Adapter around the existing heuristic policy (baseline / deterministic)."""

    last_meta: DecisionMeta = field(
        default_factory=lambda: DecisionMeta(policy_mode="heuristic", valid=True)
    )

    def decide_next_action(self, user: User, video: Video) -> UserAction:
        self.last_meta = DecisionMeta(policy_mode="heuristic", valid=True)
        return decide_action(user, video)


def _extract_first_json_object(text: str) -> str:
    """
    Best-effort extraction of the first JSON object from a model response.

    Many local models sometimes preprend/append extra text. We try to salvage the first {...}.
    If extraction fails, return the original text (which will then fail validation cleanly).
    """

    if not text:
        return text

    s = text.strip()

    start = s.find("{")
    if start == -1:
        return s

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    return s[start:]


@dataclass(slots=True)
class LLMDecider:
    """
    LLM-backed decider with robust fallback to heuristic

    - Builds a prompt from a versioned template (prompt_id)
    - Calls local model via LLMClient.complete(...)
    - Parses + validates output via Decision Contract
    - On failure: logs fallback reason and uses heuristic
    """

    prompt_id: str = "decision_v1"
    client: LLMClient | None = None
    timeout_s: float = 10.0
    fallback: ActionDecider = field(default_factory=HeuristicDecider)
    last_meta: DecisionMeta = field(default_factory=lambda: DecisionMeta(policy_mode="llm"))

    _warned_no_client: bool = False
    _warned_unreachable: bool = False
    _calls_total: int = 0
    _valid_total: int = 0
    _fallback_total: int = 0
    _prompt_tokens_total: int = 0
    _completion_tokens_total: int = 0
    _total_tokens_total: int = 0
    _token_estimated_calls: int = 0
    _fallback_reasons: dict[str, int] = field(default_factory=dict)
    # Request-seed state: the merged identity context (see set_request_context),
    # the per-run collision monitor, how many requests actually carried a seed,
    # and whether this client's complete() accepts one (inspected once).
    _request_context: dict[str, Any] = field(default_factory=dict)
    _seed_monitor: RequestSeedMonitor = field(default_factory=RequestSeedMonitor)
    _seeded_total: int = 0
    _accepts_seed: bool | None = None

    def set_request_context(self, **fields: Any) -> None:
        """Merge request-identity fields used to derive per-call sampling seeds.

        Runners set the run scope (experiment_seed, agent_id, stream) once per
        simulation run; the engine refreshes `step` every timestep and the
        call scope (call_role, draw_index, attempt) before each request. The
        call scope is consumed by the next request so no call can silently
        inherit another call's identity. Changing the run scope resets the
        per-run seed-collision monitor.
        """
        unknown = sorted(set(fields) - set(_CONTEXT_KEYS))
        if unknown:
            raise ValueError(f"unknown request-context field(s): {', '.join(unknown)}")
        run_scope_changed = any(
            key in fields and fields[key] != self._request_context.get(key)
            for key in _RUN_SCOPED_KEYS
        )
        if run_scope_changed:
            self._seed_monitor.reset()
        self._request_context.update(fields)

    def _next_call_identity(self) -> tuple[str, int | None]:
        """Consume the call scope and derive this request's sampling seed.

        Returns (call_role, seed); seed is None when the identity is
        incomplete (e.g. a direct decide_next_action call outside a runner),
        in which case the request goes out unseeded and is logged as such.
        """
        ctx = self._request_context
        call_role = str(ctx.pop("call_role", "") or "")
        draw_index = ctx.pop("draw_index", None)
        attempt = int(ctx.pop("attempt", 0) or 0)
        required = ("experiment_seed", "agent_id", "step")
        if not call_role or draw_index is None or any(ctx.get(key) is None for key in required):
            return call_role, None
        identity = {
            "experiment_seed": int(ctx["experiment_seed"]),
            "agent_id": str(ctx["agent_id"]),
            "step": int(ctx["step"]),
            "call_role": call_role,
            "draw_index": int(draw_index),
            "attempt": attempt,
            "stream": str(ctx.get("stream", "decision")),
        }
        seed = derive_request_seed(**identity)
        self._seed_monitor.check(seed, **identity)
        return call_role, seed

    def _client_accepts_request_seed(self) -> bool:
        if self._accepts_seed is None:
            try:
                params = inspect.signature(self.client.complete).parameters
                self._accepts_seed = "request_seed" in params or any(
                    p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
            except (TypeError, ValueError):
                self._accepts_seed = False
        return bool(self._accepts_seed)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return (len(text) + 3) // 4 if text else 0

    def _resolve_usage(self, prompt: str, raw: str) -> tuple[int, int, int, bool]:
        usage = getattr(self.client, "last_usage", None)
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                if not isinstance(total_tokens, int):
                    total_tokens = prompt_tokens + completion_tokens
                return prompt_tokens, completion_tokens, total_tokens, False

        prompt_tokens = self._estimate_tokens(prompt)
        completion_tokens = self._estimate_tokens(raw)
        return prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, True

    def _record_call(
        self,
        *,
        valid: bool,
        fallback_reason: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        token_count_estimated: bool = False,
    ) -> None:
        self._calls_total += 1
        self._valid_total += int(valid)
        self._prompt_tokens_total += prompt_tokens
        self._completion_tokens_total += completion_tokens
        self._total_tokens_total += total_tokens
        self._token_estimated_calls += int(token_count_estimated)
        if fallback_reason:
            self._fallback_total += 1
            self._fallback_reasons[fallback_reason] = (
                self._fallback_reasons.get(fallback_reason, 0) + 1
            )

    def diagnostics_snapshot(self) -> dict[str, int]:
        return {
            "llm_call_count": self._calls_total,
            "llm_valid_count": self._valid_total,
            "llm_fallback_count": self._fallback_total,
            "llm_retry_count": 0,
            "llm_seeded_request_count": self._seeded_total,
            "llm_seed_collision_count": self._seed_monitor.collisions,
            "llm_prompt_tokens": self._prompt_tokens_total,
            "llm_completion_tokens": self._completion_tokens_total,
            "llm_total_tokens": self._total_tokens_total,
            "llm_token_estimated_calls": self._token_estimated_calls,
            "llm_fallback_no_client": self._fallback_reasons.get("no_client", 0),
            "llm_fallback_timeout": self._fallback_reasons.get("timeout", 0),
            "llm_fallback_client_error": self._fallback_reasons.get("client_error", 0),
            "llm_fallback_invalid_output": self._fallback_reasons.get("invalid_output", 0),
        }

    def decide_next_action(self, user: User, video: Video) -> UserAction:
        # Consume the call identity first so even fallback paths log it and no
        # later call can inherit this call's scope.
        call_role, request_seed = self._next_call_identity()

        if self.client is None:
            if not self._warned_no_client:
                logger.warning(
                    "LLMDecider enabled but no client configured. prompt_id=%s -> fallback=no_client",
                    self.prompt_id,
                )
                self._warned_no_client = True

            # always set meta so each step log has correct info
            self._record_call(valid=False, fallback_reason="no_client")
            self.last_meta = DecisionMeta(
                policy_mode="llm",
                prompt_id=self.prompt_id,
                valid=False,
                fallback_reason="no_client",
                request_seed=request_seed,
                call_role=call_role,
            )
            return self.fallback.decide_next_action(user, video)

        prompt = render_decision_prompt(self.prompt_id, user=user, video=video)
        prompt_sha256 = _sha256_text(prompt)
        seed_sent = request_seed is not None and self._client_accepts_request_seed()

        try:
            if seed_sent:
                self._seeded_total += 1
                raw = self.client.complete(
                    prompt, timeout_s=self.timeout_s, request_seed=request_seed
                )
            else:
                raw = self.client.complete(prompt, timeout_s=self.timeout_s)
            self._warned_unreachable = False
        except TimeoutError as e:
            prompt_tokens = self._estimate_tokens(prompt)
            self._record_call(
                valid=False,
                fallback_reason="timeout",
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
                token_count_estimated=True,
            )
            # Always record meta data for CSV/analysis
            self.last_meta = DecisionMeta(
                policy_mode="llm",
                prompt_id=self.prompt_id,
                valid=False,
                fallback_reason="timeout",
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
                token_count_estimated=True,
                request_seed=request_seed,
                request_seed_sent=seed_sent,
                call_role=call_role,
                prompt_sha256=prompt_sha256,
            )

            # Log only once to avoid spamming when server is down
            if not self._warned_unreachable:
                logger.warning(
                    "LLM unreachable. prompt_id=%s err=%s -> falling back to heuristic",
                    self.prompt_id,
                    str(e),
                )
                self._warned_unreachable = True
            else:
                logger.debug(
                    "LLM still unreachable. prompt_id=%s err=%s -> fallback=timeout",
                    self.prompt_id,
                    str(e),
                )
            return self.fallback.decide_next_action(user, video)
        except Exception as e:
            logger.warning(
                "LLM call failed.prompt_id=%s err=%s -> fallback=client_error",
                self.prompt_id,
                type(e).__name__,
            )
            prompt_tokens = self._estimate_tokens(prompt)
            self._record_call(
                valid=False,
                fallback_reason="client_error",
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
                token_count_estimated=True,
            )
            self.last_meta = DecisionMeta(
                policy_mode="llm",
                prompt_id=self.prompt_id,
                valid=False,
                fallback_reason="client_error",
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
                token_count_estimated=True,
                request_seed=request_seed,
                request_seed_sent=seed_sent,
                call_role=call_role,
                prompt_sha256=prompt_sha256,
            )
            return self.fallback.decide_next_action(user, video)

        prompt_tokens, completion_tokens, total_tokens, estimated = self._resolve_usage(prompt, raw)
        candidate = _extract_first_json_object(raw)
        try:
            decision = parse_decision_json(candidate)
        except DecisionValidationError as e:
            # Keep logs minimal: don't dump prompt/response; just say why we fell back.
            logger.warning(
                "LLM output invalid. prompt_id=%s valid=false -> fallback=invalid_output (%s)",
                self.prompt_id,
                str(e).splitlines()[0],
            )
            self._record_call(
                valid=False,
                fallback_reason="invalid_output",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                token_count_estimated=estimated,
            )
            self.last_meta = DecisionMeta(
                policy_mode="llm",
                prompt_id=self.prompt_id,
                valid=False,
                fallback_reason="invalid_output",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                token_count_estimated=estimated,
                request_seed=request_seed,
                request_seed_sent=seed_sent,
                call_role=call_role,
                prompt_sha256=prompt_sha256,
                response_sha256=_sha256_text(raw),
            )
            return self.fallback.decide_next_action(user, video)
        except Exception as e:
            logger.warning(
                "LLM output parse/validate error. prompt_id=%s -> fallback=invalid_output (%s)",
                self.prompt_id,
                type(e).__name__,
            )
            self._record_call(
                valid=False,
                fallback_reason="invalid_output",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                token_count_estimated=estimated,
            )
            self.last_meta = DecisionMeta(
                policy_mode="llm",
                prompt_id=self.prompt_id,
                valid=False,
                fallback_reason="invalid_output",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                token_count_estimated=estimated,
                request_seed=request_seed,
                request_seed_sent=seed_sent,
                call_role=call_role,
                prompt_sha256=prompt_sha256,
                response_sha256=_sha256_text(raw),
            )
            return self.fallback.decide_next_action(user, video)

        logger.debug(
            "LLM decision valid. prompt_id=%s valid=true action=%s confidence=%.3f",
            self.prompt_id,
            decision.action.value,
            decision.confidence,
        )
        self._record_call(
            valid=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            token_count_estimated=estimated,
        )
        self.last_meta = DecisionMeta(
            policy_mode="llm",
            prompt_id=self.prompt_id,
            valid=True,
            llm_action=decision.action.value,
            llm_confidence=decision.confidence,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            token_count_estimated=estimated,
            request_seed=request_seed,
            request_seed_sent=seed_sent,
            call_role=call_role,
            prompt_sha256=prompt_sha256,
            response_sha256=_sha256_text(raw),
        )
        return decision.action


_LLM_DIAGNOSTIC_KEYS = (
    "llm_call_count",
    "llm_valid_count",
    "llm_fallback_count",
    "llm_retry_count",
    "llm_seeded_request_count",
    "llm_seed_collision_count",
    "llm_prompt_tokens",
    "llm_completion_tokens",
    "llm_total_tokens",
    "llm_token_estimated_calls",
    "llm_fallback_no_client",
    "llm_fallback_timeout",
    "llm_fallback_client_error",
    "llm_fallback_invalid_output",
)


def empty_llm_diagnostics() -> dict[str, int]:
    return {key: 0 for key in _LLM_DIAGNOSTIC_KEYS}


def llm_diagnostics_snapshot(decider: object) -> dict[str, int]:
    snapshot = getattr(decider, "diagnostics_snapshot", None)
    if not callable(snapshot):
        return empty_llm_diagnostics()
    values = snapshot()
    return {key: int(values.get(key, 0)) for key in _LLM_DIAGNOSTIC_KEYS}


def llm_diagnostics_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in _LLM_DIAGNOSTIC_KEYS}
