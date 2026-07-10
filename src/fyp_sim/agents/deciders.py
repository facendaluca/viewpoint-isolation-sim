from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from fyp_sim.llm.decision_contract import DecisionValidationError, parse_decision_json
from fyp_sim.llm.prompting import render_decision_prompt
from fyp_sim.models import User, UserAction, Video
from fyp_sim.policy import decide_action

logger = logging.getLogger(__name__)


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


class ActionDecider(Protocol):
    """Single deision interface used by the simulation loop."""

    def decide_next_action(self, user: User, video: Video) -> UserAction: ...


class LLMClient(Protocol):
    """Provider-agnostic LLM client interface (implemented later."""

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
            )
            return self.fallback.decide_next_action(user, video)

        prompt = render_decision_prompt(self.prompt_id, user=user, video=video)

        try:
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
        )
        return decision.action


_LLM_DIAGNOSTIC_KEYS = (
    "llm_call_count",
    "llm_valid_count",
    "llm_fallback_count",
    "llm_retry_count",
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
