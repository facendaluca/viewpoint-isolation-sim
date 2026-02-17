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

    def decide_next_action(self, user: User, video: Video) -> UserAction:
        if self.client is None:
            if not self._warned_no_client:
                logger.warning(
                    "LLMDecider enabled but no client configured. prompt_id=%s -> fallback=no_client",
                    self.prompt_id,
                )
                self._warned_no_client = True

            # always set meta so each step log has correct info
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
            # Always record meta data for CSV/analysis
            self.last_meta = DecisionMeta(
                policy_mode="llm",
                prompt_id=self.prompt_id,
                valid=False,
                fallback_reason="timeout",
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
                logger.debug("LLM still unreachable. prompt_id=%s err=%s -> fallback=timeout")
            return self.fallback.decide_next_action(user, video)
        except Exception as e:
            logger.warning(
                "LLM call failed.prompt_id=%s err=%s -> fallback=client_error",
                self.prompt_id,
                type(e).__name__,
            )
            self.last_meta = DecisionMeta(
                policy_mode="llm",
                prompt_id=self.prompt_id,
                valid=False,
                fallback_reason="client_error",
            )
            return self.fallback.decide_next_action(user, video)

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
            self.last_meta = DecisionMeta(
                policy_mode="llm",
                prompt_id=self.prompt_id,
                valid=False,
                fallback_reason="invalid_output",
            )
            return self.fallback.decide_next_action(user, video)
        except Exception as e:
            logger.warning(
                "LLM output parse/validate error. prompt_id=%s -> fallback=invalid_output (%s)",
                self.prompt_id,
                type(e).__name__,
            )
            self.last_meta = DecisionMeta(
                policy_mode="llm",
                prompt_id=self.prompt_id,
                valid=False,
                fallback_reason="invalid_output",
            )
            return self.fallback.decide_next_action(user, video)

        logger.debug(
            "LLM decision valid. prompt_id=%s valid=true action=%s confidence=%.3f",
            self.prompt_id,
            decision.action.value,
            decision.confidence,
        )
        self.last_meta = DecisionMeta(
            policy_mode="llm",
            prompt_id=self.prompt_id,
            valid=True,
            llm_action=decision.action.value,
            llm_confidence=decision.confidence,
        )
        return decision.action
