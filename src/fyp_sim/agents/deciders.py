from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from fyp_sim.llm.decision_contract import DecisionValidationError, parse_decision_json
from fyp_sim.llm.prompting import render_decision_prompt
from fyp_sim.models import User, UserAction, Video
from fyp_sim.policy import decide_action

logger = logging.getLogger(__name__)


class ActionDecider(Protocol):
    """Single deision interface used by the simulation loop."""

    def decide_next_action(self, user: User, video: Video) -> UserAction: ...


class LLMClient(Protocol):
    """Provider-agnostic LLM client interface (implemented later."""

    def complete(self, prompt: str, *, timeout_s: float) -> str: ...


@dataclass(slots=True)
class HeuristicDecider:
    """Adapter around the existing heuristic policy (baseline / deterministic)."""

    def decide_next_action(self, user: User, video: Video) -> UserAction:
        return decide_action(user, video)


def _extract_first_json_object(text: str) -> str:
    """
    Best-effort extraction of the first JSON object from a model response.

    Many local models sometimes preprend/append extra text. We try to salvage the first {...}.
    If extraction fails, return the original text (which will then fail validation cleanly).
    """
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]
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
                return text[start : i + 1]

    return text


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

    _warned_no_client: bool = False

    def decide_next_action(self, user: User, video: Video) -> UserAction:
        if self.client is None:
            if not self._warned_no_client:
                logger.warning(
                    "LLMDecider enabled but no client configured. prompt_id=%s -> fallback=no_client",
                    self.prompt_id,
                )
                self._warned_no_client = True
            return self.fallback.decide_next_action(user, video)

        prompt = render_decision_prompt(self.prompt_id, user=user, video=video)

        try:
            raw = self.client.complete(prompt, timeout_s=self.timeout_s)
        except TimeoutError as e:
            logger.warning(
                "LLM call timed out. prompt_id=%s: %s -> fallback=timeout (%s)",
                self.prompt_id,
                str(e),
            )
            return self.fallback.decide_next_action(user, video)
        except Exception as e:
            logger.warning(
                "LLM call failed.prompt_id=%s: %s -> fallback=client_error (%s)",
                self.prompt_id,
                type(e).__name__,
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
            return self.fallback.decide_next_action(user, video)
        except Exception as e:
            logger.warning(
                "LLM output parse/validate error. prompt_id=%s -> fallback=invalid_output (%s)",
                self.prompt_id,
                type(e).__name__,
            )
            return self.fallback.decide_next_action(user, video)

        logger.info(
            "LLM decision valid. prompt_id=%s valid=true action=%s confidence=%.3f",
            self.prompt_id,
            decision.action.value,
            decision.confidence,
        )
        return decision.action
