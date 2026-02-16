from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from fyp_sim.models import User, UserAction, Video
from fyp_sim.policy import decide_action

logger = logging.getLogger(__name__)


class ActionDecider(Protocol):
    """Single deision interface used by the simulation loop."""

    def decide_next_action(self, user: User, video: Video) -> UserAction: ...


class LLMClient(Protocol):
    """Provider-agnostic LLM client interface (implemented later."""

    def complete(self, prompt: str) -> str: ...


@dataclass(slots=True)
class HeuristicDecider:
    """Adapter around the existing heuristic policy (baseline / deterministic)."""

    def decide_next_action(self, user: User, video: Video) -> UserAction:
        return decide_action(user, video)


@dataclass(slots=True)
class LLMDecider:
    """
    LLM-backed decider scaffold.
    """

    prompt_id: str = "decision_v1"
    client: LLMClient | None = None
    fallback: ActionDecider = HeuristicDecider()

    __warned_no_client: bool = False

    def decide_next_action(self, user: User, video: Video) -> UserAction:
        if self.client is None:
            # Avoide spamming logs: warn once per run
            if not self.__warned_no_client:
                logger.warning(
                    "LLMDecider enabled but no client configured. prompt_id=%s -> falling back to heuristic.",
                    self.prompt_id,
                )
                self.__warned_no_client = True
            return self.fallback.decide_next_action(user, video)

        # TODO: prompt build -> client.complete -> parse/validate -> fallback on error
        logger.warning(
            "LLMDecider client provided but LLM call not implemented yet. prompt_id=%s -> falling back.",
            self.prompt_id,
        )
        return self.fallback.decide_next_action(user, video)
