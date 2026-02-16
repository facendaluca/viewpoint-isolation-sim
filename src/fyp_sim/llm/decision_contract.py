from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from fyp_sim.models import UserAction

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).with_name("decision.schema.json")


class DecisionValidationError(ValueError):
    """Raised when LLM output fails schema validation."""

    def __init__(self, messages: list[str]):
        super().__init__("\n".join(messages))
        self.messages = messages


@dataclass(frozen=True, slots=True)
class LLMDecision:
    action: UserAction
    confidence: float
    reason: str | None = None
    watch_time_s: int | None = None
    rewatch: bool | None = None
    share: bool | None = None
    notes: str | None = None


def _load_schema() -> dict[str, Any]:
    try:
        return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise RuntimeError(f"Decision schema file missing at: {_SCHEMA_PATH}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Decision schema file is invalid JSON at: {_SCHEMA_PATH}") from e


def _build_validator() -> Draft202012Validator:
    schema = _load_schema()
    # Validate the schema itself
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


_VALIDATOR = _build_validator()


def _format_error(err: ValidationError) -> str:
    # Build a deterministic "$.path.to.field" representation
    path = "$"
    for part in err.path:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return f"{path}: {err.message}"


def validate_decision(decision: Mapping[str, Any]) -> None:
    """
    Validate a parsed decision dict against the Decision Contract.

    Strictness:
    - Unknown fields rejected (schema additionalProperties=false)
    - Type mismatches rejected
    - Enum violations rejected
    """
    errors = list(_VALIDATOR.iter_errors(decision))
    if not errors:
        return

    # Deterministic ordering for stable logs/tests
    errors.sort(key=lambda e: (list(e.path), e.message))
    messages = [_format_error(e) for e in errors]

    logger.error(
        "Invalid LLM decision JSON (%d error(s))\n%s",
        len(messages),
        "\n".join(messages),
    )
    raise DecisionValidationError(messages)


def parse_decision(decision: Mapping[str, Any]) -> LLMDecision:
    """Validate + coerce a decicion dict into a typed LLMDecision."""
    validate_decision(decision)

    action_raw = decision["action"]
    # Should be safe after schema validation, but keep a clear fallback
    try:
        action = UserAction(str(action_raw))
    except ValueError:
        raise DecisionValidationError([f"Invalid action enum value: {action_raw}"]) from None

    return LLMDecision(
        action=action,
        confidence=float(decision["confidence"]),
        reason=decision.get("reason"),
        watch_time_s=decision.get("watch_time_s"),
        rewatch=decision.get("rewatch"),
        share=decision.get("share"),
        notes=decision.get("notes"),
    )


def parse_decision_json(text: str) -> LLMDecision:
    """Validate a JSON string and return a typed LLMDecision."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        msg = f"$.<json>: invalid JSON ({e.msg}) at line {e.lineno}, col {e.colno}"
        logger.error("Invalid LLM decision JSON:\n%s", msg)
        raise DecisionValidationError([msg]) from None

    # schema expects an object, this gives a clearer error than a cryptic stack trace
    if not isinstance(data, dict):
        msg = f"$.<root>: expected object, got {type(data).__name__}"
        logger.error("Invalid LLM decision JSON:\n%s", msg)
        raise DecisionValidationError([msg])

    return parse_decision(data)
