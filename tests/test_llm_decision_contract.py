import pytest

from fyp_sim.llm.decision_contract import DecisionValidationError, LLMDecision, parse_decision
from fyp_sim.models import UserAction


def test_valid_decision_passes() -> None:
    d = parse_decision(
        {
            "action": "Watch",
            "confidence": 0.9,
            "reason": "high interest",
            "watch_time_s": 12,
            "rewatch": False,
            "share": False,
        }
    )

    assert isinstance(d, LLMDecision)
    assert d.action == UserAction.WATCH
    assert d.confidence == pytest.approx(0.9)
    assert d.reason == "high interest"
    assert d.watch_time_s == 12
    assert d.rewatch is False
    assert d.share is False
    assert d.notes is None


def test_missing_required_field_fails() -> None:
    # Missing "action" (required)
    with pytest.raises(DecisionValidationError) as exc:
        parse_decision(
            {
                "confidence": 0.9,
                "reason": "no action provided",
            }
        )

    msg = str(exc.value)
    assert "required property" in msg
    assert "action" in msg


def test_invalid_enum_fails() -> None:
    # 'action' must be one of: Avoid | Sample | Watch
    with pytest.raises(DecisionValidationError) as exc:
        parse_decision(
            {
                "action": "Swim",
                "confidence": 0.5,
            }
        )

    msg = str(exc.value)
    assert "$.action" in msg  # path formatiing from validator
    assert "not one of" in msg
