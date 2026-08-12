from __future__ import annotations

import pytest
from deploy.smoke_lean import (
    _require_mathlib_declaration,
    _require_transition,
    _require_verified,
)


def test_verified_smoke_requires_completed_checker_backing() -> None:
    result = {
        "execution": {"status": "COMPLETED"},
        "output": {"conclusion": "TRUE"},
        "verification_record_uri": "artifact://sha256/example",
    }

    _require_verified(result, environment="MATHLIB")

    for mutation in (
        {**result, "execution": {"status": "ERROR"}},
        {**result, "output": {"conclusion": "UNKNOWN"}},
        {**result, "verification_record_uri": None},
    ):
        with pytest.raises(RuntimeError):
            _require_verified(mutation, environment="MATHLIB")


def test_transition_smoke_keeps_rejection_distinct_from_execution_failure() -> None:
    rejected = {
        "execution": {"status": "COMPLETED"},
        "output": {
            "accepted": False,
            "completed": False,
            "successor_states": [],
            "diagnostics": [{"severity": "ERROR", "message": "tactic failed"}],
        },
    }

    _require_transition(rejected, accepted=False, completed=False)

    without_diagnostics = {
        **rejected,
        "output": {**rejected["output"], "diagnostics": []},
    }
    with pytest.raises(RuntimeError, match="actionable diagnostics"):
        _require_transition(without_diagnostics, accepted=False, completed=False)


def test_transition_smoke_requires_exactly_one_bound_successor_on_acceptance() -> None:
    accepted = {
        "execution": {"status": "COMPLETED"},
        "output": {
            "accepted": True,
            "completed": True,
            "successor_states": [{"state_uri": "artifact://sha256/example"}],
            "diagnostics": [],
        },
    }

    _require_transition(accepted, accepted=True, completed=True)

    duplicate = {
        **accepted,
        "output": {
            **accepted["output"],
            "successor_states": accepted["output"]["successor_states"] * 2,
        },
    }
    with pytest.raises(RuntimeError, match="successor binding"):
        _require_transition(duplicate, accepted=True, completed=True)


def test_mathlib_declaration_smoke_requires_the_exact_declaration() -> None:
    result = {
        "execution": {"status": "COMPLETED"},
        "output": {"declarations": [{"name": "irrational_sqrt_two"}]},
    }
    _require_mathlib_declaration(result)

    for mutation in (
        {**result, "execution": {"status": "ERROR"}},
        {**result, "output": {"declarations": []}},
        {**result, "output": {"declarations": [{"name": "Nat.add"}]}},
    ):
        with pytest.raises(RuntimeError):
            _require_mathlib_declaration(mutation)
