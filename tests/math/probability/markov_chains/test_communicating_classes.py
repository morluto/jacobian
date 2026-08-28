"""Tests for Markov chain communicating class decomposition."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.probability.markov_chains import _operations
from jacobian.math.probability.markov_chains._models import (
    CommunicatingClassesResult,
    TransitionMatrixRequest,
)
from jacobian.math.probability.markov_chains._operations import (
    compute_communicating_classes,
)
from jacobian.math.probability.markov_chains._tools import TOOLS

_C = CanonicalRational.from_fraction


def _matrix(rows: list[list[Fraction]]) -> TransitionMatrixRequest:
    return TransitionMatrixRequest(
        matrix=tuple(tuple(_C(f) for f in row) for row in rows)
    )


def test_operation_in_catalog() -> None:
    ids = {tool.operation_id for tool in TOOLS}
    assert "probability.markov_chain.communicating_classes.compute" in ids


def test_absorbing_chain_has_transient_and_closed_classes() -> None:
    # 0 -> 1 -> 1 (absorbing)
    result = compute_communicating_classes(
        _matrix([[Fraction(0), Fraction(1)], [Fraction(0), Fraction(1)]])
    )
    assert len(result.classes) == 2
    assert result.classes[0] == ((0,), False)  # transient
    assert result.classes[1] == ((1,), True)  # closed


def test_irreducible_chain_has_single_closed_class() -> None:
    # Self-loops on both states
    result = compute_communicating_classes(
        _matrix([[Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 2), Fraction(1, 2)]])
    )
    assert len(result.classes) == 1
    assert result.classes[0][1] is True  # closed


def test_single_state_absorbing() -> None:
    result = compute_communicating_classes(_matrix([[Fraction(1)]]))
    assert len(result.classes) == 1
    assert result.classes[0] == ((0,), True)


def test_three_state_chain_with_transient_state() -> None:
    # 0 -> 1 -> 2 -> 2 (chain with absorbing state 2)
    result = compute_communicating_classes(
        _matrix(
            [
                [Fraction(0), Fraction(1), Fraction(0)],
                [Fraction(0), Fraction(0), Fraction(1)],
                [Fraction(0), Fraction(0), Fraction(1)],
            ]
        )
    )
    assert len(result.classes) == 3
    # All singletons, first two transient, last closed
    assert all(len(cls[0]) == 1 for cls in result.classes)
    assert result.classes[0][1] is False  # transient
    assert result.classes[1][1] is False  # transient
    assert result.classes[2][1] is True  # closed


def test_state_class_indices() -> None:
    result = compute_communicating_classes(
        _matrix([[Fraction(0), Fraction(1)], [Fraction(0), Fraction(1)]])
    )
    assert result.state_class == (0, 1)
    assert len(result.state_class) == 2


def test_two_closed_classes() -> None:
    # Identity matrix - two absorbing states
    result = compute_communicating_classes(
        _matrix([[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]])
    )
    assert len(result.classes) == 2
    assert all(cls[1] for cls in result.classes)  # both closed


def test_producer_derives_scc_partition_once(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _matrix([[Fraction(0), Fraction(1)], [Fraction(0), Fraction(1)]])
    original = _operations._derive_communicating_classes
    calls = 0

    def counted(matrix):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(matrix)

    monkeypatch.setattr(_operations, "_derive_communicating_classes", counted)
    result = compute_communicating_classes(request)

    assert result.classes == (((0,), False), ((1,), True))
    assert calls == 1


def test_explicit_verifier_rejects_forged_scc_claim() -> None:
    produced = compute_communicating_classes(
        _matrix([[Fraction(0), Fraction(1)], [Fraction(0), Fraction(1)]])
    )
    payload = produced.model_dump(mode="json")
    payload["classes"] = [((0,), True), ((1,), True)]
    forged = CommunicatingClassesResult.model_validate(payload)

    assert produced.classes != forged.classes

    payload = produced.model_dump(mode="json")
    payload["transition_matrix"] = [[{"num": "2", "den": "1"}]]
    with pytest.raises(ValidationError, match="must sum to one"):
        CommunicatingClassesResult.model_validate(payload)
