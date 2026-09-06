"""Exact contract tests for containment and equality of rational ideals."""

from __future__ import annotations

import time
from fractions import Fraction
from typing import Literal, cast

import pytest

from jacobian._exact import CanonicalRational
from jacobian._execution import current_request_execution, request_execution
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.ideals import operations
from jacobian.math.polynomials.ideals._models import (
    IdealComputationBudget,
    IdealContainmentLedger,
    IdealContainmentRequest,
    IdealEqualityRequest,
)
from jacobian.math.polynomials.ideals.operations import (
    ideal_containment,
    ideal_equality,
    verify_ideal_containment,
    verify_ideal_equality,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(
    variables: tuple[str, ...], terms: dict[tuple[int, ...], int | Fraction]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
                    exponents=exponents,
                )
                for exponents, coefficient in sorted(terms.items(), reverse=True)
                if coefficient
            )
        ),
    )


def _ideal(
    variables: tuple[str, ...], *generators: dict[tuple[int, ...], int | Fraction]
) -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(
        variables=variables,
        generators=tuple(_polynomial(variables, item) for item in generators),
    )


def test_containment_returns_complete_positive_ledger() -> None:
    source = _ideal(("x", "y"), {(2, 0): 1}, {(1, 1): 1})
    target = _ideal(("x", "y"), {(1, 0): 1})

    result = ideal_containment(source, target)

    assert result.ledger is not None
    assert result.ledger.contained
    assert len(result.ledger.normal_forms) == 2
    assert all(not item.polynomial.terms for item in result.ledger.normal_forms)


def test_containment_stops_at_first_nonzero_obstruction() -> None:
    source = _ideal(
        ("x", "y"),
        {(2, 0): 1},
        {(0, 1): 1},
        {(1, 1): 1},
    )
    target = _ideal(("x", "y"), {(1, 0): 1})

    result = ideal_containment(source, target)

    assert result.ledger is not None
    assert not result.ledger.contained
    assert result.ledger.first_obstruction_index == 1
    assert len(result.ledger.normal_forms) == 2
    assert result.ledger.normal_forms[-1] == _polynomial(("x", "y"), {(0, 1): 1})


def test_equality_is_invariant_under_order_redundancy_and_rescaling() -> None:
    left = _ideal(("x", "y"), {(1, 0): 1}, {(0, 1): 1})
    right = _ideal(
        ("x", "y"),
        {(0, 1): 2},
        {(1, 0): -3},
        {(1, 0): 1, (0, 1): 1},
    )

    result = ideal_equality(left, right)

    assert result.equal is True
    assert result.left_in_right is not None and result.left_in_right.contained
    assert result.right_in_left is not None and result.right_in_left.contained
    assert len(result.right_in_left.normal_forms) == 3


def test_unequal_ideals_retain_both_directions() -> None:
    square = _ideal(("x",), {(2,): 1})
    linear = _ideal(("x",), {(1,): 1})

    result = ideal_equality(square, linear)

    assert result.equal is False
    assert result.left_in_right is not None and result.left_in_right.contained
    assert result.right_in_left is not None and not result.right_in_left.contained
    assert result.right_in_left.first_obstruction_index == 0
    assert result.right_in_left.normal_forms[0] == _polynomial(("x",), {(1,): 1})


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_ideal_relation_is_order_independent_while_ledgers_retain_order(
    order: str,
) -> None:
    source = _ideal(
        ("x", "y"),
        {(2, 0): 1, (0, 1): -1},
        {(1, 1): 1, (0, 0): -1},
    )
    target = _ideal(
        ("x", "y"),
        {(1, 1): 1, (0, 0): -1},
        {(0, 2): 1, (1, 0): -1},
        {(2, 0): 1, (0, 1): -1},
    )

    result = ideal_containment(
        source,
        target,
        cast(Literal["lex", "grlex", "grevlex"], order),
    )

    assert result.ledger is not None and result.ledger.contained
    assert result.monomial_order == order


def test_relation_results_round_trip_without_replaying_math() -> None:
    left = _ideal(("x",), {(1,): 1})
    result = ideal_equality(left, _ideal(("x",), {(1,): 2}))

    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_serialized_relation_verifiers_reject_forged_ledgers() -> None:
    ideal = _ideal(("x",), {(1,): 1})
    containment = ideal_containment(ideal, ideal)
    decoded_containment = type(containment).model_validate_json(
        containment.model_dump_json()
    )
    assert verify_ideal_containment(decoded_containment)

    containment_payload = decoded_containment.model_dump(mode="json")
    containment_payload["ledger"]["contained"] = False
    containment_payload["ledger"]["first_obstruction_index"] = 0
    containment_payload["ledger"]["normal_forms"][0] = _polynomial(
        ("x",), {(1,): 1}
    ).model_dump(mode="json")
    forged_containment = type(containment).model_validate(containment_payload)
    assert not verify_ideal_containment(forged_containment)

    equality = ideal_equality(ideal, ideal)
    decoded_equality = type(equality).model_validate_json(equality.model_dump_json())
    assert verify_ideal_equality(decoded_equality)

    equality_payload = decoded_equality.model_dump(mode="json")
    equality_payload["equal"] = False
    equality_payload["right_in_left"]["contained"] = False
    equality_payload["right_in_left"]["first_obstruction_index"] = 0
    equality_payload["right_in_left"]["normal_forms"][0] = _polynomial(
        ("x",), {(1,): 1}
    ).model_dump(mode="json")
    forged_equality = type(equality).model_validate(equality_payload)
    assert not verify_ideal_equality(forged_equality)


def test_zero_and_unit_ideals_obey_containment_extremes() -> None:
    zero = _ideal(("x",), {})
    unit = _ideal(("x",), {(0,): 1})
    proper = _ideal(("x",), {(1,): 1})

    zero_in_proper = ideal_containment(zero, proper).ledger
    proper_in_unit = ideal_containment(proper, unit).ledger
    unit_in_proper = ideal_containment(unit, proper).ledger
    assert zero_in_proper is not None and zero_in_proper.contained
    assert proper_in_unit is not None and proper_in_unit.contained
    assert unit_in_proper is not None and not unit_in_proper.contained
    assert ideal_equality(zero, zero).equal is True
    assert ideal_equality(zero, proper).equal is False


def test_native_polynomial_api_exports_relations() -> None:
    from jacobian.math.polynomials import ideal_containment as public_containment
    from jacobian.math.polynomials import ideal_equality as public_equality

    assert public_containment is ideal_containment
    assert public_equality is ideal_equality


def test_request_rejects_mixed_ordered_rings_before_work() -> None:
    with pytest.raises(ValueError, match="same ordered ring"):
        IdealContainmentRequest(
            source=_ideal(("x",), {(1,): 1}),
            target=_ideal(("y",), {(1,): 1}),
        )
    with pytest.raises(ValueError, match="same ordered ring"):
        IdealEqualityRequest(
            left=_ideal(("x",), {(1,): 1}),
            right=_ideal(("y",), {(1,): 1}),
        )


def test_native_relation_admission_rejects_oversized_source() -> None:
    generators: tuple[dict[tuple[int, ...], int | Fraction], ...] = tuple(
        {(1,): index + 1} for index in range(33)
    )
    source = _ideal(("x",), *generators)
    target = _ideal(("x",), {(1,): 1})

    with pytest.raises(OperationDomainValidationError, match="generator"):
        ideal_containment(source, target)


def test_relation_kernel_uses_the_bound_request_deadline() -> None:
    started = time.monotonic()

    with request_execution(started):
        kernel_deadline = operations._bind_relation_deadline(
            IdealComputationBudget(wall_seconds=1)
        )
        execution = current_request_execution()

    assert execution is not None
    assert execution.deadline is not None
    assert execution.deadline == kernel_deadline


def test_ledger_rejects_a_false_positive_shape() -> None:
    with pytest.raises(ValueError, match="only zero normal forms"):
        IdealContainmentLedger(
            contained=True,
            normal_forms=(_polynomial(("x",), {(1,): 1}),),
        )


def test_backend_failure_does_not_become_noncontainment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> dict[str, object]:
        raise operations._SympyKernelError("backend failure")

    monkeypatch.setattr(operations, "_run_sympy_kernel", fail)
    source = _ideal(("x",), {(1,): 1})

    with pytest.raises(RuntimeError, match="failed without producing"):
        ideal_containment(source, source)
