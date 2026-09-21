"""Exact native tropical scalar arithmetic."""

from __future__ import annotations

from fractions import Fraction

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.tropical._models import AddBranch, InfinityCase
from jacobian.math.polynomials.tropical.values import (
    TropicalScalar,
    TropicalSemiring,
    require_scalar_budget,
)


def _admit_add_operands(
    semiring: TropicalSemiring, left: TropicalScalar, right: TropicalScalar
) -> None:
    """Enforce the admitted tropical-plus domain once per call."""

    if not isinstance(semiring, TropicalSemiring):
        raise OperationDomainValidationError(
            location=("semiring",),
            code="tropical.scalar_add_semiring_type",
            message="semiring must be a tropical semiring value",
        )
    for label, scalar in (("left", left), ("right", right)):
        if not isinstance(scalar, TropicalScalar):
            raise OperationDomainValidationError(
                location=(label,),
                code="tropical.scalar_add_operand_type",
                message="tropical-plus operands must be tropical scalar values",
            )
        if scalar.semiring != semiring:
            raise OperationDomainValidationError(
                location=(label,),
                code="tropical.scalar_add_semiring_mismatch",
                message="tropical-plus operands must carry the request semiring",
            )
        # Native callers may bypass value validation; re-establish the
        # licensed-infinity and integrality contract the kernel relies on.
        if scalar.kind == "FINITE":
            if scalar.value is None:
                raise OperationDomainValidationError(
                    location=(label,),
                    code="tropical.scalar_add_finite_missing_value",
                    message="a finite scalar must carry its value",
                )
            if semiring.base == "ZZ" and scalar.value.den != 1:
                raise OperationDomainValidationError(
                    location=(label,),
                    code="tropical.scalar_add_nonintegral_integer_scalar",
                    message="a ZZ tropical scalar must be integral",
                )
        else:
            if scalar.value is not None:
                raise OperationDomainValidationError(
                    location=(label,),
                    code="tropical.scalar_add_infinite_carries_value",
                    message="an infinite scalar carries no value",
                )
            licensed = "MIN_PLUS" if scalar.kind == "POSITIVE_INFINITY" else "MAX_PLUS"
            if semiring.convention != licensed:
                raise OperationDomainValidationError(
                    location=(label,),
                    code="tropical.scalar_add_unlicensed_infinity",
                    message="only the semiring-licensed infinity is an element",
                )
        try:
            require_scalar_budget(scalar)
        except ValueError as error:
            from jacobian.catalog.models import OperationResourceAdmissionError

            raise OperationResourceAdmissionError(
                location=(label,),
                code="tropical.scalar_add_scalar_budget",
                message=str(error),
            ) from error


def _order_key(
    semiring: TropicalSemiring, scalar: TropicalScalar
) -> tuple[int, Fraction]:
    """Return a comparison key under the semiring order.

    Finite scalars compare by value in both conventions; the licensed
    infinity is the greatest element of MIN_PLUS and the least of MAX_PLUS,
    which is exactly the additive-identity absorption law.
    """

    if scalar.kind == "FINITE":
        if scalar.value is None:
            raise RuntimeError("finite tropical scalar has no rational value")
        return (0, scalar.value.as_fraction())
    if semiring.convention == "MIN_PLUS":
        return (1, Fraction(0))
    return (-1, Fraction(0))


def tropical_scalar_add(
    semiring: TropicalSemiring, left: TropicalScalar, right: TropicalScalar
) -> tuple[TropicalScalar, AddBranch, InfinityCase]:
    """Return ``left tropical-plus right`` with its recorded branch."""

    _admit_add_operands(semiring, left, right)
    left_key = _order_key(semiring, left)
    right_key = _order_key(semiring, right)
    if semiring.convention == "MIN_PLUS":
        left_wins = left_key <= right_key
        right_wins = right_key <= left_key
    else:
        left_wins = left_key >= right_key
        right_wins = right_key >= left_key
    if left_wins and right_wins:
        branch: AddBranch = "TIE"
        winner = left
    elif left_wins:
        branch = "LEFT"
        winner = left
    else:
        branch = "RIGHT"
        winner = right
    left_infinite = left.kind != "FINITE"
    right_infinite = right.kind != "FINITE"
    infinity_case: InfinityCase = (
        "BOTH_INFINITE"
        if left_infinite and right_infinite
        else "LEFT_INFINITE"
        if left_infinite
        else "RIGHT_INFINITE"
        if right_infinite
        else "NONE"
    )
    return (
        TropicalScalar._from_kernel(
            semiring=semiring, kind=winner.kind, value=winner.value
        ),
        branch,
        infinity_case,
    )


__all__ = ["tropical_scalar_add"]
