"""Native exact Lie-bracket operations over QQ."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lie_algebras._models import (
    MAX_ELEMENT_COEFFICIENT_DIGITS,
    MAX_STRUCTURE_COEFFICIENT_DIGITS,
    BracketPairContribution,
    FiniteDimensionalLieAlgebra,
    LieAlgebraElement,
    LieBracketResult,
    StructureConstant,
)


def _as_algebra(
    value: FiniteDimensionalLieAlgebra | Mapping[str, Any],
) -> FiniteDimensionalLieAlgebra:
    return (
        value
        if isinstance(value, FiniteDimensionalLieAlgebra)
        else FiniteDimensionalLieAlgebra.model_validate(value)
    )


def _as_element(value: LieAlgebraElement | Mapping[str, Any]) -> LieAlgebraElement:
    return (
        value
        if isinstance(value, LieAlgebraElement)
        else LieAlgebraElement.model_validate(value)
    )


def _run_admission(admission: Any, *, location: tuple[str | int, ...]) -> None:
    try:
        admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=location,
            code="lie_algebra.admission",
            message=str(exc),
        ) from exc


def _bracket_table(
    algebra: FiniteDimensionalLieAlgebra,
) -> dict[tuple[int, int], dict[int, Fraction]]:
    """Expand the ordered constants into the full antisymmetric bracket table."""
    table: dict[tuple[int, int], dict[int, Fraction]] = {}
    for constant in algebra.structure_constants:
        value = constant.coefficient.as_fraction()
        table.setdefault((constant.i, constant.j), {})[constant.k] = value
        table.setdefault((constant.j, constant.i), {})[constant.k] = -value
    return table


def _admit_lie_algebra(algebra: FiniteDimensionalLieAlgebra) -> None:
    """Establish antisymmetry and every basis-triple Jacobi identity."""
    for index, constant in enumerate(algebra.structure_constants):
        _run_admission(
            lambda constant=constant: require_bounded_rational(
                constant.coefficient,
                max_digits=MAX_STRUCTURE_COEFFICIENT_DIGITS,
                label="structure constant",
            ),
            location=("algebra", "structure_constants", index),
        )
    dimension = len(algebra.basis)
    table = _bracket_table(algebra)

    def _pair(first: int, second: int) -> dict[int, Fraction]:
        if first == second:
            return {}
        return table.get((first, second), {})

    for first in range(dimension):
        for second in range(dimension):
            for third in range(dimension):
                accumulator: dict[int, Fraction] = {}
                for outer_first, outer_second, inner in (
                    (first, second, third),
                    (second, third, first),
                    (third, first, second),
                ):
                    for middle, outer_value in _pair(outer_first, outer_second).items():
                        for target, inner_value in _pair(middle, inner).items():
                            accumulator[target] = (
                                accumulator.get(target, Fraction(0))
                                + outer_value * inner_value
                            )
                if any(value != 0 for value in accumulator.values()):
                    raise OperationDomainValidationError(
                        location=("algebra", "structure_constants"),
                        code="lie_algebra.jacobi_identity",
                        message="structure constants must satisfy the Jacobi identity",
                    )


def _admit_bracket_operands(
    algebra: FiniteDimensionalLieAlgebra,
    left: LieAlgebraElement,
    right: LieAlgebraElement,
) -> None:
    _admit_lie_algebra(algebra)
    if left.basis != algebra.basis or right.basis != algebra.basis:
        raise OperationDomainValidationError(
            location=("left",),
            code="lie_algebra.element_basis",
            message="bracket elements must use the algebra's ordered basis",
        )
    for label, element in (("left", left), ("right", right)):
        for index, coordinate in enumerate(element.coordinates):
            _run_admission(
                lambda coordinate=coordinate, label=label: require_bounded_rational(
                    coordinate,
                    max_digits=MAX_ELEMENT_COEFFICIENT_DIGITS,
                    label=f"{label} coordinate",
                ),
                location=(label, "coordinates", index),
            )


def lie_bracket(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
    left: LieAlgebraElement | Mapping[str, Any],
    right: LieAlgebraElement | Mapping[str, Any],
) -> LieBracketResult:
    """Compute the exact bracket of two Lie-algebra elements with a ledger."""
    algebra_value = _as_algebra(algebra)
    left_value = _as_element(left)
    right_value = _as_element(right)
    _admit_bracket_operands(algebra_value, left_value, right_value)
    dimension = len(algebra_value.basis)
    table = _bracket_table(algebra_value)
    left_coords = [coordinate.as_fraction() for coordinate in left_value.coordinates]
    right_coords = [coordinate.as_fraction() for coordinate in right_value.coordinates]
    totals = [Fraction(0)] * dimension
    ledger: list[BracketPairContribution] = []
    for first in range(dimension):
        for second in range(first + 1, dimension):
            pair = (
                left_coords[first] * right_coords[second]
                - left_coords[second] * right_coords[first]
            )
            if pair == 0:
                continue
            row_terms: list[StructureConstant] = []
            for target, value in sorted(table.get((first, second), {}).items()):
                scaled = pair * value
                totals[target] += scaled
                row_terms.append(
                    StructureConstant.model_construct(
                        i=first,
                        j=second,
                        k=target,
                        coefficient=CanonicalRational.from_fraction(scaled),
                    )
                )
            if not row_terms:
                continue
            ledger.append(
                BracketPairContribution.model_construct(
                    i=first,
                    j=second,
                    pair_coefficient=CanonicalRational.from_fraction(pair),
                    terms=tuple(row_terms),
                )
            )
    bracket = LieAlgebraElement.model_construct(
        basis=algebra_value.basis,
        coordinates=tuple(CanonicalRational.from_fraction(value) for value in totals),
    )
    return LieBracketResult._from_kernel(
        algebra_value,
        left_value,
        right_value,
        bracket=bracket,
        ledger=tuple(ledger),
    )


__all__ = ["lie_bracket"]
