"""Standard-monomial enumeration for generic fibers."""

from __future__ import annotations

from jacobian.math.polynomials.maps._models import (
    MAX_GENERIC_FIBER_STANDARD_MONOMIALS,
)


class StandardMonomialLimitError(ValueError):
    """The exact quotient exceeds the admitted monomial cardinality."""


def _divides(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return all(a <= b for a, b in zip(left, right, strict=True))


def _has_pure_power(
    leading_exponents: tuple[tuple[int, ...], ...],
    variable_index: int,
) -> bool:
    return any(
        exponents[variable_index] > 0
        and all(
            exponent == 0
            for index, exponent in enumerate(exponents)
            if index != variable_index
        )
        for exponents in leading_exponents
    )


def enumerate_standard_monomials(
    leading_exponents: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...] | None:
    """Return the bounded standard-monomial complement for one leading ideal."""

    variable_count = len(leading_exponents[0])
    if not all(
        _has_pure_power(leading_exponents, variable_index)
        for variable_index in range(variable_count)
    ):
        return None
    origin = (0,) * variable_count
    found = {origin}
    frontier = [origin]
    while frontier:
        current = frontier.pop()
        for variable_index in range(variable_count):
            shifted = list(current)
            shifted[variable_index] += 1
            candidate = tuple(shifted)
            if candidate in found or any(
                _divides(leading, candidate) for leading in leading_exponents
            ):
                continue
            if len(found) >= MAX_GENERIC_FIBER_STANDARD_MONOMIALS:
                raise StandardMonomialLimitError(
                    "generic-fiber quotient exceeds the standard-monomial bound"
                )
            found.add(candidate)
            frontier.append(candidate)
    return tuple(sorted(found))


__all__ = ["StandardMonomialLimitError", "enumerate_standard_monomials"]
