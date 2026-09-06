"""Exact native operations for finitely generated abelian groups."""

from __future__ import annotations

from math import gcd, lcm

from jacobian.math.groups.abelian._models import (
    AbelianElement,
    AbelianPresentation,
    AbelianQuotient,
    AbelianSubgroup,
    ElementEqualResult,
    ElementOrderResult,
    ElementReduceResult,
    PresentationNormalizeRequest,
    PresentationNormalizeResult,
    QuotientResult,
    SubgroupGeneratedResult,
)


def normalize_presentation(
    invariant_factors: tuple[int, ...],
) -> PresentationNormalizeResult:
    from sympy import Matrix, diag
    from sympy.matrices.normalforms import smith_normal_form

    matrix = Matrix(diag(*invariant_factors))
    smith = smith_normal_form(matrix, domain=None)
    factors = tuple(int(smith[i, i]) for i in range(min(smith.rows, smith.cols)))
    cleaned = tuple(factor for factor in factors if factor > 1)
    return PresentationNormalizeResult(
        source=PresentationNormalizeRequest(invariant_factors=invariant_factors),
        presentation=AbelianPresentation(invariant_factors=cleaned),
    )


def reduce_element(
    invariant_factors: tuple[int, ...], coordinates: tuple[int, ...]
) -> ElementReduceResult:
    reduced = tuple(
        coordinate % factor
        for coordinate, factor in zip(coordinates, invariant_factors, strict=True)
    )
    group = AbelianPresentation(invariant_factors=invariant_factors)
    return ElementReduceResult(
        group=group,
        coordinates=coordinates,
        reduced=AbelianElement(group=group, coordinates=reduced),
    )


def elements_equal(
    invariant_factors: tuple[int, ...],
    coordinates_a: tuple[int, ...],
    coordinates_b: tuple[int, ...],
) -> ElementEqualResult:
    reduced_a = tuple(
        coordinate % factor
        for coordinate, factor in zip(coordinates_a, invariant_factors, strict=True)
    )
    reduced_b = tuple(
        coordinate % factor
        for coordinate, factor in zip(coordinates_b, invariant_factors, strict=True)
    )
    group = AbelianPresentation(invariant_factors=invariant_factors)
    return ElementEqualResult(
        group=group,
        elements_a=AbelianElement(
            group=group,
            coordinates=tuple(
                coordinate % factor
                for coordinate, factor in zip(coordinates_a, invariant_factors, strict=True)
            ),
        ),
        elements_b=AbelianElement(
            group=group,
            coordinates=tuple(
                coordinate % factor
                for coordinate, factor in zip(coordinates_b, invariant_factors, strict=True)
            ),
        ),
        equal=reduced_a == reduced_b,
    )


def verify_elements_equal(claim: ElementEqualResult) -> bool:
    """Check the equality relation asserted by a serialized element claim."""

    try:
        return (
            claim.elements_a.group == claim.group
            and claim.elements_b.group == claim.group
            and (claim.elements_a.coordinates == claim.elements_b.coordinates)
            is claim.equal
        )
    except (TypeError, ValueError):
        return False


def verify_element_reduction(claim: ElementReduceResult) -> bool:
    """Check a reduced coordinate claim against its retained group source."""

    try:
        return reduce_element(claim.group.invariant_factors, claim.coordinates).reduced == claim.reduced
    except (TypeError, ValueError):
        return False


def element_order(
    invariant_factors: tuple[int, ...], coordinates: tuple[int, ...]
) -> ElementOrderResult:
    reduced = [
        coordinate % factor
        for coordinate, factor in zip(coordinates, invariant_factors, strict=True)
    ]
    order = 1
    for coordinate, factor in zip(reduced, invariant_factors, strict=True):
        if coordinate != 0:
            order = lcm(order, factor // gcd(coordinate, factor))
    group = AbelianPresentation(invariant_factors=invariant_factors)
    return ElementOrderResult(
        group=group,
        element=AbelianElement(
            group=group,
            coordinates=tuple(
                coordinate % factor
                for coordinate, factor in zip(coordinates, invariant_factors, strict=True)
            ),
        ),
        order=order,
    )


def verify_element_order(claim: ElementOrderResult) -> bool:
    """Check the order relation asserted by a serialized element claim."""

    try:
        return (
            claim.element.group == claim.group
            and element_order(
                claim.group.invariant_factors,
                claim.element.coordinates,
            ).order == claim.order
        )
    except (TypeError, ValueError):
        return False


def _smith_diagonal(augmented_rows: list[list[int]]) -> list[int]:
    """Compute diagonal entries of the exact integer Smith normal form."""
    from sympy import Matrix
    from sympy.matrices.normalforms import smith_normal_form

    smith = smith_normal_form(Matrix(augmented_rows), domain=None)
    return [abs(int(smith[i, i])) for i in range(min(smith.rows, smith.cols))]


def generated_subgroup(
    invariant_factors: tuple[int, ...], generators: tuple[tuple[int, ...], ...]
) -> SubgroupGeneratedResult:
    """Compute the index of a subgroup generated by given elements."""
    factors = invariant_factors
    rows = [[0] * len(factors) for _ in factors]
    for index, factor in enumerate(factors):
        rows[index][index] = factor
    for generator in generators:
        for index, coordinate in enumerate(generator):
            rows[index].append(coordinate)
    diagonal = _smith_diagonal(rows)
    index = 1
    for factor in diagonal:
        if factor > 1:
            index *= factor
    group = AbelianPresentation(invariant_factors=invariant_factors)
    subgroup = AbelianSubgroup(
        group=group,
        generators=tuple(
            AbelianElement(
                group=group,
                coordinates=tuple(
                    coordinate % factor
                    for coordinate, factor in zip(generator, invariant_factors, strict=True)
                ),
            )
            for generator in generators
        ),
    )
    return SubgroupGeneratedResult(
        subgroup=subgroup,
        index=index,
    )


def quotient_group(
    invariant_factors: tuple[int, ...],
    subgroup_generators: tuple[tuple[int, ...], ...],
) -> QuotientResult:
    """Compute the quotient group via Smith normal form."""
    factors = invariant_factors
    rows = [[0] * len(factors) for _ in factors]
    for index, factor in enumerate(factors):
        rows[index][index] = factor
    for generator in subgroup_generators:
        for index, coordinate in enumerate(generator):
            rows[index].append(coordinate)
    diagonal = _smith_diagonal(rows)
    quotient_factors = tuple(factor for factor in diagonal if factor > 1)
    order = 1
    for factor in quotient_factors:
        order *= factor
    group = AbelianPresentation(invariant_factors=invariant_factors)
    subgroup = AbelianSubgroup(
        group=group,
        generators=tuple(
            AbelianElement(
                group=group,
                coordinates=tuple(
                    coordinate % factor
                    for coordinate, factor in zip(generator, invariant_factors, strict=True)
                ),
            )
            for generator in subgroup_generators
        ),
    )
    return QuotientResult(
        quotient=AbelianQuotient(
            group=group,
            subgroup=subgroup,
            invariant_factors=quotient_factors,
        ),
        quotient_order=order,
    )


def verify_generated_subgroup(claim: SubgroupGeneratedResult) -> bool:
    """Check the index relation asserted by a serialized subgroup claim."""

    try:
        return (
            generated_subgroup(
                claim.subgroup.group.invariant_factors,
                tuple(element.coordinates for element in claim.subgroup.generators),
            ).index
            == claim.index
        )
    except (TypeError, ValueError):
        return False


def verify_quotient_group(claim: QuotientResult) -> bool:
    """Check the quotient presentation asserted by a serialized claim."""

    try:
        expected = quotient_group(
            claim.quotient.group.invariant_factors,
            tuple(element.coordinates for element in claim.quotient.subgroup.generators),
        )
        return (
            expected.quotient.invariant_factors == claim.quotient.invariant_factors
            and expected.quotient_order == claim.quotient_order
        )
    except (TypeError, ValueError):
        return False


def verify_presentation_normalization(claim: PresentationNormalizeResult) -> bool:
    """Check the canonical presentation against its retained raw factors."""

    try:
        return normalize_presentation(claim.source.invariant_factors) == claim
    except (TypeError, ValueError):
        return False


__all__ = [
    "element_order",
    "elements_equal",
    "generated_subgroup",
    "normalize_presentation",
    "quotient_group",
    "reduce_element",
    "verify_element_order",
    "verify_element_reduction",
    "verify_elements_equal",
    "verify_generated_subgroup",
    "verify_presentation_normalization",
    "verify_quotient_group",
]
