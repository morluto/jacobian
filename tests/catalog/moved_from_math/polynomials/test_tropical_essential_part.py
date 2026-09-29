"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/polynomials/test_tropical_essential_part.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

import json
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.polynomials.tropical import (
    TropicalPolynomial,
    TropicalPolynomialEssentialPart,
    TropicalPolynomialTerm,
    TropicalScalar,
    TropicalSemiring,
)
from jacobian.math.polynomials.tropical._models import EssentialPartRequest


def _polynomial(
    variables: tuple[str, ...],
    exponents: tuple[tuple[int, ...], ...],
    coefficients: tuple[Fraction | int, ...],
    convention: str = "MIN_PLUS",
) -> TropicalPolynomial:
    semiring = TropicalSemiring(convention=convention, base="QQ")  # type: ignore[arg-type]
    return TropicalPolynomial(
        semiring=semiring,
        variables=variables,
        terms=tuple(
            TropicalPolynomialTerm(
                exponents=exponent,
                coefficient=TropicalScalar(
                    semiring=semiring,
                    kind="FINITE",
                    value=CanonicalRational.from_fraction(Fraction(coefficient)),
                ),
            )
            for exponent, coefficient in zip(exponents, coefficients, strict=True)
        ),
    )


def _feasible_by_fourier_motzkin(
    rows: tuple[tuple[tuple[Fraction, ...], Fraction], ...],
) -> bool:
    """Independent exact feasibility check for small rational inequalities."""
    current = list(rows)
    dimension = len(rows[0][0]) if rows else 0
    for _axis in range(dimension):
        positive, negative, zero = [], [], []
        for coefficients, bound in current:
            lead = coefficients[0]
            if lead > 0:
                positive.append((coefficients, bound))
            elif lead < 0:
                negative.append((coefficients, bound))
            else:
                zero.append((coefficients[1:], bound))
        combined = list(zero)
        for upper, upper_bound in positive:
            for lower, lower_bound in negative:
                up_scale, low_scale = -lower[0], upper[0]
                coefficients = tuple(
                    up_scale * upper[i] + low_scale * lower[i]
                    for i in range(1, len(upper))
                )
                combined.append(
                    (coefficients, up_scale * upper_bound + low_scale * lower_bound)
                )
        for coefficients, bound in combined:
            if not any(coefficients) and bound < 0:
                return False
        current = list(dict.fromkeys(combined))
    return all(any(coefficients) or bound >= 0 for coefficients, bound in current)


def _inequality_oracle(poly: TropicalPolynomial) -> tuple[int, ...]:
    attained = []
    maximize = poly.semiring.convention == "MAX_PLUS"
    for index, term in enumerate(poly.terms):
        term_rows = []
        for other_index, other in enumerate(poly.terms):
            if index == other_index:
                continue
            sign = 1 if maximize else -1
            # c_i + a_i.x >=/<= c_j + a_j.x, respectively.
            coefficients = tuple(
                Fraction(sign * (other.exponents[axis] - term.exponents[axis]))
                for axis in range(len(poly.variables))
            )
            bound = -sign * (
                other.coefficient.value.as_fraction()
                - term.coefficient.value.as_fraction()
            )
            term_rows.append((coefficients, bound))
        if _feasible_by_fourier_motzkin(tuple(term_rows)):
            attained.append(index)
    return tuple(attained)


def _assert_face_incidence(
    poly: TropicalPolynomial, result: TropicalPolynomialEssentialPart
) -> None:
    for face in result.finite_faces:
        normal = tuple(value.as_fraction() for value in face.normal)
        offset = face.offset.as_fraction()
        incident = tuple(
            index
            for index, term in enumerate(poly.terms)
            if sum(
                normal[axis] * exponent
                for axis, exponent in enumerate(
                    (*term.exponents, term.coefficient.value.as_fraction())
                )
            )
            == offset
        )
        assert incident == face.source_term_indices
    for face in result.face_incidence:
        for parent_index in face.maximal_finite_face_indices:
            parent_terms = result.finite_faces[parent_index].source_term_indices
            assert set(face.source_term_indices).issubset(parent_terms)


def _incidence_signature(result: TropicalPolynomialEssentialPart):
    return tuple(
        (
            face.dimension,
            face.source_term_indices,
            face.maximal_finite_face_indices,
        )
        for face in result.face_incidence
    )


def test_public_manifest_example_executes_as_a_source_bound_result() -> None:
    tool = next(
        operation
        for operation in BUILTIN_TOOLS
        if operation.operation_id == "tropical.polynomial.essential_part.compute"
    )
    request = EssentialPartRequest.model_validate_json(
        json.dumps(tool.examples[0].input), strict=True
    )

    result = tool.run(request)

    assert result.essential_term_indices == tuple(range(5))
    assert result.source == request.polynomial
    assert result.polynomial == request.polynomial
    assert (
        TropicalPolynomialEssentialPart.model_validate_json(
            result.model_dump_json(), strict=True
        )
        == result
    )
