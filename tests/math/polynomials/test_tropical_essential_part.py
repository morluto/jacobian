from __future__ import annotations

import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.tropical import (
    TropicalPolynomial,
    TropicalPolynomialEssentialPart,
    TropicalPolynomialTerm,
    TropicalScalar,
    TropicalSemiring,
    tropical_polynomial_essential_part,
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


@pytest.mark.parametrize("convention", ["MIN_PLUS", "MAX_PLUS"])
def test_square_center_tie_is_retained_and_all_face_incidence_is_source_bound(
    convention: str,
) -> None:
    poly = _polynomial(
        ("x", "y"),
        ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
        (0, 0, 0, 0, 0),
        convention,
    )
    result = tropical_polynomial_essential_part(poly)

    assert result.essential_term_indices == _inequality_oracle(poly) == (0, 1, 2, 3, 4)
    assert result.inessential_term_indices == ()
    assert result.polynomial.terms == poly.terms
    # The center is never uniquely active: its affine value is the average of
    # the opposite corner values. The issue's attained/tie-inclusive contract
    # intentionally retains it because all five terms tie at (0, 0).
    assert len(result.finite_faces) == 1
    assert len(result.face_incidence) == 9
    assert any(
        face.dimension == 2 and face.source_term_indices == (0, 1, 2, 3, 4)
        for face in result.face_incidence
    )
    assert all(len(face.normal) == 3 for face in result.hull_facets)
    expected = (
        (0, (0,), (0,)),
        (0, (1,), (0,)),
        (0, (3,), (0,)),
        (0, (4,), (0,)),
        (1, (0, 1), (0,)),
        (1, (0, 3), (0,)),
        (1, (1, 4), (0,)),
        (1, (3, 4), (0,)),
        (2, (0, 1, 2, 3, 4), (0,)),
    )
    assert _incidence_signature(result) == expected
    _assert_face_incidence(poly, result)


@pytest.mark.parametrize(
    ("convention", "coefficients", "active", "normal_sign"),
    [
        ("MIN_PLUS", (0, 0, 10, 0), (0, 1, 3), -1),
        ("MAX_PLUS", (0, 0, -10, 0), (0, 1, 3), 1),
    ],
)
def test_rank_deficient_lift_uses_relative_facets_and_exact_inequality_oracle(
    convention: str,
    coefficients: tuple[int, ...],
    active: tuple[int, ...],
    normal_sign: int,
) -> None:
    poly = _polynomial(
        ("x", "y", "z"),
        ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)),
        coefficients,
        convention,
    )
    result = tropical_polynomial_essential_part(poly)

    assert result.lifted_affine_dimension == 2 < len(poly.variables) + 1
    assert result.essential_term_indices == _inequality_oracle(poly) == active
    assert result.inessential_term_indices == (2,)
    assert any(row[-2].as_fraction() == 0 for row in result.affine_equalities)
    assert _incidence_signature(result) == (
        (0, (0,), (0,)),
        (0, (3,), (0,)),
        (1, (0, 1, 3), (0,)),
    )
    for face in result.finite_faces:
        assert face.normal[-1].as_fraction() * normal_sign > 0
        assert face.source_term_indices
    _assert_face_incidence(poly, result)


def test_affine_height_relation_makes_all_terms_tie_somewhere() -> None:
    poly = _polynomial(
        ("x", "y", "z"),
        ((0, 0, 0), (1, 0, 0), (2, 0, 0)),
        (0, 2, 4),
    )
    result = tropical_polynomial_essential_part(poly)

    assert result.essential_term_indices == _inequality_oracle(poly) == (0, 1, 2)
    assert len(result.finite_faces) == 1
    assert len(result.face_incidence) == 3
    assert result.finite_faces[0].source_term_indices == (0, 1, 2)
    _assert_face_incidence(poly, result)


def test_fifty_term_rank_one_input_fits_the_exact_dd_pair_envelope() -> None:
    poly = _polynomial(
        ("x", "y", "z", "w"),
        tuple((index, 0, 0, 0) for index in range(50)),
        (0,) * 50,
    )
    result = tropical_polynomial_essential_part(poly)

    assert result.essential_term_indices == tuple(range(50))
    assert len(result.finite_faces) == 1
    assert len(result.face_incidence) == 3


def test_sixty_four_term_lift_rejected_before_hull_by_candidate_pair_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poly = _polynomial(
        ("x", "y", "z", "w"),
        tuple((index, 0, 0, 0) for index in range(64)),
        (0,) * 64,
    )

    def unexpected_hull(*_args, **_kwargs):
        pytest.fail("the DD hull backend must not run after preflight rejection")

    monkeypatch.setattr(
        "jacobian.math.polynomials.tropical.essential_part.points_to_facets",
        unexpected_hull,
    )
    with pytest.raises(OperationResourceAdmissionError, match="candidate pairs"):
        tropical_polynomial_essential_part(poly)


def test_face_work_is_rejected_before_hull_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exponents = tuple(
        sorted(
            (index, (index * index) % 31, (index * 7) % 31, (index * 11) % 31)
            for index in range(50)
        )
    )
    poly = _polynomial(("x", "y", "z", "w"), exponents, (0,) * len(exponents))

    def unexpected_hull(*_args, **_kwargs):
        pytest.fail("the hull backend must not run after face-work rejection")

    monkeypatch.setattr(
        "jacobian.math.polynomials.tropical.essential_part.points_to_facets",
        unexpected_hull,
    )
    with pytest.raises(OperationResourceAdmissionError, match="face-work"):
        tropical_polynomial_essential_part(poly)


def test_face_incidence_records_a_face_shared_by_two_maximal_finite_faces() -> None:
    poly = _polynomial(
        ("x", "y"),
        ((0, 0), (0, 1), (1, 1), (2, 1), (3, 2)),
        (-3, 1, 3, 2, -1),
    )

    result = tropical_polynomial_essential_part(poly)

    assert result.essential_term_indices == _inequality_oracle(poly) == (0, 1, 3, 4)
    assert tuple(
        (face.dimension, face.source_term_indices, face.maximal_finite_face_indices)
        for face in result.face_incidence
    ) == (
        (0, (0,), (0, 1)),
        (0, (1,), (1,)),
        (0, (3,), (0,)),
        (0, (4,), (0, 1)),
        (1, (0, 1), (1,)),
        (1, (0, 3), (0,)),
        (1, (0, 4), (0, 1)),
        (1, (1, 4), (1,)),
        (1, (3, 4), (0,)),
        (2, (0, 1, 4), (1,)),
        (2, (0, 3, 4), (0,)),
    )
    _assert_face_incidence(poly, result)


@pytest.mark.parametrize(
    ("exponents", "expected_facets"),
    [
        (
            tuple((t, t**2, t**3, t**4) for t in range(6)),
            9,
        ),
        (
            tuple(
                sorted(
                    tuple(
                        2 + (1 if axis == active else 0) * direction
                        for axis in range(4)
                    )
                    for active in range(3)
                    for direction in (-1, 1)
                )
            ),
            8,
        ),
    ],
    ids=("rank-four-moment-curve", "embedded-octahedron"),
)
def test_rank_aware_facet_bound_admits_intrinsic_hull_facets(
    exponents: tuple[tuple[int, ...], ...], expected_facets: int
) -> None:
    poly = _polynomial(("x", "y", "z", "w"), exponents, (0,) * len(exponents))

    result = tropical_polynomial_essential_part(poly)

    assert result.lifted_affine_dimension == (4 if expected_facets == 9 else 3)
    assert len(result.hull_facets) == expected_facets
    assert result.essential_term_indices == _inequality_oracle(poly)
    _assert_face_incidence(poly, result)


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
