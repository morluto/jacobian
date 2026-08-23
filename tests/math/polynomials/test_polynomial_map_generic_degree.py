"""Exact generic-degree contracts for bounded rational polynomial maps."""

from __future__ import annotations

import shutil
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.maps import RationalPolynomialMap, _singular
from jacobian.math.polynomials.maps._models import (
    GenericDegreeRequest,
    GenericDegreeResult,
)
from jacobian.math.polynomials.maps._operations import compute_generic_degree
from jacobian.math.polynomials.maps._tools import TOOLS
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(
    variables: tuple[str, ...],
    terms: dict[tuple[int, ...], int | Fraction],
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


def _map(
    variables: tuple[str, ...],
    *components: dict[tuple[int, ...], int | Fraction],
) -> RationalPolynomialMap:
    return RationalPolynomialMap(
        input_variables=variables,
        output_polynomials=tuple(
            _polynomial(variables, component) for component in components
        ),
    )


def _compute(polynomial_map: RationalPolynomialMap) -> GenericDegreeResult:
    return compute_generic_degree(GenericDegreeRequest(polynomial_map=polynomial_map))


requires_singular = pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="Singular 4.4 backend is not installed",
)


def test_operation_is_one_admitted_atomic_generic_fiber_computation() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "polynomial.map.generic_degree.compute"
    )

    assert operation.version == "1"
    assert operation.examples
    request = operation.request_type.model_validate(operation.examples[0].input)
    assert request.polynomial_map.input_variables == ("x", "y")
    description = GenericDegreeRequest.model_json_schema()["properties"][
        "polynomial_map"
    ]["description"]
    assert "96 aggregate terms" in description
    assert "65536 encoded bytes" in description
    assert "Bezout" in description


def test_missing_backend_is_operational_unavailability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_singular.shutil, "which", lambda _name: None)

    result = _compute(_map(("x",), {(1,): 1}))

    assert result.outcome == "UNAVAILABLE"
    assert result.degree is None
    assert result.evidence is None
    assert result.source == _map(("x",), {(1,): 1})


def test_request_rejects_unproved_dimension_degree_and_height() -> None:
    with pytest.raises(ValidationError, match="3-variable"):
        GenericDegreeRequest(
            polynomial_map=_map(
                ("w", "x", "y", "z"),
                {(1, 0, 0, 0): 1},
            )
        )
    with pytest.raises(ValidationError, match="3-component"):
        GenericDegreeRequest(
            polynomial_map=_map(
                ("x",),
                {(1,): 1},
                {(1,): 2},
                {(1,): 3},
                {(1,): 4},
            )
        )
    with pytest.raises(ValidationError, match="total degree 8"):
        GenericDegreeRequest(polynomial_map=_map(("x",), {(9,): 1}))
    with pytest.raises(ValidationError, match="64-digit"):
        GenericDegreeRequest(polynomial_map=_map(("x",), {(1,): int("1" * 65)}))


def test_request_bounds_component_and_aggregate_support() -> None:
    monomials = [
        (a, b, c) for a in range(9) for b in range(9 - a) for c in range(9 - a - b)
    ]
    with pytest.raises(ValidationError, match="48-term"):
        GenericDegreeRequest(
            polynomial_map=_map(
                ("x", "y", "z"),
                dict.fromkeys(monomials[:49], 1),
            )
        )
    with pytest.raises(ValidationError, match="96-term"):
        GenericDegreeRequest(
            polynomial_map=_map(
                ("x", "y", "z"),
                *(
                    dict.fromkeys(monomials[offset : offset + 33], 1)
                    for offset in (0, 33, 66)
                ),
            )
        )


def test_request_accepts_the_exact_degree_and_bezout_boundary() -> None:
    request = GenericDegreeRequest(
        polynomial_map=_map(
            ("x", "y", "z"),
            {(8, 0, 0): 1},
            {(0, 8, 0): 1},
            {(0, 0, 8): 1},
        )
    )
    assert request.polynomial_map.input_variables == ("x", "y", "z")


@requires_singular
@pytest.mark.requires_backend("singular")
@pytest.mark.parametrize(
    ("polynomial_map", "degree"),
    [
        (_map(("x", "y"), {(1, 0): 1}, {(0, 1): 1}), 1),
        (
            _map(
                ("x", "y"),
                {(1, 0): 1, (0, 1): 1},
                {(0, 1): 1},
            ),
            1,
        ),
        (_map(("x", "y"), {(2, 0): 1}, {(0, 1): 1}), 2),
        (_map(("x", "y"), {(2, 0): 1}, {(0, 3): 1}), 6),
    ],
    ids=("identity", "triangular", "quadratic", "degrees_two_and_three"),
)
def test_known_generic_degrees(
    polynomial_map: RationalPolynomialMap,
    degree: int,
) -> None:
    result = _compute(polynomial_map)

    assert result.outcome == "GENERICALLY_FINITE"
    assert result.degree == degree
    assert result.evidence is not None
    assert len(result.evidence.standard_monomials) == degree


@requires_singular
@pytest.mark.requires_backend("singular")
def test_same_generic_ideal_distinguishes_both_nonfinite_outcomes() -> None:
    non_dominant = _compute(_map(("x", "y"), {(1, 0): 1}, {}))
    positive_dimensional = _compute(_map(("x", "y"), {(1, 0): 1}))

    assert non_dominant.outcome == "NOT_DOMINANT"
    assert non_dominant.degree is None
    assert positive_dimensional.outcome == "DOMINANT_NOT_GENERICALLY_FINITE"
    assert positive_dimensional.degree is None
    assert non_dominant.evidence is not None
    assert positive_dimensional.evidence is not None


@requires_singular
@pytest.mark.requires_backend("singular")
def test_rectangular_map_relations_establish_non_dominance() -> None:
    result = _compute(_map(("x",), {(1,): 1}, {(2,): 1}))

    assert result.outcome == "NOT_DOMINANT"
    assert result.degree is None


@requires_singular
@pytest.mark.requires_backend("singular")
def test_generic_degree_is_invariant_under_linear_coordinate_changes() -> None:
    source_changed = _compute(
        _map(
            ("x", "y"),
            {(0, 2): 1},
            {(1, 0): 1},
        )
    )
    target_changed = _compute(
        _map(
            ("x", "y"),
            {(2, 0): 1, (0, 1): 1},
            {(0, 1): 1},
        )
    )

    assert source_changed.degree == 2
    assert target_changed.degree == 2


@requires_singular
@pytest.mark.requires_backend("singular")
def test_atlas_weighted_lift_k1_d2_has_generic_degree_three() -> None:
    # This is the exact crater-map fixture pinned by the issue's Atlas source.
    result = _compute(
        _map(
            ("x", "y", "z"),
            {
                (3, 3, 1): 1,
                (2, 4, 0): 3,
                (2, 2, 1): 3,
                (1, 3, 0): 7,
                (1, 1, 1): 3,
                (0, 2, 0): 4,
                (0, 0, 1): 1,
            },
            {
                (3, 2, 1): 3,
                (2, 3, 0): 9,
                (2, 1, 1): 6,
                (1, 2, 0): 12,
                (1, 0, 1): 3,
                (0, 1, 0): 1,
            },
            {
                (3, 0, 1): -1,
                (2, 1, 0): -3,
                (1, 0, 0): 2,
            },
        )
    )

    assert result.outcome == "GENERICALLY_FINITE"
    assert result.degree == 3
    assert result.evidence is not None
    assert len(result.evidence.standard_monomials) == 3


@pytest.fixture(scope="module")
def quadratic_result() -> GenericDegreeResult:
    if shutil.which("Singular") is None:
        pytest.skip("Singular 4.4 backend is not installed")
    return _compute(_map(("x", "y"), {(2, 0): 1}, {(0, 1): 1}))


def test_branch_specialization_does_not_replace_generic_degree(
    quadratic_result: GenericDegreeResult,
) -> None:
    # Over target x-coordinate 0, x^2=0 has one distinct point but length two.
    assert quadratic_result.outcome == "GENERICALLY_FINITE"
    assert quadratic_result.degree == 2


def test_result_round_trip_preserves_axes_and_evidence(
    quadratic_result: GenericDegreeResult,
) -> None:
    replayed = GenericDegreeResult.model_validate_json(
        quadratic_result.model_dump_json()
    )

    assert replayed == quadratic_result
    assert replayed.source.input_variables == ("x", "y")


def test_forged_degree_and_source_are_rejected(
    quadratic_result: GenericDegreeResult,
) -> None:
    forged_degree = quadratic_result.model_dump(mode="json")
    forged_degree["degree"] = 1
    with pytest.raises(ValidationError, match="does not match"):
        GenericDegreeResult.model_validate(forged_degree)

    forged_source = quadratic_result.model_dump(mode="json")
    forged_source["source"]["output_polynomials"][0]["polynomial"]["terms"][0][
        "coefficient"
    ]["num"] = "3"
    with pytest.raises(ValidationError, match=r"source|reconstruct|reduce"):
        GenericDegreeResult.model_validate(forged_source)


@requires_singular
@pytest.mark.requires_backend("singular")
def test_source_names_cannot_collide_with_generic_target_parameters() -> None:
    result = _compute(_map(("t1",), {(2,): 1}))

    assert result.degree == 2
    assert result.evidence is not None
    assert result.evidence.source_variable_order == ("t1",)
    assert result.evidence.target_parameters == ("t1",)
