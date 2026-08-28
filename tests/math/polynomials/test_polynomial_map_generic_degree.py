"""Exact generic-degree contracts for bounded rational polynomial maps."""

from __future__ import annotations

import json
import shutil
from fractions import Fraction
from typing import Any, cast

import pytest
import sympy
from tests.math.polynomials._support import polynomial_validation_error

import jacobian.math.polynomials.maps.operations as operations
from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.maps import (
    RationalPolynomialMap,
    _generic_degree,
)
from jacobian.math.polynomials.maps._generic_degree import (
    GenericFiberReplayLimitError,
    enumerate_standard_monomials,
    require_certificate_reconstructs_from_source,
    validate_generic_fiber_certificate,
)
from jacobian.math.polynomials.maps._models import (
    MAX_GENERIC_DEGREE_ENCODED_MAP_BYTES,
    GenericDegreeRequest,
    GenericDegreeResult,
    GenericFiberCertificate,
    GenericFiberPolynomial,
    GenericFiberTerm,
)
from jacobian.math.polynomials.maps._singular import SingularGenericFiberResult
from jacobian.math.polynomials.maps._tools import TOOLS
from jacobian.math.polynomials.maps.operations import generic_degree
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _run_generic_degree(request: GenericDegreeRequest) -> GenericDegreeResult:
    return generic_degree(request.polynomial_map, request.resource_budget)


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
    return _run_generic_degree(GenericDegreeRequest(polynomial_map=polynomial_map))


def _generic_fiber_coefficient(
    exponents: tuple[int, int],
    numerator: int = 1,
) -> RationalFunction:
    one = SparseRationalPolynomial(
        terms=(
            RationalPolynomialTerm(
                coefficient=CanonicalRational.from_fraction(Fraction(1)),
                exponents=(0, 0),
            ),
        )
    )
    return RationalFunction(
        variables=("t1", "t2"),
        numerator=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(numerator)),
                    exponents=exponents,
                ),
            )
        ),
        denominator=one,
    )


def _fiber_polynomial(
    terms: tuple[tuple[tuple[int, int], RationalFunction], ...],
) -> GenericFiberPolynomial:
    return GenericFiberPolynomial(
        terms=tuple(
            GenericFiberTerm(coefficient=coefficient, source_exponents=exponents)
            for exponents, coefficient in terms
        )
    )


def _identity_certificate() -> GenericFiberCertificate:
    empty = _fiber_polynomial(())
    unit = _fiber_polynomial((((0, 0), _generic_fiber_coefficient((0, 0))),))
    y_minus_t2 = _fiber_polynomial(
        (
            ((0, 1), _generic_fiber_coefficient((0, 0))),
            ((0, 0), _generic_fiber_coefficient((0, 1), -1)),
        )
    )
    x_minus_t1 = _fiber_polynomial(
        (
            ((1, 0), _generic_fiber_coefficient((0, 0))),
            ((0, 0), _generic_fiber_coefficient((1, 0), -1)),
        )
    )
    return GenericFiberCertificate(
        target_parameters=("t1", "t2"),
        source_variable_order=("x", "y"),
        basis=(y_minus_t2, x_minus_t1),
        basis_from_source=((empty, unit), (unit, empty)),
        standard_monomials=((0, 0),),
    )


def _identity_source() -> RationalPolynomialMap:
    return _map(("x", "y"), {(1, 0): 1}, {(0, 1): 1})


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

    assert operation.examples
    request = operation.request_type.model_validate(operation.examples[0].input)
    assert request.polynomial_map.input_variables == ("x", "y")
    description = GenericDegreeRequest.model_json_schema()["properties"][
        "polynomial_map"
    ]["description"]
    assert "96 aggregate terms" in description
    assert f"{MAX_GENERIC_DEGREE_ENCODED_MAP_BYTES} encoded bytes" in description
    assert "Bezout" in description
    assert set(
        GenericDegreeRequest.model_json_schema()["$defs"][
            "GenericDegreeComputationBudget"
        ]["properties"]
    ) == {"wall_seconds"}
    assert {"method", "backend_version"}.isdisjoint(
        GenericDegreeResult.model_json_schema()["properties"]
    )
    assert (
        "singular" not in json.dumps(GenericDegreeRequest.model_json_schema()).lower()
    )
    assert "singular" not in operation.description.lower()


def test_missing_backend_is_operational_unavailability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    result = _compute(_map(("x",), {(1,): 1}))

    assert result.outcome == "UNAVAILABLE"
    assert result.degree is None
    assert result.evidence is None
    assert result.source == _map(("x",), {(1,): 1})


def test_request_rejects_unproved_dimension_degree_and_height() -> None:
    with pytest.raises(OperationDomainValidationError):
        _run_generic_degree(
            GenericDegreeRequest(
                polynomial_map=_map(
                    ("w", "x", "y", "z"),
                    {(1, 0, 0, 0): 1},
                )
            )
        )
    with pytest.raises(OperationDomainValidationError):
        _run_generic_degree(
            GenericDegreeRequest(
                polynomial_map=_map(
                    ("x",),
                    {(1,): 1},
                    {(1,): 2},
                    {(1,): 3},
                    {(1,): 4},
                )
            )
        )
    with pytest.raises(OperationDomainValidationError):
        _run_generic_degree(
            GenericDegreeRequest(polynomial_map=_map(("x",), {(9,): 1}))
        )
    with pytest.raises(OperationDomainValidationError):
        _run_generic_degree(
            GenericDegreeRequest(polynomial_map=_map(("x",), {(1,): int("1" * 65)}))
        )


def test_request_bounds_component_and_aggregate_support() -> None:
    monomials = [
        (a, b, c) for a in range(9) for b in range(9 - a) for c in range(9 - a - b)
    ]
    with pytest.raises(OperationDomainValidationError):
        _run_generic_degree(
            GenericDegreeRequest(
                polynomial_map=_map(
                    ("x", "y", "z"),
                    dict.fromkeys(monomials[:49], 1),
                )
            )
        )
    with pytest.raises(OperationDomainValidationError):
        _run_generic_degree(
            GenericDegreeRequest(
                polynomial_map=_map(
                    ("x", "y", "z"),
                    *(
                        cast(
                            dict[tuple[int, ...], int | Fraction],
                            dict.fromkeys(monomials[offset : offset + 33], 1),
                        )
                        for offset in (0, 33, 66)
                    ),
                )
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
@pytest.mark.scale
def test_accepted_triangular_map_returns_replayable_degree_128() -> None:
    result = _compute(
        _map(
            ("x", "y", "z"),
            {(8, 0, 0): 1},
            {(1, 0, 0): 1, (0, 8, 0): 1},
            {(0, 1, 0): 1, (0, 0, 2): 1},
        )
    )

    assert result.outcome == "GENERICALLY_FINITE"
    assert result.degree == 128
    assert result.evidence is not None
    assert len(result.evidence.standard_monomials) == 128
    assert GenericDegreeResult.model_validate_json(result.model_dump_json()) == result


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
@pytest.mark.exhaustive
@pytest.mark.timeout(180)
def test_atlas_weighted_lift_k1_d2_has_generic_degree_three() -> None:
    # This is a slow exact Singular regression, retained in the dedicated
    # Singular/exhaustive lane rather than the ordinary math owner lane.
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
    x, y = sympy.symbols("x y")
    special_fiber = sympy.groebner(
        [x**2, y],
        x,
        y,
        order="lex",
        domain=sympy.QQ,
    )
    standard_monomials = (sympy.Integer(1), x)

    assert {polynomial.as_expr() for polynomial in special_fiber.polys} == {x**2, y}
    assert (
        tuple(special_fiber.reduce(monomial)[1] for monomial in standard_monomials)
        == standard_monomials
    )
    assert special_fiber.reduce(x**2)[1] == 0
    assert sympy.Poly(x**2, x, domain=sympy.QQ).sqf_part().degree() == 1
    assert len(standard_monomials) == 2
    assert quadratic_result.outcome == "GENERICALLY_FINITE"
    assert quadratic_result.degree == 2


@pytest.mark.parametrize(
    "backend",
    (
        SingularGenericFiberResult(outcome="COMPUTED", backend_version="4.4.1"),
        SingularGenericFiberResult(
            outcome="COMPUTED",
            certificate=None,
            dimension=0,
            vector_dimension=None,
            backend_version="4.4.1",
        ),
    ),
)
def test_malformed_computed_backend_state_is_a_typed_error(
    backend: SingularGenericFiberResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operations,
        "run_singular_generic_fiber",
        lambda *_args: backend,
    )

    result = _compute(_map(("x",), {(1,): 1}))

    assert result.outcome == "ERROR"
    assert result.evidence is None
    assert result.degree is None


def test_result_round_trip_preserves_axes_and_evidence(
    quadratic_result: GenericDegreeResult,
) -> None:
    replayed = GenericDegreeResult.model_validate_json(
        quadratic_result.model_dump_json()
    )

    assert replayed == quadratic_result
    assert replayed.source.input_variables == ("x", "y")


def test_standard_monomial_enumeration_admits_the_sparse_leading_ideal() -> None:
    leading = ((64, 0), (1, 1), (0, 64))

    monomials = enumerate_standard_monomials(leading)

    assert monomials is not None
    assert len(monomials) == 127
    assert monomials == tuple(sorted(monomials))
    assert all(
        not all(
            bound <= exponent for bound, exponent in zip(lead, monomial, strict=True)
        )
        for lead in leading
        for monomial in monomials
    )
    members = set(monomials)
    assert all(
        tuple(
            exponent - 1 if index == variable else exponent
            for index, exponent in enumerate(monomial)
        )
        in members
        for monomial in monomials
        for variable in range(2)
        if monomial[variable]
    )


def test_standard_monomial_enumeration_rejects_unbounded_quotients() -> None:
    with pytest.raises(GenericFiberReplayLimitError, match="standard-monomial bound"):
        enumerate_standard_monomials(((23, 0), (0, 23)))
    with pytest.raises(GenericFiberReplayLimitError, match="standard-monomial bound"):
        enumerate_standard_monomials(((512, 0), (0, 512)))
    assert enumerate_standard_monomials(((1, 1), (2, 1))) is None


def test_certificate_verification_known_answer() -> None:
    outcome, degree = validate_generic_fiber_certificate(
        _identity_source(),
        _identity_certificate(),
    )

    assert outcome == "GENERICALLY_FINITE"
    assert degree == 1


@pytest.mark.parametrize(
    ("constant", "message"),
    (
        ("MAX_GENERIC_FIBER_REPLAY_REDUCTION_STEPS", "reduction-step"),
        ("MAX_GENERIC_FIBER_REPLAY_COEFFICIENT_OPERATIONS", "coefficient-operation"),
        ("MAX_GENERIC_FIBER_REPLAY_COEFFICIENT_PRODUCTS", "coefficient-product"),
    ),
    ids=("reduction-steps", "coefficient-operations", "coefficient-products"),
)
def test_declared_replay_limits_are_consulted_during_replay(
    constant: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_generic_degree, constant, 0)

    with pytest.raises(GenericFiberReplayLimitError, match=message):
        validate_generic_fiber_certificate(
            _identity_source(),
            _identity_certificate(),
        )


@requires_singular
@pytest.mark.requires_backend("singular")
def test_source_names_cannot_collide_with_generic_target_parameters() -> None:
    result = _compute(_map(("t1",), {(2,): 1}))

    assert result.degree == 2
    assert result.evidence is not None
    assert result.evidence.source_variable_order == ("t1",)
    assert result.evidence.target_parameters == ("t1",)


def _power_map_certificate(power: int) -> GenericFiberCertificate:
    empty = _fiber_polynomial(())
    unit = _fiber_polynomial((((0, 0), _generic_fiber_coefficient((0, 0))),))
    y_minus_t2 = _fiber_polynomial(
        (
            ((0, 1), _generic_fiber_coefficient((0, 0))),
            ((0, 0), _generic_fiber_coefficient((0, 1), -1)),
        )
    )
    x_power_minus_t1 = _fiber_polynomial(
        (
            ((power, 0), _generic_fiber_coefficient((0, 0))),
            ((0, 0), _generic_fiber_coefficient((1, 0), -1)),
        )
    )
    return GenericFiberCertificate(
        target_parameters=("t1", "t2"),
        source_variable_order=("x", "y"),
        basis=(y_minus_t2, x_power_minus_t1),
        basis_from_source=((empty, unit), (unit, empty)),
        standard_monomials=tuple((index, 0) for index in range(power)),
    )


def _power_map_result(power: int) -> GenericDegreeResult:
    return GenericDegreeResult(
        outcome="GENERICALLY_FINITE",
        source=_map(("x", "y"), {(power, 0): 1}, {(0, 1): 1}),
        degree=power,
        evidence=_power_map_certificate(power),
    )


def test_consistent_results_round_trip_without_replaying_at_deserialization() -> None:
    for power in (2, 3):
        result = _power_map_result(power)
        assert result.evidence is not None
        require_certificate_reconstructs_from_source(result.source, result.evidence)
        replayed = GenericDegreeResult.model_validate(result.model_dump(mode="json"))

        assert replayed == result


def test_serialized_evidence_cannot_be_presented_against_a_different_source() -> None:
    result = _power_map_result(3)
    forged = result.model_dump(mode="json")
    forged["source"] = _power_map_result(2).source.model_dump(mode="json")

    replayed = GenericDegreeResult.model_validate(forged)
    assert result.evidence is not None
    assert replayed.source != result.source
    with pytest.raises(ValueError, match="reconstruct"):
        require_certificate_reconstructs_from_source(
            _power_map_result(2).source,
            result.evidence,
        )


def test_serialized_coefficient_support_is_counted_before_nested_construction() -> None:
    nested_terms = [
        {"coefficient": {"num": "1", "den": "1"}, "exponents": [index, 0]}
        for index in range(3)
    ]
    polynomial = {
        "terms": [
            {
                "coefficient": {
                    "numerator": {"terms": nested_terms},
                    "denominator": {"terms": nested_terms},
                },
                "source_exponents": [1999 - index],
            }
            for index in range(2000)
        ],
    }
    payload: dict[str, Any] = {
        "target_parameters": ["t1", "t2"],
        "source_variable_order": ["x", "y"],
        "monomial_order": "LEX",
        "basis": [polynomial],
        "basis_from_source": [[polynomial]],
        "standard_monomials": [],
    }

    with polynomial_validation_error():
        GenericFiberCertificate.model_validate(payload)
