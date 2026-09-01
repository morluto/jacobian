"""Admission boundaries for exact plane semialgebraic component profiles."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from itertools import product
from math import gcd
from threading import Event
from time import monotonic

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis.intervals import ClosedRationalInterval, RationalBox
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue
from jacobian.math.polynomials.real_algebra import compute_plane_component_profile
from jacobian.math.polynomials.real_algebra._plane_component_bounds import (
    MAX_PLANE_COMPONENT_PROJECTED_COEFFICIENT_DIGITS,
    plane_projection_coefficient_bound,
)
from jacobian.math.polynomials.real_algebra._plane_component_models import (
    MAX_PLANE_COMPONENT_COEFFICIENT_DIGITS,
    MAX_PLANE_COMPONENT_POINT_COEFFICIENT_DIGITS,
    MAX_PLANE_COMPONENT_POINT_DEGREE,
    MAX_PLANE_COMPONENT_POLYNOMIALS,
    MAX_PLANE_COMPONENT_SAMPLE_COEFFICIENT_DIGITS,
    MAX_PLANE_COMPONENT_SAMPLE_DEGREE,
    MAX_PLANE_COMPONENT_SAMPLES,
    MAX_PLANE_COMPONENT_TERMS_PER_POLYNOMIAL,
    MAX_PLANE_COMPONENT_TOTAL_DEGREE,
    MAX_PLANE_COMPONENT_TOTAL_TERMS,
    MAX_PLANE_COMPONENTS,
    IsolatedRealPlanePoint,
    PlaneComponentProfileComputed,
    PlaneComponentProfileRequest,
    PlaneComponentProfileResult,
    PlaneSemialgebraicComponent,
    PlaneSemialgebraicSet,
    PlaneSign,
    PlaneSignCondition,
)
from jacobian.math.polynomials.real_algebra._plane_components import (
    PLANE_COMPONENT_WALL_SECONDS,
)
from jacobian.math.polynomials.real_algebra._qepcad_plane_worker import (
    _MAX_FORMULA_CHARACTERS,
    _qepcad_formula,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.process import bounded_process_cancellation

_MONOMIALS_THROUGH_DEGREE_FOUR = tuple(
    sorted(
        (
            (x_degree, y_degree)
            for x_degree in range(MAX_PLANE_COMPONENT_TOTAL_DEGREE + 1)
            for y_degree in range(MAX_PLANE_COMPONENT_TOTAL_DEGREE + 1 - x_degree)
        ),
        reverse=True,
    )
)


def _q(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _polynomial(
    terms: tuple[tuple[int, tuple[int, int]], ...],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("x", "y"),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(coefficient=_q(coefficient), exponents=exponents)
                for coefficient, exponents in terms
            )
        ),
    )


def _dense_polynomial(term_count: int, leading_coefficient: int) -> RationalPolynomial:
    return _polynomial(
        tuple(
            (leading_coefficient if index == 0 else 1, exponents)
            for index, exponents in enumerate(
                _MONOMIALS_THROUGH_DEGREE_FOUR[:term_count]
            )
        )
    )


def _large_denominator_polynomial(offset: int) -> RationalPolynomial:
    denominators = []
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47):
        denominator = prime
        while denominator * prime < 10**32:
            denominator *= prime
        denominators.append(denominator)

    terms = []
    for exponents, denominator in zip(
        _MONOMIALS_THROUGH_DEGREE_FOUR,
        denominators,
        strict=True,
    ):
        numerator = denominator - offset - 1
        while gcd(numerator, denominator) != 1:
            numerator -= 1
        terms.append(
            RationalPolynomialTerm(
                coefficient=_q(Fraction(numerator, denominator)),
                exponents=exponents,
            )
        )
    return RationalPolynomial(
        variables=("x", "y"),
        polynomial=SparseRationalPolynomial(terms=tuple(terms)),
    )


def _whole_plane_request(
    polynomials: tuple[RationalPolynomial, ...],
    *,
    samples: tuple[IsolatedRealPlanePoint, ...] = (),
) -> PlaneComponentProfileRequest:
    return PlaneComponentProfileRequest(
        semialgebraic_set=PlaneSemialgebraicSet(
            axis=("x", "y"),
            polynomials=polynomials,
            sign_conditions=tuple(
                PlaneSignCondition(signs=signs)
                for signs in product(tuple(PlaneSign), repeat=len(polynomials))
            ),
        ),
        samples=samples,
    )


def _profile(
    request: PlaneComponentProfileRequest,
) -> PlaneComponentProfileResult:
    return compute_plane_component_profile(
        request.semialgebraic_set,
        request.samples,
    )


_SAMPLE_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19)


def _coordinate_value(
    *, index: int, degree: int, leading_coefficient: int
) -> RealAlgebraicValue:
    constant = index + 1 if degree == 1 else _SAMPLE_PRIMES[index]
    return RealAlgebraicValue._from_admitted_polynomial(
        polynomial=(
            str(leading_coefficient),
            *("0" for _ in range(degree - 1)),
            str(-constant),
        ),
        real_root_index=1 if degree % 2 == 0 else 0,
    )


def _sample(
    index: int,
    *,
    degree: int = 1,
    leading_coefficient: int = 1,
) -> IsolatedRealPlanePoint:
    lower = _q(0)
    upper = _q(1_000)
    coordinate = _coordinate_value(
        index=index,
        degree=degree,
        leading_coefficient=leading_coefficient,
    )
    return IsolatedRealPlanePoint(
        axis=("x", "y"),
        coordinates=(coordinate, coordinate),
        isolating_box=RationalBox(
            domain="QQ",
            variables=("x", "y"),
            intervals=(
                ClosedRationalInterval(lower=lower, upper=upper),
                ClosedRationalInterval(lower=lower, upper=upper),
            ),
        ),
    )


def _formula_marker_point(index: int) -> IsolatedRealPlanePoint:
    coefficient_base = 5 * 10 ** (MAX_PLANE_COMPONENT_POINT_COEFFICIENT_DIGITS - 2)

    def coordinate(adjustment: int) -> RealAlgebraicValue:
        coefficients = []
        for degree in range(MAX_PLANE_COMPONENT_POINT_DEGREE, -1, -1):
            if degree == MAX_PLANE_COMPONENT_POINT_DEGREE:
                coefficient = (
                    10 ** (MAX_PLANE_COMPONENT_POINT_COEFFICIENT_DIGITS - 1) + 1
                )
            elif degree == 0:
                coefficient = -(
                    5 * 10 ** (MAX_PLANE_COMPONENT_POINT_COEFFICIENT_DIGITS - 1) + 2
                )
            else:
                coefficient = (
                    ((-1) ** degree)
                    * 2
                    * (coefficient_base + (adjustment if degree == 15 else 0))
                )
            coefficients.append(str(coefficient))
        return RealAlgebraicValue._from_admitted_polynomial(
            polynomial=tuple(coefficients),
            real_root_index=0,
        )

    interval = ClosedRationalInterval(lower=_q(-1), upper=_q(Fraction(-1, 2)))
    return IsolatedRealPlanePoint(
        axis=("x", "y"),
        coordinates=(coordinate(2 * index), coordinate(2 * index + 1)),
        isolating_box=RationalBox(
            domain="QQ",
            variables=("x", "y"),
            intervals=(interval, interval),
        ),
    )


def test_degree_and_coefficient_height_boundaries_are_inclusive() -> None:
    largest_coefficient = 10**MAX_PLANE_COMPONENT_COEFFICIENT_DIGITS - 1
    boundary = _whole_plane_request(
        (
            _polynomial(
                (
                    (1, (MAX_PLANE_COMPONENT_TOTAL_DEGREE, 0)),
                    (largest_coefficient, (0, 0)),
                )
            ),
        )
    )

    assert _profile(boundary).outcome.status == "COMPUTED"

    for polynomial in (
        _polynomial(((1, (MAX_PLANE_COMPONENT_TOTAL_DEGREE + 1, 0)),)),
        _polynomial(((10**MAX_PLANE_COMPONENT_COEFFICIENT_DIGITS, (1, 0)),)),
    ):
        with pytest.raises(OperationDomainValidationError):
            _profile(_whole_plane_request((polynomial,)))


def test_request_schema_exposes_the_runtime_polynomial_envelope() -> None:
    schema = PlaneComponentProfileRequest.model_json_schema()
    polynomial = schema["properties"]["semialgebraic_set"]["properties"]["polynomials"][
        "items"
    ]
    terms = polynomial["properties"]["polynomial"]["properties"]["terms"]
    term = terms["items"]
    exponents = term["properties"]["exponents"]
    coefficient = term["properties"]["coefficient"]["properties"]

    assert terms["maxItems"] == MAX_PLANE_COMPONENT_TERMS_PER_POLYNOMIAL
    assert {tuple(entry["const"]) for entry in exponents["oneOf"]} == {
        (x_degree, y_degree)
        for x_degree in range(MAX_PLANE_COMPONENT_TOTAL_DEGREE + 1)
        for y_degree in range(MAX_PLANE_COMPONENT_TOTAL_DEGREE + 1 - x_degree)
    }
    assert coefficient["num"]["pattern"] == (
        rf"^(?:0|-?[1-9][0-9]{{0,{MAX_PLANE_COMPONENT_COEFFICIENT_DIGITS - 1}}})$"
    )
    assert coefficient["den"]["pattern"] == (
        rf"^[1-9][0-9]{{0,{MAX_PLANE_COMPONENT_COEFFICIENT_DIGITS - 1}}}$"
    )

    raw = _whole_plane_request((_polynomial(((1, (1, 0)),)),)).model_dump(mode="json")
    candidates = []
    over_terms = deepcopy(raw)
    over_terms["semialgebraic_set"]["polynomials"][0]["polynomial"]["terms"].extend(
        {
            "coefficient": {"num": "1", "den": "1"},
            "exponents": [0, 0],
        }
        for _ in range(MAX_PLANE_COMPONENT_TERMS_PER_POLYNOMIAL)
    )
    candidates.append(over_terms)
    over_degree = deepcopy(raw)
    over_degree["semialgebraic_set"]["polynomials"][0]["polynomial"]["terms"][0][
        "exponents"
    ] = [MAX_PLANE_COMPONENT_TOTAL_DEGREE + 1, 0]
    candidates.append(over_degree)
    over_height = deepcopy(raw)
    over_height["semialgebraic_set"]["polynomials"][0]["polynomial"]["terms"][0][
        "coefficient"
    ]["num"] = "1" + "0" * MAX_PLANE_COMPONENT_COEFFICIENT_DIGITS
    candidates.append(over_height)

    for candidate in candidates:
        with pytest.raises(ValidationError):
            PlaneComponentProfileRequest.model_validate(candidate)


def test_term_and_total_term_boundaries_reject_before_backend_execution() -> None:
    boundary_polynomial = _dense_polynomial(
        MAX_PLANE_COMPONENT_TERMS_PER_POLYNOMIAL,
        1,
    )
    assert (
        _profile(_whole_plane_request((boundary_polynomial,))).outcome.status
        == "COMPUTED"
    )

    raw = _whole_plane_request((boundary_polynomial,)).model_dump(mode="json")
    raw_terms = raw["semialgebraic_set"]["polynomials"][0]["polynomial"]["terms"]
    raw_terms.append(
        {
            "coefficient": {"num": "1", "den": "1"},
            "exponents": [MAX_PLANE_COMPONENT_TOTAL_DEGREE + 1, 0],
        }
    )
    with pytest.raises(ValidationError, match="term"):
        PlaneComponentProfileRequest.model_validate(raw)

    boundary_counts = (12, 12, 12, 12)
    assert sum(boundary_counts) == MAX_PLANE_COMPONENT_TOTAL_TERMS
    boundary = _whole_plane_request(
        tuple(
            _dense_polynomial(term_count, leading_coefficient)
            for leading_coefficient, term_count in enumerate(boundary_counts, start=1)
        )
    )
    assert _profile(boundary).outcome.status == "COMPUTED"

    above = _whole_plane_request(
        tuple(
            _dense_polynomial(term_count, leading_coefficient)
            for leading_coefficient, term_count in enumerate((13, 12, 12, 12), start=1)
        )
    )
    with pytest.raises(OperationDomainValidationError, match="48 terms"):
        _profile(above)


def test_plane_dimension_polynomial_and_sign_row_bounds_reject_raw_excess() -> None:
    polynomials = tuple(
        _polynomial(((leading_coefficient, (1, 0)),))
        for leading_coefficient in range(1, MAX_PLANE_COMPONENT_POLYNOMIALS + 1)
    )
    boundary = _whole_plane_request(polynomials)
    assert len(boundary.semialgebraic_set.polynomials) == (
        MAX_PLANE_COMPONENT_POLYNOMIALS
    )
    raw = boundary.model_dump(mode="json")

    over_dimension = boundary.model_dump(mode="json")
    over_dimension["semialgebraic_set"]["axis"].append("z")
    with pytest.raises(ValidationError, match="axis"):
        PlaneComponentProfileRequest.model_validate(over_dimension)

    over_polynomials = boundary.model_dump(mode="json")
    over_polynomials["semialgebraic_set"]["polynomials"].append(
        raw["semialgebraic_set"]["polynomials"][0]
    )
    with pytest.raises(ValidationError, match="polynomial family"):
        PlaneComponentProfileRequest.model_validate(over_polynomials)

    assert len(raw["semialgebraic_set"]["sign_conditions"]) == 3**4
    raw["semialgebraic_set"]["sign_conditions"].append(
        raw["semialgebraic_set"]["sign_conditions"][0]
    )
    with pytest.raises(ValidationError, match="sign table"):
        PlaneComponentProfileRequest.model_validate(raw)


def test_sample_count_degree_and_height_envelopes_are_preflighted() -> None:
    samples = tuple(_sample(index) for index in range(MAX_PLANE_COMPONENT_SAMPLES))
    assert (
        _profile(_whole_plane_request((), samples=samples)).outcome.status == "COMPUTED"
    )

    raw = _whole_plane_request((), samples=samples).model_dump(mode="json")
    raw["samples"].append(raw["samples"][0])
    with pytest.raises(ValidationError, match="samples"):
        PlaneComponentProfileRequest.model_validate(raw)

    degree_boundary = _sample(0, degree=MAX_PLANE_COMPONENT_SAMPLE_DEGREE)
    assert (
        _profile(_whole_plane_request((), samples=(degree_boundary,))).outcome.status
        == "COMPUTED"
    )
    raw_over_degree = _whole_plane_request((), samples=(degree_boundary,)).model_dump(
        mode="json"
    )
    raw_over_degree["samples"][0]["coordinates"][0]["polynomial"] = [
        "1",
        *("0" for _ in range(MAX_PLANE_COMPONENT_SAMPLE_DEGREE)),
        "-2",
    ]
    with pytest.raises(ValidationError, match="coordinate polynomial"):
        PlaneComponentProfileRequest.model_validate(raw_over_degree)

    height_boundary = _sample(
        0,
        leading_coefficient=10**MAX_PLANE_COMPONENT_SAMPLE_COEFFICIENT_DIGITS - 1,
    )
    assert (
        _profile(_whole_plane_request((), samples=(height_boundary,))).outcome.status
        == "COMPUTED"
    )
    raw_over_height = _whole_plane_request((), samples=(height_boundary,)).model_dump(
        mode="json"
    )
    raw_over_height["samples"][0]["coordinates"][0]["polynomial"][0] = str(
        10**MAX_PLANE_COMPONENT_SAMPLE_COEFFICIENT_DIGITS
    )
    with pytest.raises(ValidationError, match="coefficient"):
        PlaneComponentProfileRequest.model_validate(raw_over_height)


def test_projection_cell_envelope_distinguishes_admitted_and_rejected_families() -> (
    None
):
    degenerate = _profile(
        _whole_plane_request(
            (),
            samples=tuple(_sample(index, degree=5) for index in range(8)),
        )
    )
    assert degenerate.outcome.status == "COMPUTED"
    nondegenerate = PlaneSemialgebraicSet(
        axis=("x", "y"),
        polynomials=(_polynomial(((1, (1, 0)),)),),
        sign_conditions=(PlaneSignCondition(signs=(PlaneSign.POSITIVE,)),),
    )
    with pytest.raises(OperationDomainValidationError, match="CAD cell bound"):
        _profile(
            PlaneComponentProfileRequest(
                semialgebraic_set=nondegenerate,
                samples=tuple(_sample(index, degree=5) for index in range(8)),
            )
        )


def test_maximal_sign_table_fits_the_worker_formula_envelope() -> None:
    polynomials = tuple(
        _large_denominator_polynomial(offset)
        for offset in range(MAX_PLANE_COMPONENT_POLYNOMIALS)
    )
    semialgebraic_set = PlaneSemialgebraicSet(
        axis=("x", "y"),
        polynomials=polynomials,
        sign_conditions=tuple(
            PlaneSignCondition(signs=signs)
            for signs in product(tuple(PlaneSign), repeat=len(polynomials))
        ),
    )

    formula = _qepcad_formula(semialgebraic_set, ())

    assert len(formula) > 2 * 1024 * 1024
    assert len(formula) <= _MAX_FORMULA_CHARACTERS


def test_refinement_envelopes_cover_every_declared_coordinate_marker() -> None:
    polynomials = tuple(
        _large_denominator_polynomial(offset)
        for offset in range(MAX_PLANE_COMPONENT_POLYNOMIALS)
    )
    semialgebraic_set = PlaneSemialgebraicSet(
        axis=("x", "y"),
        polynomials=polynomials,
        sign_conditions=tuple(
            PlaneSignCondition(signs=signs)
            for signs in product(tuple(PlaneSign), repeat=len(polynomials))
        ),
    )
    points = tuple(
        _formula_marker_point(index)
        for index in range(MAX_PLANE_COMPONENTS + MAX_PLANE_COMPONENT_SAMPLES)
    )
    markers = tuple(
        polynomial for point in points for polynomial in point.coordinate_polynomials
    )

    formula = _qepcad_formula(semialgebraic_set, points)

    assert len({marker.model_dump_json() for marker in markers}) == 272
    assert len(formula) > 9 * 1024 * 1024
    assert len(formula) <= _MAX_FORMULA_CHARACTERS
    assert (
        plane_projection_coefficient_bound((markers[1], markers[3]))
        == MAX_PLANE_COMPONENT_PROJECTED_COEFFICIENT_DIGITS
        == 16_420
    )


def test_component_result_cardinality_has_an_exact_structural_bound() -> None:
    representatives = tuple(
        sorted(
            (_sample(index) for index in range(MAX_PLANE_COMPONENTS + 1)),
            key=lambda point: point.model_dump_json(),
        )
    )
    boundary = PlaneComponentProfileComputed(
        components=tuple(
            PlaneSemialgebraicComponent(
                component_id=index,
                representative=representative,
            )
            for index, representative in enumerate(
                representatives[:MAX_PLANE_COMPONENTS]
            )
        ),
        sample_dispositions=(),
    )

    assert len(boundary.components) == MAX_PLANE_COMPONENTS
    with pytest.raises(ValidationError):
        PlaneComponentProfileComputed(
            components=tuple(
                PlaneSemialgebraicComponent(
                    component_id=index,
                    representative=representative,
                )
                for index, representative in enumerate(representatives)
            ),
            sample_dispositions=(),
        )


def test_owner_deadline_includes_time_before_plane_component_admission() -> None:
    request = _whole_plane_request(())

    with (
        request_execution(monotonic() - PLANE_COMPONENT_WALL_SECONDS - 1),
        pytest.raises(OperationExecutionTimeoutError, match="semantic admission"),
    ):
        _profile(request)


def test_owner_checkpoint_observes_cancellation_before_direct_result() -> None:
    cancellation = Event()
    cancellation.set()

    with (
        bounded_process_cancellation(cancellation),
        pytest.raises(OperationExecutionCancelledError, match="cancelled"),
    ):
        _profile(_whole_plane_request(()))
