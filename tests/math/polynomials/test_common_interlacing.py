"""Exact contract evidence for common polynomial interlacing."""

from __future__ import annotations

import threading
import time
from fractions import Fraction
from itertools import combinations_with_replacement, product

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_execution,
)
from jacobian.canonical import CanonicalLimits, canonicalize_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.algebraic_numbers.real import (
    RealAlgebraicValue,
    compare_real_algebraic,
)
from jacobian.math.polynomials.real_algebra import common_interlacing_profile
from jacobian.math.polynomials.real_algebra._common_interlacing_models import (
    CommonInterlacingDoesNotExist,
    CommonInterlacingExists,
    CommonInterlacingProfile,
    CommonInterlacingRequest,
    EmptyGapObstruction,
    LabelledRationalPolynomial,
    NonRealRootObstruction,
    PolynomialRootReference,
)
from jacobian.math.polynomials.real_algebra._common_interlacing_process import (
    _verify_declared_factors,
    run_common_interlacing_profile,
)
from jacobian.math.polynomials.real_algebra._tools import (
    TOOLS,
    compute_common_interlacing_profile,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.process import bounded_process_cancellation


def _polynomial(
    *terms: tuple[int | Fraction, int],
    variable: str = "x",
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=(variable,),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
                    exponents=(exponent,),
                )
                for coefficient, exponent in sorted(
                    terms,
                    key=lambda item: item[1],
                    reverse=True,
                )
                if coefficient
            )
        ),
    )


def _source(
    label: str,
    *terms: tuple[int | Fraction, int],
    variable: str = "x",
) -> LabelledRationalPolynomial:
    return LabelledRationalPolynomial(
        label=label,
        polynomial=_polynomial(*terms, variable=variable),
    )


def _quadratic(label: str, lower: int, upper: int) -> LabelledRationalPolynomial:
    return _source(
        label,
        (1, 2),
        (-(lower + upper), 1),
        (lower * upper, 0),
    )


def _split_source(label: str, roots: tuple[int, ...]) -> LabelledRationalPolynomial:
    coefficients = [1]
    for root in roots:
        product_coefficients = [0] * (len(coefficients) + 1)
        for index, coefficient in enumerate(coefficients):
            product_coefficients[index] += coefficient
            product_coefficients[index + 1] -= root * coefficient
        coefficients = product_coefficients
    degree = len(roots)
    return _source(
        label,
        *(
            (coefficient, degree - index)
            for index, coefficient in enumerate(coefficients)
        ),
    )


def _multiply_integer_polynomials(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    product_coefficients = [0] * (len(left) + len(right) - 1)
    for left_index, left_coefficient in enumerate(left):
        for right_index, right_coefficient in enumerate(right):
            product_coefficients[left_index + right_index] += (
                left_coefficient * right_coefficient
            )
    return tuple(product_coefficients)


def _request(*family: LabelledRationalPolynomial) -> CommonInterlacingRequest:
    return CommonInterlacingRequest(family=family)


def _referenced_value(
    result: CommonInterlacingProfile,
    reference: PolynomialRootReference,
) -> RealAlgebraicValue:
    return (
        result.root_profiles[reference.source_index]
        .roots[reference.distinct_root_index]
        .value
    )


def _error_code(exception: pytest.ExceptionInfo[OperationDomainValidationError]) -> str:
    return str(exception.value.errors()[0]["type"])


def test_quadratic_family_returns_complete_attained_gap() -> None:
    result = compute_common_interlacing_profile(
        _request(
            _source("inner", (1, 2), (-1, 0)),
            _source("outer", (1, 2), (-4, 0)),
        )
    )

    assert result.status == "EXISTS"
    assert isinstance(result.outcome, CommonInterlacingExists)
    assert tuple(profile.source_index for profile in result.root_profiles) == (0, 1)
    assert tuple(len(profile.roots) for profile in result.root_profiles) == (2, 2)
    (gap,) = result.outcome.gaps
    assert gap.gap_index == 0
    assert gap.lower == PolynomialRootReference(
        source_index=0,
        distinct_root_index=0,
    )
    assert gap.upper == PolynomialRootReference(
        source_index=0,
        distinct_root_index=1,
    )
    assert (
        compare_real_algebraic(
            _referenced_value(result, gap.lower),
            _referenced_value(result, gap.upper),
        ).order
        == "LT"
    )


def test_issue_false_negative_fixture_really_has_a_common_gap() -> None:
    # The issue listed this as an empty-gap case, but its first interval is
    # [max(-1, -3), min(1, 3)] = [-1, 1].  Preserve the mathematics rather
    # than baking that typo into the operation.
    result = common_interlacing_profile(
        (
            _source("unit", (1, 2), (-1, 0)),
            _source("triple", (1, 2), (-9, 0)),
        )
    )

    assert result.status == "EXISTS"
    assert isinstance(result.outcome, CommonInterlacingExists)
    (gap,) = result.outcome.gaps
    assert gap.lower.source_index == 0
    assert gap.upper.source_index == 0
    lower = result.root_profiles[gap.lower.source_index].roots[
        gap.lower.distinct_root_index
    ]
    upper = result.root_profiles[gap.upper.source_index].roots[
        gap.upper.distinct_root_index
    ]
    assert lower.isolating_interval.lower.as_fraction() == Fraction(-1)
    assert lower.isolating_interval.upper.as_fraction() == Fraction(-1)
    assert upper.isolating_interval.lower.as_fraction() == Fraction(1)
    assert upper.isolating_interval.upper.as_fraction() == Fraction(1)


def test_repeated_root_expands_multiplicity_and_allows_singleton_gap() -> None:
    result = common_interlacing_profile(
        (
            _source("split", (1, 2), (-1, 0)),
            _source("repeated", (1, 2), (2, 1), (1, 0)),
        )
    )

    assert result.root_profiles[1].roots[0].multiplicity == 2
    assert isinstance(result.outcome, CommonInterlacingExists)
    (gap,) = result.outcome.gaps
    lower = _referenced_value(result, gap.lower)
    upper = _referenced_value(result, gap.upper)
    assert compare_real_algebraic(lower, upper).order == "EQ"
    assert gap.lower.source_index == 0
    assert gap.upper.source_index == 1


def test_coincident_endpoint_ties_choose_the_lowest_source_index() -> None:
    result = common_interlacing_profile(
        (
            _quadratic("first", -1, 1),
            _quadratic("second", -1, 1),
            _quadratic("third", -2, 2),
        )
    )

    assert isinstance(result.outcome, CommonInterlacingExists)
    (gap,) = result.outcome.gaps
    assert gap.lower.source_index == 0
    assert gap.upper.source_index == 0


def test_first_nonreal_source_is_the_deterministic_obstruction() -> None:
    result = common_interlacing_profile(
        (
            _quadratic("real", -1, 1),
            _source("first-complex", (1, 2), (1, 0)),
            _source("second-complex", (1, 2), (4, 0)),
        )
    )

    assert isinstance(result.outcome, CommonInterlacingDoesNotExist)
    obstruction = result.outcome.obstruction
    assert isinstance(obstruction, NonRealRootObstruction)
    assert obstruction.source_index == 1
    assert obstruction.real_root_multiplicity == 0
    assert obstruction.nonreal_root_multiplicity == 2
    assert len(result.root_profiles) == 3


def test_first_empty_gap_retains_attained_tie_provenance() -> None:
    result = common_interlacing_profile(
        (
            _quadratic("left", -1, 1),
            _quadratic("right-first", 3, 4),
            _quadratic("right-tie", 3, 5),
        )
    )

    assert isinstance(result.outcome, CommonInterlacingDoesNotExist)
    obstruction = result.outcome.obstruction
    assert isinstance(obstruction, EmptyGapObstruction)
    assert obstruction.gap_index == 0
    assert obstruction.maximum_lower.source_index == 1
    assert obstruction.minimum_upper.source_index == 0
    assert (
        compare_real_algebraic(
            _referenced_value(result, obstruction.maximum_lower),
            _referenced_value(result, obstruction.minimum_upper),
        ).order
        == "GT"
    )


def test_irreducible_factor_roots_are_compared_on_one_exact_axis() -> None:
    # (x^2 - 2)(x - 3) and (x^2 - 3)(x - 2) require comparisons between
    # roots from distinct irreducible factors and distinct source polynomials.
    result = common_interlacing_profile(
        (
            _source("sqrt-two", (1, 3), (-3, 2), (-2, 1), (6, 0)),
            _source("sqrt-three", (1, 3), (-2, 2), (-3, 1), (6, 0)),
        )
    )

    assert isinstance(result.outcome, CommonInterlacingExists)
    assert tuple(root.multiplicity for root in result.root_profiles[0].roots) == (
        1,
        1,
        1,
    )
    assert len(result.outcome.gaps) == 2
    first, second = result.outcome.gaps
    assert (first.lower.source_index, first.upper.source_index) == (0, 0)
    assert (second.lower.source_index, second.upper.source_index) == (1, 1)
    for gap in result.outcome.gaps:
        assert (
            compare_real_algebraic(
                _referenced_value(result, gap.lower),
                _referenced_value(result, gap.upper),
            ).order
            != "GT"
        )


def test_returned_factor_rows_reconstruct_the_retained_source() -> None:
    # ((x^2 - 2)^2)(x - 3) has two conjugate roots from one repeated
    # irreducible factor and one rational root from another.  Reconstructing
    # the source catches a lost conjugate, factor identity, or multiplicity.
    source = _source(
        "mixed-repeated",
        (1, 5),
        (-3, 4),
        (-4, 3),
        (12, 2),
        (4, 1),
        (-12, 0),
    )
    result = common_interlacing_profile(
        (source, source.model_copy(update={"label": "same-source"}))
    )
    parsed = CommonInterlacingProfile.model_validate_json(
        result.model_dump_json(),
        strict=True,
    )

    factor_rows: dict[tuple[str, ...], list[int]] = {}
    for root in parsed.root_profiles[0].roots:
        factor_rows.setdefault(root.value.polynomial, []).append(root.multiplicity)

    reconstructed: tuple[int, ...] = (1,)
    for polynomial, multiplicities in factor_rows.items():
        factor = tuple(int(coefficient) for coefficient in polynomial)
        assert len(multiplicities) == len(factor) - 1
        assert len(set(multiplicities)) == 1
        for _ in range(multiplicities[0]):
            reconstructed = _multiply_integer_polynomials(reconstructed, factor)

    retained_terms = parsed.family[0].polynomial.polynomial.terms
    retained_degree = retained_terms[0].exponents[0]
    retained_source = [0] * (retained_degree + 1)
    for term in retained_terms:
        coefficient = term.coefficient.as_fraction()
        assert coefficient.denominator == 1
        retained_source[retained_degree - term.exponents[0]] = coefficient.numerator
    assert reconstructed == tuple(retained_source)


def test_worker_factor_multiplicity_is_capped_before_expansion() -> None:
    source = _source("linear", (1, 1), (-1, 0))

    with pytest.raises(ValueError, match="multiplicity exceeds source degree"):
        _verify_declared_factors(source, [([1, 0], 10**100)])


def test_aggregate_source_bounds_are_checked_before_worker_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = tuple(_source(f"source-{index}", (1, 32), (1, 0)) for index in range(8))

    def unexpected_worker(*_args: object, **_kwargs: object) -> None:
        pytest.fail("aggregate source bounds were checked in the worker")

    monkeypatch.setattr("jacobian.process.run_bounded_process", unexpected_worker)

    with pytest.raises(OperationDomainValidationError, match="total-degree bound"):
        run_common_interlacing_profile(family)


def test_linear_sources_have_an_exists_profile_with_no_gaps() -> None:
    result = common_interlacing_profile(
        (
            _source("zero", (1, 1)),
            _source("one", (1, 1), (-1, 0)),
        )
    )

    assert isinstance(result.outcome, CommonInterlacingExists)
    assert result.outcome.gaps == ()


def test_request_result_and_endpoint_values_round_trip_and_compose() -> None:
    request = _request(
        _source("inner", (1, 2), (-1, 0)),
        _source("outer", (1, 2), (-4, 0)),
    )
    parsed_request = CommonInterlacingRequest.model_validate_json(
        request.model_dump_json(),
        strict=True,
    )
    result = compute_common_interlacing_profile(parsed_request)
    parsed_result = CommonInterlacingProfile.model_validate_json(
        result.model_dump_json(),
        strict=True,
    )

    assert parsed_request == request
    assert parsed_result == result
    assert isinstance(parsed_result.outcome, CommonInterlacingExists)
    gap = parsed_result.outcome.gaps[0]
    lower = RealAlgebraicValue.model_validate_json(
        _referenced_value(parsed_result, gap.lower).model_dump_json(),
        strict=True,
    )
    upper = RealAlgebraicValue.model_validate_json(
        _referenced_value(parsed_result, gap.upper).model_dump_json(),
        strict=True,
    )
    assert compare_real_algebraic(lower, upper).order == "LT"


def test_trusted_producer_does_not_weaken_caller_authored_algebraic_values() -> None:
    with pytest.raises(ValidationError, match="irreducible"):
        RealAlgebraicValue(
            polynomial=("1", "0", "-1"),
            real_root_index=0,
        )
    with pytest.raises(ValidationError, match="root_index"):
        RealAlgebraicValue(
            polynomial=("1", "0", "1"),
            real_root_index=0,
        )


@pytest.mark.parametrize(
    ("family", "code"),
    [
        (
            (
                _source("duplicate", (1, 2), (-1, 0)),
                _source("duplicate", (1, 2), (-4, 0)),
            ),
            "polynomial.common_interlacing_duplicate_label",
        ),
        (
            (
                _source("monic", (1, 2), (-1, 0)),
                _source("not-monic", (2, 2), (-1, 0)),
            ),
            "polynomial.common_interlacing_monic",
        ),
        (
            (
                _source("quadratic", (1, 2), (-1, 0)),
                _source("cubic", (1, 3), (-1, 0)),
            ),
            "polynomial.common_interlacing_common_degree",
        ),
        (
            (
                _source("zero"),
                _source("linear", (1, 1)),
            ),
            "polynomial.common_interlacing_positive_degree",
        ),
        (
            (
                _source("x", (1, 2), (-1, 0)),
                _source("y", (1, 2), (-1, 0), variable="y"),
            ),
            "polynomial.common_interlacing_source_ring",
        ),
    ],
)
def test_semantic_source_contract_rejects_malformed_families(
    family: tuple[LabelledRationalPolynomial, ...],
    code: str,
) -> None:
    with pytest.raises(OperationDomainValidationError) as exception:
        common_interlacing_profile(family)
    assert _error_code(exception) == code


def test_request_raw_preflight_rejects_oversized_axes_before_nested_parsing() -> None:
    invalid_member = {
        "label": "source",
        "polynomial": {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": "not-an-integer", "den": "1"},
                        "exponents": [0],
                    }
                ]
            },
        },
    }
    with pytest.raises(
        ValidationError,
        match="common interlacing admits at most 8 family members",
    ):
        CommonInterlacingRequest.model_validate({"family": [invalid_member] * 9})

    over_degree = {
        "family": [
            {
                "label": label,
                "polynomial": {
                    "variables": ["x"],
                    "polynomial": {
                        "terms": [
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "exponents": [33],
                            }
                        ]
                    },
                },
            }
            for label in ("a", "b")
        ]
    }
    with pytest.raises(ValidationError, match="degree-32"):
        CommonInterlacingRequest.model_validate(over_degree)


def test_source_degree_and_factor_degree_boundaries() -> None:
    degree_32 = common_interlacing_profile(
        (
            _source("a", (1, 32)),
            _source("b", (1, 32)),
        )
    )
    assert isinstance(degree_32.outcome, CommonInterlacingExists)
    assert degree_32.root_profiles[0].roots[0].multiplicity == 32
    assert len(degree_32.outcome.gaps) == 31

    factor_degree_8 = common_interlacing_profile(
        (
            _source("a", (1, 8), (1, 0)),
            _source("b", (1, 8), (1, 0)),
        )
    )
    assert isinstance(factor_degree_8.outcome, CommonInterlacingDoesNotExist)
    assert isinstance(factor_degree_8.outcome.obstruction, NonRealRootObstruction)

    with pytest.raises(OperationDomainValidationError) as exception:
        common_interlacing_profile(
            (
                _source("a", (1, 33)),
                _source("b", (1, 33)),
            )
        )
    assert _error_code(exception) == "polynomial.common_interlacing_source_degree"

    with pytest.raises(OperationDomainValidationError) as exception:
        common_interlacing_profile(
            (
                _source("a", (1, 9), (-2, 0)),
                _source("b", (1, 9), (-2, 0)),
            )
        )
    assert _error_code(exception) == "polynomial.common_interlacing_factor_degree"


def test_root_free_high_degree_factor_reports_nonreal_obstruction() -> None:
    # Eisenstein at 2 makes x^10 + 2x^2 + 2 irreducible, and its value is
    # strictly positive on R.  No RealAlgebraicValue is needed for this
    # factor, so its degree should not narrow the exact obstruction result.
    result = common_interlacing_profile(
        (
            _source("first", (1, 10), (2, 2), (2, 0)),
            _source("second", (1, 10), (2, 2), (2, 0)),
        )
    )

    assert isinstance(result.outcome, CommonInterlacingDoesNotExist)
    assert isinstance(result.outcome.obstruction, NonRealRootObstruction)
    assert result.outcome.obstruction.source_index == 0
    assert result.outcome.obstruction.real_root_multiplicity == 0
    assert result.outcome.obstruction.nonreal_root_multiplicity == 10


def test_isolation_work_is_rejected_after_factorization_before_root_profiles() -> None:
    prime = 10**30 + 57
    with pytest.raises(OperationDomainValidationError) as exception:
        common_interlacing_profile(
            (
                _source("first", (1, 20), (prime, 0)),
                _source("second", (1, 20), (prime, 0)),
            )
        )

    assert _error_code(exception) == "polynomial.common_interlacing_isolation_work"


def test_expired_request_deadline_stops_before_backend_launch() -> None:
    family = (
        _source("first", (1, 2), (-1, 0)),
        _source("second", (1, 2), (-1, 0)),
    )

    with (
        request_execution(time.monotonic() - 3601.0),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        common_interlacing_profile(family)


def test_active_worker_cancellation_is_preserved_as_execution_state() -> None:
    family = tuple(
        _split_source(f"source-{index}", tuple(range(-8, 8))) for index in range(8)
    )
    cancellation = threading.Event()
    timer = threading.Timer(0.1, cancellation.set)
    started = time.monotonic()
    timer.start()
    try:
        with (
            bounded_process_cancellation(cancellation),
            pytest.raises(OperationExecutionCancelledError),
        ):
            common_interlacing_profile(family)
    finally:
        timer.cancel()
        timer.join()

    assert time.monotonic() - started < 5.0


def _large_primitive_height_source(
    label: str,
    denominators: tuple[int, ...],
) -> LabelledRationalPolynomial:
    degree = len(denominators)
    return _source(
        label,
        (1, degree),
        *(
            (Fraction(1, denominator), degree - index - 1)
            for index, denominator in enumerate(denominators)
        ),
    )


def test_primitive_height_boundary_is_computed_after_clearing_denominators() -> None:
    base = 9 * 10**63
    height_256 = tuple(base + offset for offset in (1, 2, 3, 5))
    accepted = common_interlacing_profile(
        (
            _large_primitive_height_source("a", height_256),
            _large_primitive_height_source("b", height_256),
        )
    )
    assert len(accepted.root_profiles) == 2

    # The four pairwise-coprime 64-digit denominators have a 256-digit
    # product. Adding the coprime denominator 11 raises the primitive source
    # height to 257 digits without exceeding any raw coefficient bound.
    height_257 = (*height_256, 11)
    with pytest.raises(OperationDomainValidationError) as exception:
        common_interlacing_profile(
            (
                _large_primitive_height_source("a", height_257),
                _large_primitive_height_source("b", height_257),
            )
        )
    assert _error_code(exception) == "polynomial.common_interlacing_primitive_height"


def test_maximal_family_and_root_axes_serialize_within_the_canonical_limit() -> None:
    # Eight degree-16 split sources attain the family and total-degree ceilings
    # while producing all 128 distinct source-root rows.
    family = tuple(
        _split_source(f"source-{index}", tuple(range(-8, 8))) for index in range(8)
    )
    result = common_interlacing_profile(family)

    assert tuple(profile.source_index for profile in result.root_profiles) == tuple(
        range(8)
    )
    assert sum(len(profile.roots) for profile in result.root_profiles) == 128
    assert all(
        root.multiplicity == 1
        for profile in result.root_profiles
        for root in profile.roots
    )
    encoded = canonicalize_json(
        result.model_dump(mode="json"),
        limits=CanonicalLimits(),
    )
    assert len(encoded) <= CanonicalLimits().max_output_bytes


def test_schema_uses_canonical_polynomials_and_discriminated_outcomes() -> None:
    request_schema = CommonInterlacingRequest.model_json_schema()
    labelled = request_schema["$defs"]["LabelledRationalPolynomial"]
    assert labelled["properties"]["polynomial"]["$ref"].endswith("/RationalPolynomial")
    assert "UnivariatePolynomialRequest" not in request_schema["$defs"]

    result_schema = CommonInterlacingProfile.model_json_schema()
    outcome = result_schema["properties"]["outcome"]
    assert outcome["discriminator"]["propertyName"] == "status"
    does_not_exist = result_schema["$defs"]["CommonInterlacingDoesNotExist"]
    assert (
        does_not_exist["properties"]["obstruction"]["discriminator"]["propertyName"]
        == "kind"
    )


def test_catalog_declaration_is_discoverable_and_example_executes() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "polynomial.real.common_interlacing_profile.compute"
    )

    assert "common interlacer" in operation.discovery_terms
    request = operation.request_type.model_validate(operation.examples[0].input)
    result = operation.run(request)
    assert isinstance(result, CommonInterlacingProfile)
    assert result.status == "EXISTS"


@pytest.mark.exhaustive
def test_small_split_quadratics_match_the_gap_criterion_exhaustively() -> None:
    root_pairs = tuple(combinations_with_replacement(range(-2, 3), 2))
    for (left_lower, left_upper), (right_lower, right_upper) in product(
        root_pairs,
        repeat=2,
    ):
        result = common_interlacing_profile(
            (
                _quadratic("left", left_lower, left_upper),
                _quadratic("right", right_lower, right_upper),
            )
        )
        expected = max(left_lower, right_lower) <= min(left_upper, right_upper)
        assert (result.status == "EXISTS") is expected, (
            left_lower,
            left_upper,
            right_lower,
            right_upper,
        )
