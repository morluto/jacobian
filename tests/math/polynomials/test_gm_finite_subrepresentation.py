from fractions import Fraction

import pytest

from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.derivations import _weight_operations
from jacobian.math.polynomials.derivations._weight_models import (
    PolynomialWeightAction,
    PolynomialWeightSubrepresentationRequest,
    PolynomialWeightSubrepresentationResult,
)
from jacobian.math.polynomials.derivations._weight_operations import (
    _admit_subrepresentation_support,
    _project_generators_by_weight,
    _rref_coefficient_digit_bound,
    _subrepresentation_result_size_bound,
    diagonal_weight_action,
    gm_generated_subrepresentation,
)
from jacobian.math.polynomials.values import RationalPolynomial


def _poly(
    variables: tuple[str, ...], *terms: tuple[int, tuple[int, ...]]
) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": variables,
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": coefficient, "den": 1},
                        "exponents": exponents,
                    }
                    for coefficient, exponents in sorted(
                        terms, key=lambda item: item[1], reverse=True
                    )
                ]
            },
        }
    )


def _coefficient_map(polynomial: RationalPolynomial) -> dict[tuple[int, ...], Fraction]:
    return {
        tuple(term.exponents): term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }


def _poly_with_denominators(
    variables: tuple[str, ...], terms: tuple[tuple[int, tuple[int, ...]], ...]
) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": variables,
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": 1, "den": denominator},
                        "exponents": exponents,
                    }
                    for denominator, exponents in sorted(
                        terms, key=lambda item: item[1], reverse=True
                    )
                ]
            },
        }
    )


def _prime_denominators(start: int, count: int) -> tuple[int, ...]:
    primes = []
    candidate = start
    while len(primes) < count:
        if all(candidate % divisor for divisor in range(2, int(candidate**0.5) + 1)):
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def _sparse_same_weight_seeds(
    denominators: tuple[int, ...],
) -> tuple[PolynomialWeightAction, tuple[RationalPolynomial, ...]]:
    variables = tuple(f"x{index}" for index in range(7))
    action = PolynomialWeightAction(variables=variables, weights=(0,) * len(variables))
    generators = tuple(
        _poly_with_denominators(
            variables,
            tuple(
                (denominators[column], (row, column, 0, 0, 0, 0, 0))
                for column in range(16)
            ),
        )
        for row in range(16)
    )
    return action, generators


def test_mixed_weight_seed_generates_exact_minimal_stable_span_and_action() -> None:
    result = gm_generated_subrepresentation(
        {
            "action": {"variables": ["x", "y", "z"], "weights": [1, -1, 0]},
            "generators": [
                _poly(("x", "y", "z"), (1, (1, 0, 0)), (1, (0, 1, 0)), (2, (0, 0, 1))),
                _poly(("x", "y", "z"), (1, (1, 0, 0))),
            ],
            "parameter": "lambda",
        }
    )

    # Independent character-projection oracle: x, y, and z span exactly the
    # weight-1, weight--1, and weight-0 projections of the input generators.
    assert result.weights == (-1, 0, 1)
    assert [_coefficient_map(p) for p in result.basis] == [
        {(0, 1, 0): Fraction(1)},
        {(0, 0, 1): Fraction(1)},
        {(1, 0, 0): Fraction(1)},
    ]
    assert [[c.as_fraction() for c in row] for row in result.generator_coordinates] == [
        [Fraction(1), Fraction(2), Fraction(1)],
        [Fraction(0), Fraction(0), Fraction(1)],
    ]
    assert tuple(
        tuple(entry.terms[0].exponents if entry.terms else () for entry in row)
        for row in result.matrix
    ) == (
        ((-1,), (), ()),
        ((), (0,), ()),
        ((), (), (1,)),
    )
    assert all(
        len(row) == len(result.basis) for row in result.matrix
    )  # the representation retains the ordered closure basis axes
    round_trip = PolynomialWeightSubrepresentationResult.model_validate_json(
        result.model_dump_json()
    )
    for polynomial, weight in zip(round_trip.basis, round_trip.weights, strict=True):
        consumed = diagonal_weight_action(
            round_trip.action.model_dump(),
            polynomial.model_dump(),
            round_trip.parameter,
        )
        assert all(term.exponents[-1] == weight for term in consumed.coaction.terms)


def test_same_weight_generators_are_reduced_to_a_deterministic_basis() -> None:
    result = gm_generated_subrepresentation(
        {
            "action": {"variables": ["x", "y"], "weights": [2, 2]},
            "generators": [
                _poly(("x", "y"), (1, (1, 0)), (1, (0, 1))),
                _poly(("x", "y"), (1, (1, 0)), (3, (0, 1))),
            ],
        }
    )
    assert result.weights == (2, 2)
    assert [_coefficient_map(p) for p in result.basis] == [
        {(1, 0): Fraction(1)},
        {(0, 1): Fraction(1)},
    ]
    assert [[c.as_fraction() for c in row] for row in result.generator_coordinates] == [
        [Fraction(1), Fraction(1)],
        [Fraction(1), Fraction(3)],
    ]


def test_output_admission_covers_all_serialized_result_fields() -> None:
    result = gm_generated_subrepresentation(
        {
            "action": {"variables": ["x", "y"], "weights": [1, -1]},
            "generators": [
                _poly(("x", "y"), (1, (1, 0)), (1, (0, 1))),
            ],
            "parameter": "lambda",
        }
    )
    actual_size = len(encode_strict_json(result.model_dump(mode="json")))
    support, _ = _admit_subrepresentation_support(
        result.action, result.generators, result.parameter
    )
    projections = _project_generators_by_weight(result.action, result.generators)
    admitted_bound = _subrepresentation_result_size_bound(
        result.action,
        result.generators,
        result.parameter,
        basis_terms=sum(
            len(support[weight]) * len(rows) for weight, rows in projections.items()
        ),
        dimension=sum(len(rows) for rows in projections.values()),
        coefficient_digits=_rref_coefficient_digit_bound(projections, support),
    )
    assert actual_size <= admitted_bound
    assert admitted_bound <= CanonicalLimits().max_output_bytes


def test_output_preflight_boundary_and_pre_rref_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = "lambda"
    accepted_action, accepted_generators = _sparse_same_weight_seeds(
        _prime_denominators(10_009, 16)
    )
    support, sources = _admit_subrepresentation_support(
        accepted_action, accepted_generators, parameter
    )
    projections = _project_generators_by_weight(accepted_action, accepted_generators)
    coefficient_digits = _rref_coefficient_digit_bound(projections, support)
    basis_terms = sum(
        len(support[weight]) * len(rows) for weight, rows in sources.items()
    )
    dimension = sum(len(rows) for rows in sources.values())
    admitted_bound = _subrepresentation_result_size_bound(
        accepted_action,
        accepted_generators,
        parameter,
        basis_terms=basis_terms,
        dimension=dimension,
        coefficient_digits=coefficient_digits,
    )
    assert 8 * 1024 * 1024 < admitted_bound <= CanonicalLimits().max_output_bytes
    result = gm_generated_subrepresentation(
        {
            "action": accepted_action.model_dump(),
            "generators": [generator.model_dump() for generator in accepted_generators],
            "parameter": parameter,
        }
    )
    actual_size = len(encode_strict_json(result.model_dump(mode="json")))
    assert actual_size <= admitted_bound

    rejected_action, rejected_generators = _sparse_same_weight_seeds(
        _prime_denominators(100_003, 16)
    )

    def fail_if_rref_runs(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("serialized-output admission must precede RREF")

    monkeypatch.setattr(
        _weight_operations, "_weight_projection_rref", fail_if_rref_runs
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        gm_generated_subrepresentation(
            {
                "action": rejected_action.model_dump(),
                "generators": [
                    generator.model_dump() for generator in rejected_generators
                ],
                "parameter": parameter,
            }
        )
    assert error.value.errors()[0]["type"] == "gm_subrepresentation.output_budget"


def test_empty_zero_and_duplicate_generators_have_canonical_degenerate_results() -> (
    None
):
    action = {"variables": ["x"], "weights": [0]}
    empty = gm_generated_subrepresentation({"action": action, "generators": []})
    assert empty.basis == ()
    assert empty.weights == ()
    assert empty.matrix == ()
    assert empty.generator_coordinates == ()

    zero = _poly(("x",))
    degenerate = gm_generated_subrepresentation(
        {"action": action, "generators": [zero, zero, _poly(("x",), (1, (0,)))]}
    )
    assert degenerate.weights == (0,)
    assert [_coefficient_map(p) for p in degenerate.basis] == [{(0,): Fraction(1)}]
    assert [
        [c.as_fraction() for c in row] for row in degenerate.generator_coordinates
    ] == [[Fraction(0)], [Fraction(0)], [Fraction(1)]]


def test_support_is_admitted_before_projection_and_catalog_publishes_operation() -> (
    None
):
    terms = tuple((1, (degree,)) for degree in range(64, 0, -1))
    with pytest.raises(OperationResourceAdmissionError):
        gm_generated_subrepresentation(
            {
                "action": {"variables": ["x"], "weights": [1]},
                "generators": [_poly(("x",), *terms) for _ in range(5)],
            }
        )

    operation = Catalog.open().operation(
        "algebraic_group.gm.finite_subrepresentation.compute"
    )
    assert operation is not None
    assert operation.request_type.__name__ == "PolynomialWeightSubrepresentationRequest"
    assert operation.result_type.__name__ == "PolynomialWeightSubrepresentationResult"
    parsed = PolynomialWeightSubrepresentationRequest.model_validate(
        {
            "action": {"variables": ["x"], "weights": [3]},
            "generators": [_poly(("x",), (1, (1,)))],
        }
    )
    assert operation.run(parsed).weights == (3,)
