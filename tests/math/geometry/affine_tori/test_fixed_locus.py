"""Definition, oracle, boundary, and public tests for affine-torus fixed loci."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import gcd, lcm
from random import Random
from time import monotonic
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian._execution import (
    OperationExecutionTimeoutError,
    current_request_execution,
    request_execution,
)
from jacobian.canonical import canonicalize_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.affine_tori import (
    RationalAffineTorusMap,
    affine_torus_fixed_locus,
)
from jacobian.math.geometry.affine_tori import _bounds as affine_bounds
from jacobian.math.geometry.affine_tori import operations as affine_operations
from jacobian.math.geometry.affine_tori._bounds import (
    AFFINE_TORUS_FIXED_LOCUS_WALL_SECONDS,
    MAX_AFFINE_TORUS_RESULT_BYTES,
    MAX_AFFINE_TORUS_WORK_UNITS,
    build_affine_torus_plan,
)
from jacobian.math.geometry.affine_tori._models import (
    AffineTorusFixedLocusRequest,
    AffineTorusFixedLocusResult,
    NonemptyAffineTorusFixedLocus,
)


def _payload(
    linear_part: tuple[tuple[int, ...], ...],
    translation: tuple[Fraction, ...],
) -> dict[str, Any]:
    dimension = len(linear_part)
    return {
        "torus": {"dimension": dimension},
        "linear_part": {
            "row_count": dimension,
            "column_count": dimension,
            "entries": [[str(value) for value in row] for row in linear_part],
        },
        "translation": {
            "torus": {"dimension": dimension},
            "coordinates": [
                {"num": str(value.numerator), "den": str(value.denominator)}
                for value in translation
            ],
        },
    }


def _source(
    linear_part: tuple[tuple[int, ...], ...],
    translation: tuple[Fraction, ...],
) -> RationalAffineTorusMap:
    return RationalAffineTorusMap.model_validate(_payload(linear_part, translation))


def _matrix_multiply(
    left: tuple[tuple[int, ...], ...],
    right: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    columns = len(right[0]) if right else 0
    middle = len(right)
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(middle))
            for column in range(columns)
        )
        for row in range(len(left))
    )


def _matrix_fraction_multiply(
    left: tuple[tuple[int | Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    columns = len(right[0]) if right else 0
    middle = len(right)
    return tuple(
        tuple(
            sum(
                (
                    Fraction(left[row][index]) * right[index][column]
                    for index in range(middle)
                ),
                Fraction(0),
            )
            for column in range(columns)
        )
        for row in range(len(left))
    )


def _displacement(
    linear_part: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(value - int(row == column) for column, value in enumerate(entries))
        for row, entries in enumerate(linear_part)
    )


def _determinant_two(matrix: tuple[tuple[int, ...], ...]) -> int:
    assert len(matrix) == 2 and all(len(row) == 2 for row in matrix)
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def _add_points(
    left: tuple[Fraction, ...], right: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    return tuple((a + b) % 1 for a, b in zip(left, right, strict=True))


def _scale_point(scalar: int, point: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    return tuple((scalar * value) % 1 for value in point)


def _finite_result_points(
    result: AffineTorusFixedLocusResult,
) -> set[tuple[Fraction, ...]]:
    assert isinstance(result.outcome, NonemptyAffineTorusFixedLocus)
    family = result.outcome.fixed_locus
    assert family.identity_component.parameter_dimension == 0
    base = tuple(value.as_fraction() for value in family.base_point.coordinates)
    generators = tuple(
        tuple(value.as_fraction() for value in point.coordinates)
        for point in family.component_generators
    )
    orders = tuple(int(value) for value in family.finite_components.generator_orders)
    points: set[tuple[Fraction, ...]] = set()
    for coefficients in product(*(range(order) for order in orders)):
        point = base
        for coefficient, generator in zip(coefficients, generators, strict=True):
            point = _add_points(point, _scale_point(coefficient, generator))
        points.add(point)
    return points


def test_identity_map_has_the_whole_torus_and_zero_dimensional_presentation() -> None:
    source = _source(((1, 0), (0, 1)), (Fraction(0), Fraction(0)))

    result = affine_torus_fixed_locus(source)

    assert result.source == source
    assert isinstance(result.outcome, NonemptyAffineTorusFixedLocus)
    family = result.outcome.fixed_locus
    assert family.identity_component.parameter_dimension == 2
    assert family.identity_component.embedding.entries == (("1", "0"), ("0", "1"))
    assert family.component_generators == ()
    assert family.finite_components.relation_matrix.row_count == 0
    assert family.finite_components.relation_matrix.column_count == 0
    assert family.finite_components.relation_matrix.entries == ()
    assert family.finite_components.component_count == "1"


def test_translated_identity_returns_the_first_primitive_obstruction() -> None:
    source = _source(((1, 0), (0, 1)), (Fraction(1, 3), Fraction(1, 2)))

    result = affine_torus_fixed_locus(source)

    assert result.outcome.status == "EMPTY"
    assert result.outcome.obstruction.coefficients == ("1", "0")
    assert result.outcome.obstruction_pairing.as_fraction() == Fraction(1, 3)


def test_zero_linear_map_returns_its_unique_fixed_point() -> None:
    translation = (Fraction(1, 3), Fraction(2, 5))
    result = affine_torus_fixed_locus(_source(((0, 0), (0, 0)), translation))

    assert isinstance(result.outcome, NonemptyAffineTorusFixedLocus)
    family = result.outcome.fixed_locus
    assert (
        tuple(value.as_fraction() for value in family.base_point.coordinates)
        == translation
    )
    assert family.identity_component.parameter_dimension == 0
    assert family.identity_component.embedding.entries == ((), ())
    # The standard image-saturation generators are retained even when their
    # classes have order one; C=I makes the trivial component group explicit.
    assert family.finite_components.relation_matrix.entries == (("1", "0"), ("0", "1"))
    assert family.finite_components.generator_orders == ("1", "1")
    assert family.finite_components.invariant_factors == ()
    assert family.finite_components.component_count == "1"


def test_positive_dimensional_locus_with_two_components() -> None:
    result = affine_torus_fixed_locus(
        _source(((3, 0), (0, 1)), (Fraction(0), Fraction(0)))
    )

    assert isinstance(result.outcome, NonemptyAffineTorusFixedLocus)
    family = result.outcome.fixed_locus
    assert family.identity_component.parameter_dimension == 1
    assert family.identity_component.embedding.entries == (("0",), ("1",))
    assert tuple(
        value.as_fraction() for value in family.component_generators[0].coordinates
    ) == (Fraction(1, 2), Fraction(0))
    assert family.finite_components.relation_matrix.entries == (("2",),)
    assert family.finite_components.generator_orders == ("2",)
    assert family.finite_components.invariant_factors == ("2",)
    assert family.finite_components.component_count == "2"


def test_nonunimodular_infinite_order_map_is_inside_the_public_domain() -> None:
    source = _source(((2, 1), (0, 2)), (Fraction(1, 7), Fraction(2, 7)))

    result = affine_torus_fixed_locus(source)

    assert isinstance(result.outcome, NonemptyAffineTorusFixedLocus)
    assert result.outcome.fixed_locus.identity_component.parameter_dimension == 0
    assert _finite_result_points(result) == {(Fraction(1, 7), Fraction(5, 7))}


@pytest.mark.parametrize(
    ("linear_part", "translation", "expected_pairing"),
    (
        (
            (
                (1, 0, 0, 0),
                (6, 0, 1, 0),
                (-6, -1, -1, 0),
                (-2, 1, 0, 1),
            ),
            (Fraction(1, 3), Fraction(2, 3), Fraction(2, 3), Fraction(0)),
            Fraction(1, 3),
        ),
        (
            (
                (1, 0, 0, 0),
                (0, 0, -1, 0),
                (-6, 1, 0, 0),
                (3, 0, 1, 1),
            ),
            (Fraction(3, 4), Fraction(1, 4), Fraction(3, 4), Fraction(0)),
            Fraction(3, 4),
        ),
    ),
)
def test_paper_empty_examples_return_the_first_coordinate_character(
    linear_part: tuple[tuple[int, ...], ...],
    translation: tuple[Fraction, ...],
    expected_pairing: Fraction,
) -> None:
    result = affine_torus_fixed_locus(_source(linear_part, translation))

    assert result.outcome.status == "EMPTY"
    assert result.outcome.obstruction.coefficients == ("1", "0", "0", "0")
    assert result.outcome.obstruction_pairing.as_fraction() == expected_pairing


def test_nonempty_result_reconstructs_defining_integer_and_rational_identities() -> (
    None
):
    linear_part = ((3, 0, 2), (0, 4, 3), (0, 0, 1))
    source = _source(linear_part, (Fraction(0), Fraction(0), Fraction(0)))
    result = affine_torus_fixed_locus(source)
    assert isinstance(result.outcome, NonemptyAffineTorusFixedLocus)
    family = result.outcome.fixed_locus
    displacement = _displacement(linear_part)
    identity_embedding = tuple(
        tuple(int(value) for value in row)
        for row in family.identity_component.embedding.entries
    )
    assert _matrix_multiply(displacement, identity_embedding) == (
        (0,),
        (0,),
        (0,),
    )
    component_lifts = tuple(
        tuple(
            generator.coordinates[row].as_fraction()
            for generator in family.component_generators
        )
        for row in range(3)
    )
    image_saturation = _matrix_fraction_multiply(displacement, component_lifts)
    assert image_saturation == (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(0)),
    )
    image_coordinates = ((2, 0, 2), (0, 3, 3))
    assert _matrix_fraction_multiply(
        image_saturation,
        tuple(tuple(Fraction(value) for value in row) for row in image_coordinates),
    ) == tuple(tuple(Fraction(value) for value in row) for row in displacement)

    relation_matrix = tuple(
        tuple(int(value) for value in row)
        for row in family.finite_components.relation_matrix.entries
    )
    # A relation column must send every component lift to an integral vector.
    lift_relations = _matrix_fraction_multiply(
        component_lifts,
        tuple(tuple(Fraction(value) for value in row) for row in relation_matrix),
    )
    assert all(value.denominator == 1 for row in lift_relations for value in row)
    determinant = abs(_determinant_two(relation_matrix))
    assert determinant == int(family.finite_components.component_count) == 6
    assert family.finite_components.invariant_factors == ("6",)


def test_empty_character_replays_the_obstruction_definition() -> None:
    linear_part = (
        (1, 0, 0, 0),
        (6, 0, 1, 0),
        (-6, -1, -1, 0),
        (-2, 1, 0, 1),
    )
    translation = (Fraction(1, 3), Fraction(2, 3), Fraction(2, 3), Fraction(0))
    result = affine_torus_fixed_locus(_source(linear_part, translation))
    assert result.outcome.status == "EMPTY"
    character = tuple(int(value) for value in result.outcome.obstruction.coefficients)

    displacement = _displacement(linear_part)
    assert tuple(
        sum(character[row] * displacement[row][column] for row in range(4))
        for column in range(4)
    ) == (0, 0, 0, 0)
    assert gcd(*(abs(value) for value in character)) == 1
    pairing = sum(
        (
            coefficient * value
            for coefficient, value in zip(character, translation, strict=True)
        ),
        Fraction(0),
    )
    assert pairing % 1 == result.outcome.obstruction_pairing.as_fraction() != 0


def test_random_small_full_rank_maps_match_a_grid_congruence_oracle() -> None:
    random = Random(2443)
    checked = 0
    while checked < 40:
        displacement = tuple(
            tuple(random.randint(-2, 2) for _ in range(2)) for _ in range(2)
        )
        determinant = _determinant_two(displacement)
        if determinant == 0 or abs(determinant) > 8:
            continue
        denominators = (random.randint(1, 3), random.randint(1, 3))
        translation = tuple(
            Fraction(random.randrange(denominator), denominator)
            for denominator in denominators
        )
        linear_part = tuple(
            tuple(displacement[row][column] + int(row == column) for column in range(2))
            for row in range(2)
        )
        result = affine_torus_fixed_locus(_source(linear_part, translation))
        actual = _finite_result_points(result)
        grid_denominator = abs(determinant) * lcm(
            *(value.denominator for value in translation)
        )
        expected: set[tuple[Fraction, ...]] = set()
        for numerators in product(range(grid_denominator), repeat=2):
            point = tuple(
                Fraction(numerator, grid_denominator) for numerator in numerators
            )
            if all(
                (
                    sum(
                        displacement[row][column]
                        * Fraction(numerators[column], grid_denominator)
                        for column in range(2)
                    )
                    + translation[row]
                ).denominator
                == 1
                for row in range(2)
            ):
                expected.add(point)
        assert actual == expected
        assert len(actual) == abs(determinant)
        checked += 1


def test_integral_basis_change_covariance_for_a_finite_fixed_locus() -> None:
    linear_part = ((3, 1), (0, 2))
    translation = (Fraction(1, 3), Fraction(1, 4))
    source_result = affine_torus_fixed_locus(_source(linear_part, translation))
    change = ((1, 1), (0, 1))
    inverse = ((1, -1), (0, 1))

    conjugated = _matrix_multiply(_matrix_multiply(change, linear_part), inverse)
    translated = tuple(
        sum(
            (change[row][column] * translation[column] for column in range(2)),
            Fraction(0),
        )
        % 1
        for row in range(2)
    )
    conjugated_result = affine_torus_fixed_locus(_source(conjugated, translated))
    transformed_points = {
        tuple(
            sum(
                (change[row][column] * point[column] for column in range(2)),
                Fraction(0),
            )
            % 1
            for row in range(2)
        )
        for point in _finite_result_points(source_result)
    }
    assert _finite_result_points(conjugated_result) == transformed_points


def test_zero_dimensional_matrices_and_discriminated_result_round_trip() -> None:
    for source in (
        _source(((1, 0), (0, 1)), (Fraction(0), Fraction(0))),
        _source(((0, 0), (0, 0)), (Fraction(1, 3), Fraction(2, 5))),
    ):
        result = affine_torus_fixed_locus(source)
        restored = AffineTorusFixedLocusResult.model_validate(
            result.model_dump(mode="json")
        )
        assert restored == result


def test_contradictory_discriminator_and_component_metadata_fail_closed() -> None:
    result = affine_torus_fixed_locus(_source(((3,),), (Fraction(0),))).model_dump(
        mode="json"
    )
    contradictory_status = dict(result)
    contradictory_status["outcome"] = {
        **result["outcome"],
        "status": "EMPTY",
    }
    with pytest.raises(ValidationError):
        AffineTorusFixedLocusResult.model_validate(contradictory_status)

    contradictory_count = dict(result)
    outcome = dict(result["outcome"])
    family = dict(outcome["fixed_locus"])
    components = dict(family["finite_components"])
    components["component_count"] = "3"
    family["finite_components"] = components
    outcome["fixed_locus"] = family
    contradictory_count["outcome"] = outcome
    with pytest.raises(ValidationError, match="component count"):
        AffineTorusFixedLocusResult.model_validate(contradictory_count)

    contradictory_order = dict(result)
    outcome = dict(result["outcome"])
    family = dict(outcome["fixed_locus"])
    components = dict(family["finite_components"])
    components["generator_orders"] = ["3"]
    family["finite_components"] = components
    outcome["fixed_locus"] = family
    contradictory_order["outcome"] = outcome
    with pytest.raises(ValidationError, match="generator orders"):
        AffineTorusFixedLocusResult.model_validate(contradictory_order)


@pytest.mark.parametrize(
    "mutation",
    ("rows", "columns", "digits", "coordinates", "nested_component"),
)
def test_raw_shape_and_digit_preflight_precedes_nested_parsing(mutation: str) -> None:
    payload = _payload(((1,),), (Fraction(0),))
    if mutation == "rows":
        payload["linear_part"]["entries"] = [[object()] for _ in range(17)]
    elif mutation == "columns":
        payload["linear_part"]["entries"] = [[object() for _ in range(17)]]
    elif mutation == "digits":
        payload["linear_part"]["entries"] = [["1" * 33]]
    elif mutation == "coordinates":
        payload["translation"]["coordinates"] = [object() for _ in range(17)]
    else:
        payload["translation"]["coordinates"] = [{"num": [[[[object()]]]], "den": "1"}]

    with pytest.raises(ValidationError) as error:
        RationalAffineTorusMap.model_validate(payload)

    assert error.value.errors()[0]["type"] in {
        "affine_torus.raw_digits",
        "affine_torus.raw_shape",
        "affine_torus.raw_size",
        "affine_torus.raw_type",
    }


def test_admission_precharges_every_phase_before_the_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(((3, 0), (0, 4)), (Fraction(0), Fraction(0)))
    plan = build_affine_torus_plan(source, deadline=monotonic() + 30)
    charges = dict(plan.work_units_by_category)
    assert set(charges) == {
        "source_conversion",
        "hnf",
        "rank_minor_selection",
        "rational_solves",
        "snf",
        "normalization",
        "serialization",
    }
    assert all(amount > 0 for amount in charges.values())
    assert plan.work_units == sum(charges.values()) <= MAX_AFFINE_TORUS_WORK_UNITS

    monkeypatch.setattr(
        affine_bounds, "MAX_AFFINE_TORUS_WORK_UNITS", plan.work_units - 1
    )
    backend_called = False

    def forbidden_kernel(*_args: object, **_kwargs: object) -> None:
        nonlocal backend_called
        backend_called = True

    monkeypatch.setattr(
        affine_operations, "compute_fixed_locus_kernel", forbidden_kernel
    )
    with pytest.raises(OperationDomainValidationError, match="work budget"):
        affine_torus_fixed_locus(source)
    assert not backend_called


def test_dispatch_start_and_all_phases_share_one_deadline() -> None:
    source = _source(((3,),), (Fraction(0),))
    started_at = monotonic() - AFFINE_TORUS_FIXED_LOCUS_WALL_SECONDS - 1

    with request_execution(started_at):
        with pytest.raises(OperationExecutionTimeoutError, match="semantic admission"):
            affine_torus_fixed_locus(source)
        execution = current_request_execution()
        assert execution is not None
        assert execution.deadline == pytest.approx(
            started_at + AFFINE_TORUS_FIXED_LOCUS_WALL_SECONDS
        )


def test_max_dimension_adversary_is_deterministic_and_below_one_mibibyte() -> None:
    dimension = 16
    height = 10**31 - 19
    displacement = tuple(
        tuple(
            height - row - column if column >= row else 0 for column in range(dimension)
        )
        for row in range(dimension)
    )
    linear_part = tuple(
        tuple(
            displacement[row][column] + int(row == column)
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    source = _source(linear_part, (Fraction(0),) * dimension)
    plan = build_affine_torus_plan(source, deadline=monotonic() + 30)

    first = affine_torus_fixed_locus(source)
    second = affine_torus_fixed_locus(source)
    encoded = canonicalize_json(first.model_dump(mode="json"))

    assert first == second
    assert len(encoded) <= plan.result_bytes_upper_bound
    assert len(encoded) < MAX_AFFINE_TORUS_RESULT_BYTES


def test_public_tool_validates_and_executes_its_example() -> None:
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "affine_torus.fixed_locus.compute"
    )
    request = AffineTorusFixedLocusRequest.model_validate(tool.examples[0].input)
    schema = tool.request_type.model_json_schema()
    linear_schema = schema["$defs"]["RationalAffineTorusMap"]["properties"][
        "linear_part"
    ]

    result = tool.run(request)

    assert linear_schema["properties"]["row_count"]["maximum"] == 16
    assert linear_schema["properties"]["column_count"]["maximum"] == 16
    assert linear_schema["properties"]["entries"]["maxItems"] == 16
    assert linear_schema["properties"]["entries"]["items"]["maxItems"] == 16
    assert isinstance(result, AffineTorusFixedLocusResult)
    assert result.outcome.status == "NONEMPTY"
    assert result.outcome.fixed_locus.finite_components.component_count == "2"
