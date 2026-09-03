"""Definition, oracle, boundary, and public tests for affine-torus fixed loci."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from fractions import Fraction
from itertools import product
from math import gcd, lcm
from random import Random
from time import monotonic
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from tests.fixtures.accounting import assert_charged_work_parity

from jacobian._execution import (
    OperationExecutionTimeoutError,
    current_request_execution,
    request_execution,
)
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry import affine_tori
from jacobian.math.geometry.affine_tori import (
    AffineTorusFixedLocusOutcome,
    AffineTorusFixedLocusResult,
    EmptyAffineTorusFixedLocus,
    NonemptyAffineTorusFixedLocus,
    RationalAffineTorusMap,
    RationalTorusPoint,
    affine_torus_fixed_locus,
)
from jacobian.math.geometry.affine_tori import _flint as affine_flint
from jacobian.math.geometry.affine_tori._bounds import (
    AFFINE_TORUS_FIXED_LOCUS_WALL_SECONDS,
    build_affine_torus_plan,
)
from jacobian.math.geometry.affine_tori._flint import (
    compute_fixed_locus_kernel as _compute_fixed_locus_kernel,
)
from jacobian.math.geometry.affine_tori._kernel_types import (
    AffineTorusKernelSource,
    NonemptyFixedLocusKernel,
)
from jacobian.math.geometry.affine_tori._models import AffineTorusFixedLocusRequest
from jacobian.math.geometry.affine_tori.values import MAX_AFFINE_TORUS_POINT_DIGITS


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


def test_fixed_locus_result_contracts_are_public_owner_symbols() -> None:
    expected = {
        "AffineTorusFixedLocusOutcome": AffineTorusFixedLocusOutcome,
        "AffineTorusFixedLocusResult": AffineTorusFixedLocusResult,
        "EmptyAffineTorusFixedLocus": EmptyAffineTorusFixedLocus,
        "NonemptyAffineTorusFixedLocus": NonemptyAffineTorusFixedLocus,
    }

    assert set(expected) <= set(affine_tori.__all__)
    assert all(getattr(affine_tori, name) is value for name, value in expected.items())


def _source(
    linear_part: tuple[tuple[int, ...], ...],
    translation: tuple[Fraction, ...],
) -> RationalAffineTorusMap:
    return RationalAffineTorusMap.model_validate(_payload(linear_part, translation))


def _kernel_source(source: RationalAffineTorusMap) -> AffineTorusKernelSource:
    return AffineTorusKernelSource(
        dimension=source.torus.dimension,
        linear_part=tuple(
            tuple(int(value) for value in row) for row in source.linear_part.entries
        ),
        translation=tuple(
            coordinate.as_fraction() for coordinate in source.translation.coordinates
        ),
    )


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


def _finite_kernel_points(
    kernel: NonemptyFixedLocusKernel,
) -> set[tuple[Fraction, ...]]:
    points: set[tuple[Fraction, ...]] = set()
    for coefficients in product(
        *(range(order) for order in kernel.generator_orders)
    ):
        point = kernel.base_point
        for coefficient, generator in zip(
            coefficients, kernel.component_generators, strict=True
        ):
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


def test_zero_dimensional_torus_has_its_single_exact_fixed_point() -> None:
    source = _source((), ())

    result = affine_torus_fixed_locus(source)

    assert result.source == source
    assert isinstance(result.outcome, NonemptyAffineTorusFixedLocus)
    family = result.outcome.fixed_locus
    assert family.ambient_torus.dimension == 0
    assert family.base_point.coordinates == ()
    assert family.identity_component.parameter_dimension == 0
    assert family.identity_component.embedding.row_count == 0
    assert family.identity_component.embedding.column_count == 0
    assert family.identity_component.embedding.entries == ()
    assert family.component_generators == ()
    assert family.finite_components.generator_count == 0
    assert family.finite_components.relation_matrix.entries == ()
    assert family.finite_components.component_count == "1"
    assert (
        AffineTorusFixedLocusResult.model_validate_json(
            encode_strict_json(result.model_dump(mode="json")),
            strict=True,
        )
        == result
    )


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


def test_base_point_height_accounts_for_minor_times_translation_denominator() -> None:
    source = _source(((17,),), (Fraction(1, 2),))
    plan = build_affine_torus_plan(source, deadline=monotonic() + 30)

    result = affine_torus_fixed_locus(source)

    assert isinstance(result.outcome, NonemptyAffineTorusFixedLocus)
    family = result.outcome.fixed_locus
    assert family.base_point.coordinates[0].as_fraction() == Fraction(31, 32)
    assert family.finite_components.component_count == "16"
    assert plan.bounds_for_rank(1).rational_intermediate_height >= 32


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


def test_component_relation_may_land_nontrivially_in_identity_subtorus() -> None:
    result = affine_torus_fixed_locus(
        _source(((1, 0), (-4, -4)), (Fraction(0), Fraction(0)))
    )

    assert isinstance(result.outcome, NonemptyAffineTorusFixedLocus)
    family = result.outcome.fixed_locus
    assert family.identity_component.embedding.entries == (("5",), ("-4",))
    generator = tuple(
        value.as_fraction() for value in family.component_generators[0].coordinates
    )
    assert generator == (Fraction(3, 4), Fraction(0))
    assert family.finite_components.relation_matrix.entries == (("1",),)
    assert family.finite_components.generator_orders == ("1",)
    assert family.finite_components.invariant_factors == ()
    assert family.finite_components.component_count == "1"

    # A component relation vanishes modulo the connected subtorus, not
    # necessarily as a point of the ambient torus. Here the nonintegral
    # generator is the image of parameter 3/4 under t |-> (5t,-4t).
    parameter = Fraction(3, 4)
    assert (
        tuple(
            (int(row[0]) * parameter) % 1
            for row in family.identity_component.embedding.entries
        )
        == generator
    )


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
    relation_points = _matrix_fraction_multiply(
        component_lifts,
        tuple(tuple(Fraction(value) for value in row) for row in relation_matrix),
    )
    # Each relation is zero in the component quotient: modulo the ambient
    # integer lattice it lies in the connected identity subtorus. These two
    # relation points use the zero identity parameter; the regression above
    # covers the essential nonzero-parameter case.
    identity_parameters = ((Fraction(0), Fraction(0)),)
    identity_relation_points = _matrix_fraction_multiply(
        identity_embedding,
        identity_parameters,
    )
    assert tuple(tuple(value % 1 for value in row) for row in relation_points) == tuple(
        tuple(value % 1 for value in row) for row in identity_relation_points
    )
    determinant = abs(_determinant_two(relation_matrix))
    assert determinant == int(family.finite_components.component_count) == 6
    assert family.finite_components.invariant_factors == ("6",)


def test_empty_character_satisfies_the_obstruction_definition() -> None:
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


def test_empty_result_deserialization_does_not_replay_obstruction_theorem() -> None:
    source = _source(
        [[1]],
        [Fraction(1, 3)],
    )
    result = affine_torus_fixed_locus(source)
    assert isinstance(result.outcome, EmptyAffineTorusFixedLocus)
    payload = result.model_dump(mode="json")
    payload["outcome"]["obstruction_pairing"] = {"num": "1", "den": "2"}

    assert AffineTorusFixedLocusResult.model_validate(payload)


def test_nonempty_result_deserialization_does_not_replay_kernel_dimension() -> None:
    source = _source(((1,),), (Fraction(0),))
    result = affine_torus_fixed_locus(source).model_dump(mode="json")
    outcome = dict(result["outcome"])
    family = dict(outcome["fixed_locus"])
    identity_component = dict(family["identity_component"])
    identity_component["parameter_dimension"] = 0
    identity_component["embedding"] = {
        "domain": "ZZ",
        "row_count": 1,
        "column_count": 0,
        "entries": [[]],
    }
    family["identity_component"] = identity_component
    outcome["fixed_locus"] = family
    result["outcome"] = outcome

    assert AffineTorusFixedLocusResult.model_validate(result)


def test_random_small_full_rank_kernels_match_a_grid_congruence_oracle() -> None:
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
        source = _source(linear_part, translation)
        kernel = _compute_fixed_locus_kernel(_kernel_source(source))
        assert isinstance(kernel, NonemptyFixedLocusKernel)
        actual = _finite_kernel_points(kernel)
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


def test_integral_shear_carries_a_positive_dimensional_fixed_locus() -> None:
    source_result = affine_torus_fixed_locus(
        _source(((3, 0), (0, 1)), (Fraction(1, 3), Fraction(0)))
    )
    sheared_result = affine_torus_fixed_locus(
        _source(((3, -2), (0, 1)), (Fraction(1, 3), Fraction(0)))
    )

    assert isinstance(source_result.outcome, NonemptyAffineTorusFixedLocus)
    assert isinstance(sheared_result.outcome, NonemptyAffineTorusFixedLocus)
    source_family = source_result.outcome.fixed_locus
    sheared_family = sheared_result.outcome.fixed_locus
    assert source_family.identity_component.embedding.entries == (("0",), ("1",))
    assert sheared_family.identity_component.embedding.entries == (("1",), ("1",))
    assert (
        sheared_family.finite_components.component_count
        == source_family.finite_components.component_count
        == "2"
    )
    assert tuple(
        value.as_fraction() for value in sheared_family.base_point.coordinates
    ) == tuple(value.as_fraction() for value in source_family.base_point.coordinates)


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


def test_component_relation_theorem_is_not_replayed_during_deserialization() -> None:
    result = affine_torus_fixed_locus(_source(((3,),), (Fraction(0),))).model_dump(
        mode="json"
    )
    outcome = dict(result["outcome"])
    family = dict(outcome["fixed_locus"])
    generator = dict(family["component_generators"][0])
    generator["coordinates"] = [{"num": "1", "den": "3"}]
    family["component_generators"] = [generator]
    outcome["fixed_locus"] = family
    result["outcome"] = outcome

    assert AffineTorusFixedLocusResult.model_validate(result)


def test_saturated_annihilator_theorem_is_not_replayed_during_deserialization() -> None:
    result = affine_torus_fixed_locus(
        _source(
            ((2, 0, -2), (0, 3, -2), (0, 0, 1)),
            (Fraction(0), Fraction(0), Fraction(0)),
        )
    ).model_dump(mode="json")
    outcome = dict(result["outcome"])
    family = dict(outcome["fixed_locus"])
    generator = dict(family["component_generators"][1])
    generator["coordinates"] = [
        {"num": "0", "den": "1"},
        {"num": "1", "den": "4"},
        {"num": "0", "den": "1"},
    ]
    family["component_generators"] = [generator]
    family["finite_components"] = {
        "generator_count": 1,
        "relation_matrix": {
            "domain": "ZZ",
            "row_count": 1,
            "column_count": 1,
            "entries": [["2"]],
        },
        "generator_orders": ["2"],
        "invariant_factors": ["2"],
        "component_count": "2",
    }
    outcome["fixed_locus"] = family
    result["outcome"] = outcome

    assert AffineTorusFixedLocusResult.model_validate(result)


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
    assert AffineTorusFixedLocusResult.model_validate(contradictory_count)

    contradictory_order = dict(result)
    outcome = dict(result["outcome"])
    family = dict(outcome["fixed_locus"])
    components = dict(family["finite_components"])
    components["generator_orders"] = ["3"]
    family["finite_components"] = components
    outcome["fixed_locus"] = family
    contradictory_order["outcome"] = outcome
    assert AffineTorusFixedLocusResult.model_validate(contradictory_order)


@pytest.mark.parametrize(
    "mutation",
    ("rows", "columns", "digits", "coordinates", "nested_component"),
)
def test_raw_shape_and_digit_preflight_precedes_nested_parsing(mutation: str) -> None:
    payload = _payload(((1,),), (Fraction(0),))
    if mutation == "rows":
        payload["linear_part"]["entries"] = [[object()] for _ in range(65)]
    elif mutation == "columns":
        payload["linear_part"]["entries"] = [[object() for _ in range(65)]]
    elif mutation == "digits":
        payload["linear_part"]["entries"] = [["1" * 501]]
    elif mutation == "coordinates":
        payload["translation"]["coordinates"] = [object() for _ in range(65)]
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


def test_request_schema_matches_sign_aware_affine_scalar_digit_bounds() -> None:
    schema = AffineTorusFixedLocusRequest.model_json_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    source_schema = schema["$defs"]["RationalAffineTorusMap"]
    linear_scalar = source_schema["properties"]["linear_part"]["properties"]["entries"][
        "items"
    ]["items"]
    rational_scalar = source_schema["properties"]["translation"]["properties"][
        "coordinates"
    ]["items"]["properties"]

    assert linear_scalar["pattern"] == r"^(?:0|-?[1-9][0-9]{0,499})$"
    assert linear_scalar["maxLength"] == 501
    assert rational_scalar["num"]["pattern"] == linear_scalar["pattern"]
    assert rational_scalar["num"]["maxLength"] == 501
    assert rational_scalar["den"]["pattern"] == r"^[1-9][0-9]{0,499}$"
    assert rational_scalar["den"]["maxLength"] == 500

    valid_payload = {
        "affine_map": _payload(
            ((-(10**500 - 1),),),
            (Fraction(10**500 - 2, 10**500 - 1),),
        )
    }
    assert not list(validator.iter_errors(valid_payload))
    AffineTorusFixedLocusRequest.model_validate(valid_payload)

    for field, invalid in (
        ("linear", "-" + "1" * 501),
        ("numerator", "1" * 501),
        ("denominator", "1" * 501),
        ("denominator", "-1"),
    ):
        invalid_payload = {"affine_map": _payload(((1,),), (Fraction(1, 2),))}
        if field == "linear":
            invalid_payload["affine_map"]["linear_part"]["entries"] = [[invalid]]
        else:
            component = invalid_payload["affine_map"]["translation"]["coordinates"][0]
            component["num" if field == "numerator" else "den"] = invalid
        assert list(validator.iter_errors(invalid_payload)), (field, invalid)
        with pytest.raises(ValidationError):
            AffineTorusFixedLocusRequest.model_validate(invalid_payload)


def test_point_schema_matches_sign_aware_carrier_digit_bounds() -> None:
    digit_cap = MAX_AFFINE_TORUS_POINT_DIGITS
    schema = RationalTorusPoint.model_json_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    rational_scalar = schema["properties"]["coordinates"]["items"]
    components = rational_scalar["properties"]

    assert components["num"]["pattern"] == (
        rf"^(?:0|-?[1-9][0-9]{{0,{digit_cap - 1}}})$"
    )
    assert components["num"]["maxLength"] == digit_cap + 1
    assert components["den"]["pattern"] == rf"^[1-9][0-9]{{0,{digit_cap - 1}}}$"
    assert components["den"]["maxLength"] == digit_cap

    valid_point = {
        "torus": {"dimension": 1},
        "coordinates": [{"num": "1" + "0" * (digit_cap - 1), "den": "9" * digit_cap}],
    }
    assert not list(validator.iter_errors(valid_point))
    RationalTorusPoint.model_validate(valid_point)

    for component, invalid in (
        ("num", "1" * (digit_cap + 1)),
        ("num", "-" + "1" * (digit_cap + 1)),
        ("den", "1" * (digit_cap + 1)),
        ("den", "-1"),
    ):
        invalid_point: dict[str, Any] = {
            "torus": {"dimension": 1},
            "coordinates": [{"num": "1", "den": "2"}],
        }
        invalid_point["coordinates"][0][component] = invalid
        assert list(validator.iter_errors(invalid_point)), (component, invalid)
        with pytest.raises(ValidationError):
            RationalTorusPoint.model_validate(invalid_point)

    result_schema = AffineTorusFixedLocusResult.model_json_schema()
    result_scalar = result_schema["$defs"]["RationalTorusPoint"]["properties"][
        "coordinates"
    ]["items"]
    assert result_scalar == rational_scalar


def test_admission_records_the_maintained_backend_structure() -> None:
    source = _source(((3, 0), (0, 4)), (Fraction(0), Fraction(0)))
    plan = build_affine_torus_plan(source, deadline=monotonic() + 30)
    envelope = plan.backend_envelope

    assert dict(envelope.primitive_call_limits) == {
        "integer_rank": 5,
        "integer_hnf": 4,
        "rational_solve": 3,
        "integer_snf": 1,
        "rational_inverse": 1,
        "integer_multiply": 1,
    }
    assert envelope.maximum_integer_rows == 2
    assert envelope.maximum_integer_columns == 4
    assert envelope.maximum_integer_height >= plan.displacement_height
    assert envelope.maximum_rational_rows == 2
    assert envelope.maximum_rational_columns == 2
    assert envelope.maximum_rational_height >= 1
    assert plan.worker_input_bytes_upper_bound > 0


def test_near_envelope_kernel_fits_the_observed_backend_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dimension = 16
    rank = 8
    height = (10**32 - 1) // 2
    displacement = [[0] * dimension for _ in range(dimension)]
    for index in range(rank):
        displacement[index][index] = height - index
    # This shear makes (-2, 0, ..., 1, 0, ...) a primitive invariant
    # character. Its pairing with b is -1, so the nonempty path performs a
    # nonzero integral character lift as well as all three augmented HNFs.
    displacement[rank][0] = 2 * height
    linear_part = tuple(
        tuple(
            displacement[row][column] + int(row == column)
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    translation = (Fraction(1, 2),) + (Fraction(0),) * (dimension - 1)
    source = _source(linear_part, translation)
    plan = build_affine_torus_plan(source, deadline=monotonic() + 30)
    envelope = plan.backend_envelope
    executed: Counter[str] = Counter()
    integer_operands: list[tuple[int, int, int]] = []
    rational_operands: list[tuple[int, int, int]] = []

    def integer_height(matrix: Any) -> int:
        return max(
            (
                abs(int(matrix[row, column]))
                for row in range(matrix.nrows())
                for column in range(matrix.ncols())
            ),
            default=0,
        )

    def rational_height(matrix: Any) -> int:
        return max(
            (
                max(abs(int(matrix[row, column].p)), int(matrix[row, column].q))
                for row in range(matrix.nrows())
                for column in range(matrix.ncols())
            ),
            default=0,
        )

    def observe_integer(name: str, primitive: Callable[..., Any]) -> Callable[..., Any]:
        def observed(*matrices: Any) -> Any:
            executed[name] += 1
            integer_operands.extend(
                (matrix.nrows(), matrix.ncols(), integer_height(matrix))
                for matrix in matrices
            )
            return primitive(*matrices)

        return observed

    def observe_rational(
        name: str, primitive: Callable[..., Any]
    ) -> Callable[..., Any]:
        def observed(*matrices: Any) -> Any:
            executed[name] += 1
            rational_operands.extend(
                (matrix.nrows(), matrix.ncols(), rational_height(matrix))
                for matrix in matrices
            )
            return primitive(*matrices)

        return observed

    for name in (
        "integer_rank",
        "integer_hnf",
        "integer_snf",
        "integer_multiply",
    ):
        attribute = f"_{name}"
        monkeypatch.setattr(
            affine_flint,
            attribute,
            observe_integer(name, getattr(affine_flint, attribute)),
        )
    for name in ("rational_solve", "rational_inverse"):
        attribute = f"_{name}"
        monkeypatch.setattr(
            affine_flint,
            attribute,
            observe_rational(name, getattr(affine_flint, attribute)),
        )

    result = _compute_fixed_locus_kernel(_kernel_source(source))

    assert isinstance(result, NonemptyFixedLocusKernel)
    assert_charged_work_parity(
        charged=dict(envelope.primitive_call_limits), executed=executed
    )
    assert set(executed) == {
        "integer_rank",
        "integer_hnf",
        "rational_solve",
        "integer_snf",
        "rational_inverse",
        "integer_multiply",
    }
    assert all(
        rows <= envelope.maximum_integer_rows
        and columns <= envelope.maximum_integer_columns
        and height <= envelope.maximum_integer_height
        for rows, columns, height in integer_operands
    )
    assert all(
        rows <= envelope.maximum_rational_rows
        and columns <= envelope.maximum_rational_columns
        and height <= envelope.maximum_rational_height
        for rows, columns, height in rational_operands
    )


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


def test_max_dimension_adversary_is_deterministic_and_transport_bounded() -> None:
    dimension = 16
    height = 10**32 - 1
    linear_part = tuple(
        tuple(
            -height if column == row else height - row - column if column > row else 0
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    source = _source(linear_part, (Fraction(0),) * dimension)
    plan = build_affine_torus_plan(source, deadline=monotonic() + 30)

    first = affine_torus_fixed_locus(source)
    second = affine_torus_fixed_locus(source)
    encoded = encode_strict_json(first.model_dump(mode="json"))

    assert first == second
    assert len(encoded) <= plan.worker_stdout_bytes_upper_bound
    assert (
        AffineTorusFixedLocusResult.model_validate_json(encoded, strict=True) == first
    )


def test_admission_uses_the_exact_attainable_rank_not_the_nonzero_row_count() -> None:
    # A rank-one displacement with every row non-zero (H * ones) must not be
    # rejected through a fictitious rank-three minor bound: the actual result
    # has a ~500-digit component count and fits the canonical carriers.
    height = 10**499
    linear = tuple(
        tuple(height + int(row == column) for column in range(3)) for row in range(3)
    )
    source = _source(linear, (Fraction(0), Fraction(0), Fraction(0)))

    plan = build_affine_torus_plan(source, deadline=monotonic() + 300)
    kernel = _compute_fixed_locus_kernel(_kernel_source(source))

    assert tuple(bounds.rank for bounds in plan.rank_bounds) == (1,)
    assert isinstance(kernel, NonemptyFixedLocusKernel)
    assert len(kernel.component_generators) == plan.rank_bounds[0].rank
    assert kernel.component_count == height
    assert plan.bounds_for_rank(1).source_minor_height >= height


def test_zero_translation_uses_selected_generator_height_not_minor_bound() -> None:
    height = 10**499
    source = _source(
        tuple(
            tuple((height + 1) if row == column else 0 for column in range(3))
            for row in range(3)
        ),
        (Fraction(0),) * 3,
    )

    plan = build_affine_torus_plan(source, deadline=monotonic() + 300)
    bounds = plan.bounds_for_rank(3)

    assert bounds.source_minor_height > height
    assert bounds.component_generator_height == height


def test_point_height_is_revalidated_after_the_selected_translated_solve() -> None:
    determinant_scale = 10**399
    translation_denominator = 10**499
    source = _source(
        (
            (determinant_scale + 1, 1),
            (1, determinant_scale + 1),
        ),
        (Fraction(1, translation_denominator), Fraction(0)),
    )

    with pytest.raises(
        OperationDomainValidationError,
        match="exact fixed-locus point bound exceeds",
    ):
        build_affine_torus_plan(source, deadline=monotonic() + 300)


def test_admission_uses_the_selected_translated_solve_height() -> None:
    height = 10**499
    linear = ((height + 1, 0, 0), (0, height + 1, 0), (0, 0, 1))
    source = _source(
        linear,
        (Fraction(1, height), Fraction(0), Fraction(0)),
    )

    plan = build_affine_torus_plan(source, deadline=monotonic() + 300)
    result = affine_torus_fixed_locus(source)

    assert isinstance(result.outcome, NonemptyAffineTorusFixedLocus)
    assert result.outcome.fixed_locus.base_point.coordinates[0].as_fraction() == (
        Fraction(height * height - 1, height * height)
    )
    assert plan.bounds_for_rank(2).base_point_component_height == height * height


def test_translated_identity_is_empty_without_charging_the_translation_lcm() -> None:
    # Three pairwise-coprime 500-digit denominators make the global lcm far
    # larger than the point carrier, but a translated identity is empty and
    # produces no base point, so admission must not reject it via that lcm.
    denominators = (10**499, 10**499 + 1, 10**499 + 2)
    linear = tuple(tuple(int(row == column) for column in range(3)) for row in range(3))
    translation = tuple(Fraction(1, denominator) for denominator in denominators)
    source = _source(linear, translation)

    plan = build_affine_torus_plan(source, deadline=monotonic() + 300)
    result = affine_torus_fixed_locus(source)

    assert tuple(bounds.rank for bounds in plan.rank_bounds) == (0,)
    assert plan.bounds_for_rank(0).base_point_component_height == 1
    assert result.outcome.status == "EMPTY"


def test_empty_locus_is_classified_before_large_nonempty_point_solve() -> None:
    denominators = (10**499, 10**499 + 1, 10**499 + 3)
    linear = (
        (2, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    )
    source = _source(
        linear,
        (
            Fraction(0),
            *(Fraction(1, denominator) for denominator in denominators),
        ),
    )

    plan = build_affine_torus_plan(source, deadline=monotonic() + 300)
    result = affine_torus_fixed_locus(source)

    bounds = plan.bounds_for_rank(1)
    assert len(str(bounds.obstruction_pairing_height)) > MAX_AFFINE_TORUS_POINT_DIGITS
    assert isinstance(result.outcome, EmptyAffineTorusFixedLocus)
    assert result.outcome.obstruction_pairing.as_fraction() == Fraction(
        1, denominators[0]
    )


def test_dimension_envelope_agrees_with_the_matrix_carrier() -> None:
    # The affine-matrix schema and preflight must not advertise a dimension the
    # reused integer-matrix carrier cannot parse: a 33-axis linear part is
    # unrepresentable and must be rejected, while a 17-axis identity is a small
    # well-formed request that the derived admission accepts.
    seventeen_identity = [
        [int(row == column) for column in range(17)] for row in range(17)
    ]
    source = _source(seventeen_identity, (Fraction(0),) * 17)
    plan = build_affine_torus_plan(source, deadline=monotonic() + 30)
    assert plan.dimension == 17

    oversized = {
        "torus": {"dimension": 33},
        "linear_part": {
            "row_count": 33,
            "column_count": 33,
            "entries": [
                [str(int(row == column)) for column in range(33)] for row in range(33)
            ],
        },
        "translation": {
            "torus": {"dimension": 33},
            "coordinates": [{"num": "0", "den": "1"} for _ in range(33)],
        },
    }
    with pytest.raises(ValidationError):
        RationalAffineTorusMap.model_validate(oversized)
