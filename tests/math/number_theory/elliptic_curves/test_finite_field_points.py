import json
import math
from fractions import Fraction

import pytest
import rfc8785
from pydantic import ValidationError

from jacobian.canonical import CanonicalLimits
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
)
from jacobian.math.number_theory.elliptic_curves import (
    finite_field as finite_field_module,
)
from jacobian.math.number_theory.elliptic_curves.finite_field import (
    FiniteFieldEllipticPoint,
    FiniteFieldShortWeierstrassCurve,
    FiniteFieldZetaFunctionResult,
    finite_field_cardinality,
    finite_field_curve_base_change,
    finite_field_discriminant,
    finite_field_extension_counts,
    finite_field_frobenius,
    finite_field_group_structure,
    finite_field_isogeny_class,
    finite_field_isomorphism,
    finite_field_point_add,
    finite_field_point_check,
    finite_field_point_negate,
    finite_field_point_order,
    finite_field_point_scalar,
    finite_field_points,
    finite_field_quadratic_twist,
    finite_field_zeta_function,
    finite_field_zeta_polynomial,
)
from jacobian.math.polynomials.values import require_canonical_rational_function


def test_finite_field_group_identities_and_cardinality() -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=one, coefficient_b=one
    )
    points = finite_field_points(curve).points
    assert len(points) == 9
    for point in points:
        assert finite_field_point_add(
            curve, point, finite_field_point_negate(curve, point).point
        ).point.at_infinity
    result = finite_field_cardinality(curve)
    assert result.cardinality == len(points)
    assert result.trace == 5 + 1 - len(points)


def test_frobenius_data_and_supersingularity_match_direct_f5_oracle() -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )

    def curve(a: int, b: int) -> FiniteFieldShortWeierstrassCurve:
        return FiniteFieldShortWeierstrassCurve(
            field=field,
            coefficient_a=FiniteFieldElement(presentation=field, coordinates=(a,)),
            coefficient_b=FiniteFieldElement(presentation=field, coordinates=(b,)),
        )

    def direct_count(a: int, b: int) -> int:
        return 1 + sum(
            1
            for x in range(5)
            for y in range(5)
            if (y * y - x * x * x - a * x - b) % 5 == 0
        )

    for coefficients in ((1, 1), (0, 1), (2, 1)):
        a, b = coefficients
        result = finite_field_frobenius(curve(a, b))
        count = direct_count(a, b)
        trace = 6 - count
        assert result.curve == curve(a, b)
        assert result.cardinality == count
        assert result.trace == trace
        assert result.determinant == 5
        assert result.characteristic_polynomial.coefficients == (5, -trace, 1)
        assert result.discriminant == trace * trace - 20
        assert result.classification == (
            "SUPERSINGULAR" if trace % 5 == 0 else "ORDINARY"
        )


def test_frobenius_accepts_field_5003_with_admitted_character_sum() -> None:
    field = FiniteFieldPresentation(
        characteristic=5003, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    result = finite_field_frobenius(
        FiniteFieldShortWeierstrassCurve(
            field=field, coefficient_a=one, coefficient_b=one
        )
    )
    assert result.cardinality > 0
    assert result.characteristic_polynomial.coefficients[0] == 5003


def test_isogeny_class_decision_matches_independent_finite_field_counts() -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )

    def curve(a: int, b: int) -> FiniteFieldShortWeierstrassCurve:
        return FiniteFieldShortWeierstrassCurve(
            field=field,
            coefficient_a=FiniteFieldElement(presentation=field, coordinates=(a,)),
            coefficient_b=FiniteFieldElement(presentation=field, coordinates=(b,)),
        )

    def direct_count(a: int, b: int) -> int:
        return 1 + sum(
            1
            for x in range(5)
            for y in range(5)
            if (y * y - x * x * x - a * x - b) % 5 == 0
        )

    first, other_in_class, different_class = curve(1, 1), curve(1, 4), curve(2, 1)
    same = finite_field_isogeny_class(first, other_in_class)
    different = finite_field_isogeny_class(first, different_class)

    first_count = direct_count(1, 1)
    same_count = direct_count(1, 4)
    different_count = direct_count(2, 1)
    assert same.first_cardinality == first_count == 9
    assert same.second_cardinality == same_count == 9
    assert same.first_trace == 5 + 1 - first_count
    assert same.second_trace == 5 + 1 - same_count
    assert same.first_frobenius_polynomial == (5, -same.first_trace, 1)
    assert same.second_frobenius_polynomial == (5, -same.second_trace, 1)
    assert same.same_isogeny_class is True
    assert different.first_cardinality == first_count
    assert different.second_cardinality == different_count == 7
    assert different.same_isogeny_class is False


def test_model_isomorphism_matches_independent_complete_scaling_oracle() -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )

    def curve(a: int, b: int) -> FiniteFieldShortWeierstrassCurve:
        return FiniteFieldShortWeierstrassCurve(
            field=field,
            coefficient_a=FiniteFieldElement(presentation=field, coordinates=(a,)),
            coefficient_b=FiniteFieldElement(presentation=field, coordinates=(b,)),
        )

    def scaling_oracle(source: tuple[int, int], target: tuple[int, int]) -> int | None:
        a, b = source
        target_a, target_b = target
        return next(
            (
                u
                for u in range(1, 5)
                if (pow(u, 4, 5) * a) % 5 == target_a
                and (pow(u, 6, 5) * b) % 5 == target_b
            ),
            None,
        )

    source = curve(1, 1)
    # u=2 transports (1,1) to (u^4,u^6)=(1,4) in F5.
    target = curve(1, 4)
    unrelated = curve(2, 1)
    positive = finite_field_isomorphism(source, target)
    negative = finite_field_isomorphism(source, unrelated)
    assert scaling_oracle((1, 1), (1, 4)) == 2
    assert positive.isomorphic is True
    assert positive.scaling is not None
    u = positive.scaling.coordinates[0]
    assert (pow(u, 4, 5), pow(u, 6, 5)) == (1, 4)
    assert scaling_oracle((1, 1), (2, 1)) is None
    assert negative.isomorphic is False
    assert negative.scaling is None


def test_model_isomorphism_public_example_composes_through_dispatch() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("elliptic_curve.finite_field.isomorphism.decide")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output["isomorphic"] is True
    assert result.output["scaling"] is not None


def test_model_isomorphism_bounds_complete_search_before_field_arithmetic(
    monkeypatch,
) -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=one, coefficient_b=one
    )
    monkeypatch.setattr(finite_field_module, "MAX_FINITE_FIELD_ISOMORPHISM_ORDER", 4)
    monkeypatch.setattr(
        finite_field_module,
        "_decode_field_element",
        lambda *_args: pytest.fail("field arithmetic must follow search admission"),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        finite_field_isomorphism(curve, curve)
    assert error.value.errors()[0]["type"] == (
        "elliptic_curve.finite_field.isomorphism_order_bound"
    )


def test_canonical_quadratic_twist_flips_trace_over_prime_field() -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )

    def element(value: int) -> FiniteFieldElement:
        return FiniteFieldElement(presentation=field, coordinates=(value,))

    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=element(1), coefficient_b=element(1)
    )
    twist = finite_field_quadratic_twist(curve)
    # 2 is the first encoded nonsquare in F5, so d^2 A=4 and d^3 B=3.
    assert twist.coefficient_a.coordinates == (4,)
    assert twist.coefficient_b.coordinates == (3,)

    def direct_count(a: int, b: int) -> int:
        return 1 + sum(
            (y * y - x * x * x - a * x - b) % 5 == 0 for x in range(5) for y in range(5)
        )

    source_order = direct_count(1, 1)
    twist_order = direct_count(4, 3)
    assert source_order == 9
    assert twist_order == 3
    assert source_order + twist_order == 2 * (5 + 1)


def test_quadratic_twist_composes_with_extension_count_consumer() -> None:
    field = FiniteFieldPresentation(
        characteristic=5,
        modulus_coefficients=(2, 0, 1),
        generator="a",
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1, 0))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=one, coefficient_b=one
    )
    twist = finite_field_quadratic_twist(curve)
    source_count = finite_field_cardinality(curve)
    twist_count = finite_field_cardinality(twist)
    assert source_count.trace + twist_count.trace == 0
    assert source_count.cardinality + twist_count.cardinality == 2 * (25 + 1)


def test_quadratic_twist_rejects_field_order_before_search(monkeypatch) -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=one, coefficient_b=one
    )
    monkeypatch.setattr(finite_field_module, "MAX_FINITE_FIELD_TWIST_ORDER", 4)
    monkeypatch.setattr(
        finite_field_module,
        "_power",
        lambda *_args: pytest.fail("field search must follow order admission"),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        finite_field_module.finite_field_quadratic_twist(curve)
    assert error.value.errors()[0]["type"] == (
        "elliptic_curve.finite_field.twist_order_bound"
    )


@pytest.mark.parametrize(
    ("prime", "coefficient_a", "coefficient_b", "expected_factors"),
    [(5, 1, 1, (9,)), (5, 1, 0, (2, 2)), (7, 1, 1, (5,))],
)
def test_group_structure_and_generators_match_independent_exhaustive_oracle(
    prime: int,
    coefficient_a: int,
    coefficient_b: int,
    expected_factors: tuple[int, ...],
) -> None:
    field = FiniteFieldPresentation(
        characteristic=prime, modulus_coefficients=(0, 1), generator="a"
    )
    curve = FiniteFieldShortWeierstrassCurve(
        field=field,
        coefficient_a=FiniteFieldElement(
            presentation=field, coordinates=(coefficient_a,)
        ),
        coefficient_b=FiniteFieldElement(
            presentation=field, coordinates=(coefficient_b,)
        ),
    )

    def add(
        left: tuple[int, int] | None, right: tuple[int, int] | None
    ) -> tuple[int, int] | None:
        if left is None:
            return right
        if right is None:
            return left
        x, y = left
        u, v = right
        if x == u and (y + v) % prime == 0:
            return None
        if left == right:
            slope = (3 * x * x + coefficient_a) * pow(2 * y, -1, prime) % prime
        else:
            slope = (v - y) * pow(u - x, -1, prime) % prime
        new_x = (slope * slope - x - u) % prime
        return new_x, (slope * (x - new_x) - y) % prime

    points: set[tuple[int, int] | None] = {None}
    points.update(
        (x, y)
        for x in range(prime)
        for y in range(prime)
        if (y * y - x * x * x - coefficient_a * x - coefficient_b) % prime == 0
    )
    result = finite_field_group_structure(curve)
    factors = tuple(result.group.invariant_factors)
    assert factors == expected_factors
    assert len(result.generators) == len(factors)
    assert all(generator.curve == curve for generator in result.generators)

    generated: set[tuple[int, int] | None] = {None}
    for generator, order in zip(result.generators, factors, strict=True):
        assert not generator.at_infinity
        assert generator.x is not None and generator.y is not None
        point = (generator.x.coordinates[0], generator.y.coordinates[0])
        multiples: set[tuple[int, int] | None] = set()
        multiple = None
        for _ in range(order):
            multiples.add(multiple)
            multiple = add(multiple, point)
        assert multiple is None
        assert len(multiples) == order
        generated = {add(left, right) for left in generated for right in multiples}

    assert len(generated) == len(points) == math.prod(factors)
    assert generated == points


def test_group_structure_operation_is_published_and_serializable() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("elliptic_curve.finite_field.group_structure.compute")
    assert operation is not None
    invocation = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    result = operation.result_type.model_validate_json(json.dumps(invocation.output))
    assert tuple(result.group.invariant_factors) == (9,)
    assert len(result.generators) == 1


def test_group_structure_rejects_work_before_point_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=one, coefficient_b=one
    )
    monkeypatch.setattr(finite_field_module, "MAX_FINITE_FIELD_GROUP_STRUCTURE_WORK", 1)
    monkeypatch.setattr(
        finite_field_module,
        "_enumerate_admitted_points",
        lambda *_args: pytest.fail("group work must be admitted before enumeration"),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        finite_field_group_structure(curve)
    assert error.value.errors()[0]["type"] == (
        "elliptic_curve.finite_field.group_structure_work_bound"
    )


def test_isogeny_class_requires_identical_field_presentation() -> None:
    first_field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    second_field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="b"
    )
    first = FiniteFieldShortWeierstrassCurve(
        field=first_field,
        coefficient_a=FiniteFieldElement(presentation=first_field, coordinates=(1,)),
        coefficient_b=FiniteFieldElement(presentation=first_field, coordinates=(1,)),
    )
    second = FiniteFieldShortWeierstrassCurve(
        field=second_field,
        coefficient_a=FiniteFieldElement(presentation=second_field, coordinates=(1,)),
        coefficient_b=FiniteFieldElement(presentation=second_field, coordinates=(1,)),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        finite_field_isogeny_class(first, second)
    assert error.value.errors()[0]["type"] == (
        "elliptic_curve.finite_field.isogeny_field_mismatch"
    )


def test_point_orders_match_independent_repeated_addition_and_witnesses() -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    curve = FiniteFieldShortWeierstrassCurve(
        field=field,
        coefficient_a=FiniteFieldElement(presentation=field, coordinates=(1,)),
        coefficient_b=FiniteFieldElement(presentation=field, coordinates=(1,)),
    )

    def add(left, right):
        if left is None:
            return right
        if right is None:
            return left
        x1, y1 = left
        x2, y2 = right
        if x1 == x2 and (y1 + y2) % 5 == 0:
            return None
        if left == right:
            if y1 == 0:
                return None
            slope = (3 * x1 * x1 + 1) * pow(2 * y1, -1, 5) % 5
        else:
            slope = (y2 - y1) * pow(x2 - x1, -1, 5) % 5
        x3 = (slope * slope - x1 - x2) % 5
        return (x3, (slope * (x1 - x3) - y1) % 5)

    for point in finite_field_points(curve).points:
        coordinates = (
            None
            if point.at_infinity
            else (point.x.coordinates[0], point.y.coordinates[0])
        )
        multiple = None
        expected_order = None
        for scalar in range(1, 10):
            multiple = add(multiple, coordinates)
            if multiple is None:
                expected_order = scalar
                break
        assert expected_order is not None

        result = finite_field_point_order(curve, point)
        assert result.group_cardinality == 9
        assert result.order == expected_order
        assert result.annihilating_multiple.at_infinity
        assert tuple(w.prime for w in result.prime_divisor_witnesses) == tuple(
            prime
            for prime in range(2, expected_order + 1)
            if expected_order % prime == 0
            and all(prime % divisor for divisor in range(2, int(prime**0.5) + 1))
        )
        for witness in result.prime_divisor_witnesses:
            assert witness.reduced_scalar == result.order // witness.prime
            assert not witness.reduced_multiple.at_infinity


def test_point_order_operation_is_published_and_serializable() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("elliptic_curve.finite_field.point.order.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output["group_cardinality"] == 9
    assert result.output["order"] == 9
    assert result.output["annihilating_multiple"]["at_infinity"] is True
    assert [
        (witness["prime"], witness["reduced_scalar"])
        for witness in result.output["prime_divisor_witnesses"]
    ] == [(3, 3)]


def test_point_order_rejects_field_above_exact_counting_envelope() -> None:
    prime = 5003
    field = FiniteFieldPresentation(
        characteristic=prime, modulus_coefficients=(0, 1), generator="a"
    )
    curve = FiniteFieldShortWeierstrassCurve(
        field=field,
        coefficient_a=FiniteFieldElement(presentation=field, coordinates=(1,)),
        coefficient_b=FiniteFieldElement(presentation=field, coordinates=(1,)),
    )
    point = FiniteFieldEllipticPoint.affine(
        curve,
        FiniteFieldElement(presentation=field, coordinates=(0,)),
        FiniteFieldElement(presentation=field, coordinates=(1,)),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        finite_field_point_order(curve, point)
    assert error.value.errors()[0]["type"] == (
        "elliptic_curve.finite_field.enumeration_bound"
    )


def test_isogeny_class_operation_is_published_and_serializable() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("elliptic_curve.finite_field.isogeny_class.decide")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output["same_isogeny_class"] is False
    assert result.output["first_frobenius_polynomial"] == [5, 3, 1]
    assert result.output["second_frobenius_polynomial"] == [5, 1, 1]


@pytest.mark.parametrize("prime", [5, 7, 11])
def test_point_enumeration_is_exact_against_exhaustive_integer_oracle(
    prime: int,
) -> None:
    field = FiniteFieldPresentation(
        characteristic=prime, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=one, coefficient_b=one
    )
    expected = {
        (x, y)
        for x in range(prime)
        for y in range(prime)
        if (y * y - (x * x * x + x + 1)) % prime == 0
    }
    actual = finite_field_points(curve)
    coordinates = {
        (point.x.coordinates[0], point.y.coordinates[0])
        for point in actual.points
        if not point.at_infinity and point.x is not None and point.y is not None
    }
    assert coordinates == expected
    assert len(actual.points) == len(expected) + 1
    assert sum(point.at_infinity for point in actual.points) == 1


def test_scaled_point_enumeration_uses_exact_character_count_oracle() -> None:
    """The larger accepted field checks every output point and exact coverage."""
    prime = 10_007
    field = FiniteFieldPresentation(
        characteristic=prime, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=one, coefficient_b=one
    )
    expected_count = 1
    for x in range(prime):
        rhs = (x * x * x + x + 1) % prime
        if rhs == 0:
            expected_count += 1
        elif pow(rhs, (prime - 1) // 2, prime) == 1:
            expected_count += 2

    point_set = finite_field_points(curve)
    assert len(point_set.points) == expected_count
    assert len(rfc8785.dumps(point_set.model_dump(mode="json"))) <= (
        CanonicalLimits().max_output_bytes
    )
    assert sum(point.at_infinity for point in point_set.points) == 1
    affine = [point for point in point_set.points if not point.at_infinity]
    pairs = {(point.x.coordinates[0], point.y.coordinates[0]) for point in affine}
    assert len(pairs) == len(affine)
    assert all((y * y - (x * x * x + x + 1)) % prime == 0 for x, y in pairs)


def test_point_enumeration_rejects_output_bound_before_expansion() -> None:
    prime = 20_011
    field = FiniteFieldPresentation(
        characteristic=prime, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=one, coefficient_b=one
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        finite_field_points(curve)
    assert error.value.errors()[0]["type"] == (
        "elliptic_curve.finite_field.enumeration_output_bound"
    )


def test_point_enumeration_rejects_singular_curve() -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    zero = FiniteFieldElement(presentation=field, coordinates=(0,))
    singular = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=zero, coefficient_b=zero
    )
    with pytest.raises(OperationDomainValidationError) as error:
        finite_field_points(singular)
    assert error.value.errors()[0]["type"] == (
        "elliptic_curve.finite_field.singular_curve"
    )


def test_f5_point_group_law_matches_independent_oracle_and_hand_examples() -> None:
    """Check the group law against an integer-coordinate F5 oracle."""
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=one, coefficient_b=one
    )

    def affine(x: int, y: int) -> FiniteFieldEllipticPoint:
        return FiniteFieldEllipticPoint.affine(
            curve,
            FiniteFieldElement(presentation=field, coordinates=(x,)),
            FiniteFieldElement(presentation=field, coordinates=(y,)),
        )

    identity = FiniteFieldEllipticPoint.infinity(curve)
    p = affine(0, 1)
    q = affine(2, 1)
    assert finite_field_point_add(curve, p, p).point == affine(4, 2)
    assert finite_field_point_add(curve, p, q).point == affine(3, 4)
    assert finite_field_point_add(curve, identity, p).point == p
    assert (
        finite_field_point_add(
            curve, p, finite_field_point_negate(curve, p).point
        ).point
        == identity
    )

    def oracle(left: tuple[int, int] | None, right: tuple[int, int] | None):
        if left is None:
            return right
        if right is None:
            return left
        x1, y1 = left
        x2, y2 = right
        if x1 == x2 and (y1 + y2) % 5 == 0:
            return None
        if left == right:
            slope = ((3 * x1 * x1 + 1) * pow(2 * y1, -1, 5)) % 5
        else:
            slope = ((y2 - y1) * pow(x2 - x1, -1, 5)) % 5
        x3 = (slope * slope - x1 - x2) % 5
        return (x3, (slope * (x1 - x3) - y1) % 5)

    def coordinates(point: FiniteFieldEllipticPoint):
        if point.at_infinity:
            return None
        assert point.x is not None and point.y is not None
        return (point.x.coordinates[0], point.y.coordinates[0])

    expected = {
        (x, y)
        for x in range(5)
        for y in range(5)
        if (y * y - (x * x * x + x + 1)) % 5 == 0
    }
    points = finite_field_points(curve).points
    assert {coordinates(point) for point in points if not point.at_infinity} == expected
    assert sum(point.at_infinity for point in points) == 1
    for left in points:
        for right in points:
            actual = finite_field_point_add(curve, left, right).point
            assert coordinates(actual) == oracle(coordinates(left), coordinates(right))


def test_finite_field_addition_is_published_and_runs_through_catalog() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("elliptic_curve.finite_field.point.add.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output["point"]["at_infinity"] is False
    assert result.output["point"]["x"]["coordinates"] == ["3"]
    assert result.output["point"]["y"]["coordinates"] == ["4"]


def test_finite_field_scalar_multiplication_rejects_work_above_bound() -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=one, coefficient_b=one
    )
    point = FiniteFieldEllipticPoint.affine(
        curve,
        FiniteFieldElement(presentation=field, coordinates=(0,)),
        one,
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        finite_field_point_scalar(curve, point, 1_000_001)
    assert error.value.errors()[0]["type"] == "elliptic_curve.finite_field.scalar_bound"


def test_extension_counts_match_direct_count_over_f25() -> None:
    base = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    extension = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(2, 0, 1), generator="b"
    )
    one_base = FiniteFieldElement(presentation=base, coordinates=(1,))
    curve = FiniteFieldShortWeierstrassCurve(
        field=base, coefficient_a=one_base, coefficient_b=one_base
    )

    def multiply(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        # In F25 = F5[b]/(b^2 + 2), b^2 = 3.
        return (
            (left[0] * right[0] + 3 * left[1] * right[1]) % 5,
            (left[0] * right[1] + left[1] * right[0]) % 5,
        )

    def add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        return ((left[0] + right[0]) % 5, (left[1] + right[1]) % 5)

    one = (1, 0)
    direct_count = 1  # The point at infinity.
    for x0 in range(5):
        for x1 in range(5):
            x = (x0, x1)
            rhs = add(add(multiply(multiply(x, x), x), x), one)
            for y0 in range(5):
                for y1 in range(5):
                    if multiply((y0, y1), (y0, y1)) == rhs:
                        direct_count += 1

    result = finite_field_extension_counts(curve, 2)
    assert extension.degree == 2
    assert result.base_cardinality == 9
    assert result.base_trace == -3
    assert result.counts[0].cardinality == result.base_cardinality
    assert result.counts[1].cardinality == direct_count == 27
    assert result.counts[1].frobenius_power_sum == -1


def test_point_enumeration_over_directly_presented_extension_field() -> None:
    """Directly presented F25 curves are supported; no base-field lift occurs."""
    # This is a curve declared over this exact F25 presentation. It does not
    # claim a canonical transport from a curve over F5: that needs an explicit
    # typed embedding between the two field presentations.
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(2, 0, 1), generator="b"
    )
    curve = FiniteFieldShortWeierstrassCurve(
        field=field,
        coefficient_a=FiniteFieldElement(presentation=field, coordinates=(1, 0)),
        coefficient_b=FiniteFieldElement(presentation=field, coordinates=(1, 0)),
    )

    def add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        return ((left[0] + right[0]) % 5, (left[1] + right[1]) % 5)

    def multiply(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        # The declared modulus is b^2 + 2, hence b^2 = 3 in F25.
        return (
            (left[0] * right[0] + 3 * left[1] * right[1]) % 5,
            (left[0] * right[1] + left[1] * right[0]) % 5,
        )

    expected: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    one = (1, 0)
    for x0 in range(5):
        for x1 in range(5):
            x = (x0, x1)
            rhs = add(add(multiply(multiply(x, x), x), x), one)
            for y0 in range(5):
                for y1 in range(5):
                    y = (y0, y1)
                    if multiply(y, y) == rhs:
                        expected.add((x, y))

    result = finite_field_points(curve)
    actual = {
        (point.x.coordinates, point.y.coordinates)
        for point in result.points
        if not point.at_infinity and point.x is not None and point.y is not None
    }
    assert len(result.points) == 27
    assert actual == expected
    assert all(point.curve == curve for point in result.points)
    # The independently enumerated order is 27; Lagrange then checks every
    # returned affine point through the public group operation.
    for point in result.points:
        assert finite_field_point_scalar(curve, point, 27).point.at_infinity


def test_zeta_numerator_matches_independent_f5_and_f25_counts() -> None:
    base = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    base_curve = FiniteFieldShortWeierstrassCurve(
        field=base,
        coefficient_a=FiniteFieldElement(presentation=base, coordinates=(1,)),
        coefficient_b=FiniteFieldElement(presentation=base, coordinates=(1,)),
    )
    # Direct enumeration by the defining equation, independent of the
    # character-sum kernel used by finite_field_zeta_polynomial.
    f5_count = 1 + sum(
        pow(y, 2, 5) == (pow(x, 3, 5) + x + 1) % 5 for x in range(5) for y in range(5)
    )
    f5 = finite_field_zeta_polynomial(base_curve)
    assert f5_count == f5.cardinality == 9
    assert f5.trace == -3
    assert f5.numerator.coefficients == (5, 3, 1)

    full_zeta = finite_field_zeta_function(base_curve)
    assert full_zeta.curve == base_curve
    assert full_zeta.cardinality == f5_count
    assert full_zeta.trace == -3
    assert full_zeta.zeta_function.variables == ("T",)
    assert {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in full_zeta.zeta_function.numerator.terms
    } == {2: 1, 1: Fraction(3, 5), 0: Fraction(1, 5)}
    assert {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in full_zeta.zeta_function.denominator.terms
    } == {2: 1, 1: Fraction(-6, 5), 0: Fraction(1, 5)}
    assert require_canonical_rational_function(full_zeta.zeta_function) == (
        full_zeta.zeta_function
    )
    assert (
        FiniteFieldZetaFunctionResult.model_validate_json(full_zeta.model_dump_json())
        == full_zeta
    )
    with pytest.raises(ValidationError):
        FiniteFieldZetaFunctionResult.model_validate(
            {
                **full_zeta.model_dump(),
                "trace": full_zeta.trace + 1,
            }
        )
    with pytest.raises(ValidationError):
        FiniteFieldZetaFunctionResult.model_validate(
            {
                **full_zeta.model_dump(),
                "zeta_function": f5.numerator,
            }
        )

    extension = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(2, 0, 1), generator="b"
    )
    extension_curve = FiniteFieldShortWeierstrassCurve(
        field=extension,
        coefficient_a=FiniteFieldElement(presentation=extension, coordinates=(1, 0)),
        coefficient_b=FiniteFieldElement(presentation=extension, coordinates=(1, 0)),
    )

    def f25_multiply(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        # b^2 = 3 for the declared modulus b^2 + 2.
        return (
            (left[0] * right[0] + 3 * left[1] * right[1]) % 5,
            (left[0] * right[1] + left[1] * right[0]) % 5,
        )

    def f25_add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        return ((left[0] + right[0]) % 5, (left[1] + right[1]) % 5)

    one = (1, 0)
    direct_f25_count = 1
    for x0 in range(5):
        for x1 in range(5):
            x = (x0, x1)
            rhs = f25_add(f25_add(f25_multiply(f25_multiply(x, x), x), x), one)
            for y0 in range(5):
                for y1 in range(5):
                    y = (y0, y1)
                    direct_f25_count += f25_multiply(y, y) == rhs
    f25 = finite_field_zeta_polynomial(extension_curve)
    assert direct_f25_count == f25.cardinality == 27
    assert f25.trace == -1
    assert f25.numerator.coefficients == (25, 1, 1)

    # The zeta numerator's trace predicts the quadratic extension count,
    # independently matched above by F25 point enumeration.
    q, linear, _constant = f5.numerator.coefficients
    a = -linear
    assert direct_f25_count == q**2 + 1 - (a**2 - 2 * q)


def test_zeta_polynomial_is_publicly_discoverable_and_exact() -> None:
    tool = Catalog.open().operation(
        "elliptic_curve.finite_field.zeta_polynomial.compute"
    )
    assert tool is not None
    result = invoke_operation(tool.operation_id, tool.examples[0].input, Catalog.open())
    assert result.output["cardinality"] == 9
    assert result.output["trace"] == -3
    assert result.output["numerator"]["coefficients"] == ["5", "3", "1"]


def test_full_zeta_function_is_publicly_discoverable() -> None:
    tool = Catalog.open().operation("elliptic_curve.finite_field.zeta.compute")
    assert tool is not None
    result = invoke_operation(tool.operation_id, tool.examples[0].input, Catalog.open())
    assert result.output["cardinality"] == 9
    assert result.output["trace"] == -3
    assert result.output["zeta_function"]["variables"] == ["T"]


def test_curve_and_point_transport_along_explicit_f5_to_f25_embedding() -> None:
    base = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    extension = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(2, 0, 1), generator="b"
    )
    curve = FiniteFieldShortWeierstrassCurve(
        field=base,
        coefficient_a=FiniteFieldElement(presentation=base, coordinates=(1,)),
        coefficient_b=FiniteFieldElement(presentation=base, coordinates=(1,)),
    )
    point = FiniteFieldEllipticPoint.affine(
        curve,
        FiniteFieldElement(presentation=base, coordinates=(0,)),
        FiniteFieldElement(presentation=base, coordinates=(1,)),
    )
    # The source modulus is X, so its generator maps to zero in F25.
    embedding = finite_field_module.FieldEmbedding(
        source=base,
        target=extension,
        generator_image=FiniteFieldElement(presentation=extension, coordinates=(0, 0)),
    )
    result = finite_field_curve_base_change(curve, embedding, point)
    assert result.curve.field == extension
    assert result.curve.coefficient_a.coordinates == (1, 0)
    assert result.curve.coefficient_b.coordinates == (1, 0)
    assert result.point is not None
    assert result.point.curve == result.curve
    assert result.point.x is not None and result.point.x.coordinates == (0, 0)
    assert result.point.y is not None and result.point.y.coordinates == (1, 0)
    assert finite_field_point_check(result.curve, result.point).on_curve
    assert finite_field_point_scalar(result.curve, result.point, 9).point.at_infinity

    operation = Catalog.open().operation(
        "elliptic_curve.finite_field.base_change.compute"
    )
    assert operation is not None
    wire_result = invoke_operation(
        operation.operation_id,
        {
            "curve": curve.model_dump(mode="json"),
            "embedding": embedding.model_dump(mode="json"),
            "point": point.model_dump(mode="json"),
        },
        Catalog.open(),
    )
    assert wire_result.output["curve"]["field"] == extension.model_dump(mode="json")
    assert wire_result.output["point"]["y"]["coordinates"] == ["1", "0"]

    invalid = finite_field_module.FieldEmbedding(
        source=base,
        target=extension,
        generator_image=FiniteFieldElement(presentation=extension, coordinates=(1, 0)),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        finite_field_curve_base_change(curve, invalid)
    assert error.value.errors()[0]["type"] == (
        "finite_field.embedding_generator_not_root"
    )


def test_base_change_public_example_is_advertised_and_runs() -> None:
    operation = Catalog.open().operation(
        "elliptic_curve.finite_field.base_change.compute"
    )
    assert operation is not None
    assert operation.examples[0].name == "base_change_f5_to_f25"
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, Catalog.open()
    )
    assert result.output["curve"]["field"]["modulus_coefficients"] == [
        "2",
        "0",
        "1",
    ]
    assert result.output["point"] is None


@pytest.mark.parametrize(("prime", "a", "b"), [(5, 1, 1), (7, 2, 3), (11, 0, 4)])
def test_character_sum_count_matches_independent_point_enumeration(
    prime: int, a: int, b: int
) -> None:
    field = FiniteFieldPresentation(
        characteristic=prime, modulus_coefficients=(0, 1), generator="a"
    )
    curve = FiniteFieldShortWeierstrassCurve(
        field=field,
        coefficient_a=FiniteFieldElement(presentation=field, coordinates=(a,)),
        coefficient_b=FiniteFieldElement(presentation=field, coordinates=(b,)),
    )

    count = finite_field_extension_counts(curve, 1)
    enumerated = finite_field_points(curve)
    assert count.base_cardinality == len(enumerated.points)
    assert count.base_trace == prime + 1 - len(enumerated.points)


def test_large_admitted_prime_field_count_uses_character_sum_boundary() -> None:
    field = FiniteFieldPresentation(
        characteristic=4093, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=one, coefficient_b=one
    )

    result = finite_field_extension_counts(curve, 1)
    assert result.base_cardinality == 4100
    assert result.base_trace == -6


def test_extension_count_degree_is_bounded_before_counting() -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=one, coefficient_b=one
    )
    with pytest.raises(OperationDomainValidationError) as error:
        finite_field_extension_counts(curve, 65)
    assert error.value.errors()[0]["type"] == (
        "elliptic_curve.finite_field.extension_degree_bound"
    )


def test_extension_count_catalog_example_runs_with_typed_output() -> None:
    operation = Catalog.open().operation(
        "elliptic_curve.finite_field.extension_counts.compute"
    )
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, Catalog.open()
    )
    validated = operation.result_type.model_validate_json(json.dumps(result.output))
    assert validated.counts[0].cardinality == 9
    assert validated.counts[1].cardinality == 27


def test_native_curve_consumers_reject_missing_authored_fields() -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    missing_coefficient = FiniteFieldShortWeierstrassCurve.model_construct(
        field=field, coefficient_a=one
    )
    with pytest.raises(OperationDomainValidationError):
        finite_field_points(missing_coefficient)

    malformed_coefficient = FiniteFieldElement.model_construct(coordinates=(1,))
    with pytest.raises(OperationDomainValidationError):
        finite_field_discriminant(field, malformed_coefficient, one)


def test_native_point_consumer_rejects_forged_coordinate_axis() -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=one, coefficient_b=one
    )
    forged_coordinate = FiniteFieldElement.model_construct(
        presentation=field, coordinates=(1, 2)
    )
    forged = FiniteFieldEllipticPoint.model_construct(
        curve=curve, at_infinity=False, x=one, y=forged_coordinate
    )
    with pytest.raises(OperationDomainValidationError) as error:
        finite_field_point_negate(curve, forged)
    assert error.value.errors()[0]["type"] == (
        "elliptic_curve.finite_field.point_coordinates"
    )
