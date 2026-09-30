"""Executed-path regressions for the Weyl weight and orbit boundaries.

Each test fails against the corresponding defect and passes on the repair; the
negative control is this file run against unmodified `main`.
"""

from __future__ import annotations

import pytest

from jacobian._execution import OperationBackendError
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems import (
    WeylElement,
    weight_lattice_vector,
)
from jacobian.math.groups.root_systems import operations as rs_operations
from jacobian.math.groups.root_systems._models import (
    MAX_LATTICE_OUTPUT_COORDINATE_BITS,
    MAX_REFLECTION_REPRESENTABLE,
    WeylParabolicWeightOrbitResult,
)
from jacobian.math.groups.root_systems.operations import (
    weyl_antidominant_representative,
    weyl_parabolic_weight_orbit,
    weyl_weight_orbit,
)

A1 = ((2,),)
A2 = ((2, -1), (-1, 2))
G2 = ((2, -3), (-1, 2))


def _a1(weight: int) -> int:
    return weight


def test_trivial_parabolic_admits_a_large_exact_source_weight() -> None:
    """A parabolic with no generators reflects nothing, so the source is the orbit.

    The ambient Weyl norm is a bound for reflections that are never performed;
    applying it to the trivial subgroup refuses a representable singleton orbit.
    """
    large = MAX_REFLECTION_REPRESENTABLE + 1
    assert large.bit_length() <= MAX_LATTICE_OUTPUT_COORDINATE_BITS
    weight = weight_lattice_vector(A1, (large,))

    result = weyl_parabolic_weight_orbit(weight, ())

    assert result.orbit == (result.weight,)
    assert result.weight == (large,)

    revived = WeylParabolicWeightOrbitResult.model_validate(result.model_dump())
    assert revived == result


def test_nontrivial_parabolic_still_bounds_every_image() -> None:
    """The representability bound is unchanged once a reflection is performed."""
    weight = weight_lattice_vector(A1, (MAX_REFLECTION_REPRESENTABLE,))

    result = weyl_parabolic_weight_orbit(weight, (0,))

    assert max(abs(value) for value in result.orbit[0]) <= MAX_REFLECTION_REPRESENTABLE
    with pytest.raises(OperationDomainValidationError) as error:
        weyl_parabolic_weight_orbit(
            weight_lattice_vector(A1, (MAX_REFLECTION_REPRESENTABLE + 1,)),
            (0,),
        )
    assert (
        error.value.errors()[0]["type"] == "root_system.weight_orbit_coordinate_bound"
    )


def test_parabolic_rejects_a_negative_native_generator_index() -> None:
    """Python would treat -1 as the first generator and report a wrong orbit."""
    weight = weight_lattice_vector(A1, (1,))

    with pytest.raises(OperationDomainValidationError) as error:
        weyl_parabolic_weight_orbit(weight, (-1,))

    assert error.value.errors()[0]["type"] == "root_system.parabolic_simple_indices"
    assert error.value.errors()[0]["loc"] == ("simple_root_indices",)


def test_parabolic_result_model_rejects_a_negative_index() -> None:
    weight = weight_lattice_vector(A1, (1,))
    result = weyl_parabolic_weight_orbit(weight, (0,))
    payload = result.model_dump()
    payload["simple_root_indices"] = (-1,)

    with pytest.raises(ValueError, match="greater_than_equal"):
        WeylParabolicWeightOrbitResult.model_validate(payload)


def test_weight_orbit_admits_a_representable_complete_orbit() -> None:
    """A worst-case norm estimate is a growth envelope, not a representability gate.

    For the G2 datum and this weight every one of the 12 images stays inside
    the interoperable bound, while the norm estimate does not.
    """
    weight = (1_801_439_850_948_198, 1_801_439_850_948_198)
    estimate = rs_operations._weight_coordinate_bounds(
        G2, weight, enforce_interoperable_bound=False
    )
    assert any(bound > MAX_REFLECTION_REPRESENTABLE for bound in estimate)

    result = weyl_weight_orbit(G2, weight)

    assert len(result.orbit) == 12
    assert all(
        abs(value) <= MAX_REFLECTION_REPRESENTABLE
        for image in result.orbit
        for value in image
    )


def test_weight_orbit_still_refuses_an_unrepresentable_orbit() -> None:
    with pytest.raises((OperationBackendError, OperationDomainValidationError)):
        weyl_weight_orbit(A1, (MAX_REFLECTION_REPRESENTABLE * 8 + 1,))
    # a representable image set is admitted, an unbounded one is not
    admitted = weyl_weight_orbit(A1, (MAX_REFLECTION_REPRESENTABLE // 2,))
    assert sorted(abs(value) for image in admitted.orbit for value in image) == [
        MAX_REFLECTION_REPRESENTABLE // 2,
        MAX_REFLECTION_REPRESENTABLE // 2,
    ]


def test_antidominant_representative_bounds_every_intermediate_reflection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The antidominant path must check what the dominant path checks.

    ``_weight_reflect`` performs no check, so without the intermediate bound
    this input grows to three times the interoperable limit mid-traversal and is
    only noticed because the final answer happens to be small.
    """
    weight = (1, -MAX_REFLECTION_REPRESENTABLE)
    reflect = rs_operations._weight_reflect
    observed: list[int] = []

    def recording_reflect(
        value: tuple[int, ...], index: int, rows: tuple[tuple[int, ...], ...]
    ) -> tuple[int, ...]:
        reflected = reflect(value, index, rows)
        observed.extend(abs(item) for item in reflected)
        return reflected

    monkeypatch.setattr(rs_operations, "_weight_reflect", recording_reflect)
    if True:
        result = weyl_antidominant_representative(G2, weight)

    assert result.antidominant_weight.coordinates == (
        -1,
        -MAX_REFLECTION_REPRESENTABLE + 1,
    )
    assert max(observed) <= 3 * MAX_REFLECTION_REPRESENTABLE
    # The intermediate norm bound is the invariant one, not the public limit.
    bounds = rs_operations._weight_coordinate_bounds(
        G2, weight, enforce_interoperable_bound=False
    )
    assert all(bound > MAX_REFLECTION_REPRESENTABLE for bound in bounds)
    assert all(observed[index] <= bound for index, bound in enumerate(bounds))


def test_integer_inverse_reports_the_caller_location() -> None:
    """The singular branch must name the same location as its sibling."""
    with pytest.raises(OperationDomainValidationError) as error:
        rs_operations._integer_inverse(((1, 1), (0, 0)), "element")

    assert error.value.errors()[0]["type"] == "root_system.noninvertible_weyl_action"
    assert error.value.errors()[0]["loc"] == ("element", "root_action")

    with pytest.raises(OperationDomainValidationError) as error:
        rs_operations._integer_inverse(((1, 1), (0, 0)), "lower")

    assert error.value.errors()[0]["loc"] == ("lower", "root_action")


def test_singular_weyl_action_is_refused_with_the_action_location() -> None:
    """The reported reproduction: a shape-valid but singular root action."""
    from jacobian.math.groups.root_systems._models import CartanMatrix
    from jacobian.math.matrices.values import IntegerMatrix

    for entries in (A2, G2):
        element = WeylElement.model_construct(
            matrix=CartanMatrix(
                matrix=IntegerMatrix(
                    row_count=len(entries),
                    column_count=len(entries),
                    entries=entries,
                ),
                simple_root_axis=tuple(range(len(entries))),
            ),
            root_action=IntegerMatrix(
                row_count=2, column_count=2, entries=((1, 1), (0, 0))
            ),
        )
        with pytest.raises(OperationDomainValidationError) as error:
            rs_operations._admit_weyl_element_value(element)
        assert error.value.errors()[0]["loc"] == ("element", "root_action")
        assert error.value.errors()[0]["type"] in {
            "root_system.action_not_root_automorphism",
            "root_system.action_not_weyl_element",
        }
