"""Exact weight-orbit output and resource refusal across the public boundary."""

import json

import pytest
from jsonschema import Draft202012Validator

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.groups.root_systems import (
    WeightLatticeVector,
    WeylParabolicWeightOrbitResult,
    WeylWeightOrbitResult,
    weight_lattice_vector,
    weyl_parabolic_weight_orbit,
    weyl_weight_orbit,
)
from jacobian.math.groups.root_systems import operations as rs_operations
from jacobian.math.groups.root_systems._models import (
    MAX_LATTICE_COORDINATE_BITS,
    MAX_REFLECTION_REPRESENTABLE,
)

_PARABOLIC = "weyl_group.parabolic_weight.orbit.compute"
_FULL_ORBIT = "weyl_group.weight.orbit.compute"
_G2 = ((2, -3), (-1, 2))


@pytest.mark.parametrize(
    "coordinate",
    [0, MAX_REFLECTION_REPRESENTABLE + 1, -((1 << MAX_LATTICE_COORDINATE_BITS) - 1)],
)
def test_trivial_parabolic_round_trips_the_produced_large_weight(
    coordinate: int,
) -> None:
    catalog = Catalog.open()
    produced = invoke_operation(
        "root_system.weight_lattice.vector.compute",
        {"matrix": [[2]], "coordinates": [str(coordinate)]},
        catalog,
    )
    weight_payload = json.loads(produced.model_dump_json())["output"]
    native_weight = WeightLatticeVector.model_validate_json(json.dumps(weight_payload))
    native = weyl_parabolic_weight_orbit(native_weight, ())
    public = invoke_operation(
        _PARABOLIC, {"weight": weight_payload, "simple_root_indices": []}, catalog
    )

    assert public.output == native.model_dump(mode="json")
    assert public.output["orbit"] == [[str(coordinate)]]
    assert public.output["weight"] == [str(coordinate)]
    descriptor = catalog.inspect(_PARABOLIC)
    assert descriptor is not None
    Draft202012Validator(descriptor.output_schema).validate(public.output)
    assert (
        WeylParabolicWeightOrbitResult.model_validate_json(json.dumps(public.output))
        == native
    )
    # Each orbit image remains exact coordinates for the typed weight consumer.
    image_weight = {
        "datum": public.output["datum"],
        "coordinates": public.output["orbit"][0],
    }
    assert (
        invoke_operation(
            _PARABOLIC, {"weight": image_weight, "simple_root_indices": []}, catalog
        ).output
        == public.output
    )


@pytest.mark.parametrize("parabolic", [False, True])
def test_representable_g2_orbit_has_native_and_public_parity(parabolic: bool) -> None:
    native: WeylParabolicWeightOrbitResult | WeylWeightOrbitResult
    scale = MAX_REFLECTION_REPRESENTABLE // 5
    weight = (scale, scale)
    # The twelve G2 images of (1,1), from its two simple reflection matrices.
    unit_orbit = (
        (-5, 2),
        (-5, 3),
        (-4, 1),
        (-4, 3),
        (-1, -1),
        (-1, 2),
        (1, -2),
        (1, 1),
        (4, -3),
        (4, -1),
        (5, -3),
        (5, -2),
    )
    expected = tuple(tuple(scale * value for value in image) for image in unit_orbit)
    if parabolic:
        native_weight = weight_lattice_vector(_G2, weight)
        native = weyl_parabolic_weight_orbit(native_weight, (0, 1))
        operation_id = _PARABOLIC
        payload = {
            "weight": native_weight.model_dump(mode="json"),
            "simple_root_indices": [0, 1],
        }
    else:
        native = weyl_weight_orbit(_G2, weight)
        operation_id = _FULL_ORBIT
        payload = {"matrix": [list(row) for row in _G2], "weight": list(weight)}
    public = invoke_operation(operation_id, payload, Catalog.open())

    assert native.orbit == expected
    assert public.output == native.model_dump(mode="json")


def test_proper_parabolic_checks_only_its_executed_reflections() -> None:
    weight = weight_lattice_vector(_G2, (1, -MAX_REFLECTION_REPRESENTABLE))
    native = weyl_parabolic_weight_orbit(weight, (0,))
    public = invoke_operation(
        _PARABOLIC,
        {"weight": weight.model_dump(mode="json"), "simple_root_indices": [0]},
        Catalog.open(),
    )

    assert native.orbit == (
        (-1, -MAX_REFLECTION_REPRESENTABLE + 1),
        (1, -MAX_REFLECTION_REPRESENTABLE),
    )
    assert public.output == native.model_dump(mode="json")


@pytest.mark.parametrize(
    ("indices", "weight"),
    [
        (None, (1, -MAX_REFLECTION_REPRESENTABLE)),
        ((0, 1), (1, -MAX_REFLECTION_REPRESENTABLE)),
        ((1,), (1, -MAX_REFLECTION_REPRESENTABLE)),
        (None, (MAX_REFLECTION_REPRESENTABLE // 5 + 1,) * 2),
        ((0, 1), (MAX_REFLECTION_REPRESENTABLE // 5 + 1,) * 2),
    ],
)
def test_oversized_executed_orbit_is_a_public_resource_refusal(
    indices: tuple[int, ...] | None,
    weight: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reflected_weights: list[tuple[int, ...]] = []
    reflect = rs_operations._weight_reflect

    def recording_reflect(
        value: tuple[int, ...], index: int, rows: tuple[tuple[int, ...], ...]
    ) -> tuple[int, ...]:
        reflected_weights.append(value)
        return reflect(value, index, rows)

    monkeypatch.setattr(rs_operations, "_weight_reflect", recording_reflect)
    if indices is not None:
        native_weight = weight_lattice_vector(_G2, weight)
        operation_id = _PARABOLIC
        payload = {
            "weight": native_weight.model_dump(mode="json"),
            "simple_root_indices": list(indices),
        }
        with pytest.raises(OperationResourceAdmissionError) as native_error:
            weyl_parabolic_weight_orbit(native_weight, indices)
    else:
        operation_id = _FULL_ORBIT
        payload = {"matrix": [list(row) for row in _G2], "weight": list(weight)}
        with pytest.raises(OperationResourceAdmissionError) as native_error:
            weyl_weight_orbit(_G2, weight)
    with pytest.raises(OperationResourceAdmissionError) as public_error:
        invoke_operation(operation_id, payload, Catalog.open())

    for error in (native_error.value, public_error.value):
        assert error.errors()[0]["type"] == "root_system.weight_orbit_coordinate_bound"
        assert error.errors()[0]["loc"] == ("weight",)
    # Both paths must refuse before chamber normalization or weight-orbit BFS.
    assert reflected_weights == []
