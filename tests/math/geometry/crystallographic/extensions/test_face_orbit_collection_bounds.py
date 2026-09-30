"""Reject oversized raw face-orbit ledgers before expanding nested errors."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from pydantic import ValidationError
from tests.math.geometry.crystallographic.extensions.test_validator_repair_boundaries import (
    _real_complex,
)

from jacobian.math.geometry.crystallographic.extensions._models import (
    MAX_FACE_ORBIT_VERTICES,
    BieberbachFaceOrbitComplex,
)
from jacobian.math.geometry.polytopes._models import MAX_COMPUTED_FACETS

_LIMITS = (
    ("vertex_orbits", MAX_COMPUTED_FACETS),
    ("edge_orbit_representatives", MAX_COMPUTED_FACETS),
    ("orbit_maps", MAX_COMPUTED_FACETS * MAX_FACE_ORBIT_VERTICES),
    ("boundary_1_to_0", 2 * MAX_COMPUTED_FACETS * MAX_FACE_ORBIT_VERTICES),
    ("boundary_2_to_1", MAX_COMPUTED_FACETS),
)


@pytest.fixture(scope="module")
def real_complex() -> BieberbachFaceOrbitComplex:
    return _real_complex()


@pytest.mark.parametrize(("field", "limit"), _LIMITS)
@pytest.mark.parametrize("wire", [False, True])
def test_oversized_malformed_collection_stops_before_nested_errors(
    real_complex: BieberbachFaceOrbitComplex, field: str, limit: int, wire: bool
) -> None:
    payload = real_complex.model_dump(mode="json" if wire else "python")
    payload[field] = [{}] * (limit + 1)

    with pytest.raises(ValidationError) as error:
        if wire:
            BieberbachFaceOrbitComplex.model_validate_json(json.dumps(payload))
        else:
            BieberbachFaceOrbitComplex.model_validate(payload)

    assert error.value.error_count() == 1
    detail = error.value.errors(include_input=False)[0]
    assert detail["type"] == "crystallographic.extension.face_orbit_bound"
    assert detail["loc"] == (field,)


@pytest.mark.parametrize("wire", [False, True])
def test_oversized_malformed_vertex_orbit_stops_before_integer_errors(
    real_complex: BieberbachFaceOrbitComplex, wire: bool
) -> None:
    payload = real_complex.model_dump(mode="json" if wire else "python")
    payload["vertex_orbits"] = [[{}] * (MAX_FACE_ORBIT_VERTICES + 1)]

    with pytest.raises(ValidationError) as error:
        if wire:
            BieberbachFaceOrbitComplex.model_validate_json(json.dumps(payload))
        else:
            BieberbachFaceOrbitComplex.model_validate(payload)

    assert error.value.error_count() == 1
    assert error.value.errors(include_input=False)[0]["type"] == (
        "crystallographic.extension.face_orbit_bound"
    )


class _UnexpectedIterable:
    def __iter__(self) -> Iterator[object]:
        raise AssertionError("unadmitted iterables must not be consumed")


class _UnexpectedList(list[object]):
    def __len__(self) -> int:
        raise AssertionError("list subclasses must be refused before length checks")

    def __iter__(self) -> Iterator[object]:
        raise AssertionError("list subclasses must not be consumed")


@pytest.mark.parametrize("field", [field for field, _ in _LIMITS])
@pytest.mark.parametrize("container", [_UnexpectedIterable, _UnexpectedList])
def test_collection_type_is_checked_before_length_or_iteration(
    real_complex: BieberbachFaceOrbitComplex,
    field: str,
    container: type[_UnexpectedIterable] | type[_UnexpectedList],
) -> None:
    payload = real_complex.model_dump()
    payload[field] = container()

    with pytest.raises(ValidationError) as error:
        BieberbachFaceOrbitComplex.model_validate(payload)

    assert error.value.error_count() == 1
    assert error.value.errors(include_input=False)[0]["type"] == (
        "crystallographic.extension.face_orbit_collection"
    )


@pytest.mark.parametrize("container", [_UnexpectedIterable, _UnexpectedList])
def test_inner_orbit_type_is_checked_before_length_or_iteration(
    real_complex: BieberbachFaceOrbitComplex,
    container: type[_UnexpectedIterable] | type[_UnexpectedList],
) -> None:
    payload = real_complex.model_dump()
    payload["vertex_orbits"] = [container()]

    with pytest.raises(ValidationError) as error:
        BieberbachFaceOrbitComplex.model_validate(payload)

    assert error.value.error_count() == 1
    assert error.value.errors(include_input=False)[0]["type"] == (
        "crystallographic.extension.face_orbit_collection"
    )


def test_real_canonical_and_json_values_remain_accepted(
    real_complex: BieberbachFaceOrbitComplex,
) -> None:
    assert (
        BieberbachFaceOrbitComplex.model_validate(
            real_complex.model_dump(), strict=True
        )
        == real_complex
    )
    assert (
        BieberbachFaceOrbitComplex.model_validate_json(
            real_complex.model_dump_json(), strict=True
        )
        == real_complex
    )
    payload = real_complex.model_dump()
    for field, _ in _LIMITS:
        payload[field] = list(payload[field])
    assert BieberbachFaceOrbitComplex.model_validate(payload) == real_complex


def test_bounded_malformed_item_still_receives_nested_validation(
    real_complex: BieberbachFaceOrbitComplex,
) -> None:
    payload = real_complex.model_dump()
    payload["orbit_maps"] = [{}]

    with pytest.raises(ValidationError) as error:
        BieberbachFaceOrbitComplex.model_validate(payload)

    assert error.value.error_count() == 6
    assert {item["type"] for item in error.value.errors(include_input=False)} == {
        "missing"
    }


@pytest.mark.parametrize(
    ("field", "scalar_field"),
    (
        ("orbit_maps", "source_vertex_index"),
        ("boundary_1_to_0", "coefficient"),
        ("boundary_2_to_1", "incidence_index"),
    ),
)
def test_json_scalar_containers_reach_leaf_rejection_without_projection(
    real_complex: BieberbachFaceOrbitComplex, field: str, scalar_field: str
) -> None:
    payload = real_complex.model_dump(mode="json")
    malformed = [[0] * 1000]
    payload[field][0][scalar_field] = malformed

    with pytest.raises(ValidationError) as error:
        BieberbachFaceOrbitComplex.model_validate_json(json.dumps(payload), strict=True)

    assert error.value.error_count() == 1
    detail = error.value.errors()[0]
    assert detail["type"] == "int_type"
    assert detail["loc"] == (field, 0, scalar_field)
    assert detail["input"] == malformed
    assert type(detail["input"]) is list
    assert type(detail["input"][0]) is list


@pytest.mark.parametrize("field", ["orbit_maps", "boundary_1_to_0", "boundary_2_to_1"])
def test_json_oversized_malformed_translation_is_refused_before_projection(
    real_complex: BieberbachFaceOrbitComplex, field: str
) -> None:
    payload = real_complex.model_dump(mode="json")
    payload[field][0]["lattice_translation"] = [{}] * 5

    with pytest.raises(ValidationError) as error:
        BieberbachFaceOrbitComplex.model_validate_json(json.dumps(payload), strict=True)

    assert error.value.error_count() == 1
    assert error.value.errors(include_input=False)[0]["type"] == (
        "crystallographic.extension.face_orbit_bound"
    )


@pytest.mark.parametrize("field", ["orbit_maps", "boundary_1_to_0", "boundary_2_to_1"])
def test_json_unknown_row_keys_are_bounded_before_projection(
    real_complex: BieberbachFaceOrbitComplex, field: str
) -> None:
    payload = real_complex.model_dump(mode="json")
    payload[field][0]["unknown"] = [[0] * 1000]

    with pytest.raises(ValidationError) as error:
        BieberbachFaceOrbitComplex.model_validate_json(json.dumps(payload), strict=True)

    assert error.value.error_count() == 1
    assert error.value.errors(include_input=False)[0]["type"] == (
        "crystallographic.extension.face_orbit_bound"
    )
