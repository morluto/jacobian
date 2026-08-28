from __future__ import annotations

import copy

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation


def _example_payload(operation_id: str) -> dict[str, object]:
    operation = Catalog.open().operation(operation_id)
    assert operation is not None
    assert operation.examples
    return copy.deepcopy(operation.examples[0].input)


def test_universal_algebra_assignment_admission_is_typed() -> None:
    operation_id = "universal_algebra.term.evaluate.compute"
    payload = _example_payload(operation_id)
    payload["assignment"] = [0]

    with pytest.raises(OperationDomainValidationError) as caught:
        invoke_operation(operation_id, payload, Catalog.open())

    assert caught.value.errors() == (
        {
            "loc": ("assignment",),
            "type": "universal_algebra.assignment_coverage",
            "msg": "assignment must cover exactly the referenced variables",
        },
    )


def test_edge_path_continuity_admission_is_typed() -> None:
    operation_id = "topology.simplicial.edge_path.word.compute"
    payload = _example_payload(operation_id)
    path = payload["path"]
    assert isinstance(path, list)
    path[1] = {"edge_index": 2, "orientation": 1}

    with pytest.raises(OperationDomainValidationError) as caught:
        invoke_operation(operation_id, payload, Catalog.open())

    assert caught.value.errors()[0]["loc"] == ("path",)
    assert caught.value.errors()[0]["type"] == "topology.edge_path.path_continuity"


def test_simplicial_complex_admission_is_typed() -> None:
    operation_id = "topology.simplicial_complex.canonicalize"
    payload = {
        "vertices": ["a", "b", "isolated"],
        "facets": [["a", "b"]],
    }

    with pytest.raises(OperationDomainValidationError) as caught:
        invoke_operation(operation_id, payload, Catalog.open())

    assert caught.value.errors() == (
        {
            "loc": ("facets",),
            "type": "topology.require_request_complex_5",
            "msg": (
                "every vertex must occur in a facet; use a singleton facet for an "
                "isolated vertex"
            ),
        },
    )


def test_symbolic_dynamics_enumeration_admission_is_typed() -> None:
    operation_id = "symbolic_dynamics.block_language.compute"
    payload = {
        "shift": {
            "alphabet": list("abcdefghijklmnop"),
            "forbidden_blocks": [],
            "two_sided": True,
        },
        "block_length": 5,
    }

    with pytest.raises(OperationDomainValidationError) as caught:
        invoke_operation(operation_id, payload, Catalog.open())

    assert caught.value.errors()[0]["loc"] == ("block_length",)
    assert (
        caught.value.errors()[0]["type"]
        == "symbolic_dynamics.block_enumeration_not_admitted"
    )


@pytest.mark.parametrize(
    ("operation_id", "payload", "location", "code"),
    (
        (
            "group.element_order.compute",
            {"degree": 3, "generator": [0, 0, 1]},
            ("generator",),
            "group.generator_permutation",
        ),
        (
            "group.orbit.compute",
            {
                "group": {"degree": 2, "generators": [[1, 0]]},
                "point": 3,
            },
            ("point",),
            "group.point_out_of_range",
        ),
        (
            "group.conjugacy_classes.compute",
            {"degree": 3, "generators": [[0, 0, 1]]},
            ("generators",),
            "group.generator_permutation",
        ),
        (
            "group.stabilizer.compute",
            {
                "group": {"degree": 2, "generators": [[1, 0]]},
                "point": 3,
            },
            ("point",),
            "group.point_out_of_range",
        ),
    ),
)
def test_group_semantic_admission_is_owned_by_native_operations(
    operation_id: str,
    payload: dict[str, object],
    location: tuple[str, ...],
    code: str,
) -> None:
    with pytest.raises(OperationDomainValidationError) as caught:
        invoke_operation(operation_id, payload, Catalog.open())

    assert caught.value.errors()[0]["loc"] == location
    assert caught.value.errors()[0]["type"] == code


def test_euclidean_segment_admission_is_typed() -> None:
    operation_id = "geometry.euclidean.segment_ratio.compute"
    payload = _example_payload(operation_id)
    payload["segment2"] = [
        {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
        {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
    ]

    with pytest.raises(OperationDomainValidationError) as caught:
        invoke_operation(operation_id, payload, Catalog.open())

    assert caught.value.errors()[0]["loc"] == ("second",)
    assert caught.value.errors()[0]["type"] == "geometry.second_segment_nonzero"


@pytest.mark.parametrize(
    ("operation_id", "payload", "location", "code"),
    (
        (
            "rational.compute.reciprocal",
            {"value": {"num": "0", "den": "1"}},
            ("value",),
            "arithmetic.reciprocal_requires_nonzero",
        ),
        (
            "rational.compute.quotient",
            {
                "left": {"num": "1", "den": "2"},
                "right": {"num": "0", "den": "1"},
            },
            ("right",),
            "arithmetic.division_requires_nonzero_divisor",
        ),
    ),
)
def test_rational_arithmetic_admission_is_typed(
    operation_id: str,
    payload: dict[str, object],
    location: tuple[str, ...],
    code: str,
) -> None:
    with pytest.raises(OperationDomainValidationError) as caught:
        invoke_operation(operation_id, payload, Catalog.open())

    assert caught.value.errors()[0]["loc"] == location
    assert caught.value.errors()[0]["type"] == code


def test_orthogonal_recurrence_admission_is_typed() -> None:
    operation_id = "orthogonal_polynomial.recurrence.compute"
    payload = _example_payload(operation_id)
    family = payload["family"]
    assert isinstance(family, dict)
    polynomials = family["polynomials"]
    assert isinstance(polynomials, list)
    first = polynomials[0]
    assert isinstance(first, dict)
    first["squared_norm"] = {"num": "0", "den": "1"}
    family["is_quasi_definite"] = False
    family["is_positive_definite"] = False

    with pytest.raises(OperationDomainValidationError) as caught:
        invoke_operation(operation_id, payload, Catalog.open())

    assert caught.value.errors()[0]["loc"] == ("family",)
    assert caught.value.errors()[0]["type"] == "moments_orthogonal.zero_norm"
