"""Cross-owner seam: serialized CE differentials drive GF(p) matrix consumers.

An MCP caller holding ``DifferentialMatrix.model_dump()["matrix"]`` — the
canonical {prime, entries, columns} value — must be able to pass it unchanged
as every ``prime_field.matrix`` consumer's ``matrix`` value without
reshaping its contents.
"""

from __future__ import annotations

from typing import Any

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation

_SL2_GF5 = {
    "prime": 5,
    "dimension": 3,
    "structure_constants": [
        [[0, 0, 0], [0, 0, 1], [3, 0, 0]],
        [[0, 0, 4], [0, 0, 0], [0, 2, 0]],
        [[2, 0, 0], [0, 3, 0], [0, 0, 0]],
    ],
}


def _serialized_degree_two_differential() -> dict[str, Any]:
    complex_output = invoke_operation(
        "lie_algebra.chevalley_eilenberg.complex.compute",
        {"lie_algebra": _SL2_GF5},
        Catalog.open(),
    ).output
    differential = next(d for d in complex_output["differentials"] if d["degree"] == 2)
    return dict(differential["matrix"])


def test_serialized_differential_drives_every_public_matrix_consumer() -> None:
    serialized = _serialized_degree_two_differential()
    assert serialized == {
        "prime": 5,
        "entries": [[0, 2, 0], [0, 0, 3], [4, 0, 0]],
        "columns": 3,
    }
    catalog = Catalog.open()
    payload = {"matrix": serialized}

    rank_output = invoke_operation(
        "prime_field.matrix.rank.compute", payload, catalog
    ).output
    assert rank_output["rank"] == 3

    rref_output = invoke_operation(
        "prime_field.matrix.rref.compute", payload, catalog
    ).output
    assert rref_output["rref_matrix"]["entries"] == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    assert rref_output["pivot_columns"] == [0, 1, 2]

    nullspace_output = invoke_operation(
        "prime_field.matrix.nullspace.compute", payload, catalog
    ).output
    assert nullspace_output["nullity"] == 0


def test_homology_of_the_same_complex_is_exact() -> None:
    homology_output = invoke_operation(
        "lie_algebra.homology.compute",
        {"lie_algebra": _SL2_GF5},
        Catalog.open(),
    ).output
    assert tuple(group["betti"] for group in homology_output["groups"]) == (
        1,
        0,
        0,
        1,
    )
