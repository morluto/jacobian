"""Outcome vocabulary should retrieve witness-bearing integer normal forms."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationMatchRequest


@pytest.mark.parametrize(
    "query",
    [
        "Given integer matrix A and integer vector b, determine the exact additive order of b modulo the column lattice of A and return integer multipliers witnessing m b = A x, or a separating obstruction if no positive m exists.",
        "Find the least positive integer multiplier that puts a vector in an integer column lattice, with a witness.",
        "Determine whether a class in an integer matrix cokernel is torsion and compute its order with unimodular transformations.",
    ],
)
def test_integer_lattice_order_retrieves_certified_smith(query: str) -> None:
    matches = Catalog.open().match(OperationMatchRequest(need=query, limit=10)).matches
    assert "matrix.normal_form.smith.certified.compute" in {
        m.operation_id for m in matches
    }


def test_column_lattice_basis_retrieves_hermite_via_explicit_transpose() -> None:
    matches = (
        Catalog.open()
        .match(
            OperationMatchRequest(
                need="Integer column lattice canonical basis and membership with a unimodular transformation via transpose",
                limit=10,
            )
        )
        .matches
    )
    assert "lattice.hermite_normal_form.compute" in {m.operation_id for m in matches}


@pytest.mark.parametrize(
    ("query", "operation_id"),
    [
        (
            "Sidon set extension profile with uncovered differences",
            "combinatorics.integer_set.sidon.extension_profile.compute",
        ),
        (
            "subset sum residue profile modulo an integer",
            "additive.subset_sum.residue_profile.compute",
        ),
        ("lattice rank and Gram matrix", "lattice.rank_gram.compute"),
        ("order of a permutation", "group.element_order.compute"),
    ],
)
def test_neighboring_mathematical_intents_remain_discoverable(
    query: str, operation_id: str
) -> None:
    matches = Catalog.open().match(OperationMatchRequest(need=query, limit=5)).matches
    assert operation_id in {m.operation_id for m in matches}
