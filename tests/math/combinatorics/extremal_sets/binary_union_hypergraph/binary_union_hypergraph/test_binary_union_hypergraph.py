from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.extremal_sets.binary_union_hypergraph._models import (
    BinaryUnionHypergraphRequest,
)
from jacobian.math.combinatorics.extremal_sets.binary_union_hypergraph.operations import (
    compute_binary_union_hypergraph,
)


def test_simple_union() -> None:
    """{1} union {2} = {1, 2}."""
    result = compute_binary_union_hypergraph(((1,), (2,), (1, 2)))
    # i=0, j=1, k=2: {1} union {2} = {1,2} -> one relation
    assert result.relation_count == 1


def test_no_relations() -> None:
    """Disjoint sets with no union relation."""
    result = compute_binary_union_hypergraph(((1,), (2,), (3,)))
    assert result.relation_count == 0


def test_multiple_relations() -> None:
    """{1} union {2} = {1,2}, {1} union {3} = {1,3}, etc."""
    result = compute_binary_union_hypergraph(((1,), (2,), (3,), (1, 2), (1, 3), (2, 3)))
    # {1} union {2}={1,2}, {1} union {3}={1,3}, {2} union {3}={2,3}
    assert result.relation_count == 3


def test_result_preserves_source() -> None:
    result = compute_binary_union_hypergraph(((1,), (2,), (1, 2)))
    assert result.sets == ((1,), (2,), (1, 2))


def test_oversized_family_is_rejected_before_relation_scan() -> None:
    family = tuple((value,) for value in range(257))

    with pytest.raises(OperationDomainValidationError, match="256-vertex"):
        compute_binary_union_hypergraph(family)
    with pytest.raises(ValidationError, match="256-vertex"):
        BinaryUnionHypergraphRequest(sets=family)


def test_finite_set_inputs_must_be_canonical_and_distinct() -> None:
    with pytest.raises(OperationDomainValidationError, match="strictly increasing"):
        compute_binary_union_hypergraph(((2, 1), (3,)))
    with pytest.raises(OperationDomainValidationError, match="distinct sets"):
        compute_binary_union_hypergraph(((1,), (1,)))
