"""MathTool declarations for the issue-#1739 integer-lattice operations."""

from __future__ import annotations

from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.lattices._lattice_operations import (
    compute_canonical_basis,
    compute_direct_sum,
    compute_discriminant_group,
    compute_dual,
    compute_orthogonal_complement,
    compute_orthogonal_sum,
    compute_rank_gram,
    compute_saturation,
    compute_sublattice_index,
)
from jacobian.math.lattices._models import (
    CanonicalBasisResult,
    DirectSumRequest,
    DirectSumResult,
    DiscriminantGroupRequest,
    DiscriminantGroupResult,
    DualRequest,
    DualResult,
    IntegerLattice,
    OrthogonalComplementRequest,
    OrthogonalComplementResult,
    OrthogonalSumRequest,
    OrthogonalSumResult,
    RankGramRequest,
    RankGramResult,
    SaturationResult,
    SublatticeIndexRequest,
    SublatticeIndexResult,
)

__all__ = ["LATTICE_STRUCTURE_OPERATIONS"]


def _lattice(ambient: int, basis: list[list[int]]) -> dict[str, object]:
    """Return a JSON-serialable IntegerLattice payload for examples."""
    return {
        "ambient_dimension": ambient,
        "basis": {"entries": [[str(v) for v in row] for row in basis]},
    }


RANK_GRAM_OPERATION: MathTool[Any, Any] = MathTool(
    operation_id="lattice.rank_gram.compute",
    version="1",
    title="Compute exact rank, Gram matrix, and squared covolume of a lattice",
    description=(
        "For a full-row-rank integer lattice basis under the standard bilinear "
        "form, return the exact rank, labelled Gram matrix B B^T, and squared "
        "covolume det(G)."
    ),
    request_type=RankGramRequest,
    result_type=RankGramResult,
    run=compute_rank_gram,
    tags=("lattice", "gram", "exact-integer", "bounded"),
    examples=(
        example(
            "identity_lattice",
            "Compute the Gram matrix and covolume of ZZ^2.",
            {"lattice": _lattice(2, [[1, 0], [0, 1]])},
        ),
    ),
)

CANONICAL_BASIS_OPERATION: MathTool[Any, Any] = MathTool(
    operation_id="lattice.canonical_basis.compute",
    version="1",
    title="Compute the canonical HNF basis of a lattice",
    description=(
        "Return the row Hermite normal form canonical basis of a lattice and "
        "its exact left unimodular transformation."
    ),
    request_type=IntegerLattice,
    result_type=CanonicalBasisResult,
    run=compute_canonical_basis,
    tags=("lattice", "hermite-normal-form", "canonical-basis", "exact-integer"),
    examples=(
        example(
            "identity_lattice",
            "Canonical basis of ZZ^2.",
            _lattice(2, [[1, 0], [0, 1]]),
        ),
    ),
)

DUAL_OPERATION: MathTool[Any, Any] = MathTool(
    operation_id="lattice.dual.compute",
    version="1",
    title="Compute the exact rational dual lattice",
    description=(
        "For a full-rank integer lattice, return the exact rational dual basis "
        "L^* = {x in span_Q(L) : <x,L> subset ZZ} and its dual Gram matrix."
    ),
    request_type=DualRequest,
    result_type=DualResult,
    run=compute_dual,
    tags=("lattice", "dual", "exact-rational", "bounded"),
    examples=(
        example(
            "identity_lattice",
            "Dual of ZZ^2 is itself.",
            {"lattice": _lattice(2, [[1, 0], [0, 1]])},
        ),
    ),
)

SATURATION_OPERATION: MathTool[Any, Any] = MathTool(
    operation_id="lattice.saturation.compute",
    version="1",
    title="Compute the saturation (primitive closure) of a lattice",
    description=(
        "Return the canonical HNF basis of sat(L) = span_Q(L) cap ZZ^n, the "
        "integer inclusion matrix, and the finite saturation index."
    ),
    request_type=IntegerLattice,
    result_type=SaturationResult,
    run=compute_saturation,
    tags=("lattice", "saturation", "exact-integer", "bounded"),
    examples=(
        example(
            "identity_lattice",
            "Saturation of ZZ^2 is itself.",
            _lattice(2, [[1, 0], [0, 1]]),
        ),
    ),
)

SUBLATTICE_INDEX_OPERATION: MathTool[Any, Any] = MathTool(
    operation_id="lattice.sublattice_index.compute",
    version="1",
    title="Compute sublattice index and quotient invariant factors",
    description=(
        "For a supplied sublattice inclusion, return the finite index and the "
        "Smith invariant factors of the quotient parent / sublattice."
    ),
    request_type=SublatticeIndexRequest,
    result_type=SublatticeIndexResult,
    run=compute_sublattice_index,
    tags=("lattice", "smith-normal-form", "quotient", "exact-integer"),
    examples=(
        example(
            "double_sublattice",
            "Index of 2 ZZ inside ZZ.",
            {
                "sublattice": _lattice(1, [[2]]),
                "parent": _lattice(1, [[1]]),
                "embedding": {"entries": [["2"]]},
            },
        ),
    ),
)

DISCRIMINANT_GROUP_OPERATION: MathTool[Any, Any] = MathTool(
    operation_id="lattice.discriminant_group.compute",
    version="1",
    title="Compute the discriminant group and pairing order",
    description=(
        "For a nondegenerate integer lattice, return the discriminant order "
        "|det G| and the Smith invariant factors of L^*/L."
    ),
    request_type=DiscriminantGroupRequest,
    result_type=DiscriminantGroupResult,
    run=compute_discriminant_group,
    tags=("lattice", "discriminant-group", "exact-integer", "bounded"),
    examples=(
        example(
            "identity_lattice",
            "Discriminant group of ZZ^2 is trivial.",
            {"lattice": _lattice(2, [[1, 0], [0, 1]])},
        ),
    ),
)

ORTHOGONAL_COMPLEMENT_OPERATION: MathTool[Any, Any] = MathTool(
    operation_id="lattice.orthogonal_complement.compute",
    version="1",
    title="Compute the rational orthogonal complement",
    description=(
        "Return a canonical rational basis for the orthogonal complement of a "
        "lattice in QQ^n under the standard bilinear form."
    ),
    request_type=OrthogonalComplementRequest,
    result_type=OrthogonalComplementResult,
    run=compute_orthogonal_complement,
    tags=("lattice", "orthogonal-complement", "exact-rational", "bounded"),
    examples=(
        example(
            "line_in_plane",
            "Orthogonal complement of a line in QQ^2.",
            {"lattice": _lattice(2, [[1, 0]])},
        ),
    ),
)

DIRECT_SUM_OPERATION: MathTool[Any, Any] = MathTool(
    operation_id="lattice.direct_sum.compute",
    version="1",
    title="Compute the direct sum of two lattices",
    description=(
        "Return the block-diagonal direct sum of two integer lattices under "
        "the standard bilinear form."
    ),
    request_type=DirectSumRequest,
    result_type=DirectSumResult,
    run=compute_direct_sum,
    tags=("lattice", "direct-sum", "exact-integer", "bounded"),
    examples=(
        example(
            "z2_plus_z2",
            "Direct sum of ZZ^2 with ZZ^2 is ZZ^4.",
            {
                "first": _lattice(2, [[1, 0], [0, 1]]),
                "second": _lattice(2, [[1, 0], [0, 1]]),
            },
        ),
    ),
)

ORTHOGONAL_SUM_OPERATION: MathTool[Any, Any] = MathTool(
    operation_id="lattice.orthogonal_sum.compute",
    version="1",
    title="Compute the orthogonal sum of two lattices",
    description=(
        "Return the block-diagonal orthogonal sum of two integer lattices "
        "under the standard bilinear form."
    ),
    request_type=OrthogonalSumRequest,
    result_type=OrthogonalSumResult,
    run=compute_orthogonal_sum,
    tags=("lattice", "orthogonal-sum", "exact-integer", "bounded"),
    examples=(
        example(
            "z2_plus_z2",
            "Orthogonal sum of ZZ^2 with ZZ^2 is ZZ^4.",
            {
                "first": _lattice(2, [[1, 0], [0, 1]]),
                "second": _lattice(2, [[1, 0], [0, 1]]),
            },
        ),
    ),
)

LATTICE_STRUCTURE_OPERATIONS: MathTools = (
    RANK_GRAM_OPERATION,
    CANONICAL_BASIS_OPERATION,
    DUAL_OPERATION,
    SATURATION_OPERATION,
    SUBLATTICE_INDEX_OPERATION,
    DISCRIMINANT_GROUP_OPERATION,
    ORTHOGONAL_COMPLEMENT_OPERATION,
    DIRECT_SUM_OPERATION,
    ORTHOGONAL_SUM_OPERATION,
)
