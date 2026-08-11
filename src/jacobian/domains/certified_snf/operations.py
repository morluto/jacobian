"""Transformation-certified Smith normal-form operation."""

from __future__ import annotations

from typing import Any

from jacobian.contracts.certified_snf import (
    CertifiedSmithNormalFormRequest,
    CertifiedSmithNormalFormResult,
)
from jacobian.domains._certified_snf import certificate_from_reduction, smith_reduce
from jacobian.domains._examples import example
from jacobian.operations import ComputedSuccess, MaterializedOperation


def _certified_smith(
    request: CertifiedSmithNormalFormRequest,
) -> ComputedSuccess[CertifiedSmithNormalFormResult]:
    source = [[int(value) for value in row] for row in request.matrix.entries]
    reduction = smith_reduce(
        source,
        row_count=request.matrix.row_count,
        column_count=request.matrix.column_count,
    )
    return ComputedSuccess(
        CertifiedSmithNormalFormResult(
            certificate=certificate_from_reduction(reduction)
        )
    )


CERTIFIED_SNF_CAPABILITIES: tuple[MaterializedOperation[Any, Any, Any, Any], ...] = (
    MaterializedOperation(
        capability_id="matrix.normal_form.smith.certified.compute",
        title="Compute a transformation-certified Smith normal form",
        description=(
            "Compute the canonical Smith diagonal D and explicit unimodular "
            "matrices U and V satisfying D = U A V for one bounded integer matrix."
        ),
        request_model=CertifiedSmithNormalFormRequest,
        result_model=CertifiedSmithNormalFormResult,
        implementation=_certified_smith,
        relation_id="matrix.normal_form.smith.certified.relation",
        tags=(
            "matrix",
            "integer",
            "smith-normal-form",
            "unimodular-transformation",
            "certificate",
            "exact",
            "bounded",
        ),
        resource_reason=(
            "the complete U, D, and V transformation certificate is retained for "
            "independent replay and downstream integral-homology binding"
        ),
        invocation_examples=(
            example(
                "certified_smith_two_by_two",
                "Compute D, U, and V for a two-by-two integer matrix.",
                {
                    "matrix": {
                        "row_count": 2,
                        "column_count": 2,
                        "entries": [["2", "4"], ["6", "8"]],
                    }
                },
            ),
        ),
        version="4",
    ),
)

__all__ = ["CERTIFIED_SNF_CAPABILITIES"]
