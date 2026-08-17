"""Transformation-certified Smith normal-form operation."""

from __future__ import annotations

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.certified_snf._models import (
    CertifiedSmithNormalFormRequest,
    CertifiedSmithNormalFormResult,
)
from jacobian.math.certified_snf.operations import (
    certificate_from_reduction,
    smith_reduce,
)


def _certified_smith(
    request: CertifiedSmithNormalFormRequest,
) -> CertifiedSmithNormalFormResult:
    source = [[int(value) for value in row] for row in request.matrix.entries]
    reduction = smith_reduce(
        source,
        row_count=request.matrix.row_count,
        column_count=request.matrix.column_count,
    )
    return CertifiedSmithNormalFormResult(
        certificate=certificate_from_reduction(reduction)
    )


CERTIFIED_SNF_OPERATIONS: MathTools = (
    MathTool(
        operation_id="matrix.normal_form.smith.certified.compute",
        version="4",
        title="Compute a transformation-certified Smith normal form",
        description=(
            "Compute the canonical Smith diagonal D and explicit unimodular "
            "matrices U and V satisfying D = U A V for one integer matrix of "
            "at most 16 by 16."
        ),
        request_type=CertifiedSmithNormalFormRequest,
        result_type=CertifiedSmithNormalFormResult,
        run=_certified_smith,
        tags=(
            "matrix",
            "integer",
            "smith-normal-form",
            "unimodular-transformation",
            "certificate",
            "exact",
            "bounded",
        ),
        examples=(
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
    ),
)

__all__ = ["CERTIFIED_SNF_OPERATIONS"]
