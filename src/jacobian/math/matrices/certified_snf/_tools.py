"""Transformation-certified Smith normal forms."""

from __future__ import annotations

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.matrices.certified_snf._models import (
    CertifiedSmithNormalFormRequest,
    CertifiedSmithNormalFormResult,
)
from jacobian.math.matrices.certified_snf.operations import (
    smith_normal_form_certificate,
)


def _certified_smith(
    request: CertifiedSmithNormalFormRequest,
) -> CertifiedSmithNormalFormResult:
    return CertifiedSmithNormalFormResult._from_kernel(
        certificate=smith_normal_form_certificate(request.matrix)
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="matrix.normal_form.smith.certified.compute",
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

__all__ = ["TOOLS"]
