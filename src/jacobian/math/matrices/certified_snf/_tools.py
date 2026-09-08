"""Transformation-certified Smith normal forms."""

from __future__ import annotations

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.matrices.certified_snf._models import (
    CertifiedSmithNormalFormRequest,
    CertifiedSmithNormalFormResult,
    PolynomialSmithRequest,
)
from jacobian.math.matrices.certified_snf.operations import (
    smith_normal_form_certificate,
)
from jacobian.math.matrices.certified_snf.polynomial import (
    PolynomialSmithDecomposition,
    polynomial_smith_decomposition,
)


def _polynomial_smith(request: PolynomialSmithRequest) -> PolynomialSmithDecomposition:
    return polynomial_smith_decomposition(request.matrix)


def _certified_smith(
    request: CertifiedSmithNormalFormRequest,
) -> CertifiedSmithNormalFormResult:
    return CertifiedSmithNormalFormResult._from_kernel(
        certificate=smith_normal_form_certificate(request.matrix)
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="matrix.normal_form.smith.polynomial.compute",
        title="Compute Smith decomposition over QQ[t] with unimodular maps",
        description=(
            "Return a monic divisibility diagonal D and polynomial unimodular "
            "U,V with D=UAV for a rectangular matrix over QQ[t]. Retain the "
            "polynomial ring, empty axes and nonzero rational units. Full "
            "transform production is admitted by degree, rational-height, "
            "Euclidean-work and exact-output bounds."
        ),
        request_type=PolynomialSmithRequest,
        result_type=PolynomialSmithDecomposition,
        run=_polynomial_smith,
        discovery_terms=(
            "polynomial presentation module decomposition",
            "polynomial cokernel torsion invariant factors",
            "polynomial unimodular source and target coordinates",
            "rectangular polynomial Smith normal form",
        ),
        tags=(
            "matrix",
            "polynomial",
            "smith-normal-form",
            "unimodular-transformation",
            "exact",
            "bounded",
        ),
        examples=(
            OperationExample(
                name="polynomial_torsion",
                description="The map t: QQ[t] to QQ[t] has cokernel QQ[t]/(t).",
                input={
                    "matrix": {
                        "variables": ["t"],
                        "row_count": 1,
                        "column_count": 1,
                        "entries": [
                            [
                                {
                                    "variables": ["t"],
                                    "polynomial": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "1", "den": "1"},
                                                "exponents": [1],
                                            },
                                        ]
                                    },
                                }
                            ]
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.normal_form.smith.certified.compute",
        title="Compute a transformation-certified Smith normal form",
        description=(
            "Compute the canonical Smith diagonal D and explicit unimodular "
            "matrices U and V satisfying D = U A V for one integer matrix of "
            "at most 16 by 16."
        ),
        discovery_terms=(
            "integer column lattice membership",
            "additive order of a vector modulo an integer lattice",
            "cokernel torsion class free coordinates separating obstruction",
            "least positive integer multiplier and integer multipliers witness",
            "unimodular row and column transformations",
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
            OperationExample(
                name="certified_smith_two_by_two",
                description="Compute D, U, and V for a two-by-two integer matrix.",
                input={
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
