"""Catalog declaration for exact cyclic linear-system profiles."""

from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.matrices.cyclic_linear._models import (
    CyclicRationalRankKernelProfile,
    CyclicRationalRankKernelProfileRequest,
)
from jacobian.math.matrices.cyclic_linear.operations import (
    CyclicRankKernelAdmissionError,
    cyclic_rational_rank_kernel_profile,
)


def _compute(
    request: CyclicRationalRankKernelProfileRequest,
) -> CyclicRationalRankKernelProfile:
    try:
        return cyclic_rational_rank_kernel_profile(
            request.symbol, enforce_transport_limit=True
        )
    except CyclicRankKernelAdmissionError as error:
        raise OperationDomainValidationError(
            location=("symbol",),
            code=f"matrix.cyclic.{error.reason}",
            message=str(error),
        ) from error


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="matrix.cyclic.rational_rank_kernel_profile.compute",
        title="Compute an exact rational cyclic rank and kernel profile",
        description=(
            "Decompose a bounded rational block-circulant linear map into every "
            "rational Galois component QQ[x]/(Phi_d), returning the exact "
            "component matrices, ranks, nullities, nonzero-minor witnesses, "
            "kernel bases, CRT idempotents, and reconstructed global rational "
            "kernel. The period is at most 128, both expanded axes are at most "
            "128, and coefficient numerators and denominators have at most 64 "
            "decimal digits each (excluding an optional numerator sign). Exact "
            "elimination height, scalar-bit work, reconstruction height, and "
            "result output are admitted from the source before execution."
        ),
        request_type=CyclicRationalRankKernelProfileRequest,
        result_type=CyclicRationalRankKernelProfile,
        run=_compute,
        tags=(
            "matrix",
            "cyclic",
            "block-circulant",
            "cyclotomic",
            "rank",
            "kernel",
            "exact",
        ),
        discovery_terms=("Fourier mode", "circulant matrix", "root of unity"),
        examples=(
            example(
                "first_difference_on_six_cycle",
                "The scalar symbol x-1 on C_6 drops rank only on the trivial component.",
                {
                    "symbol": {
                        "period": 6,
                        "target_block_dimension": 1,
                        "source_block_dimension": 1,
                        "entries": [
                            {
                                "target_coordinate": 0,
                                "source_coordinate": 0,
                                "shift": 0,
                                "coefficient": {"num": "-1", "den": "1"},
                            },
                            {
                                "target_coordinate": 0,
                                "source_coordinate": 0,
                                "shift": 1,
                                "coefficient": {"num": "1", "den": "1"},
                            },
                        ],
                    }
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
