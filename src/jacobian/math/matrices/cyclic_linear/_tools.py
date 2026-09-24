"""Catalog declaration for exact cyclic linear-system profiles."""

from typing import Any

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.matrices.cyclic_linear._models import (
    CyclicRationalRankKernelProfile,
    CyclicRationalRankKernelProfileRequest,
    CyclotomicElementMapRequest,
    CyclotomicFieldInclusion,
    CyclotomicFieldInclusionCompositionRequest,
    CyclotomicFieldInclusionRequest,
    RationalCyclotomicElement,
)
from jacobian.math.matrices.cyclic_linear.operations import (
    CyclicRankKernelAdmissionError,
    apply_cyclotomic_field_inclusion,
    compose_cyclotomic_field_inclusions,
    cyclic_rational_rank_kernel_profile,
    cyclotomic_field_inclusion,
)


def _compute(
    request: CyclicRationalRankKernelProfileRequest,
) -> CyclicRationalRankKernelProfile:
    try:
        return cyclic_rational_rank_kernel_profile(request.symbol)
    except CyclicRankKernelAdmissionError as error:
        raise OperationDomainValidationError(
            location=("symbol",),
            code=f"matrix.cyclic.{error.reason}",
            message=str(error),
        ) from error


def _inclusion(request: CyclotomicFieldInclusionRequest) -> CyclotomicFieldInclusion:
    try:
        return cyclotomic_field_inclusion(request)
    except CyclicRankKernelAdmissionError as error:
        raise OperationDomainValidationError(
            location=("source",),
            code=f"matrix.cyclic.{error.reason}",
            message=str(error),
        ) from error


def _compose(
    request: CyclotomicFieldInclusionCompositionRequest,
) -> CyclotomicFieldInclusion:
    try:
        return compose_cyclotomic_field_inclusions(request)
    except CyclicRankKernelAdmissionError as error:
        raise OperationDomainValidationError(
            location=("first",),
            code=f"matrix.cyclic.{error.reason}",
            message=str(error),
        ) from error


def _map_element(request: CyclotomicElementMapRequest) -> RationalCyclotomicElement:
    try:
        return apply_cyclotomic_field_inclusion(request)
    except CyclicRankKernelAdmissionError as error:
        raise OperationDomainValidationError(
            location=("inclusion",),
            code=f"matrix.cyclic.{error.reason}",
            message=str(error),
        ) from error


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="matrix.cyclic.cyclotomic_inclusion.compute",
        title="Compute the standard cyclotomic field inclusion",
        description=(
            "Return the standard exact inclusion QQ(zeta_n) -> QQ(zeta_m) for "
            "n dividing m, with the reduced target power-basis coordinates of "
            "zeta_n mapped to zeta_m^(m/n). This operation does not represent "
            "arbitrary embeddings. Both field orders are at most 128."
        ),
        request_type=CyclotomicFieldInclusionRequest,
        result_type=CyclotomicFieldInclusion,
        run=_inclusion,
        tags=("number-theory", "cyclotomic", "field-inclusion", "exact"),
        discovery_terms=("cyclotomic field map", "root of unity inclusion"),
        examples=(
            OperationExample(
                name="third_roots_in_sixth_roots",
                description="The standard inclusion sends zeta_3 to zeta_6 squared = zeta_6 - 1.",
                input={"source": {"order": 3}, "target": {"order": 6}},
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.cyclic.cyclotomic_inclusion.compose",
        title="Compose standard cyclotomic field inclusions",
        description=(
            "Compose standard inclusions QQ(zeta_n) -> QQ(zeta_m) -> QQ(zeta_r) "
            "when the intermediate parent matches and n divides m divides r."
        ),
        request_type=CyclotomicFieldInclusionCompositionRequest,
        result_type=CyclotomicFieldInclusion,
        run=_compose,
        tags=("number-theory", "cyclotomic", "composition", "exact"),
        discovery_terms=("compose cyclotomic field maps",),
        examples=(
            OperationExample(
                name="compose_orders_three_six_twelve",
                description="Compose the standard inclusions through the order-six field.",
                input={
                    "first": {
                        "source": {"order": 3},
                        "target": {"order": 6},
                        "generator_image": [
                            {"num": "-1", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                    "second": {
                        "source": {"order": 6},
                        "target": {"order": 12},
                        "generator_image": [
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.cyclic.cyclotomic_element.map",
        title="Map an exact element through a standard cyclotomic inclusion",
        description=(
            "Apply a typed standard QQ(zeta_n) -> QQ(zeta_m) inclusion to an "
            "exact power-basis element and return reduced target coordinates. "
            "The element, map, degree, work, and exact output height are bounded."
        ),
        request_type=CyclotomicElementMapRequest,
        result_type=RationalCyclotomicElement,
        run=_map_element,
        tags=("number-theory", "cyclotomic", "field-map", "exact"),
        discovery_terms=("apply cyclotomic field map", "transport algebraic value"),
        examples=(
            OperationExample(
                name="map_zeta_three_to_order_six",
                description="Map zeta_3 in QQ(zeta_3) to zeta_6 - 1 in QQ(zeta_6).",
                input={
                    "inclusion": {
                        "source": {"order": 3},
                        "target": {"order": 6},
                        "generator_image": [
                            {"num": "-1", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                    "element": {
                        "field": {"order": 3},
                        "coefficients_ascending": [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
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
            OperationExample(
                name="first_difference_on_six_cycle",
                description="The scalar symbol x-1 on C_6 drops rank only on the trivial component.",
                input={
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


# Preserve the pre-existing cyclic-profile manifest position for native callers.
TOOLS = (TOOLS[-1], *TOOLS[:-1])

__all__ = ["TOOLS"]
