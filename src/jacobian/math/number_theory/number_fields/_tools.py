"""Number field operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.number_theory.number_fields._binary_power_sum import (
    BinaryPowerSumAdmissionError,
    BinaryPowerSumGapProfile,
    NumberFieldBinaryPowerSumGapProfileRequest,
)
from jacobian.math.number_theory.number_fields._discriminant_process import (
    compute_nf_discriminant,
)
from jacobian.math.number_theory.number_fields._models import (
    NumberFieldDiscriminantResult,
    NumberFieldEmbeddingsRequest,
    NumberFieldRealEmbeddingOrderRequest,
    NumberFieldRequest,
)
from jacobian.math.number_theory.number_fields._real_embedding_order import (
    NumberFieldRealEmbeddingOrderError,
)
from jacobian.math.number_theory.number_fields.operations import (
    NumberFieldEmbeddingAdmissionError,
    binary_power_sum_gap_profile,
    compare_real_embedding_elements,
    embeddings,
)
from jacobian.math.number_theory.number_fields.values import (
    NumberFieldEmbeddingProfile,
    SimpleNumberFieldRealEmbeddingOrder,
)


def nf_operation[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


def _compute_embeddings(
    request: NumberFieldEmbeddingsRequest,
) -> NumberFieldEmbeddingProfile:
    try:
        return embeddings(request.field)
    except NumberFieldEmbeddingAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("field",),
            code=f"number_field.embeddings.{exc.reason}",
            message=str(exc),
        ) from exc


def _compare_real_embedding_elements(
    request: NumberFieldRealEmbeddingOrderRequest,
) -> SimpleNumberFieldRealEmbeddingOrder:
    try:
        return compare_real_embedding_elements(request.left, request.right)
    except NumberFieldRealEmbeddingOrderError as exc:
        raise OperationDomainValidationError(
            location=("left", "embedding_record"),
            code=f"number_field.real_embedding_order.{exc.reason}",
            message=str(exc),
        ) from exc


def _compute_binary_power_sum_gap_profile(
    request: NumberFieldBinaryPowerSumGapProfileRequest,
) -> BinaryPowerSumGapProfile:
    try:
        return binary_power_sum_gap_profile(request.base, request.exponent_count)
    except BinaryPowerSumAdmissionError as exc:
        base_reasons = {
            "base_interval",
            "embedding_record_not_recognized",
        }
        raise OperationDomainValidationError(
            location=("base",)
            if exc.reason in base_reasons or exc.reason.startswith("embedding_")
            else ("exponent_count",),
            code=f"number_field.binary_power_sum.{exc.reason}",
            message=str(exc),
        ) from exc


TOOLS: tuple[MathTool[Any, Any], ...] = (
    nf_operation(
        "number_field.discriminant.compute",
        "Compute the discriminant of a number field",
        "Compute the field discriminant of one canonical SimpleNumberFieldPresentation in an isolated SymPy worker, or return UNKNOWN if its bounded execution cannot establish a result.",
        NumberFieldRequest,
        NumberFieldDiscriminantResult,
        compute_nf_discriminant,
        "number-field",
        "discriminant",
        "exact",
        examples=(
            example(
                "quadratic_disc",
                "Discriminant of x^2-2.",
                {
                    "field": {
                        "domain": "QQ",
                        "coefficients_descending": ["1", "0", "-2"],
                    }
                },
            ),
        ),
    ),
    nf_operation(
        "number_field.embeddings.compute",
        "Compute every exact embedding of a simple number field",
        "Return all real and complex embeddings of one bounded primitive irreducible ZZ presentation of QQ(alpha), ordered by exact roots, with certified rational isolation, signature, conjugate-pair grouping, and defining-polynomial discriminant. Degree is at most 8, coefficients have at most 256 digits, and a request-deadline-bound one-shot worker performs recognition, exact signature, conditional elimination, and one all-root isolation pass within the admitted 32,768-bit refinement, 2,097,152-bit resultant-storage, and 10,485,760-byte result envelopes.",
        NumberFieldEmbeddingsRequest,
        NumberFieldEmbeddingProfile,
        _compute_embeddings,
        "number-field",
        "embedding",
        "algebraic-number",
        "exact",
        examples=(
            example(
                "gaussian_field",
                "Both embeddings of QQ(i), grouped as one conjugate pair.",
                {
                    "field": {
                        "domain": "QQ",
                        "coefficients_descending": ["1", "0", "1"],
                    }
                },
            ),
        ),
    ),
    nf_operation(
        "number_field.real_embedding.element_order.compare",
        "Compare field elements under one selected real embedding",
        "Return the exact LT, EQ, or GT order of two reduced elements of one "
        "simple number field at one exact real embedding record. The consumer "
        "recomputes the complete bounded embedding profile and requires an "
        "exact producer-record match before canonical SymPy quotient-field "
        "arithmetic and selected-image real-root isolation. Equality is exact "
        "in QQ[x]/(f), independently of the selected real order.",
        NumberFieldRealEmbeddingOrderRequest,
        SimpleNumberFieldRealEmbeddingOrder,
        _compare_real_embedding_elements,
        "number-field",
        "embedding",
        "order",
        "exact",
        examples=(
            example(
                "rational_field",
                "Compare 3/2 and 1 at the unique embedding of QQ.",
                {
                    "left": {
                        "element": {
                            "presentation": {
                                "domain": "QQ",
                                "coefficients_descending": ["1", "-1"],
                            },
                            "coefficients_ascending": [{"num": "3", "den": "2"}],
                        },
                        "embedding_record": {
                            "kind": "REAL",
                            "embedding": {
                                "kind": "REAL",
                                "presentation": {
                                    "domain": "QQ",
                                    "coefficients_descending": ["1", "-1"],
                                },
                                "root": {
                                    "polynomial": ["1", "-1"],
                                    "real_root_index": 0,
                                },
                            },
                            "isolating_interval": {
                                "lower": {"num": "1", "den": "1"},
                                "upper": {"num": "1", "den": "1"},
                                "interval_type": "SINGLETON",
                            },
                        },
                    },
                    "right": {
                        "element": {
                            "presentation": {
                                "domain": "QQ",
                                "coefficients_descending": ["1", "-1"],
                            },
                            "coefficients_ascending": [{"num": "1", "den": "1"}],
                        },
                        "embedding_record": {
                            "kind": "REAL",
                            "embedding": {
                                "kind": "REAL",
                                "presentation": {
                                    "domain": "QQ",
                                    "coefficients_descending": ["1", "-1"],
                                },
                                "root": {
                                    "polynomial": ["1", "-1"],
                                    "real_root_index": 0,
                                },
                            },
                            "isolating_interval": {
                                "lower": {"num": "1", "den": "1"},
                                "upper": {"num": "1", "den": "1"},
                                "interval_type": "SINGLETON",
                            },
                        },
                    },
                },
            ),
        ),
    ),
    nf_operation(
        "number_field.real_embedding.binary_power_sum_gap_profile.compute",
        "Compute an exact real-embedded binary power-sum gap profile",
        "For one structurally bound field element q, selected real embedding, "
        "and exponent count m, require 1 < sigma(q) < 2 and return every "
        "distinct sum of epsilon_i*q^i in exact increasing embedded order. "
        "The result partitions all 2^m indexed bit vectors by exact quotient-"
        "field equality, retains every collision representation, reconstructs "
        "every adjacent exact gap with positive rational enclosure evidence, "
        "and reports exact multiplicity and least/largest-gap summaries. The "
        "finite exhaustive envelope admits at most 4,096 representations and "
        "preflights field growth, comparisons, and complete result bytes.",
        NumberFieldBinaryPowerSumGapProfileRequest,
        BinaryPowerSumGapProfile,
        _compute_binary_power_sum_gap_profile,
        "number-field",
        "embedding",
        "power-sum",
        "gap-profile",
        "exact",
        examples=(
            example(
                "three_halves",
                "All eight binary power sums for q=3/2 in QQ and m=3.",
                {
                    "base": {
                        "element": {
                            "presentation": {
                                "domain": "QQ",
                                "coefficients_descending": ["1", "0"],
                            },
                            "coefficients_ascending": [{"num": "3", "den": "2"}],
                        },
                        "embedding_record": {
                            "kind": "REAL",
                            "embedding": {
                                "kind": "REAL",
                                "presentation": {
                                    "domain": "QQ",
                                    "coefficients_descending": ["1", "0"],
                                },
                                "root": {
                                    "polynomial": ["1", "0"],
                                    "real_root_index": 0,
                                },
                            },
                            "isolating_interval": {
                                "lower": {"num": "0", "den": "1"},
                                "upper": {"num": "0", "den": "1"},
                                "interval_type": "SINGLETON",
                            },
                        },
                    },
                    "exponent_count": 3,
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
