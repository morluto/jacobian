"""Number field operation declarations."""

from typing import Any

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
    MathTool(
        operation_id="number_field.discriminant.compute",
        title="Compute the discriminant of a number field",
        description="Compute the field discriminant of one canonical SimpleNumberFieldPresentation in an isolated SymPy worker. Worker non-completion raises an execution error without a discriminant claim.",
        request_type=NumberFieldRequest,
        result_type=NumberFieldDiscriminantResult,
        run=compute_nf_discriminant,
        tags=("number-field", "discriminant", "exact"),
        examples=(
            OperationExample(
                name="quadratic_disc",
                description="Discriminant of x^2-2.",
                input={
                    "field": {
                        "domain": "QQ",
                        "coefficients_descending": ["1", "0", "-2"],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="number_field.embeddings.compute",
        title="Compute every exact embedding of a simple number field",
        description="Return all real and complex embeddings of one bounded primitive irreducible ZZ presentation of QQ(alpha), ordered by exact roots, with certified rational isolation, signature, conjugate-pair grouping, and defining-polynomial discriminant. Degree is at most 8, coefficients have at most 256 digits, and a request-deadline-bound one-shot worker performs recognition, exact signature, conditional elimination, and one all-root isolation pass within the admitted 32,768-bit refinement, 2,097,152-bit resultant-storage, and 10,485,760-byte result envelopes.",
        request_type=NumberFieldEmbeddingsRequest,
        result_type=NumberFieldEmbeddingProfile,
        run=_compute_embeddings,
        tags=("number-field", "embedding", "algebraic-number", "exact"),
        examples=(
            OperationExample(
                name="gaussian_field",
                description="Both embeddings of QQ(i), grouped as one conjugate pair.",
                input={
                    "field": {
                        "domain": "QQ",
                        "coefficients_descending": ["1", "0", "1"],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="number_field.real_embedding.element_order.compare",
        title="Compare field elements under one selected real embedding",
        description="Return the exact LT, EQ, or GT order of two reduced elements of one "
        "simple number field at one exact real embedding record. The consumer "
        "recomputes the complete bounded embedding profile and requires an "
        "exact producer-record match before canonical SymPy quotient-field "
        "arithmetic and selected-image real-root isolation. Equality is exact "
        "in QQ[x]/(f), independently of the selected real order.",
        request_type=NumberFieldRealEmbeddingOrderRequest,
        result_type=SimpleNumberFieldRealEmbeddingOrder,
        run=_compare_real_embedding_elements,
        tags=("number-field", "embedding", "order", "exact"),
        examples=(
            OperationExample(
                name="rational_field",
                description="Compare 3/2 and 1 at the unique embedding of QQ.",
                input={
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
    MathTool(
        operation_id="number_field.real_embedding.binary_power_sum_gap_profile.compute",
        title="Compute an exact real-embedded binary power-sum gap profile",
        description="For one structurally bound field element q, selected real embedding, "
        "and exponent count m, require 1 < sigma(q) < 2 and return every "
        "distinct sum of epsilon_i*q^i in exact increasing embedded order. "
        "The result partitions all 2^m indexed bit vectors by exact quotient-"
        "field equality, retains every collision representation, reconstructs "
        "every adjacent exact gap with positive rational enclosure evidence, "
        "and reports exact multiplicity and least/largest-gap summaries. The "
        "finite exhaustive envelope admits at most 4,096 representations and "
        "preflights field growth, comparisons, and complete result bytes.",
        request_type=NumberFieldBinaryPowerSumGapProfileRequest,
        result_type=BinaryPowerSumGapProfile,
        run=_compute_binary_power_sum_gap_profile,
        tags=("number-field", "embedding", "power-sum", "gap-profile", "exact"),
        examples=(
            OperationExample(
                name="three_halves",
                description="All eight binary power sums for q=3/2 in QQ and m=3.",
                input={
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
