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
from jacobian.math.number_theory.number_fields._field_embedding import (
    SimpleNumberFieldEmbeddingRequest,
    SimpleNumberFieldEmbeddingResult,
    apply_simple_number_field_embedding,
)
from jacobian.math.number_theory.number_fields._models import (
    MAX_CLASS_GROUP_DEGREE,
    NumberFieldClassGroupRequest,
    NumberFieldClassGroupResult,
    NumberFieldDiscriminantResult,
    NumberFieldEmbeddingsRequest,
    NumberFieldRealEmbeddingOrderRequest,
    NumberFieldRequest,
    NumberFieldRingOfIntegersRequest,
    NumberFieldUnitGroupRequest,
    NumberFieldUnitGroupResult,
)
from jacobian.math.number_theory.number_fields._real_embedding_order import (
    NumberFieldRealEmbeddingOrderError,
)
from jacobian.math.number_theory.number_fields._relative_trace_norm import (
    MAX_RELATIVE_TRACE_NORM_DEGREE,
    NumberFieldRelativeTraceNormRequest,
    NumberFieldRelativeTraceNormResult,
    relative_trace_norm,
)
from jacobian.math.number_theory.number_fields._ring_of_integers import (
    NumberFieldRingOfIntegersResult,
)
from jacobian.math.number_theory.number_fields._ring_of_integers_process import (
    compute_nf_ring_of_integers,
)
from jacobian.math.number_theory.number_fields.operations import (
    NumberFieldEmbeddingAdmissionError,
    binary_power_sum_gap_profile,
    class_group,
    compare_real_embedding_elements,
    embeddings,
    unit_group,
)
from jacobian.math.number_theory.number_fields.values import (
    MAX_NUMBER_FIELD_DISCRIMINANT_DIGITS,
    MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
    NumberFieldEmbeddingProfile,
    SimpleNumberFieldRealEmbeddingOrder,
)


def compute_ring_of_integers(
    request: NumberFieldRingOfIntegersRequest,
) -> NumberFieldRingOfIntegersResult:
    return compute_nf_ring_of_integers(request)


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


def _compute_class_group(
    request: NumberFieldClassGroupRequest,
) -> NumberFieldClassGroupResult:
    return class_group(request.field)


def _compute_unit_group(
    request: NumberFieldUnitGroupRequest,
) -> NumberFieldUnitGroupResult:
    return unit_group(request.field)


def _compute_relative_trace_norm(
    request: NumberFieldRelativeTraceNormRequest,
) -> NumberFieldRelativeTraceNormResult:
    return relative_trace_norm(request.field, request.element)


def _apply_field_embedding(
    request: SimpleNumberFieldEmbeddingRequest,
) -> SimpleNumberFieldEmbeddingResult:
    return apply_simple_number_field_embedding(request)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="number_field.ring_of_integers.compute",
        title="Compute the ring-of-integers basis of a number field",
        description=(
            "Return one deterministic integral-basis witness for the ring of "
            "integers of a presented simple number field QQ(alpha), with every "
            "basis member represented as a canonical field element on the "
            "presentation's own ascending power basis, together with the field "
            "discriminant. The defining polynomial must be irreducible over QQ "
            "and have degree at most 31, and its monicized discriminant must "
            f"not exceed {MAX_NUMBER_FIELD_DISCRIMINANT_DIGITS} digits. "
            "Admission additionally proves the"
            "defining-polynomial discriminant factors within a bounded trial "
            "envelope (remaining cofactor at most 4096 digits and one, prime, "
            "or a prime power) and that the proved Round 2 order-enlargement "
            "work fits a fixed step envelope, so the exact backend completes "
            "instead of timing out. A field whose integral-basis coordinates "
            f"exceed the {MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS}-digit element "
            "envelope is refused with a typed coordinate-bound rejection."
        ),
        request_type=NumberFieldRingOfIntegersRequest,
        result_type=NumberFieldRingOfIntegersResult,
        run=compute_ring_of_integers,
        tags=("number-field", "ring-of-integers", "exact"),
        examples=(
            OperationExample(
                name="golden_field",
                description=(
                    "QQ(sqrt(5)) is irreducible and within the degree bound; "
                    "its ring of integers has basis 1 and (1+sqrt(5))/2."
                ),
                input={
                    "field": {
                        "domain": "QQ",
                        "coefficients_descending": ["1", "0", "-5"],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="number_field.discriminant.compute",
        title="Compute the discriminant of a number field",
        description=(
            "Compute the field discriminant of one canonical "
            "SimpleNumberFieldPresentation in an isolated SymPy worker. The "
            "defining polynomial must be irreducible and within the degree "
            f"bound, and its monicized discriminant must not exceed "
            f"{MAX_NUMBER_FIELD_DISCRIMINANT_DIGITS} digits and must factor "
            "with a cofactor of at most 4096 digits that is one, prime, or a "
            "prime power, with proved Round 2 enlargement work inside a fixed "
            "step envelope. Worker non-completion raises an execution error "
            "without a discriminant claim."
        ),
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
    MathTool(
        operation_id="number_field.class_group.compute",
        title="Compute the exact class group of a number field",
        description=(
            "Return the class number, elementary divisors, field "
            "discriminant, signature, and Hermite-normal-form ideal-class "
            "representatives of one presented simple number field, computed "
            "by the maintained PARI backend inside a request-owned killable "
            f"worker. The degree is at most {MAX_CLASS_GROUP_DEGREE}, the "
            "polynomial must be irreducible, and the PARI runtime must be "
            "installed; a missing backend is a typed resource refusal, never "
            "a mathematical negative."
        ),
        request_type=NumberFieldClassGroupRequest,
        result_type=NumberFieldClassGroupResult,
        run=_compute_class_group,
        tags=("number-field", "class-group", "ideal", "exact"),
        examples=(
            OperationExample(
                name="imaginary_quadratic_class_two",
                description=(
                    "QQ(sqrt(-5)) has class number 2, cyclic structure C2, and "
                    "one nontrivial ideal class."
                ),
                input={
                    "field": {
                        "domain": "QQ",
                        "coefficients_descending": ["1", "0", "5"],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="number_field.unit_group.compute",
        title="Compute the exact unit group of a number field",
        description=(
            "Return the unit rank r1 + r2 - 1, the torsion order and its root "
            "of unity, and the exact fundamental units of one presented "
            "simple number field, computed by the maintained PARI backend "
            "inside a request-owned killable worker. The degree is at most "
            f"{MAX_CLASS_GROUP_DEGREE}, the polynomial must be irreducible, "
            "and the PARI runtime must be installed."
        ),
        request_type=NumberFieldUnitGroupRequest,
        result_type=NumberFieldUnitGroupResult,
        run=_compute_unit_group,
        tags=("number-field", "unit-group", "fundamental-unit", "exact"),
        examples=(
            OperationExample(
                name="real_quadratic_fundamental_unit",
                description=(
                    "QQ(sqrt(2)) has rank 1 and fundamental unit 1 + sqrt(2) "
                    "up to sign."
                ),
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
        operation_id="number_field.relative_trace_norm.compute",
        title="Compute the exact relative trace and norm of a number-field element",
        description=(
            "Return the exact trace and norm of one power-basis element of a "
            "presented simple number field QQ(alpha) over QQ, read from the "
            "QQ-linear multiplication matrix (trace and determinant) and "
            "cross-checked against the complete embedding family through "
            "Newton power-sum and resultant replays, with the full "
            "multiplication matrix and characteristic polynomial retained. "
            "The defining polynomial must be irreducible of degree at most "
            f"{MAX_RELATIVE_TRACE_NORM_DEGREE}; reducible input is a domain "
            "rejection, never a zero trace or norm."
        ),
        request_type=NumberFieldRelativeTraceNormRequest,
        result_type=NumberFieldRelativeTraceNormResult,
        run=_compute_relative_trace_norm,
        tags=("number-field", "trace", "norm", "exact"),
        discovery_terms=(
            "relative trace and norm of a number-field element",
            "multiplication matrix trace and determinant",
            "embedding sum and product replay",
            "trace norm of x^3-2 element",
        ),
        examples=(
            OperationExample(
                name="cbrt2_trace_norm",
                description=(
                    "Trace 0 and norm 2 of the primitive element of QQ(cbrt(2)); "
                    "the field must be irreducible of degree at most 8."
                ),
                input={
                    "field": {
                        "domain": "QQ",
                        "coefficients_descending": ["1", "0", "0", "-2"],
                    },
                    "element": {
                        "presentation": {
                            "domain": "QQ",
                            "coefficients_descending": ["1", "0", "0", "-2"],
                        },
                        "coefficients_ascending": [
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
        operation_id="number_field.embedding.apply_exact.compute",
        title="Apply an exact simple number-field embedding",
        description=(
            "Apply a proposed QQ-embedding between simple number fields of degree at most six. "
            "The source and target presentations must be irreducible, and the supplied target-field image "
            "of the source generator must satisfy its defining polynomial exactly. Returns the reusable "
            "embedding value and the exact image of one source element. Polynomial and coordinate inputs "
            "are limited to 32 digits to bound exact coordinate growth."
        ),
        request_type=SimpleNumberFieldEmbeddingRequest,
        result_type=SimpleNumberFieldEmbeddingResult,
        run=_apply_field_embedding,
        tags=("number-field", "embedding", "exact"),
        discovery_terms=(
            "exact embedding between number fields",
            "map a number-field element by generator image",
            "field homomorphism in power-basis coordinates",
        ),
        examples=(
            OperationExample(
                name="quadratic_into_quartic",
                description=(
                    "Embed QQ(sqrt(2)) into QQ(beta), beta^4=2, by sending its generator to beta^2."
                ),
                input={
                    "source": {
                        "domain": "QQ",
                        "coefficients_descending": ["1", "0", "-2"],
                    },
                    "target": {
                        "domain": "QQ",
                        "coefficients_descending": ["1", "0", "0", "0", "-2"],
                    },
                    "generator_image": {
                        "presentation": {
                            "domain": "QQ",
                            "coefficients_descending": ["1", "0", "0", "0", "-2"],
                        },
                        "coefficients_ascending": [
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    "element": {
                        "presentation": {
                            "domain": "QQ",
                            "coefficients_descending": ["1", "0", "-2"],
                        },
                        "coefficients_ascending": [
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
