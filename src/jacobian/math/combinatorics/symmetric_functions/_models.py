"""Typed wire contracts for symmetric function operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, WithJsonSchema, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math._labels import MAX_OPAQUE_LABEL_LENGTH, OpaqueLabel
from jacobian.math.combinatorics.symmetric_functions.values import (
    MAX_PARTITION_SIZE,
    IntegerPartition,
    TableauCandidate,
)

_MAX_POINT_COORDINATE_DIGITS = 6
_MAX_POINT_COORDINATE_ABS = 10**_MAX_POINT_COORDINATE_DIGITS - 1
_MAX_SCHUR_RESULT_DIGITS = 4000
_MAX_SCHUR_PARTITION_LENGTH = 50
_MAX_SCHUR_VARIABLE_NAME_LENGTH = MAX_OPAQUE_LABEL_LENGTH

# LR tableau enumeration is exponential in the skew-cell count. The operation
# bound is intentionally separate from the 500-cell partition carrier bound.
MAX_LR_SKEW_CELLS = 8
MAX_LR_SEARCH_STATES = 100_000
MAX_LR_TABLEAU_OUTPUT_BYTES = 8_000_000
MAX_LR_TABLEAUX = 100_000
MAX_SCHUR_PRODUCT_WORK = 1_000_000
MAX_SCHUR_PRODUCT_TERMS = 22  # p(8)


def _lr_prefix_state_bound(content: IntegerPartition) -> int:
    """Count all distinct multiset-word prefixes before tableau pruning."""
    state_count = 0

    def visit(index: int, chosen: tuple[int, ...]) -> None:
        nonlocal state_count
        if index == len(content.parts):
            length = sum(chosen)
            ways = 1
            for factor in range(2, length + 1):
                ways *= factor
            for count in chosen:
                for factor in range(2, count + 1):
                    ways //= factor
            state_count += ways
            return
        for count in range(content.parts[index] + 1):
            visit(index + 1, (*chosen, count))

    visit(0, ())
    return state_count


def _lr_complete_word_bound(content: IntegerPartition) -> int:
    """Count complete words with the submitted multiplicities."""
    size = sum(content.parts)
    words = 1
    for factor in range(2, size + 1):
        words *= factor
    for multiplicity in content.parts:
        for factor in range(2, multiplicity + 1):
            words //= factor
    return words


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by symmetric-function contracts."""

    return PydanticCustomError(f"symmetric_function.{reason}", message)


def _lr_inner_content_orientation(
    left: IntegerPartition, right: IntegerPartition
) -> tuple[IntegerPartition, IntegerPartition]:
    """Orient a commutative LR lower pair by the cheaper content-prefix bound.

    ``c^outer_{inner, content}`` is symmetric in its lower pair, so the
    Schur-product kernel may swap the operands freely. Admission and the
    kernel call this one owner helper so both agree on the same orientation.
    """
    if _lr_prefix_state_bound(right) <= _lr_prefix_state_bound(left):
        return left, right
    return right, left


PointCoordinate = Annotated[
    StrictInt,
    Field(
        ge=-_MAX_POINT_COORDINATE_ABS,
        le=_MAX_POINT_COORDINATE_ABS,
        description=(
            "Canonical integer with at most "
            f"{_MAX_POINT_COORDINATE_DIGITS} decimal digits."
        ),
    ),
]
"""One bounded evaluation coordinate: ``abs(value) <= 10**6 - 1``."""


SchurVariableName = OpaqueLabel
"""One canonical bounded variable label retained in a Schur context."""


def _schur_partition_schema() -> JsonSchemaValue:
    """Project the Jacobi-Trudi part bound onto the shared partition schema."""

    schema = IntegerPartition.model_json_schema()
    schema["properties"]["parts"].update(
        maxItems=_MAX_SCHUR_PARTITION_LENGTH,
        description=(
            "Positive weakly-decreasing parts with a total size (sum) of at "
            f"most {MAX_PARTITION_SIZE}; at most {_MAX_SCHUR_PARTITION_LENGTH} "
            "parts for this operation."
        ),
    )
    return schema


class PartitionRequest(StrictModel):
    partition: IntegerPartition


class PartitionConjugateResult(StrictModel):
    conjugate: IntegerPartition


class SchurExpansionRequest(StrictModel):
    """Evaluate one Schur function at a bounded integer point.

    Preconditions published through this schema: ``variables`` and ``point``
    must have equal lengths in ``[1, 20]``, variable names must be distinct,
    each coordinate satisfies ``abs(coordinate) <= 999999``, and the partition
    total size is capped at 500.
    """

    partition: Annotated[
        IntegerPartition,
        WithJsonSchema(_schur_partition_schema()),
    ] = Field(
        description=(
            "A canonical partition of total size at most "
            f"{MAX_PARTITION_SIZE} with at most {_MAX_SCHUR_PARTITION_LENGTH} "
            "parts for the admitted Jacobi-Trudi determinant."
        )
    )
    variables: tuple[SchurVariableName, ...] = Field(
        min_length=1,
        max_length=20,
        description=(
            "Distinct variable names; the length must equal the length of "
            "point (between 1 and 20), and each name must contain at most "
            f"{_MAX_SCHUR_VARIABLE_NAME_LENGTH} characters."
        ),
        json_schema_extra={"uniqueItems": True},
    )
    point: tuple[PointCoordinate, ...] = Field(
        min_length=1,
        max_length=20,
        description=(
            "Integer evaluation coordinates, one per variable, each with at "
            f"most {_MAX_POINT_COORDINATE_DIGITS} decimal digits; the length "
            "must equal the length of variables."
        ),
    )

    @model_validator(mode="after")
    def require_matching_dimensions(self) -> Self:
        if len(self.partition.parts) > _MAX_SCHUR_PARTITION_LENGTH:
            raise _validation_error(
                "schur_partition_length_exceeded",
                "Schur evaluation partition length must not exceed "
                f"{_MAX_SCHUR_PARTITION_LENGTH}",
            )
        if len(self.variables) != len(self.point):
            raise _validation_error(
                "schur_dimensions_mismatch",
                "variables and point must have the same length",
            )
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "schur_variables_not_distinct",
                "variables must be distinct (duplicate axis)",
            )
        return self


class SchurExpansionResult(StrictModel):
    partition: IntegerPartition
    variables: tuple[SchurVariableName, ...] = Field(
        min_length=1,
        max_length=20,
        description=(
            "Distinct variable names, each containing at most "
            f"{_MAX_SCHUR_VARIABLE_NAME_LENGTH} characters."
        ),
        json_schema_extra={"uniqueItems": True},
    )
    point: tuple[PointCoordinate, ...] = Field(min_length=1, max_length=20)
    value: ExactInteger

    @model_validator(mode="after")
    def require_bounded_result(self) -> Self:
        if len(self.variables) != len(self.point):
            raise _validation_error(
                "schur_dimensions_mismatch",
                "variables and point must have the same length",
            )
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "schur_variables_not_distinct",
                "variables must be distinct (duplicate axis)",
            )
        if len(format_canonical_integer(abs(self.value))) > _MAX_SCHUR_RESULT_DIGITS:
            raise _validation_error(
                "schur_value_digits_exceeded",
                "Schur value exceeds the output digit bound",
            )
        return self


class LittlewoodRichardsonCoefficientRequest(StrictModel):
    """Compute ``c^outer_{inner, content}`` using LR tableaux.

    The reading word scans each skew row right-to-left, from top to bottom;
    every prefix must contain at least as many ``i`` as ``i+1`` for all i.
    Complete search admits at most {MAX_LR_SKEW_CELLS} skew cells and
    {MAX_LR_SEARCH_STATES} distinct content-word prefixes. Admission bounds
    the skew diagram ``|outer| - |inner|`` and the content, not the ambient
    diagrams, whose row-scan work is bounded by the shared 500-cell partition
    carrier.
    """

    outer: IntegerPartition
    inner: IntegerPartition
    content: IntegerPartition


class LittlewoodRichardsonCoefficientResult(StrictModel):
    """One exact LR coefficient, bound to its three partition arguments."""

    outer: IntegerPartition
    inner: IntegerPartition
    content: IntegerPartition
    coefficient: StrictInt = Field(ge=0)


class LittlewoodRichardsonTableauxRequest(StrictModel):
    """Enumerate all LR tableaux for one skew shape and content."""

    outer: IntegerPartition
    inner: IntegerPartition
    content: IntegerPartition


class LittlewoodRichardsonTableauxResult(StrictModel):
    """Complete LR family, bound to its skew shape and content."""

    outer: IntegerPartition
    inner: IntegerPartition
    content: IntegerPartition
    tableaux: tuple[TableauCandidate, ...] = Field(max_length=MAX_LR_TABLEAUX)

    @classmethod
    def _from_kernel(
        cls,
        request: LittlewoodRichardsonTableauxRequest,
        tableaux: tuple[TableauCandidate, ...],
    ) -> Self:
        """Construct the complete family after bounded exhaustive search."""
        return cls.model_construct(
            outer=request.outer,
            inner=request.inner,
            content=request.content,
            tableaux=tableaux,
        )


class SchurProductRequest(StrictModel):
    """Compute a complete Schur product inside the bounded LR envelope."""

    left: IntegerPartition
    right: IntegerPartition

    @model_validator(mode="after")
    def require_bounded_complete_product(self) -> Self:
        total = sum(self.left.parts) + sum(self.right.parts)
        if total > MAX_LR_SKEW_CELLS:
            raise _validation_error(
                "schur_product_size_exceeded",
                f"Schur product total degree must not exceed {MAX_LR_SKEW_CELLS}",
            )
        # Every possible outer shape of this degree is sent through the same
        # admitted LR tableau kernel. The lower pair commutes, so the kernel
        # searches with whichever operand has the cheaper content-prefix
        # bound. Charge that orientation once per candidate before generating
        # candidates or tableaux, and bound the expansion by term
        # cardinality, not transport bytes.
        candidates = _partition_count(total)
        _inner, content = _lr_inner_content_orientation(self.left, self.right)
        prefix_bound = _lr_prefix_state_bound(content)
        if prefix_bound > MAX_LR_SEARCH_STATES:
            raise _validation_error(
                "schur_product_search_states_exceeded",
                f"LR content-prefix bound must not exceed {MAX_LR_SEARCH_STATES}",
            )
        work = candidates * prefix_bound
        if work > MAX_SCHUR_PRODUCT_WORK:
            raise _validation_error(
                "schur_product_work_exceeded",
                f"complete Schur product work bound must not exceed {MAX_SCHUR_PRODUCT_WORK}",
            )
        if candidates > MAX_SCHUR_PRODUCT_TERMS:
            raise _validation_error(
                "schur_product_terms_exceeded",
                "complete Schur product admits at most "
                f"{MAX_SCHUR_PRODUCT_TERMS} candidate terms",
            )
        return self


class SchurProductTerm(StrictModel):
    partition: IntegerPartition
    coefficient: StrictInt = Field(gt=0)


class SchurProductResult(StrictModel):
    left: IntegerPartition
    right: IntegerPartition
    terms: tuple[SchurProductTerm, ...] = Field(max_length=MAX_SCHUR_PRODUCT_TERMS)

    @model_validator(mode="after")
    def require_canonical_terms(self) -> Self:
        sizes = [sum(term.partition.parts) for term in self.terms]
        if any(size != sum(self.left.parts) + sum(self.right.parts) for size in sizes):
            raise _validation_error(
                "schur_product_degree_mismatch", "term degree must equal product degree"
            )
        keys = [term.partition.parts for term in self.terms]
        if keys != sorted(set(keys), key=lambda parts: tuple(-part for part in parts)):
            # Canonical ordering is reverse lexicographic, largest first.
            raise _validation_error(
                "schur_product_terms_not_canonical",
                "Schur product terms must be unique and canonically ordered",
            )
        return self


def _partition_count(total: int) -> int:
    counts = [0] * (total + 1)
    counts[0] = 1
    for part in range(1, total + 1):
        for size in range(part, total + 1):
            counts[size] += counts[size - part]
    return counts[total]


__all__ = [
    "MAX_LR_SEARCH_STATES",
    "MAX_LR_SKEW_CELLS",
    "MAX_LR_TABLEAUX",
    "MAX_LR_TABLEAU_OUTPUT_BYTES",
    "MAX_SCHUR_PRODUCT_TERMS",
    "MAX_SCHUR_PRODUCT_WORK",
    "IntegerPartition",
    "LittlewoodRichardsonCoefficientRequest",
    "LittlewoodRichardsonCoefficientResult",
    "LittlewoodRichardsonTableauxRequest",
    "LittlewoodRichardsonTableauxResult",
    "PartitionConjugateResult",
    "PartitionRequest",
    "SchurExpansionRequest",
    "SchurExpansionResult",
    "SchurProductRequest",
    "SchurProductResult",
    "SchurProductTerm",
    "SchurVariableName",
]
