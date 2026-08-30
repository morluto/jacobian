"""Provider-independent exact matrix values."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from itertools import pairwise
from typing import Annotated, Any, Literal, Self

from pydantic import Field, GetJsonSchemaHandler, field_validator, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalInteger,
    CanonicalRational,
)
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory.algebraic_numbers.quadratic import RealQuadraticValue
from jacobian.math.number_theory.number_fields.values import (
    RealNumberFieldEmbedding,
    SimpleNumberFieldElement,
)

MAX_MATRIX_DIMENSION = 32
MAX_EXACT_LINEAR_MATRIX_AXIS = 64
# The canonical dense rational matrix retains exact sources for analysis
# results whose operations admit them by their own work and result budgets,
# so its structural order is not tied to the shared computation dimension.
# Determinant and characteristic-polynomial operations admit square matrices
# through order 128. Keep that complete public domain representable by the one
# canonical QQ matrix value while narrower operations enforce their own
# request envelopes.
MAX_RATIONAL_MATRIX_ORDER = 128
# Exact inverse admits square integer sources through order 128. Keep that
# complete public domain representable by the one canonical ZZ matrix value
# while lattice reduction and the other integer operations enforce their own
# request envelopes.
MAX_INTEGER_MATRIX_ORDER = 128
MAX_SPARSE_RATIONAL_MATRIX_AXIS = 8_192
MAX_SPARSE_RATIONAL_MATRIX_NONZEROS = 32_768
MAX_MATRIX_SCALAR_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS


def require_matrix_scalar_digits(
    entries: tuple[tuple[str | CanonicalRational, ...], ...],
    *,
    maximum: int,
    label: str,
) -> None:
    """Apply an operation-owned scalar budget to an authoritative matrix value."""

    for row in entries:
        for value in row:
            components = (value,) if isinstance(value, str) else (value.num, value.den)
            if any(len(component.lstrip("-")) > maximum for component in components):
                raise _validation_error(
                    "budget_exceeded",
                    f"{label} scalars are limited to {maximum} decimal digits",
                )


def _prepare_raw_matrix_scalar(
    scalar: object, *, label: str, scalar_domain: Literal["QQ", "ZZ"]
) -> object:
    """Reject nested scalar containers and copy one valid-shaped QQ mapping."""

    if isinstance(scalar, (list, tuple)) or (
        scalar_domain == "ZZ" and isinstance(scalar, dict)
    ):
        raise _validation_error(
            "shape_mismatch", f"{label} entries must be scalar values"
        )
    components: tuple[object, ...]
    if isinstance(scalar, dict):
        if set(scalar).difference({"num", "den"}):
            raise _validation_error(
                "shape_mismatch",
                f"{label} rational scalar contains unknown fields",
            )
        components = (scalar.get("num"), scalar.get("den"))
        if any(isinstance(component, (dict, list, tuple)) for component in components):
            raise _validation_error(
                "shape_mismatch",
                f"{label} rational scalar components must be scalar values",
            )
        normalized: object = dict(scalar)
    else:
        components = (scalar,)
        normalized = scalar
    if any(
        isinstance(component, str)
        and len(component.lstrip("-")) > MAX_MATRIX_SCALAR_DIGITS
        for component in components
    ):
        raise _validation_error(
            "budget_exceeded",
            f"{label} scalars are limited to {MAX_MATRIX_SCALAR_DIGITS} decimal digits",
        )
    return normalized


def _prepare_raw_matrix_envelope(
    data: object,
    *,
    maximum_axis: int,
    label: str,
    scalar_domain: Literal["QQ", "ZZ"],
) -> object:
    """Bound raw matrix depth and shallowly normalize its two array axes."""

    if not isinstance(data, dict):
        return data
    if set(data).difference({"domain", "entries"}):
        raise _validation_error("shape_mismatch", f"{label} contains unknown fields")
    normalized = dict(data)
    if "entries" not in data:
        return normalized
    entries = data.get("entries")
    if not isinstance(entries, (list, tuple)):
        raise _validation_error(
            "shape_mismatch", f"{label} entries must be an array of rows"
        )
    if len(entries) > maximum_axis:
        raise _validation_error(
            "budget_exceeded", f"{label} has at most {maximum_axis} rows"
        )
    normalized_rows: list[tuple[object, ...]] = []
    for row in entries:
        if not isinstance(row, (list, tuple)):
            raise _validation_error("shape_mismatch", f"{label} rows must be arrays")
        if len(row) > maximum_axis:
            raise _validation_error(
                "budget_exceeded", f"{label} has at most {maximum_axis} columns"
            )
        normalized_rows.append(
            tuple(
                _prepare_raw_matrix_scalar(
                    scalar, label=label, scalar_domain=scalar_domain
                )
                for scalar in row
            )
        )
    normalized["entries"] = tuple(normalized_rows)
    return normalized


class RationalMatrix(StrictModel):
    """One nonempty rectangular matrix over canonical rationals."""

    domain: Literal["QQ"] = "QQ"
    entries: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_MATRIX_ORDER,
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_matrix_envelope(cls, data: Any) -> Any:
        return _prepare_raw_matrix_envelope(
            data,
            maximum_axis=MAX_RATIONAL_MATRIX_ORDER,
            label="matrix",
            scalar_domain="QQ",
        )

    @model_validator(mode="after")
    def require_rectangular_nonempty_rows(self) -> Self:
        column_count = len(self.entries[0])
        if column_count == 0 or column_count > MAX_RATIONAL_MATRIX_ORDER:
            raise _validation_error(
                "budget_exceeded",
                "matrix rows must contain between 1 and "
                f"{MAX_RATIONAL_MATRIX_ORDER} entries",
            )
        if any(len(row) != column_count for row in self.entries):
            raise _validation_error(
                "budget_exceeded", "matrix rows must all have the same length"
            )
        require_matrix_scalar_digits(
            self.entries,
            maximum=MAX_MATRIX_SCALAR_DIGITS,
            label="matrix",
        )
        return self


class EmbeddedRealSimpleNumberFieldMatrix(StrictModel):
    """One matrix over a simple number field at a selected real embedding.

    Every entry retains the canonical abstract field element.  The common
    embedding selects how those elements act as real scalars; consumers must
    recognize the field and indexed root before doing mathematical work.
    """

    domain: Literal["EMBEDDED_REAL_SIMPLE_NUMBER_FIELD"] = (
        "EMBEDDED_REAL_SIMPLE_NUMBER_FIELD"
    )
    embedding: RealNumberFieldEmbedding
    entries: tuple[tuple[SimpleNumberFieldElement, ...], ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_MATRIX_ORDER,
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_matrix_envelope(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if set(data).difference({"domain", "embedding", "entries"}):
            raise _validation_error(
                "shape_mismatch",
                "embedded number-field matrix contains unknown fields",
            )
        entries = data.get("entries")
        normalized = dict(data)
        embedding = normalized.get("embedding")
        if isinstance(embedding, dict):
            normalized_embedding = dict(embedding)
            root = normalized_embedding.get("root")
            if isinstance(root, dict) and isinstance(root.get("polynomial"), list):
                normalized_root = dict(root)
                normalized_root["polynomial"] = tuple(root["polynomial"])
                normalized_embedding["root"] = normalized_root
            normalized["embedding"] = normalized_embedding
        if isinstance(entries, (list, tuple)):
            if len(entries) > MAX_RATIONAL_MATRIX_ORDER:
                raise _validation_error(
                    "budget_exceeded",
                    "embedded number-field matrices have at most "
                    f"{MAX_RATIONAL_MATRIX_ORDER} rows",
                )
            for row in entries:
                if not isinstance(row, (list, tuple)):
                    raise _validation_error(
                        "shape_mismatch",
                        "embedded number-field matrix rows must be arrays",
                    )
                if len(row) > MAX_RATIONAL_MATRIX_ORDER:
                    raise _validation_error(
                        "budget_exceeded",
                        "embedded number-field matrices have at most "
                        f"{MAX_RATIONAL_MATRIX_ORDER} columns",
                    )
                for scalar in row:
                    if isinstance(scalar, SimpleNumberFieldElement):
                        continue
                    if not isinstance(scalar, dict) or set(scalar).difference(
                        {"presentation", "coefficients_ascending"}
                    ):
                        raise _validation_error(
                            "shape_mismatch",
                            "embedded number-field matrix entries must be field elements",
                        )
            normalized["entries"] = tuple(tuple(row) for row in entries)
        return normalized

    @model_validator(mode="after")
    def require_common_embedding_and_rectangular_shape(self) -> Self:
        column_count = len(self.entries[0])
        if column_count == 0 or column_count > MAX_RATIONAL_MATRIX_ORDER:
            raise _validation_error(
                "budget_exceeded",
                "embedded number-field matrix rows must contain between 1 and "
                f"{MAX_RATIONAL_MATRIX_ORDER} entries",
            )
        if any(len(row) != column_count for row in self.entries):
            raise _validation_error(
                "shape_mismatch",
                "embedded number-field matrix rows must have equal length",
            )
        presentation = self.embedding.presentation
        if any(
            entry.presentation != presentation for row in self.entries for entry in row
        ):
            raise _validation_error(
                "embedding_presentation",
                "every matrix entry must use the selected embedding's presentation",
            )
        return self


class _RequiredExactRealMatrixDomainSchema:
    """Publish the discriminator that strict exact-real parsing requires."""

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: Any,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        schema = deepcopy(handler(core_schema))
        branches = []
        for branch in schema["oneOf"]:
            resolved = deepcopy(handler.resolve_ref_schema(branch))
            required = list(resolved.get("required", ()))
            if "domain" not in required:
                required.insert(0, "domain")
            resolved["required"] = required
            branches.append(resolved)
        schema["oneOf"] = branches
        return schema


ExactRealMatrix = Annotated[
    RationalMatrix | EmbeddedRealSimpleNumberFieldMatrix,
    Field(discriminator="domain"),
    _RequiredExactRealMatrixDomainSchema,
]


def rational_matrix_from_fractions(
    entries: tuple[tuple[Fraction, ...], ...] | list[list[Fraction]],
) -> RationalMatrix:
    """Construct the canonical dense rational matrix from exact fractions."""

    return RationalMatrix(
        entries=tuple(
            tuple(CanonicalRational.from_fraction(value) for value in row)
            for row in entries
        )
    )


class SparseRationalMatrixEntry(StrictModel):
    """One nonzero entry at a stable zero-based matrix coordinate."""

    row: int = Field(ge=0, le=MAX_SPARSE_RATIONAL_MATRIX_AXIS - 1)
    column: int = Field(ge=0, le=MAX_SPARSE_RATIONAL_MATRIX_AXIS - 1)
    value: CanonicalRational


class SparseRationalMatrix(StrictModel):
    """A dimension-retaining coordinate-sparse matrix over QQ."""

    domain: Literal["QQ"] = "QQ"
    row_count: int = Field(ge=1, le=MAX_SPARSE_RATIONAL_MATRIX_AXIS)
    column_count: int = Field(ge=1, le=MAX_SPARSE_RATIONAL_MATRIX_AXIS)
    entries: tuple[SparseRationalMatrixEntry, ...] = Field(
        default=(), max_length=MAX_SPARSE_RATIONAL_MATRIX_NONZEROS
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_sparse_envelope(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if set(data).difference({"domain", "row_count", "column_count", "entries"}):
            raise _validation_error(
                "shape_mismatch", "sparse matrix contains unknown fields"
            )
        entries = data.get("entries")
        if (
            isinstance(entries, (list, tuple))
            and len(entries) > MAX_SPARSE_RATIONAL_MATRIX_NONZEROS
        ):
            raise _validation_error(
                "budget_exceeded",
                "sparse matrix stores at most "
                f"{MAX_SPARSE_RATIONAL_MATRIX_NONZEROS} nonzeros",
            )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_canonical_coordinates(self) -> Self:
        coordinates = tuple((entry.row, entry.column) for entry in self.entries)
        if coordinates != tuple(sorted(set(coordinates))):
            raise _validation_error(
                "shape_mismatch",
                "sparse matrix coordinates must be unique and row-major sorted",
            )
        if any(
            entry.row >= self.row_count or entry.column >= self.column_count
            for entry in self.entries
        ):
            raise _validation_error(
                "shape_mismatch", "sparse matrix coordinates exceed declared axes"
            )
        if any(entry.value.as_fraction() == 0 for entry in self.entries):
            raise _validation_error(
                "shape_mismatch", "sparse matrices must not store explicit zeros"
            )
        require_matrix_scalar_digits(
            tuple((entry.value,) for entry in self.entries),
            maximum=MAX_MATRIX_SCALAR_DIGITS,
            label="sparse matrix",
        )
        return self


def sparse_rational_matrix_from_dense(matrix: RationalMatrix) -> SparseRationalMatrix:
    """Convert the canonical dense QQ matrix to dimension-retaining coordinates."""

    return SparseRationalMatrix(
        row_count=len(matrix.entries),
        column_count=len(matrix.entries[0]),
        entries=tuple(
            SparseRationalMatrixEntry(row=row, column=column, value=value)
            for row, values in enumerate(matrix.entries)
            for column, value in enumerate(values)
            if value.as_fraction()
        ),
    )


def dense_rational_matrix_from_sparse(matrix: SparseRationalMatrix) -> RationalMatrix:
    """Convert bounded sparse coordinates to the canonical dense QQ matrix."""

    if (
        matrix.row_count > MAX_RATIONAL_MATRIX_ORDER
        or matrix.column_count > MAX_RATIONAL_MATRIX_ORDER
    ):
        raise ValueError(
            "sparse matrix axes exceed the canonical dense matrix representation"
        )
    zero = CanonicalRational(num="0", den="1")
    coordinates = {(entry.row, entry.column): entry.value for entry in matrix.entries}
    return RationalMatrix(
        entries=tuple(
            tuple(
                coordinates.get((row, column), zero)
                for column in range(matrix.column_count)
            )
            for row in range(matrix.row_count)
        )
    )


class RationalVectorSpaceBasis(StrictModel):
    """A rational vector-space basis with its ambient dimension retained.

    Unlike a dense matrix, a basis may be empty.  The explicit ambient
    dimension distinguishes the zero subspace of ``QQ^n`` for different ``n``.
    """

    domain: Literal["QQ"] = "QQ"
    ambient_dimension: int = Field(ge=1, le=MAX_RATIONAL_MATRIX_ORDER)
    vectors: tuple[tuple[CanonicalRational, ...], ...] = Field(
        default=(), max_length=MAX_RATIONAL_MATRIX_ORDER
    )

    @model_validator(mode="after")
    def require_vector_shape(self) -> Self:
        if any(len(vector) != self.ambient_dimension for vector in self.vectors):
            raise _validation_error(
                "shape_mismatch",
                "each basis vector must have the declared ambient dimension",
            )
        require_matrix_scalar_digits(
            self.vectors,
            maximum=MAX_MATRIX_SCALAR_DIGITS,
            label="basis",
        )
        return self


def rational_vector_space_basis_from_fractions(
    vectors: tuple[tuple[Fraction, ...], ...] | list[list[Fraction]],
    *,
    ambient_dimension: int,
) -> RationalVectorSpaceBasis:
    """Construct a canonical rational basis, including the empty basis."""

    return RationalVectorSpaceBasis(
        ambient_dimension=ambient_dimension,
        vectors=tuple(
            tuple(CanonicalRational.from_fraction(value) for value in vector)
            for vector in vectors
        ),
    )


class RealQuadraticMatrix(StrictModel):
    """One nonempty rectangular matrix over a shared real quadratic field."""

    domain: Literal["QQ_SQRT_D"] = "QQ_SQRT_D"
    entries: tuple[tuple[RealQuadraticValue, ...], ...] = Field(
        min_length=1,
        max_length=MAX_MATRIX_DIMENSION,
        description=(
            "Nonempty rectangular rows of a+b*sqrt(d) values. Every entry "
            "must carry the same square-free positive radicand d."
        ),
    )

    @model_validator(mode="after")
    def require_rectangular_shared_field(self) -> Self:
        column_count = len(self.entries[0])
        if column_count == 0 or column_count > MAX_MATRIX_DIMENSION:
            raise _validation_error(
                "shape_mismatch", "matrix rows must contain between 1 and 32 entries"
            )
        if any(len(row) != column_count for row in self.entries):
            raise _validation_error(
                "shape_mismatch", "matrix rows must all have the same length"
            )
        radicand = self.entries[0][0].radicand
        if any(entry.radicand != radicand for row in self.entries for entry in row):
            raise _validation_error(
                "shape_mismatch",
                "every matrix entry must belong to one shared real quadratic field",
            )
        return self


class IntegerMatrix(StrictModel):
    """One nonempty rectangular matrix over exact canonical integers.

    Structural axes follow ``MAX_INTEGER_MATRIX_ORDER``. Operations whose
    admitted computation envelope is narrower, including exact linear
    operations and lattice reduction, enforce that bound in owner-local
    admission rather than on this shared value.
    """

    domain: Literal["ZZ"] = "ZZ"
    entries: tuple[tuple[CanonicalInteger, ...], ...] = Field(
        min_length=1,
        max_length=MAX_INTEGER_MATRIX_ORDER,
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_matrix_envelope(cls, data: Any) -> Any:
        return _prepare_raw_matrix_envelope(
            data,
            maximum_axis=MAX_INTEGER_MATRIX_ORDER,
            label="matrix",
            scalar_domain="ZZ",
        )

    @model_validator(mode="after")
    def require_rectangular_nonempty_rows(self) -> Self:
        column_count = len(self.entries[0])
        if column_count == 0 or column_count > MAX_INTEGER_MATRIX_ORDER:
            raise _validation_error(
                "budget_exceeded",
                "matrix rows must contain between 1 and "
                f"{MAX_INTEGER_MATRIX_ORDER} entries",
            )
        if any(len(row) != column_count for row in self.entries):
            raise _validation_error(
                "budget_exceeded", "matrix rows must all have the same length"
            )
        require_matrix_scalar_digits(
            self.entries,
            maximum=MAX_MATRIX_SCALAR_DIGITS,
            label="matrix",
        )
        return self


def integer_matrix_axis_schema(maximum_axis: int) -> JsonSchemaValue:
    """Project ``IntegerMatrix`` with an operation-local axis ceiling.

    The canonical ZZ matrix retains inverse sources through order 128, so a
    verbatim shared definition would publish ``maxItems: 128`` on every
    consumer. Narrower operations attach this schema through
    ``WithJsonSchema`` so discovery advertises the axis their validators
    enforce; validation itself stays with the canonical value plus owner-local
    admission.
    """

    schema: dict[str, Any] = IntegerMatrix.model_json_schema()
    entries = schema["properties"]["entries"]
    entries["maxItems"] = maximum_axis
    row_schema = entries.get("items")
    if isinstance(row_schema, dict):
        row_schema["maxItems"] = maximum_axis
    return schema


class SmithNormalForm(StrictModel):
    """A backend-independent positive divisibility diagonal and its metadata."""

    normal_form: IntegerMatrix
    rank: int = Field(ge=0, le=MAX_EXACT_LINEAR_MATRIX_AXIS)
    invariant_factors: tuple[CanonicalInteger, ...] = Field(
        max_length=MAX_EXACT_LINEAR_MATRIX_AXIS
    )
    transformation_available: Literal[False] = False
    convention: Literal["POSITIVE_DIVISIBILITY_DIAGONAL"] = (
        "POSITIVE_DIVISIBILITY_DIAGONAL"
    )

    @model_validator(mode="after")
    def require_invariant_factor_chain(self) -> Self:
        rows = len(self.normal_form.entries)
        columns = len(self.normal_form.entries[0])
        if len(self.invariant_factors) != self.rank:
            raise _validation_error(
                "shape_mismatch", "nonzero invariant factor count must equal rank"
            )
        if self.rank > min(rows, columns):
            raise _validation_error(
                "shape_mismatch", "Smith rank cannot exceed the matrix dimensions"
            )
        factors = tuple(
            parse_canonical_integer(value) for value in self.invariant_factors
        )
        if any(value <= 0 for value in factors):
            raise _validation_error(
                "shape_mismatch", "Smith invariant factors must be positive"
            )
        if any(right % left != 0 for left, right in pairwise(factors)):
            raise _validation_error(
                "shape_mismatch", "each Smith invariant factor must divide the next"
            )
        for row, entries in enumerate(self.normal_form.entries):
            for column, value in enumerate(entries):
                expected = factors[row] if row == column and row < self.rank else 0
                if parse_canonical_integer(value) != expected:
                    raise _validation_error(
                        "budget_exceeded",
                        "Smith normal form must contain its positive invariant "
                        "factors on the leading diagonal and zero elsewhere",
                    )
        return self

    @field_validator("invariant_factors")
    @classmethod
    def require_bounded_invariant_factors(
        cls, values: tuple[CanonicalInteger, ...]
    ) -> tuple[CanonicalInteger, ...]:
        for value in values:
            if len(value.lstrip("-")) > MAX_MATRIX_SCALAR_DIGITS:
                raise _validation_error(
                    "budget_exceeded",
                    f"matrix scalars are limited to {MAX_MATRIX_SCALAR_DIGITS} decimal digits",
                )
        return values


__all__ = [
    "MAX_EXACT_LINEAR_MATRIX_AXIS",
    "MAX_INTEGER_MATRIX_ORDER",
    "MAX_MATRIX_DIMENSION",
    "MAX_MATRIX_SCALAR_DIGITS",
    "MAX_RATIONAL_MATRIX_ORDER",
    "MAX_SPARSE_RATIONAL_MATRIX_AXIS",
    "MAX_SPARSE_RATIONAL_MATRIX_NONZEROS",
    "EmbeddedRealSimpleNumberFieldMatrix",
    "ExactRealMatrix",
    "IntegerMatrix",
    "RationalMatrix",
    "RationalVectorSpaceBasis",
    "RealQuadraticMatrix",
    "SmithNormalForm",
    "SparseRationalMatrix",
    "SparseRationalMatrixEntry",
    "dense_rational_matrix_from_sparse",
    "integer_matrix_axis_schema",
    "rational_matrix_from_fractions",
    "rational_vector_space_basis_from_fractions",
    "require_matrix_scalar_digits",
    "sparse_rational_matrix_from_dense",
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"matrix.{reason}", message)
