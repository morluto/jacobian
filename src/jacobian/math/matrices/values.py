"""Provider-independent exact matrix values."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from fractions import Fraction
from itertools import pairwise
from typing import Any, Literal, Self, cast

from pydantic import Field, field_validator, model_validator
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
    MAX_SIMPLE_NUMBER_FIELD_DEGREE,
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
# Representation envelope includes the existing 4096-state Markov domain.
# Computational order limits above remain operation-owned.
MAX_RATIONAL_MATRIX_AXIS = 4096
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


def _require_raw_matrix_envelope(  # noqa: C901
    data: object, *, maximum_axis: int, label: str, allow_shape: bool = False
) -> object:
    """Bound raw matrix depth, axes, and scalar strings before tuple copying."""

    if not isinstance(data, dict):
        return data
    allowed = (
        {"domain", "entries", "row_count", "column_count"}
        if allow_shape
        else {"domain", "entries"}
    )
    if set(data).difference(allowed):
        raise _validation_error("shape_mismatch", f"{label} contains unknown fields")
    entries = data.get("entries")
    if entries is None:
        return data
    if isinstance(entries, (str, bytes, Mapping)):
        return data
    if not isinstance(entries, (list, tuple)):
        try:
            iterator = iter(cast(Iterable[object], entries))
        except TypeError:
            return data
        materialized: list[object] = []
        for _index in range(maximum_axis + 1):
            try:
                materialized.append(next(iterator))
            except StopIteration:
                break
        else:
            raise _validation_error(
                "budget_exceeded", f"{label} has at most {maximum_axis} rows"
            )
        entries = tuple(materialized)
        data = dict(data)
        data["entries"] = entries
    if len(entries) > maximum_axis:
        raise _validation_error(
            "budget_exceeded", f"{label} has at most {maximum_axis} rows"
        )
    normalized_rows: list[object] = []
    for row in entries:
        if not isinstance(row, (list, tuple)):
            if isinstance(row, (str, bytes, Mapping)):
                normalized_rows.append(row)
                continue
            try:
                row_iterator = iter(row)
            except TypeError:
                normalized_rows.append(row)
                continue
            row_values: list[object] = []
            for _index in range(maximum_axis + 1):
                try:
                    row_values.append(next(row_iterator))
                except StopIteration:
                    break
            else:
                raise _validation_error(
                    "budget_exceeded",
                    f"{label} has at most {maximum_axis} columns",
                )
            row = tuple(row_values)
        normalized_rows.append(row)
        if len(row) > maximum_axis:
            raise _validation_error(
                "budget_exceeded", f"{label} has at most {maximum_axis} columns"
            )
        for scalar in row:
            if isinstance(scalar, (list, tuple)):
                raise _validation_error(
                    "shape_mismatch", f"{label} entries must be scalar values"
                )
            if isinstance(scalar, dict) and set(scalar).difference({"num", "den"}):
                raise _validation_error(
                    "shape_mismatch", f"{label} rational scalar contains unknown fields"
                )
            if isinstance(scalar, dict):
                for _key in ("num", "den"):
                    _val = scalar.get(_key)
                    if _val is not None and not isinstance(_val, str):
                        raise _validation_error(
                            "shape_mismatch",
                            f"{label} rational {_key} must be a string"
                            f", not {type(_val).__name__}",
                        )
            components = (
                (scalar.get("num"), scalar.get("den"))
                if isinstance(scalar, dict)
                else (scalar,)
            )
            for component in components:
                if not isinstance(component, (str, int)):
                    if hasattr(component, "num") and hasattr(component, "den"):
                        component_value: Any = component
                        component = (
                            component_value.num,
                            component_value.den,
                        )
                        for sub in component:
                            if (
                                isinstance(sub, (str, int))
                                and len(str(sub).lstrip("-")) > MAX_MATRIX_SCALAR_DIGITS
                            ):
                                raise _validation_error(
                                    "budget_exceeded",
                                    f"{label} scalars are limited to "
                                    f"{MAX_MATRIX_SCALAR_DIGITS} decimal digits",
                                )
                        continue
                    raise _validation_error(
                        "shape_mismatch",
                        f"{label} rational scalar components must be integers or strings",
                    )
                if (
                    isinstance(component, (str, int))
                    and len(str(component).lstrip("-")) > MAX_MATRIX_SCALAR_DIGITS
                ):
                    raise _validation_error(
                        "budget_exceeded",
                        f"{label} scalars are limited to "
                        f"{MAX_MATRIX_SCALAR_DIGITS} decimal digits",
                    )
    if normalized_rows and tuple(normalized_rows) != entries:
        data = dict(data)
        data["entries"] = tuple(normalized_rows)
    return data


def _infer_matrix_shape(data: Any) -> Any:
    """Retain declared axes; infer omitted dimensions only from actual rows."""
    if isinstance(data, dict):
        data = dict(data)
        entries = data.get("entries", ())
        if isinstance(entries, (list, tuple)):
            data.setdefault("row_count", len(entries))
            if not entries or isinstance(entries[0], (list, tuple)):
                data.setdefault("column_count", len(entries[0]) if entries else 0)
    return data


class RationalMatrix(StrictModel):
    """An exact QQ matrix retaining both dimensions, including empty axes."""

    domain: Literal["QQ"] = "QQ"
    row_count: int = Field(
        default_factory=int,
        ge=0,
        le=MAX_RATIONAL_MATRIX_AXIS,
        description="Row count, inferred from entries when omitted.",
    )
    column_count: int = Field(
        default_factory=int,
        ge=0,
        le=MAX_RATIONAL_MATRIX_AXIS,
        description="Column count, inferred from the first row or zero when omitted.",
    )
    entries: tuple[tuple[CanonicalRational, ...], ...] = Field(
        default=(),
        max_length=MAX_RATIONAL_MATRIX_AXIS,
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_matrix_envelope(cls, data: Any) -> Any:
        data = _require_raw_matrix_envelope(
            data,
            maximum_axis=MAX_RATIONAL_MATRIX_AXIS,
            label="matrix",
            allow_shape=True,
        )
        return canonicalize_json_containers(_infer_matrix_shape(data))

    @model_validator(mode="after")
    def require_shape_and_scalars(self) -> Self:
        if len(self.entries) != self.row_count or any(
            len(row) != self.column_count for row in self.entries
        ):
            raise _validation_error(
                "shape_mismatch", "matrix entries must match the declared shape"
            )
        require_matrix_scalar_digits(
            self.entries, maximum=MAX_MATRIX_SCALAR_DIGITS, label="matrix"
        )
        return self


def rational_matrix_from_fractions(
    entries: tuple[tuple[Fraction, ...], ...] | list[list[Fraction]],
    *,
    column_count: int | None = None,
) -> RationalMatrix:
    """Construct a QQ matrix; supply column_count to retain a zero-row domain."""
    return RationalMatrix(
        row_count=len(entries),
        column_count=column_count
        if column_count is not None
        else (len(entries[0]) if entries else 0),
        entries=tuple(
            tuple(CanonicalRational.from_fraction(value) for value in row)
            for row in entries
        ),
    )


class SparseRationalMatrixEntry(StrictModel):
    """One nonzero entry at a stable zero-based matrix coordinate."""

    row: int = Field(ge=0, le=MAX_SPARSE_RATIONAL_MATRIX_AXIS - 1)
    column: int = Field(ge=0, le=MAX_SPARSE_RATIONAL_MATRIX_AXIS - 1)
    value: CanonicalRational


class SparseRationalMatrix(StrictModel):
    """A dimension-retaining coordinate-sparse matrix over QQ."""

    domain: Literal["QQ"] = "QQ"
    row_count: int = Field(ge=0, le=MAX_SPARSE_RATIONAL_MATRIX_AXIS)
    column_count: int = Field(ge=0, le=MAX_SPARSE_RATIONAL_MATRIX_AXIS)
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
        row_count=matrix.row_count,
        column_count=matrix.column_count,
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
        matrix.row_count > MAX_RATIONAL_MATRIX_AXIS
        or matrix.column_count > MAX_RATIONAL_MATRIX_AXIS
    ):
        raise ValueError("sparse matrix axes exceed the canonical dense matrix")
    zero = CanonicalRational(num="0", den="1")
    coordinates = {(entry.row, entry.column): entry.value for entry in matrix.entries}
    return RationalMatrix(
        row_count=matrix.row_count,
        column_count=matrix.column_count,
        entries=tuple(
            tuple(
                coordinates.get((row, column), zero)
                for column in range(matrix.column_count)
            )
            for row in range(matrix.row_count)
        ),
    )


class RationalVectorSpaceBasis(StrictModel):
    """A rational vector-space basis with its ambient dimension retained.

    A basis may be empty. The explicit ambient
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
    """An exact matrix over one declared real quadratic field, including empty axes."""

    domain: Literal["QQ_SQRT_D"] = "QQ_SQRT_D"
    radicand: int = Field(
        default_factory=int,
        ge=2,
        le=1_000_000,
        description="Shared radicand; inferred from nonempty entries, required for empty matrices.",
    )
    row_count: int = Field(
        default_factory=int,
        ge=0,
        le=MAX_MATRIX_DIMENSION,
        description="Row count, inferred from entries when omitted.",
    )
    column_count: int = Field(
        default_factory=int,
        ge=0,
        le=MAX_MATRIX_DIMENSION,
        description="Column count, inferred from the first row or zero when omitted.",
    )
    entries: tuple[tuple[RealQuadraticValue, ...], ...] = Field(
        default=(),
        max_length=MAX_MATRIX_DIMENSION,
    )

    @model_validator(mode="before")
    @classmethod
    def infer_context(cls, data: Any) -> Any:
        data = _infer_matrix_shape(data)
        if isinstance(data, dict) and "radicand" not in data:
            entries = data.get("entries", ())
            if (
                isinstance(entries, (list, tuple))
                and entries
                and isinstance(entries[0], (list, tuple))
                and entries[0]
            ):
                first = entries[0][0]
                data["radicand"] = (
                    first.get("radicand")
                    if isinstance(first, dict)
                    else getattr(first, "radicand", None)
                )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_rectangular_shared_field(self) -> Self:
        if not 2 <= self.radicand <= 1_000_000:
            raise _validation_error(
                "shape_mismatch",
                "empty quadratic matrices require an explicit radicand",
            )
        if len(self.entries) != self.row_count or any(
            len(row) != self.column_count for row in self.entries
        ):
            raise _validation_error(
                "shape_mismatch", "matrix entries must match the declared shape"
            )
        if any(
            entry.radicand != self.radicand for row in self.entries for entry in row
        ):
            raise _validation_error(
                "shape_mismatch",
                "every matrix entry must belong to one shared real quadratic field",
            )
        return self


class IntegerMatrix(StrictModel):
    """One exact integer matrix, including zero-dimensional shapes.

    Missing shape fields are inferred from entries (an empty row family
    defaults to 0 columns). Supply column_count to retain a 0-by-n shape.
    Serialization always retains both dimensions. Operation bounds belong
    to native admission, not to a certificate-specific carrier.
    """

    domain: Literal["ZZ"] = "ZZ"
    row_count: int = Field(
        default_factory=int,
        ge=0,
        le=MAX_INTEGER_MATRIX_ORDER,
        description="Row count; inferred from entries when omitted.",
    )
    column_count: int = Field(
        default_factory=int,
        ge=0,
        le=MAX_INTEGER_MATRIX_ORDER,
        description="Column count; inferred from the first row when omitted, or zero for no rows.",
    )
    entries: tuple[tuple[CanonicalInteger, ...], ...] = Field(
        default=(),
        max_length=MAX_INTEGER_MATRIX_ORDER,
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_matrix_envelope(cls, data: Any) -> Any:
        data = _require_raw_matrix_envelope(
            data,
            maximum_axis=MAX_INTEGER_MATRIX_ORDER,
            label="matrix",
            allow_shape=True,
        )
        if isinstance(data, dict):
            data = dict(data)
            entries = data.get("entries", ())
            if isinstance(entries, (list, tuple)):
                data.setdefault("row_count", len(entries))
                if not entries or isinstance(entries[0], (list, tuple)):
                    data.setdefault("column_count", len(entries[0]) if entries else 0)
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_shape_and_scalars(self) -> Self:
        if len(self.entries) != self.row_count or any(
            len(row) != self.column_count for row in self.entries
        ):
            raise _validation_error(
                "shape_mismatch", "matrix entries must match the declared shape"
            )
        require_matrix_scalar_digits(
            self.entries, maximum=MAX_MATRIX_SCALAR_DIGITS, label="matrix"
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
    for axis in ("row_count", "column_count"):
        schema["properties"][axis]["maximum"] = maximum_axis
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
    convention: Literal["POSITIVE_DIVISIBILITY_DIAGONAL"] = (
        "POSITIVE_DIVISIBILITY_DIAGONAL"
    )

    @model_validator(mode="after")
    def require_invariant_factor_chain(self) -> Self:
        rows = len(self.normal_form.entries)
        columns = self.normal_form.column_count
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


class EmbeddedRealSimpleNumberFieldMatrix(StrictModel):
    """An exact matrix retaining its embedding and both axes, including empty shapes."""

    domain: Literal["EMBEDDED_REAL_SIMPLE_NUMBER_FIELD"] = (
        "EMBEDDED_REAL_SIMPLE_NUMBER_FIELD"
    )
    embedding: RealNumberFieldEmbedding
    row_count: int = Field(
        default_factory=int,
        ge=0,
        le=MAX_MATRIX_DIMENSION,
        description="Row count, inferred from entries when omitted.",
    )
    column_count: int = Field(
        default_factory=int,
        ge=0,
        le=MAX_MATRIX_DIMENSION,
        description="Column count, inferred from the first row or zero when omitted.",
    )
    entries: tuple[tuple[SimpleNumberFieldElement, ...], ...] = Field(
        default=(),
        max_length=MAX_MATRIX_DIMENSION,
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_matrix_envelope(cls, data: Any) -> Any:  # noqa: C901
        if not isinstance(data, dict):
            return data
        if set(data).difference(
            {"domain", "embedding", "entries", "row_count", "column_count"}
        ):
            raise _validation_error(
                "shape_mismatch",
                "embedded number-field matrix contains unknown fields",
            )
        entries = data.get("entries")
        normalized = dict(data)

        def materialize_bounded(value: object, *, maximum: int, message: str) -> object:
            if isinstance(value, (list, tuple, str, bytes, Mapping)):
                return value
            try:
                iterator = iter(cast(Iterable[object], value))
            except TypeError:
                return value
            values: list[object] = []
            for _index in range(maximum + 1):
                try:
                    values.append(next(iterator))
                except StopIteration:
                    break
            else:
                raise _validation_error("budget_exceeded", message)
            return tuple(values)

        def bounded_field_coefficients(value: object) -> object:
            return materialize_bounded(
                value,
                maximum=MAX_SIMPLE_NUMBER_FIELD_DEGREE + 1,
                message=(
                    "simple number-field polynomials have at most "
                    f"{MAX_SIMPLE_NUMBER_FIELD_DEGREE + 1} coefficients"
                ),
            )

        embedding = normalized.get("embedding")
        if isinstance(embedding, dict):
            normalized_embedding = dict(embedding)
            root = normalized_embedding.get("root")
            if isinstance(root, dict):
                normalized_root = dict(root)
                polynomial = bounded_field_coefficients(root.get("polynomial"))
                if isinstance(polynomial, (list, tuple)):
                    if len(polynomial) > MAX_SIMPLE_NUMBER_FIELD_DEGREE + 1:
                        raise _validation_error(
                            "budget_exceeded",
                            "simple number-field polynomials have at most "
                            f"{MAX_SIMPLE_NUMBER_FIELD_DEGREE + 1} coefficients",
                        )
                    normalized_root["polynomial"] = tuple(polynomial)
                normalized_embedding["root"] = normalized_root
            normalized["embedding"] = normalized_embedding

        entries = materialize_bounded(
            entries,
            maximum=MAX_MATRIX_DIMENSION,
            message=(
                "embedded number-field matrices have at most "
                f"{MAX_MATRIX_DIMENSION} rows"
            ),
        )
        if isinstance(entries, (list, tuple)):
            if len(entries) > MAX_MATRIX_DIMENSION:
                raise _validation_error(
                    "budget_exceeded",
                    "embedded number-field matrices have at most "
                    f"{MAX_MATRIX_DIMENSION} rows",
                )
            normalized_entries: list[tuple[object, ...]] = []
            for raw_row in entries:
                row = materialize_bounded(
                    raw_row,
                    maximum=MAX_MATRIX_DIMENSION,
                    message=(
                        "embedded number-field matrices have at most "
                        f"{MAX_MATRIX_DIMENSION} columns"
                    ),
                )
                if not isinstance(row, (list, tuple)):
                    raise _validation_error(
                        "shape_mismatch",
                        "embedded number-field matrix rows must be arrays",
                    )
                if len(row) > MAX_MATRIX_DIMENSION:
                    raise _validation_error(
                        "budget_exceeded",
                        "embedded number-field matrices have at most "
                        f"{MAX_MATRIX_DIMENSION} columns",
                    )
                normalized_row: list[object] = []
                for scalar in row:
                    if isinstance(scalar, SimpleNumberFieldElement):
                        normalized_row.append(scalar)
                        continue
                    if not isinstance(scalar, dict) or set(scalar).difference(
                        {"presentation", "coefficients_ascending"}
                    ):
                        raise _validation_error(
                            "shape_mismatch",
                            "embedded number-field matrix entries must be field elements",
                        )
                    coordinates = scalar.get("coefficients_ascending")
                    coordinates = materialize_bounded(
                        coordinates,
                        maximum=MAX_SIMPLE_NUMBER_FIELD_DEGREE,
                        message=(
                            "simple number-field elements have at most "
                            f"{MAX_SIMPLE_NUMBER_FIELD_DEGREE} coordinates"
                        ),
                    )
                    if not isinstance(coordinates, (list, tuple)):
                        raise PydanticCustomError(
                            "tuple_type", "Input should be a valid tuple"
                        )
                    if isinstance(scalar, dict):
                        scalar = dict(scalar)
                        scalar["coefficients_ascending"] = coordinates
                    for coordinate in coordinates:
                        if not isinstance(coordinate, dict):
                            raise _validation_error(
                                "shape_mismatch",
                                "embedded number-field matrix rational coordinates "
                                "must be mappings",
                            )
                        if set(coordinate).difference({"num", "den"}):
                            raise _validation_error(
                                "shape_mismatch",
                                "embedded number-field matrix rational coordinates "
                                "contain unknown fields",
                            )
                        for component in ("num", "den"):
                            if not isinstance(coordinate.get(component), (str, int)):
                                raise PydanticCustomError(
                                    "string_type", "Input should be a valid string"
                                )
                    presentation = scalar.get("presentation")
                    if not isinstance(presentation, dict):
                        raise _validation_error(
                            "shape_mismatch",
                            "embedded number-field entry presentation must be a mapping",
                        )
                    if isinstance(presentation, dict):
                        if set(presentation).difference(
                            {"domain", "coefficients_descending"}
                        ):
                            raise _validation_error(
                                "shape_mismatch",
                                "embedded number-field entry presentations contain "
                                "unknown fields",
                            )
                        coefficients = bounded_field_coefficients(
                            presentation.get("coefficients_descending")
                        )
                        if not isinstance(coefficients, (list, tuple)):
                            raise PydanticCustomError(
                                "tuple_type", "Input should be a valid tuple"
                            )
                        if any(
                            not isinstance(coefficient, (str, int))
                            for coefficient in coefficients
                        ):
                            raise PydanticCustomError(
                                "string_type", "Input should be a valid string"
                            )
                    normalized_row.append(scalar)
                normalized_entries.append(tuple(normalized_row))
            normalized["entries"] = canonicalize_json_containers(
                tuple(normalized_entries)
            )
        if isinstance(embedding, dict):
            presentation = embedding.get("presentation")
            if isinstance(presentation, dict):
                if set(presentation).difference({"domain", "coefficients_descending"}):
                    raise _validation_error(
                        "shape_mismatch",
                        "embedded number-field presentations contain unknown fields",
                    )
                coefficients = bounded_field_coefficients(
                    presentation.get("coefficients_descending")
                )
                if not isinstance(coefficients, (list, tuple)):
                    raise PydanticCustomError(
                        "tuple_type", "Input should be a valid tuple"
                    )
                if any(
                    not isinstance(coefficient, (str, int))
                    for coefficient in coefficients
                ):
                    raise PydanticCustomError(
                        "string_type", "Input should be a valid string"
                    )
            root = embedding.get("root")
            if isinstance(root, dict) and not isinstance(
                root.get("polynomial"), (list, tuple)
            ):
                raise PydanticCustomError("tuple_type", "Input should be a valid tuple")
        return _infer_matrix_shape(normalized)

    @model_validator(mode="after")
    def require_rectangular_shared_field(self) -> Self:
        if len(self.entries) != self.row_count or any(
            len(row) != self.column_count for row in self.entries
        ):
            raise _validation_error(
                "shape_mismatch", "matrix entries must match the declared shape"
            )
        presentation = self.embedding.presentation
        for row in self.entries:
            for entry in row:
                if entry.presentation != presentation:
                    raise _validation_error(
                        "embedding_presentation",
                        "every matrix entry must belong to the embedding field",
                    )
        return self


ExactRealMatrix = RationalMatrix | EmbeddedRealSimpleNumberFieldMatrix


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
