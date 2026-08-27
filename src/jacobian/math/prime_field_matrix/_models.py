"""Typed wire contracts for prime-field matrix operations."""

from __future__ import annotations

from typing import Any, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

MAX_ROWS = 256
MAX_COLUMNS = 256
MAX_PRIME = 2_147_483_647
"""Explicit conservative bound on the field prime before primality testing."""


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by prime-field-matrix contracts."""

    return PydanticCustomError(f"prime_field_matrix.{reason}", message)


class PrimeFieldMatrixRequest(StrictModel):
    """A bounded integer matrix over an explicit prime field GF(p).

    The matrix is carried as the domain-owned ``PrimeFieldMatrix`` canonical
    value so it composes unchanged with the other GF(p) producers and
    consumers. Shape rules (schema-visible): 0..256 rows, 1..256 columns,
    rectangular rows, and every entry a canonical residue in ``[0, prime)``.
    Zero rows carry an explicit column axis, matching the canonical empty
    matrix that full-rank nullspace producers return. The characteristic is
    bounded to ``MAX_PRIME`` by a pre-construction validator so no accepted
    request can reach unbounded primality or modular work.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A bounded integer matrix over an explicit prime field GF(p), "
                "as the canonical `PrimeFieldMatrix` value: `prime` must be a "
                "prime integer in [2, 2147483647], `entries` must have at "
                "most 256 rows, each row 1..256 columns, all rows the same "
                "length matching the declared `columns`, and every entry a "
                "canonical residue in [0, prime). Zero rows are permitted "
                "and carry the explicit `columns` axis."
            )
        }
    )

    matrix: PrimeFieldMatrix

    @model_validator(mode="before")
    @classmethod
    def require_bounded_prime(cls, data: Any) -> Any:
        data = canonicalize_json_containers(data)
        # Bound the characteristic BEFORE the nested canonical value is
        # constructed: PrimeFieldMatrix.__post_init__ runs the (expensive)
        # primality test, so an oversized prime must be rejected first.
        # Shared strict-JSON container canonicalization above keeps the
        # nested matrix entry rows in their declared tuple shape.
        if isinstance(data, dict):
            raw = data.get("matrix")
            if isinstance(raw, dict):
                prime = raw.get("prime")
            else:
                prime = getattr(raw, "prime", None)
            if isinstance(prime, int) and not 2 <= prime <= MAX_PRIME:
                raise _validation_error(
                    "request.prime_bound",
                    "field prime must lie in [2, "
                    f"{MAX_PRIME}] so validation work stays bounded",
                )
        return data

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.matrix.entries) > MAX_ROWS:
            raise _validation_error(
                "request.rows_bound", f"matrix has at most {MAX_ROWS} rows"
            )
        if not 1 <= self.matrix.columns <= MAX_COLUMNS:
            raise _validation_error(
                "request.columns_bound", f"matrix has at most {MAX_COLUMNS} columns"
            )
        return self


def _require_source_prime(self_prime: int, source: PrimeFieldMatrixRequest) -> None:
    if self_prime != source.matrix.prime:
        raise _validation_error(
            "result.source_prime", "result prime must equal the retained source prime"
        )


class PrimeFieldMatrixRankResult(StrictModel):
    """Rank of a matrix over GF(p), retaining its source matrix."""

    prime: int = Field(gt=1, le=MAX_PRIME)
    source: PrimeFieldMatrixRequest
    rank: int = Field(ge=0)

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        _require_source_prime(self.prime, self.source)
        if self.rank > min(len(self.source.matrix.entries), self.source.matrix.columns):
            raise _validation_error(
                "result.rank_bound",
                "rank cannot exceed either source matrix dimension",
            )
        return self

    @classmethod
    def _from_kernel(cls, request: PrimeFieldMatrixRequest, *, rank: int) -> Self:
        """Build a result after the admitted rank kernel established it."""

        return cls.model_construct(
            source=request, prime=request.matrix.prime, rank=rank
        )


class PrimeFieldRrefResult(StrictModel):
    """Reduced row-echelon form over GF(p) as the canonical matrix value.

    ``rref_matrix`` carries the exact reduced form with its pivot columns and
    rank; it feeds downstream GF(p) consumers unchanged. Parsing checks only
    its canonical shape; the producer kernel establishes reduced-form facts.
    """

    prime: int = Field(gt=1, le=MAX_PRIME)
    source: PrimeFieldMatrixRequest
    rref_matrix: PrimeFieldMatrix
    pivot_columns: tuple[int, ...]
    rank: int = Field(ge=0)

    @model_validator(mode="after")
    def require_rref_canonical(self) -> Self:
        _require_source_prime(self.prime, self.source)
        if (
            self.rref_matrix.prime != self.source.matrix.prime
            or self.rref_matrix.columns != self.source.matrix.columns
            or len(self.rref_matrix.entries) != len(self.source.matrix.entries)
        ):
            raise _validation_error(
                "result.rref_shape",
                "rref_matrix must carry the source shape and prime",
            )
        if any(
            column < 0 or column >= self.source.matrix.columns
            for column in self.pivot_columns
        ) or any(
            later <= earlier
            for earlier, later in zip(
                self.pivot_columns, self.pivot_columns[1:], strict=False
            )
        ):
            raise _validation_error(
                "result.pivot_columns",
                "pivot_columns must be a strictly increasing source-column sequence",
            )
        if self.rank != len(self.pivot_columns):
            raise _validation_error(
                "result.rank_pivots", "rank must equal the number of pivot columns"
            )
        if self.rank > min(len(self.source.matrix.entries), self.source.matrix.columns):
            raise _validation_error(
                "result.rank_bound",
                "rank cannot exceed either source matrix dimension",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: PrimeFieldMatrixRequest,
        *,
        rref_matrix: PrimeFieldMatrix,
        pivot_columns: tuple[int, ...],
    ) -> Self:
        """Build a result after the admitted RREF kernel established it."""

        return cls.model_construct(
            source=request,
            prime=request.matrix.prime,
            rref_matrix=rref_matrix,
            pivot_columns=pivot_columns,
            rank=len(pivot_columns),
        )


class PrimeFieldNullspaceResult(StrictModel):
    """Right nullspace basis over GF(p) as the canonical matrix value.

    An empty basis still carries the source ``columns`` axis inside
    ``nullspace_matrix`` so the ambient dimension of the zero subspace stays
    unambiguous and composable.
    """

    prime: int = Field(gt=1, le=MAX_PRIME)
    source: PrimeFieldMatrixRequest
    nullspace_matrix: PrimeFieldMatrix
    nullity: int = Field(ge=0)

    @model_validator(mode="after")
    def require_nullspace_canonical(self) -> Self:
        _require_source_prime(self.prime, self.source)
        if (
            self.nullspace_matrix.prime != self.source.matrix.prime
            or self.nullspace_matrix.columns != self.source.matrix.columns
        ):
            raise _validation_error(
                "result.nullspace_shape",
                "nullspace_matrix must carry the source prime and column axis",
            )
        if self.nullity != len(self.nullspace_matrix.entries):
            raise _validation_error(
                "result.nullity_basis", "nullity must equal the number of basis vectors"
            )
        if self.nullity > self.source.matrix.columns:
            raise _validation_error(
                "result.nullity_bound",
                "nullity cannot exceed the source column count",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: PrimeFieldMatrixRequest, *, nullspace_matrix: PrimeFieldMatrix
    ) -> Self:
        """Build a result after the admitted nullspace kernel established it."""

        return cls.model_construct(
            source=request,
            prime=request.matrix.prime,
            nullspace_matrix=nullspace_matrix,
            nullity=len(nullspace_matrix.entries),
        )


__all__ = [
    "MAX_COLUMNS",
    "MAX_PRIME",
    "MAX_ROWS",
    "PrimeFieldMatrixRankResult",
    "PrimeFieldMatrixRequest",
    "PrimeFieldNullspaceResult",
    "PrimeFieldRrefResult",
]
