"""Typed wire contracts for linear matroid operations over a finite field."""

from __future__ import annotations

from typing import Any, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix
from jacobian.math.matrices.finite_fields.linear_algebra import rank as prime_field_rank

MAX_GROUND_SIZE = 256
"""Schema-visible cap on the ground-set cardinality (matrix columns)."""

MAX_REPRESENTATION_ROWS = 256
"""Preserved row envelope for matroid representation and witness work."""

MAX_PRIME = 2_147_483_647
"""Explicit conservative bound on the field prime before primality testing."""


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by matroid contracts."""

    return PydanticCustomError(f"matroid.{reason}", message)


MAX_WEIGHT_DIGITS = 12
"""Schema-visible cap on decimal digits of one matroid weight entry."""


class LinearMatroid(StrictModel):
    """A linear matroid over GF(p) represented by a canonical matrix.

    The ground set ``{0, ..., columns - 1}`` indexes the columns of the
    domain-owned ``PrimeFieldMatrix``; rank and closure derive from the
    column span. The empty matroid is admitted: with zero columns the row
    axis stays declared, the rank is exactly zero, and every closure is
    empty. The characteristic is bounded before construction so no accepted
    value performs unbounded primality work.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A linear matroid over GF(p) as the canonical "
                "`PrimeFieldMatrix`: the ground set indexes matrix columns, "
                "entries are canonical residues in [0, prime), and up to 256 "
                "columns. The empty matroid (zero columns) is admitted."
            )
        }
    )

    matrix: PrimeFieldMatrix
    ground_labels: tuple[str, ...] | None = Field(
        default=None,
        description="Optional canonical labels for the matrix-column ground axis; omitted means positional labels.",
    )

    @model_validator(mode="before")
    @classmethod
    def require_bounded_declared_prime(cls, data: Any) -> Any:
        data = canonicalize_json_containers(data)
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
                    "field_prime.bound",
                    f"field prime must lie in [2, {MAX_PRIME}] so validation "
                    "work stays bounded",
                )
        return data

    @model_validator(mode="after")
    def require_bounded_ground_set(self) -> Self:
        if len(self.matrix.entries) > MAX_REPRESENTATION_ROWS:
            raise _validation_error(
                "representation_rows.bound",
                "matroid representation must have at most "
                f"{MAX_REPRESENTATION_ROWS} rows",
            )
        if self.matrix.columns > MAX_GROUND_SIZE:
            raise _validation_error(
                "ground_set.bound",
                f"ground set must hold at most {MAX_GROUND_SIZE} elements, "
                f"got {self.matrix.columns}",
            )
        return self

    @model_validator(mode="after")
    def require_ground_axis(self) -> Self:
        if self.ground_labels is not None and (
            len(self.ground_labels) != self.matrix.columns
            or len(set(self.ground_labels)) != len(self.ground_labels)
        ):
            raise _validation_error(
                "ground_axis",
                "ground labels must cover the matrix columns exactly once",
            )
        return self

    @property
    def ground_size(self) -> int:
        return self.matrix.columns

    @property
    def ground_axis(self) -> tuple[str, ...]:
        return self.ground_labels or tuple(str(i) for i in range(self.matrix.columns))


def validate_subset_indices(matroid: LinearMatroid, subset: Any) -> None:
    """Shared closure-subset admission: in-range and distinct indices.

    Used by both the wire request model and the native entry point so a
    direct kernel call can never admit indices the wire path rejects
    (negative indexing must not select columns).
    """
    for idx in subset:
        if not (0 <= idx < matroid.ground_size):
            raise ValueError("subset indices must be in 0..n-1")
    if len(set(subset)) != len(subset):
        raise ValueError("subset indices must be distinct")


class MatroidClosureRequest(StrictModel):
    """Compute the closure of a subset in a linear matroid."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute the closure of a distinct subset of the ground set of "
                "a bounded linear matroid. Ground-set elements are the columns "
                "of `matroid.matrix`, indexed from 0 through "
                "`matroid.matrix.columns - 1`."
            )
        }
    )

    matroid: LinearMatroid
    subset: tuple[StrictInt, ...] = Field(
        default=(),
        max_length=MAX_GROUND_SIZE,
        description=(
            "Distinct ground-set indices. Every index must lie in "
            "0..matroid.matrix.columns-1; at most "
            f"{MAX_GROUND_SIZE} indices are admitted."
        ),
        json_schema_extra={"uniqueItems": True},
    )


class MatroidClosureResult(MatroidClosureRequest):
    """The claimed closure (flat) of a subset in a linear matroid.

    Deserialization establishes only the retained request and bounded canonical
    result shape. Kernel output uses ``_from_kernel`` after its trusted bounded
    computation.
    """

    closure: tuple[StrictInt, ...] = Field(
        default=(),
        max_length=MAX_GROUND_SIZE,
        description=(
            "The complete flat spanned by `subset`, as distinct ground-set "
            "indices in increasing order."
        ),
    )
    rank: StrictInt = Field(
        ge=0,
        description="The exact rank of `subset` in the declared linear matroid.",
    )

    @model_validator(mode="after")
    def require_bounded_canonical_claim(self) -> Self:
        if self.closure != tuple(sorted(set(self.closure))):
            raise _validation_error(
                "closure.canonical",
                "closure indices must be distinct and in increasing order",
            )
        if any(not 0 <= index < self.matroid.ground_size for index in self.closure):
            raise _validation_error(
                "closure.indices",
                "closure indices must be in 0..matroid.matrix.columns-1",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matroid: LinearMatroid,
        subset: tuple[int, ...],
        closure: tuple[int, ...],
        rank: int,
    ) -> Self:
        """Construct trusted output of the owner-local closure kernel."""

        return cls.model_construct(
            matroid=matroid,
            subset=subset,
            closure=closure,
            rank=rank,
        )


class MaximumWeightBasisRequest(StrictModel):
    """Compute a maximum-weight basis of a bounded linear matroid.

    ``weights[i]`` is the exact integer weight of ground element ``i``; the
    tuple covers the matroid ground set exactly once in ground order.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A bounded linear matroid plus one exact integer weight per "
                "ground element in ground order. Negative and zero weights "
                "are admitted."
            )
        }
    )

    matroid: LinearMatroid
    weights: tuple[int, ...] = Field(
        max_length=MAX_GROUND_SIZE,
        description=(
            "One exact integer weight per ground element in ground order; "
            "length must equal matroid.matrix.columns."
        ),
    )

    @model_validator(mode="after")
    def require_ground_keyed_weights(self) -> Self:
        if len(self.weights) != self.matroid.ground_size:
            raise _validation_error(
                "weights.ground_coverage",
                "weights must cover the matroid ground set exactly once",
            )
        if any(type(weight) is not int for weight in self.weights):
            raise _validation_error(
                "weights.integer",
                "matroid weights must be exact integers",
            )
        if any(abs(weight) >= 10**MAX_WEIGHT_DIGITS for weight in self.weights):
            raise _validation_error(
                "weights.digits",
                "matroid weights must have fewer than "
                f"{MAX_WEIGHT_DIGITS} decimal digits",
            )
        return self


class ExchangeLedgerRow(StrictModel):
    """One fundamental-circuit exchange replay row for an outside element."""

    outside: StrictInt = Field(
        ge=0,
        description="Ground element outside the selected basis.",
    )
    circuit: tuple[StrictInt, ...] = Field(
        description=(
            "The fundamental circuit of `outside` with respect to the basis, "
            "as distinct ground indices in increasing order."
        ),
    )
    best_delta: StrictInt = Field(
        description=(
            "Maximum single-element exchange improvement "
            "weights[outside] - weights[replaced]; nonpositive at optimum."
        ),
    )


class MaximumWeightBasisResult(StrictModel):
    """One deterministic maximum-weight basis bound to its source weights.

    Deserialization establishes only the retained request and bounded
    canonical result shape. Kernel output uses ``_from_kernel`` after its
    trusted bounded computation.
    """

    matroid: LinearMatroid
    weights: tuple[int, ...] = Field(max_length=MAX_GROUND_SIZE)
    basis: tuple[StrictInt, ...] = Field(
        max_length=MAX_GROUND_SIZE,
        description="Selected basis as distinct ground indices in increasing order.",
    )
    total_weight: StrictInt = Field(
        description="Exact sum of the selected basis weights, derived from `basis`."
    )
    rank: StrictInt = Field(
        ge=0,
        description="Exact rank of the matroid, equal to the basis cardinality.",
    )
    greedy_order: tuple[StrictInt, ...] = Field(
        max_length=MAX_GROUND_SIZE,
        description=(
            "Deterministic greedy consideration order: ground elements sorted "
            "by decreasing weight with ties broken by increasing index."
        ),
    )
    exchange_ledger: tuple[ExchangeLedgerRow, ...] = Field(
        max_length=MAX_GROUND_SIZE,
        description=(
            "One fundamental-circuit row per ground element outside the basis."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_claim(self) -> Self:
        if len(self.weights) != self.matroid.ground_size:
            raise _validation_error(
                "weights.ground_coverage",
                "weights must cover the matroid ground set exactly once",
            )
        if self.basis != tuple(sorted(set(self.basis))):
            raise _validation_error(
                "basis.canonical",
                "basis indices must be distinct and in increasing order",
            )
        if any(not 0 <= index < self.matroid.ground_size for index in self.basis):
            raise _validation_error(
                "basis.indices",
                "basis indices must be in 0..matroid.matrix.columns-1",
            )
        if sorted(self.greedy_order) != list(range(self.matroid.ground_size)):
            raise _validation_error(
                "greedy_order.coverage",
                "greedy order must cover the ground set exactly once",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matroid: LinearMatroid,
        weights: tuple[int, ...],
        basis: tuple[int, ...],
        total_weight: int,
        rank: int,
        greedy_order: tuple[int, ...],
        exchange_ledger: tuple[ExchangeLedgerRow, ...],
    ) -> Self:
        """Construct trusted output of the owner-local greedy kernel."""

        return cls.model_construct(
            matroid=matroid,
            weights=weights,
            basis=basis,
            total_weight=total_weight,
            rank=rank,
            greedy_order=greedy_order,
            exchange_ledger=exchange_ledger,
        )


class MatroidIntersectionRequest(StrictModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute a maximum common independent set for two linear "
                "matroids on one labelled ground. The exact exchange kernel "
                "admits at most 256 ground elements and a derived "
                "50,000,000-unit bound covering rank probes, min-max witness "
                "work, representation rows, and O(n) result materialization "
                "for the returned independent set and rank partition."
            ),
            "admission_limits": {
                "max_ground_elements": 256,
                "max_work_units": 50_000_000,
                "work_includes": [
                    "exchange rank probes",
                    "representation rows",
                    "min-max witness ranks",
                    "result indices",
                ],
            },
        }
    )

    first: LinearMatroid
    second: LinearMatroid


class MatroidIntersectionWitness(StrictModel):
    subset: tuple[StrictInt, ...]
    rank_first: StrictInt = Field(ge=0)
    rank_second_complement: StrictInt = Field(ge=0)
    equality: StrictInt = Field(ge=0)


class MatroidIntersectionResult(StrictModel):
    first: LinearMatroid
    second: LinearMatroid
    common_independent: tuple[StrictInt, ...]
    cardinality: StrictInt = Field(ge=0)
    witness: MatroidIntersectionWitness

    @staticmethod
    def _subset_rank(matroid: LinearMatroid, subset: tuple[int, ...]) -> int:
        entries = tuple(
            tuple(row[index] for index in subset) for row in matroid.matrix.entries
        )
        matrix = PrimeFieldMatrix(
            prime=matroid.matrix.prime,
            entries=entries,
            columns=len(subset),
        )
        return prime_field_rank(matrix) if subset else 0

    @model_validator(mode="after")
    def require_witness_shape(self) -> Self:
        if (
            self.first.matrix.prime != self.second.matrix.prime
            or self.first.ground_axis != self.second.ground_axis
        ):
            raise _validation_error(
                "intersection_ground",
                "intersection result operands must share one labelled ground and field",
            )
        n = self.first.ground_size
        common = self.common_independent
        if common != tuple(sorted(set(common))) or any(
            index < 0 or index >= n for index in common
        ):
            raise _validation_error(
                "intersection_common_axis",
                "common independent indices must be sorted, distinct, and in range",
            )
        if self._subset_rank(self.first, common) != len(common) or self._subset_rank(
            self.second, common
        ) != len(common):
            raise _validation_error(
                "intersection_common_independence",
                "common_independent must be independent in both matroids",
            )
        if self.cardinality != len(common):
            raise _validation_error(
                "intersection_cardinality",
                "cardinality must equal the common independent set size",
            )
        witness_subset = self.witness.subset
        if witness_subset != tuple(sorted(set(witness_subset))) or any(
            index < 0 or index >= n for index in witness_subset
        ):
            raise _validation_error(
                "intersection_witness_axis",
                "witness subset must be sorted, distinct, and in range",
            )
        complement = tuple(index for index in range(n) if index not in witness_subset)
        rank_first = self._subset_rank(self.first, witness_subset)
        rank_second_complement = self._subset_rank(self.second, complement)
        if (
            self.witness.rank_first != rank_first
            or self.witness.rank_second_complement != rank_second_complement
        ):
            raise _validation_error(
                "intersection_witness_binding",
                "witness ranks must match the retained matroid subset axes",
            )
        if self.witness.equality != rank_first + rank_second_complement:
            raise _validation_error(
                "intersection_minmax",
                "min-max witness equality must match its two ranks",
            )
        if self.witness.equality != self.cardinality:
            raise _validation_error(
                "intersection_minmax",
                "min-max witness must equal the intersection cardinality",
            )
        return self


__all__ = [
    "ExchangeLedgerRow",
    "LinearMatroid",
    "MatroidClosureRequest",
    "MatroidClosureResult",
    "MaximumWeightBasisRequest",
    "MaximumWeightBasisResult",
    "validate_subset_indices",
]
