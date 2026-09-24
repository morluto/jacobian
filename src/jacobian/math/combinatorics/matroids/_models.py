"""Typed wire contracts for linear matroid operations over a finite field."""

from __future__ import annotations

from itertools import pairwise
from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix

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

MAX_WEIGHTED_INTERSECTION_OPT_DUAL_DIGITS = 1024
"""Maximum decimal digits admitted for private integral split intermediates."""


class GraphicMatroidRequest(StrictModel):
    """Construct the GF(2) incidence representation of a simple graph."""

    graph: SimpleUndirectedGraph


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


class MatroidWeightFunction(StrictModel):
    """Exact integer weights keyed by an explicit canonical ground axis."""

    ground_axis: tuple[str, ...] = Field(max_length=MAX_GROUND_SIZE)
    values: tuple[StrictInt, ...] = Field(max_length=MAX_GROUND_SIZE)

    @model_validator(mode="after")
    def require_ground_coverage(self) -> Self:
        if len(self.ground_axis) != len(self.values) or len(
            set(self.ground_axis)
        ) != len(self.ground_axis):
            raise _validation_error(
                "weights.ground_axis",
                "weight values must cover one unique ground axis exactly once",
            )
        if any(abs(value) >= 10**MAX_WEIGHT_DIGITS for value in self.values):
            raise _validation_error(
                "weights.digits",
                f"weights must have fewer than {MAX_WEIGHT_DIGITS} decimal digits",
            )
        return self


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

    The weight function is keyed by the exact matroid ground axis. Its row
    order is immaterial; execution canonicalizes values to the matrix-column
    order before optimization.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A bounded linear matroid plus an exact integer weight "
                "function keyed by its ground axis. Negative and zero weights "
                "are admitted; weight row order is immaterial."
            )
        }
    )

    matroid: LinearMatroid
    weight_function: MatroidWeightFunction

    @model_validator(mode="after")
    def require_ground_keyed_weights(self) -> Self:
        if set(self.weight_function.ground_axis) != set(self.matroid.ground_axis):
            raise _validation_error(
                "weights.ground_coverage",
                "weight keys must equal the exact matroid ground axis",
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
    weight_function: MatroidWeightFunction
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
        if set(self.weight_function.ground_axis) != set(self.matroid.ground_axis):
            raise _validation_error(
                "weights.ground_coverage",
                "weight keys must equal the exact matroid ground axis",
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
        weight_function: MatroidWeightFunction,
        basis: tuple[int, ...],
        total_weight: int,
        rank: int,
        greedy_order: tuple[int, ...],
        exchange_ledger: tuple[ExchangeLedgerRow, ...],
    ) -> Self:
        """Construct trusted output of the owner-local greedy kernel."""

        return cls.model_construct(
            matroid=matroid,
            weight_function=weight_function,
            basis=basis,
            total_weight=total_weight,
            rank=rank,
            greedy_order=greedy_order,
            exchange_ledger=exchange_ledger,
        )


class MaximumWeightIndependentSetRequest(StrictModel):
    """Compute a maximum-weight independent set, allowing empty output."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A bounded linear matroid and a domain-owned exact integer "
                "weight function whose explicit ground axis equals the "
                "matroid axis. Only positive-weight elements are considered; "
                "zero and negative weights are excluded from the greedy scan."
            )
        }
    )

    matroid: LinearMatroid
    weight_function: MatroidWeightFunction

    @model_validator(mode="after")
    def require_ground_keyed_weights(self) -> Self:
        if self.weight_function.ground_axis != self.matroid.ground_axis:
            raise _validation_error(
                "weights.ground_coverage",
                "weight keys must equal the exact matroid ground axis",
            )
        return self


class MaximumWeightIndependentSetResult(StrictModel):
    """Canonical selected set and exact objective, bound to source weights."""

    matroid: LinearMatroid
    weight_function: MatroidWeightFunction
    independent_set: tuple[StrictInt, ...] = Field(
        max_length=MAX_GROUND_SIZE,
        description="The selected independent ground subset in increasing index order.",
    )
    total_weight: StrictInt = Field(description="Exact sum of selected weights.")
    rank: StrictInt = Field(ge=0, description="Exact rank of the selected set.")
    greedy_order: tuple[StrictInt, ...] = Field(
        max_length=MAX_GROUND_SIZE,
        description=(
            "Positive-weight ground elements in decreasing weight order, ties "
            "by increasing index. Nonpositive elements are omitted."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_claim(self) -> Self:
        if self.weight_function.ground_axis != self.matroid.ground_axis:
            raise _validation_error(
                "weights.ground_coverage",
                "weight keys must equal the exact matroid ground axis",
            )
        if self.independent_set != tuple(sorted(set(self.independent_set))):
            raise _validation_error(
                "independent_set.canonical",
                "selected indices must be distinct and in increasing order",
            )
        if any(not 0 <= i < self.matroid.ground_size for i in self.independent_set):
            raise _validation_error(
                "independent_set.indices",
                "selected indices must lie in the matroid ground set",
            )
        if self.rank > len(self.independent_set):
            raise _validation_error(
                "independent_set.rank", "selected-set rank cannot exceed its size"
            )
        expected = tuple(
            sorted(
                (
                    i
                    for i, weight in enumerate(self.weight_function.values)
                    if weight > 0
                ),
                key=lambda i: (-self.weight_function.values[i], i),
            )
        )
        if self.greedy_order != expected:
            raise _validation_error(
                "greedy_order", "order must cover exactly positive weights canonically"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matroid: LinearMatroid,
        weight_function: MatroidWeightFunction,
        independent_set: tuple[int, ...],
        total_weight: int,
        rank: int,
        greedy_order: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            matroid=matroid,
            weight_function=weight_function,
            independent_set=independent_set,
            total_weight=total_weight,
            rank=rank,
            greedy_order=greedy_order,
        )


class MatroidWeightedIntersectionCertificateRequest(StrictModel):
    """Check a supplied integral weight-splitting certificate for a candidate."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Check whether a caller-supplied common independent set is "
                "maximum-weight. The caller supplies an exact integer split "
                "w=u+v; each source matroid's maximum-weight independent-set "
                "value for its split is recomputed by the bounded greedy "
                "rank kernel. This operation checks a certificate and does "
                "not search for an optimum or produce a split."
            ),
            "admission_limits": {
                "max_ground_elements": MAX_GROUND_SIZE,
                "max_representation_rows": MAX_REPRESENTATION_ROWS,
                "max_weight_digits": MAX_WEIGHT_DIGITS,
                "max_aggregate_rank_work": 50_000_000,
                "max_result_bytes": 8 * 1024 * 1024,
                "work_includes": [
                    "both split-weight greedy scans",
                    "both candidate feasibility ranks",
                    "source-bound certificate result serialization",
                ],
            },
        }
    )

    first: LinearMatroid
    second: LinearMatroid
    weight_function: MatroidWeightFunction
    common_independent: tuple[StrictInt, ...] = Field(max_length=MAX_GROUND_SIZE)
    first_split: MatroidWeightFunction
    second_split: MatroidWeightFunction

    @model_validator(mode="after")
    def require_certificate_axes(self) -> Self:
        if (
            self.first.matrix.prime != self.second.matrix.prime
            or self.first.ground_axis != self.second.ground_axis
        ):
            raise _validation_error(
                "weighted_intersection.ground",
                "source matroids must share one labelled ground and field",
            )
        axis = self.first.ground_axis
        if any(
            value.ground_axis != axis
            for value in (self.weight_function, self.first_split, self.second_split)
        ):
            raise _validation_error(
                "weighted_intersection.weight_axis",
                "objective and split weights must use the exact source ground axis",
            )
        if self.common_independent != tuple(
            sorted(set(self.common_independent))
        ) or any(
            not 0 <= index < self.first.ground_size for index in self.common_independent
        ):
            raise _validation_error(
                "weighted_intersection.common_set",
                "candidate indices must be sorted, distinct, and in range",
            )
        return self


class MatroidWeightedIntersectionOptimizationRequest(StrictModel):
    """Compute a maximum-weight common independent set of two matroids."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute one exact maximum-weight common independent set of "
                "two linear matroids over the same prime field and labelled "
                "ground axis. The input weights are exact integers."
            ),
            "admission_limits": {
                "max_ground_elements": MAX_GROUND_SIZE,
                "max_representation_rows": MAX_REPRESENTATION_ROWS,
                "max_weight_digits": MAX_WEIGHT_DIGITS,
                "max_dual_intermediate_digits": MAX_WEIGHTED_INTERSECTION_OPT_DUAL_DIGITS,
                "max_aggregate_exact_work": 50_000_000,
                "max_result_bytes": 8 * 1024 * 1024,
                "work_includes": [
                    "rank-oracle exchange-circuit construction",
                    "reachable-set, tight-edge, and dual-slack scans",
                    "integral weight-split intermediate arithmetic",
                    "one bounded source-field primality check",
                    "selected-matrix copies and residue validation",
                    "final common-set feasibility ranks",
                    "source-bound result serialization",
                ],
            },
        }
    )

    first: LinearMatroid
    second: LinearMatroid
    weight_function: MatroidWeightFunction

    @model_validator(mode="after")
    def require_optimizer_axes(self) -> Self:
        if (
            self.first.matrix.prime != self.second.matrix.prime
            or self.first.ground_axis != self.second.ground_axis
            or self.weight_function.ground_axis != self.first.ground_axis
        ):
            raise _validation_error(
                "weighted_intersection.ground",
                "sources and objective must share one labelled ground and field",
            )
        return self


class MatroidWeightedIntersectionOptimizationResult(StrictModel):
    """One exact source-bound maximum-weight common independent set."""

    first: LinearMatroid
    second: LinearMatroid
    weight_function: MatroidWeightFunction
    common_independent: tuple[StrictInt, ...] = Field(max_length=MAX_GROUND_SIZE)
    total_weight: StrictInt

    @model_validator(mode="after")
    def require_result_context(self) -> Self:
        request = MatroidWeightedIntersectionOptimizationRequest(
            first=self.first,
            second=self.second,
            weight_function=self.weight_function,
        )
        n = request.first.ground_size
        if self.common_independent != tuple(
            sorted(set(self.common_independent))
        ) or any(not 0 <= index < n for index in self.common_independent):
            raise _validation_error(
                "weighted_intersection.common_set",
                "candidate indices must be sorted, distinct, and in range",
            )
        expected_weight = sum(
            self.weight_function.values[index] for index in self.common_independent
        )
        if self.total_weight != expected_weight:
            raise _validation_error(
                "weighted_intersection.objective",
                "candidate total must equal its exact source weight",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        request: MatroidWeightedIntersectionOptimizationRequest,
        common_independent: tuple[int, ...],
        total_weight: int,
    ) -> Self:
        return cls.model_construct(
            first=request.first,
            second=request.second,
            weight_function=request.weight_function,
            common_independent=common_independent,
            total_weight=total_weight,
        )


class MatroidRankMultiplier(StrictModel):
    """A positive integer multiplier on one matroid rank inequality."""

    subset: tuple[StrictInt, ...] = Field(
        max_length=MAX_GROUND_SIZE,
        description="A nonempty, sorted subset in the source ground axis.",
    )
    multiplier: StrictInt = Field(
        gt=0,
        description="Positive integer coefficient; terms with zero coefficient are omitted.",
    )

    @model_validator(mode="after")
    def require_canonical_subset(self) -> Self:
        if self.subset != tuple(sorted(set(self.subset))):
            raise _validation_error(
                "rank_dual.subset",
                "rank multiplier subsets must be sorted and contain no duplicates",
            )
        return self


class MatroidWeightedIntersectionRankCertificateRequest(StrictModel):
    """Check rank-inequality dual multipliers for a common independent set."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Check a supplied primal common independent set and sparse exact "
                "dual multipliers on the two matroids' rank inequalities. The "
                "operation recomputes only the listed exact ranks, verifies the "
                "elementwise dual cover and equality of primal and dual values, "
                "and does not search for an optimum or construct the multipliers."
            ),
            "admission_limits": {
                "max_ground_elements": MAX_GROUND_SIZE,
                "max_representation_rows": MAX_REPRESENTATION_ROWS,
                "max_weight_digits": MAX_WEIGHT_DIGITS,
                "max_total_rank_terms": 2 * MAX_GROUND_SIZE,
                "max_aggregate_rank_work": 50_000_000,
                "max_result_bytes": 8 * 1024 * 1024,
                "work_includes": [
                    "both candidate feasibility ranks",
                    "each supplied rank-inequality rank",
                    "elementwise dual cover and objective checks",
                    "source-bound certificate result serialization",
                ],
            },
        }
    )

    first: LinearMatroid
    second: LinearMatroid
    weight_function: MatroidWeightFunction
    common_independent: tuple[StrictInt, ...] = Field(max_length=MAX_GROUND_SIZE)
    first_rank_terms: tuple[MatroidRankMultiplier, ...] = Field(
        max_length=MAX_GROUND_SIZE
    )
    second_rank_terms: tuple[MatroidRankMultiplier, ...] = Field(
        max_length=MAX_GROUND_SIZE
    )

    @model_validator(mode="after")
    def require_certificate_axes(self) -> Self:
        if (
            self.first.matrix.prime != self.second.matrix.prime
            or self.first.ground_axis != self.second.ground_axis
            or self.weight_function.ground_axis != self.first.ground_axis
        ):
            raise _validation_error(
                "rank_dual.ground",
                "sources and objective must share one labelled ground and field",
            )
        n = self.first.ground_size
        if self.common_independent != tuple(
            sorted(set(self.common_independent))
        ) or any(not 0 <= index < n for index in self.common_independent):
            raise _validation_error(
                "rank_dual.common_set",
                "candidate indices must be sorted, distinct, and in range",
            )
        terms = self.first_rank_terms + self.second_rank_terms
        if len(terms) > 2 * n:
            raise _validation_error(
                "rank_dual.term_count",
                "the two rank-multiplier families may contain at most twice the ground size",
            )
        coefficient_bound = max(1, n) * 10**MAX_WEIGHT_DIGITS
        if any(term.multiplier >= coefficient_bound for term in terms):
            raise _validation_error(
                "rank_dual.multiplier_bound",
                "rank multipliers exceed the coefficient bound derived from the ground and weight limits",
            )
        for family in (self.first_rank_terms, self.second_rank_terms):
            subsets = tuple(term.subset for term in family)
            ordered_subsets = tuple(
                sorted(set(subsets), key=lambda item: (len(item), item))
            )
            if subsets != ordered_subsets:
                raise _validation_error(
                    "rank_dual.term_order",
                    "each rank-multiplier family must be ordered by subset size and lexicographic indices",
                )
            if any(
                not term.subset or any(not 0 <= index < n for index in term.subset)
                for term in family
            ):
                raise _validation_error(
                    "rank_dual.subset_range",
                    "rank multiplier subsets must be nonempty and within the source ground",
                )
            if any(not set(left).issubset(right) for left, right in pairwise(subsets)):
                raise _validation_error(
                    "rank_dual.chain",
                    "each source rank-multiplier family must be a nested chain",
                )
        return self


class MatroidWeightedIntersectionRankCertificateResult(StrictModel):
    """A source-bound common-independent optimum with its rank-dual witness."""

    first: LinearMatroid
    second: LinearMatroid
    weight_function: MatroidWeightFunction
    common_independent: tuple[StrictInt, ...] = Field(max_length=MAX_GROUND_SIZE)
    first_rank_terms: tuple[MatroidRankMultiplier, ...] = Field(
        max_length=MAX_GROUND_SIZE
    )
    second_rank_terms: tuple[MatroidRankMultiplier, ...] = Field(
        max_length=MAX_GROUND_SIZE
    )
    total_weight: StrictInt
    first_split: MatroidWeightFunction
    second_split: MatroidWeightFunction

    @model_validator(mode="after")
    def require_result_context(self) -> Self:
        request = MatroidWeightedIntersectionRankCertificateRequest(
            first=self.first,
            second=self.second,
            weight_function=self.weight_function,
            common_independent=self.common_independent,
            first_rank_terms=self.first_rank_terms,
            second_rank_terms=self.second_rank_terms,
        )
        if self.total_weight != sum(
            self.weight_function.values[index] for index in self.common_independent
        ):
            raise _validation_error(
                "rank_dual.objective",
                "candidate total must equal its exact source weight",
            )
        if (
            self.first_split.ground_axis != request.first.ground_axis
            or self.second_split.ground_axis != request.first.ground_axis
            or tuple(
                a + b
                for a, b in zip(
                    self.first_split.values,
                    self.second_split.values,
                    strict=True,
                )
            )
            != self.weight_function.values
        ):
            raise _validation_error(
                "rank_dual.split",
                "derived integral split must use the source ground and sum to the objective",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        request: MatroidWeightedIntersectionRankCertificateRequest,
        total_weight: int,
        first_split: MatroidWeightFunction,
        second_split: MatroidWeightFunction,
    ) -> Self:
        return cls.model_construct(
            first=request.first,
            second=request.second,
            weight_function=request.weight_function,
            common_independent=request.common_independent,
            first_rank_terms=request.first_rank_terms,
            second_rank_terms=request.second_rank_terms,
            total_weight=total_weight,
            first_split=first_split,
            second_split=second_split,
        )


class MatroidWeightedIntersectionResult(StrictModel):
    """A source-bound optimum with a checked integral weight-splitting witness."""

    weight_function: MatroidWeightFunction
    common_independent: tuple[StrictInt, ...] = Field(max_length=MAX_GROUND_SIZE)
    total_weight: StrictInt
    first_maximizer: MaximumWeightIndependentSetResult
    second_maximizer: MaximumWeightIndependentSetResult

    @property
    def first(self) -> LinearMatroid:
        return self.first_maximizer.matroid

    @property
    def second(self) -> LinearMatroid:
        return self.second_maximizer.matroid

    @model_validator(mode="after")
    def require_canonical_certificate_shape(self) -> Self:
        first = self.first_maximizer.matroid
        second = self.second_maximizer.matroid
        n = first.ground_size
        if (
            first.matrix.prime != second.matrix.prime
            or first.ground_axis != second.ground_axis
            or self.weight_function.ground_axis != first.ground_axis
        ):
            raise _validation_error(
                "weighted_intersection.ground",
                "result sources and objective must share one labelled ground and field",
            )
        if self.common_independent != tuple(
            sorted(set(self.common_independent))
        ) or any(not 0 <= index < n for index in self.common_independent):
            raise _validation_error(
                "weighted_intersection.common_set",
                "candidate indices must be sorted, distinct, and in range",
            )
        expected_weight = sum(
            self.weight_function.values[index] for index in self.common_independent
        )
        if self.total_weight != expected_weight:
            raise _validation_error(
                "weighted_intersection.objective",
                "candidate total must equal its exact source weight",
            )
        if (
            self.first_maximizer.weight_function.ground_axis != first.ground_axis
            or self.second_maximizer.weight_function.ground_axis != second.ground_axis
            or tuple(
                a + b
                for a, b in zip(
                    self.first_maximizer.weight_function.values,
                    self.second_maximizer.weight_function.values,
                    strict=True,
                )
            )
            != self.weight_function.values
        ):
            raise _validation_error(
                "weighted_intersection.split",
                "retained integral split must sum exactly to the objective weights",
            )
        if (
            self.first_maximizer.total_weight + self.second_maximizer.total_weight
            != self.total_weight
        ):
            raise _validation_error(
                "weighted_intersection.optimality",
                "the two exact source maxima must sum to the candidate weight",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        weight_function: MatroidWeightFunction,
        common_independent: tuple[int, ...],
        total_weight: int,
        first_maximizer: MaximumWeightIndependentSetResult,
        second_maximizer: MaximumWeightIndependentSetResult,
    ) -> Self:
        return cls.model_construct(
            weight_function=weight_function,
            common_independent=common_independent,
            total_weight=total_weight,
            first_maximizer=first_maximizer,
            second_maximizer=second_maximizer,
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
                    "both final common-set feasibility ranks",
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
    rank_first_common: StrictInt = Field(
        ge=0,
        description="Exact source rank of common_independent in first.",
    )
    rank_second_common: StrictInt = Field(
        ge=0,
        description="Exact source rank of common_independent in second.",
    )
    witness: MatroidIntersectionWitness

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
        if self.cardinality != len(common):
            raise _validation_error(
                "intersection_cardinality",
                "cardinality must equal the common independent set size",
            )
        if (
            self.rank_first_common != self.cardinality
            or self.rank_second_common != self.cardinality
        ):
            raise _validation_error(
                "intersection_common_ranks",
                "both source matroid ranks of the common set must equal its cardinality",
            )
        witness_subset = self.witness.subset
        if witness_subset != tuple(sorted(set(witness_subset))) or any(
            index < 0 or index >= n for index in witness_subset
        ):
            raise _validation_error(
                "intersection_witness_axis",
                "witness subset must be sorted, distinct, and in range",
            )
        complement_size = n - len(witness_subset)
        if self.witness.rank_first > min(
            len(self.first.matrix.entries), len(witness_subset)
        ) or self.witness.rank_second_complement > min(
            len(self.second.matrix.entries), complement_size
        ):
            raise _validation_error(
                "intersection_witness_rank_bounds",
                "witness ranks must fit their retained matrix and subset axes",
            )
        if self.witness.equality != (
            self.witness.rank_first + self.witness.rank_second_complement
        ):
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

    @classmethod
    def _from_kernel(
        cls,
        *,
        first: LinearMatroid,
        second: LinearMatroid,
        common_independent: tuple[int, ...],
        cardinality: int,
        rank_first_common: int,
        rank_second_common: int,
        witness: MatroidIntersectionWitness,
    ) -> Self:
        """Construct a kernel-proved result without replaying computed ranks."""

        return cls.model_construct(
            first=first,
            second=second,
            common_independent=common_independent,
            cardinality=cardinality,
            rank_first_common=rank_first_common,
            rank_second_common=rank_second_common,
            witness=witness,
        )


class MatroidCommonBasisRequest(StrictModel):
    """Determine whether two represented matroids share a common basis."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute a maximum common independent set and exact source "
                "ranks. Return COMMON_BASIS only when the set is a basis of "
                "both sources; otherwise return NO_COMMON_BASIS with an "
                "explicit rank-mismatch or maximum-intersection deficit reason."
            ),
            "admission_limits": {
                "max_ground_elements": 256,
                "max_work_units": 50_000_000,
                "work_includes": [
                    "maximum-intersection exchange probes",
                    "common-set and min-max rank replays",
                    "both source rank computations",
                    "result materialization",
                ],
            },
        }
    )

    first: LinearMatroid
    second: LinearMatroid


class MatroidCommonBasisResult(StrictModel):
    """Closed common-basis result retaining its exact intersection witness."""

    first: LinearMatroid
    second: LinearMatroid
    common_independent: tuple[StrictInt, ...] = Field(max_length=MAX_GROUND_SIZE)
    cardinality: StrictInt = Field(ge=0)
    rank_first_common: StrictInt = Field(ge=0)
    rank_second_common: StrictInt = Field(ge=0)
    witness: MatroidIntersectionWitness
    rank_first: StrictInt = Field(ge=0)
    rank_second: StrictInt = Field(ge=0)
    status: Literal["COMMON_BASIS", "NO_COMMON_BASIS"]
    reason: Literal[
        "COMMON_BASIS",
        "SOURCE_RANK_MISMATCH",
        "MAXIMUM_COMMON_INDEPENDENT_SET_TOO_SMALL",
    ]
    common_basis: tuple[StrictInt, ...] | None = Field(max_length=MAX_GROUND_SIZE)

    @model_validator(mode="after")
    def require_exact_closed_claim(self) -> Self:
        n = self.first.ground_size
        if (
            self.first.matrix.prime != self.second.matrix.prime
            or self.first.ground_axis != self.second.ground_axis
        ):
            raise _validation_error(
                "common_basis.ground",
                "source matroids must share one labelled ground and field",
            )
        if self.common_independent != tuple(
            sorted(set(self.common_independent))
        ) or any(not 0 <= i < n for i in self.common_independent):
            raise _validation_error(
                "common_basis.common_set",
                "maximum common independent set must be sorted, distinct, and in range",
            )
        if self.cardinality != len(self.common_independent):
            raise _validation_error(
                "common_basis.cardinality", "cardinality must equal the set size"
            )
        if (
            self.rank_first_common != self.cardinality
            or self.rank_second_common != self.cardinality
        ):
            raise _validation_error(
                "common_basis.feasibility",
                "both source ranks of the maximum common set must equal its cardinality",
            )
        if self.rank_first > min(
            len(self.first.matrix.entries), n
        ) or self.rank_second > min(len(self.second.matrix.entries), n):
            raise _validation_error(
                "common_basis.rank_bounds", "source ranks exceed representation bounds"
            )
        if self.witness.subset != tuple(sorted(set(self.witness.subset))) or any(
            not 0 <= i < n for i in self.witness.subset
        ):
            raise _validation_error(
                "common_basis.witness_axis",
                "min-max witness subset must be sorted, distinct, and in range",
            )
        complement_size = n - len(self.witness.subset)
        if self.witness.rank_first > min(
            len(self.first.matrix.entries), len(self.witness.subset)
        ) or self.witness.rank_second_complement > min(
            len(self.second.matrix.entries), complement_size
        ):
            raise _validation_error(
                "common_basis.witness_rank_bounds",
                "witness ranks exceed their retained matrix and subset axes",
            )
        if (
            self.witness.equality != self.cardinality
            or self.witness.equality
            != self.witness.rank_first + self.witness.rank_second_complement
        ):
            raise _validation_error(
                "common_basis.minmax",
                "min-max rank sum must equal the exact maximum cardinality",
            )
        expected_status = (
            "COMMON_BASIS"
            if self.rank_first == self.rank_second == self.cardinality
            else "NO_COMMON_BASIS"
        )
        expected_reason = (
            "COMMON_BASIS"
            if expected_status == "COMMON_BASIS"
            else "SOURCE_RANK_MISMATCH"
            if self.rank_first != self.rank_second
            else "MAXIMUM_COMMON_INDEPENDENT_SET_TOO_SMALL"
        )
        expected_basis = (
            self.common_independent if expected_status == "COMMON_BASIS" else None
        )
        if (
            self.status != expected_status
            or self.reason != expected_reason
            or self.common_basis != expected_basis
        ):
            raise _validation_error(
                "common_basis.outcome",
                "outcome, reason, and common basis must follow exact source ranks",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        intersection: MatroidIntersectionResult,
        rank_first: int,
        rank_second: int,
    ) -> Self:
        if rank_first != rank_second:
            status = "NO_COMMON_BASIS"
            reason = "SOURCE_RANK_MISMATCH"
            basis = None
        elif intersection.cardinality != rank_first:
            status = "NO_COMMON_BASIS"
            reason = "MAXIMUM_COMMON_INDEPENDENT_SET_TOO_SMALL"
            basis = None
        else:
            status = "COMMON_BASIS"
            reason = "COMMON_BASIS"
            basis = intersection.common_independent
        return cls.model_construct(
            first=intersection.first,
            second=intersection.second,
            common_independent=intersection.common_independent,
            cardinality=intersection.cardinality,
            rank_first_common=intersection.rank_first_common,
            rank_second_common=intersection.rank_second_common,
            witness=intersection.witness,
            rank_first=rank_first,
            rank_second=rank_second,
            status=status,
            reason=reason,
            common_basis=basis,
        )


__all__ = [
    "ExchangeLedgerRow",
    "LinearMatroid",
    "MatroidClosureRequest",
    "MatroidClosureResult",
    "MatroidCommonBasisRequest",
    "MatroidCommonBasisResult",
    "MatroidRankMultiplier",
    "MatroidWeightFunction",
    "MatroidWeightedIntersectionCertificateRequest",
    "MatroidWeightedIntersectionOptimizationRequest",
    "MatroidWeightedIntersectionOptimizationResult",
    "MatroidWeightedIntersectionRankCertificateRequest",
    "MatroidWeightedIntersectionRankCertificateResult",
    "MatroidWeightedIntersectionResult",
    "MaximumWeightBasisRequest",
    "MaximumWeightBasisResult",
    "MaximumWeightIndependentSetRequest",
    "MaximumWeightIndependentSetResult",
    "validate_subset_indices",
]
