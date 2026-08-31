"""Canonical values for clause-constrained rational-flat classification."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import CanonicalizationError, CanonicalLimits
from jacobian.math._labels import OpaqueLabel
from jacobian.math.matrices.values import (
    RationalVectorSpaceBasis,
    SparseRationalMatrix,
)

MAX_RATIONAL_FLAT_AMBIENT_DIMENSION = 16
MAX_RATIONAL_FLAT_CANDIDATES = 128
MAX_RATIONAL_FLAT_FORBIDDEN_VECTORS = 128
MAX_RATIONAL_FLAT_CLAUSES = 128
MAX_RATIONAL_FLAT_CLAUSE_MEMBERSHIPS = 4_096
MAX_RATIONAL_FLAT_SYMMETRY_GENERATORS = 16
MAX_RATIONAL_FLAT_GROUP_ORDER = 10_000
MAX_RATIONAL_FLAT_RESULT_ORBITS = 100_000
MAX_RATIONAL_FLAT_RESULT_BYTES = CanonicalLimits().max_output_bytes
MAX_RATIONAL_FLAT_INPUT_COMPONENT_DIGITS = 256
MAX_RATIONAL_FLAT_MATRIX_NONZEROS = (
    MAX_RATIONAL_FLAT_CANDIDATES * MAX_RATIONAL_FLAT_AMBIENT_DIMENSION
)

CandidateIndex = Annotated[
    StrictInt,
    Field(ge=0, lt=MAX_RATIONAL_FLAT_CANDIDATES),
]
CandidateClause = Annotated[
    tuple[CandidateIndex, ...],
    Field(max_length=MAX_RATIONAL_FLAT_CANDIDATES),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"rational_flat.{reason}", message)


def _raw_component_exceeds_input_digits(value: object) -> bool:
    if isinstance(value, str):
        # A valid negative decimal has one sign plus the bounded digit body.
        # Check the total length before stripping signs so repeated leading
        # minus characters cannot evade the cheap raw-input bound.
        allowed_length = MAX_RATIONAL_FLAT_INPUT_COMPONENT_DIGITS + int(
            value.startswith("-")
        )
        if len(value) > allowed_length:
            return True
        return len(value.lstrip("-")) > MAX_RATIONAL_FLAT_INPUT_COMPONENT_DIGITS
    if isinstance(value, int) and not isinstance(value, bool):
        return bool(abs(value) >= 10**MAX_RATIONAL_FLAT_INPUT_COMPONENT_DIGITS)
    return False


def _require_raw_mapping_keys(
    data: dict[object, object],
    *,
    allowed: frozenset[str],
    reason: str,
    label: str,
) -> None:
    """Reject unknown raw fields before inspecting their nested values."""

    if len(data) > len(allowed) or any(
        not isinstance(key, str) or key not in allowed for key in data
    ):
        raise _validation_error(reason, f"{label} contains unknown fields")


def _require_raw_sparse_entry(entry: object, *, label: str) -> None:
    if not isinstance(entry, dict):
        raise _validation_error(
            f"{label}_entry_shape",
            f"every {label} matrix entry must be an object",
        )
    _require_raw_mapping_keys(
        entry,
        allowed=frozenset({"row", "column", "value"}),
        reason=f"{label}_entry_shape",
        label=f"{label} matrix entry",
    )
    if any(
        coordinate is not None
        and (not isinstance(coordinate, int) or isinstance(coordinate, bool))
        for coordinate in (entry.get("row"), entry.get("column"))
    ):
        raise _validation_error(
            f"{label}_entry_shape",
            f"{label} matrix coordinates must be integers",
        )
    value = entry.get("value")
    if not isinstance(value, dict):
        raise _validation_error(
            f"{label}_value_shape",
            f"every {label} matrix value must be a rational object",
        )
    _require_raw_mapping_keys(
        value,
        allowed=frozenset({"num", "den"}),
        reason=f"{label}_value_shape",
        label=f"{label} rational value",
    )
    components = (value.get("num"), value.get("den"))
    if any(
        component is not None and not isinstance(component, (str, int))
        for component in components
    ):
        raise _validation_error(
            f"{label}_value_shape",
            f"{label} rational components must be decimal scalars",
        )
    if any(_raw_component_exceeds_input_digits(component) for component in components):
        raise _validation_error(
            "input_component_bound",
            f"{label} rational components may contain at most "
            f"{MAX_RATIONAL_FLAT_INPUT_COMPONENT_DIGITS} decimal digits",
        )


def _require_raw_sparse_input_envelope(
    data: object,
    *,
    label: str,
    maximum_rows: int,
) -> None:
    """Reject operation-sized sparse inputs before nested scalar parsing."""

    if not isinstance(data, dict):
        return
    _require_raw_mapping_keys(
        data,
        allowed=frozenset({"domain", "row_count", "column_count", "entries"}),
        reason=f"{label}_matrix_shape",
        label=f"{label} matrix",
    )
    row_count = data.get("row_count")
    column_count = data.get("column_count")
    for field_name, value in (
        ("row_count", row_count),
        ("column_count", column_count),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise _validation_error(
                f"{label}_matrix_shape",
                f"{label} matrix {field_name} must be an integer",
            )
    domain = data.get("domain")
    if domain is not None and not isinstance(domain, str):
        raise _validation_error(
            f"{label}_matrix_shape",
            f"{label} matrix domain must be a string",
        )
    if (
        isinstance(row_count, int)
        and not isinstance(row_count, bool)
        and row_count > maximum_rows
    ):
        raise _validation_error(
            f"{label}_row_bound",
            f"{label} matrices have at most {maximum_rows} rows",
        )
    if (
        isinstance(column_count, int)
        and not isinstance(column_count, bool)
        and column_count > MAX_RATIONAL_FLAT_AMBIENT_DIMENSION
    ):
        raise _validation_error(
            f"{label}_column_bound",
            f"{label} matrices have at most "
            f"{MAX_RATIONAL_FLAT_AMBIENT_DIMENSION} columns",
        )
    entries = data.get("entries")
    if entries is None:
        return
    if not isinstance(entries, (list, tuple)):
        raise _validation_error(
            f"{label}_matrix_shape",
            f"{label} matrix entries must be an array",
        )
    if len(entries) > MAX_RATIONAL_FLAT_MATRIX_NONZEROS:
        raise _validation_error(
            f"{label}_nonzero_bound",
            f"{label} matrices store at most "
            f"{MAX_RATIONAL_FLAT_MATRIX_NONZEROS} nonzero entries",
        )
    for entry in entries:
        _require_raw_sparse_entry(entry, label=label)


def _require_raw_configuration_envelope(data: object, *, label: str) -> None:
    """Reject oversized configuration labels before canonicalizing containers."""

    if not isinstance(data, dict):
        return
    coordinate_axis = data.get("coordinate_axis")
    if (
        isinstance(coordinate_axis, (list, tuple))
        and len(coordinate_axis) > MAX_RATIONAL_FLAT_AMBIENT_DIMENSION
    ):
        raise _validation_error(
            f"{label}_axis_bound",
            f"{label} coordinate axes have at most "
            f"{MAX_RATIONAL_FLAT_AMBIENT_DIMENSION} entries",
        )
    vector_labels = data.get("vector_labels")
    if isinstance(vector_labels, (list, tuple)) and len(vector_labels) > (
        MAX_RATIONAL_FLAT_CANDIDATES
    ):
        raise _validation_error(
            f"{label}_label_bound",
            f"{label} vector labels have at most "
            f"{MAX_RATIONAL_FLAT_CANDIDATES} entries",
        )


def _require_raw_generator_envelope(data: object) -> object:
    if not isinstance(data, dict):
        return data
    _require_raw_mapping_keys(
        data,
        allowed=frozenset({"coordinate_permutation", "candidate_permutation"}),
        reason="symmetry_generator_shape",
        label="rational-flat symmetry generator",
    )
    projected = dict(data)
    for field_name, maximum, reason in (
        (
            "coordinate_permutation",
            MAX_RATIONAL_FLAT_AMBIENT_DIMENSION,
            "coordinate_permutation_bound",
        ),
        (
            "candidate_permutation",
            MAX_RATIONAL_FLAT_CANDIDATES,
            "candidate_permutation_bound",
        ),
    ):
        permutation = data.get(field_name)
        if isinstance(permutation, (list, tuple)) and len(permutation) > maximum:
            raise _validation_error(
                reason,
                f"{field_name} admits at most {maximum} entries",
            )
        if isinstance(permutation, (list, tuple)):
            if any(
                not isinstance(item, int) or isinstance(item, bool)
                for item in permutation
            ):
                raise _validation_error(
                    "symmetry_generator_shape",
                    f"every {field_name} entry must be an integer",
                )
            projected[field_name] = tuple(permutation)
    return projected


class RationalVectorConfiguration(StrictModel):
    """A finite labelled row-vector configuration over ``QQ``.

    Rows remain labelled even when they are zero, equal, or proportional.  The
    ordered coordinate axis gives every row and downstream subspace one common
    ambient interpretation.
    """

    coordinate_axis: tuple[OpaqueLabel, ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_FLAT_AMBIENT_DIMENSION,
        description=(
            "Pairwise-distinct labels for the ordered QQ coordinate axis; its "
            "length must equal vectors.column_count."
        ),
    )
    vector_labels: tuple[OpaqueLabel, ...] = Field(
        max_length=MAX_RATIONAL_FLAT_CANDIDATES,
        description=(
            "Pairwise-distinct labels in matrix-row order; the length must equal "
            "vectors.row_count, including zero and proportional labelled rows."
        ),
    )
    vectors: SparseRationalMatrix = Field(
        description=(
            "Dimension-retaining QQ matrix whose rows, including implicit zero "
            "rows, follow vector_labels and whose columns follow coordinate_axis; "
            "each numerator and denominator component may contain at most 256 "
            "decimal digits."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_configuration_envelope(cls, data: Any) -> Any:
        if isinstance(data, dict):
            _require_raw_mapping_keys(
                data,
                allowed=frozenset({"coordinate_axis", "vector_labels", "vectors"}),
                reason="configuration_shape",
                label="rational vector configuration",
            )
            axis = data.get("coordinate_axis")
            labels = data.get("vector_labels")
            for field_name, value in (
                ("coordinate_axis", axis),
                ("vector_labels", labels),
            ):
                if isinstance(value, (list, tuple)) and any(
                    not isinstance(item, str) for item in value
                ):
                    raise _validation_error(
                        "configuration_shape",
                        f"every {field_name} entry must be a string",
                    )
            if isinstance(axis, (list, tuple)) and len(axis) > (
                MAX_RATIONAL_FLAT_AMBIENT_DIMENSION
            ):
                raise _validation_error(
                    "ambient_dimension_bound",
                    "rational-flat configurations have at most "
                    f"{MAX_RATIONAL_FLAT_AMBIENT_DIMENSION} coordinates",
                )
            if isinstance(labels, (list, tuple)) and len(labels) > (
                MAX_RATIONAL_FLAT_CANDIDATES
            ):
                raise _validation_error(
                    "candidate_count_bound",
                    "rational-flat configurations have at most "
                    f"{MAX_RATIONAL_FLAT_CANDIDATES} labelled vectors",
                )
            _require_raw_sparse_input_envelope(
                data.get("vectors"),
                label="candidate",
                maximum_rows=MAX_RATIONAL_FLAT_CANDIDATES,
            )
            try:
                return canonicalize_json_containers(data)
            except CanonicalizationError as exc:
                raise _validation_error(
                    "raw_container_structure",
                    "raw rational-flat containers must be acyclic",
                ) from exc
        return data

    @model_validator(mode="after")
    def bind_labels_and_axes(self) -> Self:
        if len(set(self.coordinate_axis)) != len(self.coordinate_axis):
            raise _validation_error(
                "duplicate_coordinate_label",
                "configuration coordinate labels must be pairwise distinct",
            )
        if len(set(self.vector_labels)) != len(self.vector_labels):
            raise _validation_error(
                "duplicate_vector_label",
                "configuration vector labels must be pairwise distinct",
            )
        if self.vectors.row_count != len(self.vector_labels):
            raise _validation_error(
                "vector_label_count",
                "vector_labels must contain one label for every declared matrix row",
            )
        if self.vectors.column_count != len(self.coordinate_axis):
            raise _validation_error(
                "coordinate_axis_count",
                "the sparse matrix column count must equal the coordinate-axis length",
            )
        return self

    @property
    def vector_count(self) -> int:
        return self.vectors.row_count


class RationalFlatRankInterval(StrictModel):
    """Inclusive admitted ranks for returned candidate-generated flats."""

    minimum: StrictInt = Field(
        ge=0,
        le=MAX_RATIONAL_FLAT_AMBIENT_DIMENSION,
        description="Smallest admitted flat rank, inclusive.",
    )
    maximum: StrictInt = Field(
        ge=0,
        le=MAX_RATIONAL_FLAT_AMBIENT_DIMENSION,
        description=(
            "Largest admitted flat rank, inclusive; it cannot exceed the source "
            "coordinate-axis length."
        ),
    )

    @model_validator(mode="after")
    def require_ordered_interval(self) -> Self:
        if self.minimum > self.maximum:
            raise _validation_error(
                "rank_interval_order",
                "minimum rank must not exceed maximum rank",
            )
        return self


class RationalFlatSymmetryGenerator(StrictModel):
    """One compatible permutation of coordinates and labelled candidates.

    ``coordinate_permutation[i]`` is the image of coordinate ``i``.  The row
    action sends the old coefficient at ``i`` to that image coordinate.
    Candidate rows need only agree projectively, since nonzero rational scaling
    does not change their equations or row-matroid elements.
    """

    coordinate_permutation: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_FLAT_AMBIENT_DIMENSION,
        description=(
            "Array-form permutation of every source coordinate position; its "
            "length must equal the coordinate-axis length."
        ),
    )
    candidate_permutation: tuple[StrictInt, ...] = Field(
        max_length=MAX_RATIONAL_FLAT_CANDIDATES,
        description=(
            "Array-form permutation of every labelled candidate row; its length "
            "must equal the source candidate count."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_generator_envelope(cls, data: Any) -> Any:
        return canonicalize_json_containers(_require_raw_generator_envelope(data))


class ClauseConstrainedRationalFlatProblem(StrictModel):
    """One finite exact rational-flat classification problem.

    Each clause contains candidate-row indices and requires the closed flat to
    contain at least one of them.  A forbidden row may not lie in the flat's
    rational span.  Empty clauses are admitted and make the satisfying family
    empty.  With no rank interval, every rank from zero through the ambient
    dimension is eligible.  An empty symmetry-generator tuple is the trivial
    action.
    """

    candidates: RationalVectorConfiguration
    clauses: tuple[CandidateClause, ...] = Field(
        default=(),
        max_length=MAX_RATIONAL_FLAT_CLAUSES,
        description=(
            "Canonical distinct clauses, each a sorted set of candidate indices; "
            "an empty clause has no satisfying flat."
        ),
    )
    forbidden_vectors: SparseRationalMatrix = Field(
        description=(
            "Rows in the candidates' coordinate axis that must not belong to a "
            "returned flat's row span; zero rows therefore exclude every flat, "
            "and each rational component may contain at most 256 decimal digits."
        )
    )
    rank_interval: RationalFlatRankInterval | None = Field(
        default=None,
        description=(
            "Optional inclusive rank interval; omission means zero through the "
            "ambient dimension."
        ),
    )
    symmetry_generators: tuple[RationalFlatSymmetryGenerator, ...] = Field(
        default=(),
        max_length=MAX_RATIONAL_FLAT_SYMMETRY_GENERATORS,
        description=(
            "Generators of one paired coordinate/candidate permutation action; "
            "each must preserve candidate projective rows, the clause family, "
            "and the forbidden projective-row set."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_problem_envelope(cls, data: Any) -> Any:  # noqa: C901
        if isinstance(data, dict):
            _require_raw_mapping_keys(
                data,
                allowed=frozenset(
                    {
                        "candidates",
                        "clauses",
                        "forbidden_vectors",
                        "rank_interval",
                        "symmetry_generators",
                    }
                ),
                reason="problem_shape",
                label="rational-flat problem",
            )
            candidates = data.get("candidates")
            if isinstance(candidates, dict):
                _require_raw_configuration_envelope(candidates, label="candidate")
                _require_raw_sparse_input_envelope(
                    candidates.get("vectors"),
                    label="candidate",
                    maximum_rows=MAX_RATIONAL_FLAT_CANDIDATES,
                )
            _require_raw_sparse_input_envelope(
                data.get("forbidden_vectors"),
                label="forbidden",
                maximum_rows=MAX_RATIONAL_FLAT_FORBIDDEN_VECTORS,
            )
            clauses = data.get("clauses")
            if isinstance(clauses, (list, tuple)):
                if len(clauses) > MAX_RATIONAL_FLAT_CLAUSES:
                    raise _validation_error(
                        "clause_count_bound",
                        f"at most {MAX_RATIONAL_FLAT_CLAUSES} clauses are admitted",
                    )
                memberships = 0
                for clause in clauses:
                    if not isinstance(clause, (list, tuple)):
                        raise _validation_error(
                            "clause_shape", "every clause must be an array"
                        )
                    memberships += len(clause)
                    if memberships > MAX_RATIONAL_FLAT_CLAUSE_MEMBERSHIPS:
                        raise _validation_error(
                            "clause_membership_bound",
                            "clause memberships exceed the structural bound of "
                            f"{MAX_RATIONAL_FLAT_CLAUSE_MEMBERSHIPS}",
                        )
                    if any(
                        not isinstance(index, int) or isinstance(index, bool)
                        for index in clause
                    ):
                        raise _validation_error(
                            "raw_container_structure",
                            "raw rational-flat clause entries must be integers",
                        )
            generators = data.get("symmetry_generators")
            if isinstance(generators, (list, tuple)):
                if len(generators) > MAX_RATIONAL_FLAT_SYMMETRY_GENERATORS:
                    raise _validation_error(
                        "symmetry_generator_bound",
                        "at most "
                        f"{MAX_RATIONAL_FLAT_SYMMETRY_GENERATORS} symmetry generators "
                        "are admitted",
                    )
                for generator in generators:
                    _require_raw_generator_envelope(generator)
            rank_interval = data.get("rank_interval")
            if isinstance(rank_interval, dict):
                _require_raw_mapping_keys(
                    rank_interval,
                    allowed=frozenset({"minimum", "maximum"}),
                    reason="rank_interval_shape",
                    label="rational-flat rank interval",
                )
                for key in ("minimum", "maximum"):
                    value = rank_interval.get(key)
                    if value is not None and not isinstance(value, int):
                        raise _validation_error(
                            "rank_interval_shape",
                            "rational-flat rank interval endpoints must be integers",
                        )
            try:
                return canonicalize_json_containers(data)
            except CanonicalizationError as exc:
                raise _validation_error(
                    "raw_container_structure",
                    "raw rational-flat containers must be acyclic",
                ) from exc
        return data

    @model_validator(mode="after")
    def require_source_bound_problem(self) -> Self:
        candidate_count = self.candidates.vector_count
        normalized_clauses = tuple(
            sorted({tuple(sorted(set(clause))) for clause in self.clauses})
        )
        if any(
            index >= candidate_count
            for clause in normalized_clauses
            for index in clause
        ):
            raise _validation_error(
                "clause_candidate_index",
                "every clause index must refer to a candidate row",
            )
        if sum(map(len, normalized_clauses)) > MAX_RATIONAL_FLAT_CLAUSE_MEMBERSHIPS:
            raise _validation_error(
                "clause_membership_bound",
                "canonical clause memberships exceed the structural bound",
            )
        object.__setattr__(self, "clauses", normalized_clauses)

        ambient_dimension = len(self.candidates.coordinate_axis)
        if self.forbidden_vectors.column_count != ambient_dimension:
            raise _validation_error(
                "forbidden_coordinate_axis",
                "forbidden rows must use the candidates' ambient coordinate axis",
            )
        if self.forbidden_vectors.row_count > MAX_RATIONAL_FLAT_FORBIDDEN_VECTORS:
            raise _validation_error(
                "forbidden_count_bound",
                "at most "
                f"{MAX_RATIONAL_FLAT_FORBIDDEN_VECTORS} forbidden rows are admitted",
            )
        if (
            self.rank_interval is not None
            and self.rank_interval.maximum > ambient_dimension
        ):
            raise _validation_error(
                "rank_ambient_bound",
                "the maximum flat rank cannot exceed the ambient dimension",
            )

        canonical_generators: dict[
            tuple[tuple[int, ...], tuple[int, ...]], RationalFlatSymmetryGenerator
        ] = {}
        for generator in self.symmetry_generators:
            coordinate_permutation = tuple(generator.coordinate_permutation)
            candidate_permutation = tuple(generator.candidate_permutation)
            if len(coordinate_permutation) != ambient_dimension or sorted(
                coordinate_permutation
            ) != list(range(ambient_dimension)):
                raise _validation_error(
                    "coordinate_permutation",
                    "every coordinate generator must permute the complete ambient axis",
                )
            if len(candidate_permutation) != candidate_count or sorted(
                candidate_permutation
            ) != list(range(candidate_count)):
                raise _validation_error(
                    "candidate_permutation",
                    "every candidate generator must permute all candidate rows",
                )
            canonical_generators[(coordinate_permutation, candidate_permutation)] = (
                generator
            )
        object.__setattr__(
            self,
            "symmetry_generators",
            tuple(canonical_generators[key] for key in sorted(canonical_generators)),
        )
        return self

    @property
    def minimum_rank(self) -> int:
        return self.rank_interval.minimum if self.rank_interval is not None else 0

    @property
    def maximum_rank(self) -> int:
        return (
            self.rank_interval.maximum
            if self.rank_interval is not None
            else len(self.candidates.coordinate_axis)
        )


class RationalFlatOrbitRepresentative(StrictModel):
    """One canonical closed flat and its exact orbit--stabilizer data."""

    closed_candidate_indices: tuple[CandidateIndex, ...] = Field(
        max_length=MAX_RATIONAL_FLAT_CANDIDATES,
        description=(
            "Increasing source-row indices i for exactly those candidate vectors "
            "lying in the representative row span."
        ),
    )
    rank: StrictInt = Field(
        ge=0,
        le=MAX_RATIONAL_FLAT_AMBIENT_DIMENSION,
        description="Exact QQ dimension of the representative row span.",
    )
    row_space_basis: RationalVectorSpaceBasis = Field(
        description=(
            "Reduced-row-echelon QQ basis of the flat's row span in the source "
            "coordinate axis."
        )
    )
    annihilator_basis: RationalVectorSpaceBasis = Field(
        description=(
            "Canonical free-variable basis of column vectors x satisfying r*x=0 "
            "for every row-space basis vector r, represented in the source axis."
        )
    )
    orbit_size: StrictInt = Field(
        ge=1,
        le=MAX_RATIONAL_FLAT_GROUP_ORDER,
        description="Number of distinct closed candidate sets in this orbit.",
    )
    stabilizer_order: StrictInt = Field(
        ge=1,
        le=MAX_RATIONAL_FLAT_GROUP_ORDER,
        description=(
            "Order of the subgroup of the supplied paired action that fixes this "
            "closed candidate set."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_local_shape(self) -> Self:
        if self.closed_candidate_indices != tuple(
            sorted(set(self.closed_candidate_indices))
        ):
            raise _validation_error(
                "closed_candidate_order",
                "closed candidate indices must be distinct and increasing",
            )
        if len(self.row_space_basis.vectors) != self.rank:
            raise _validation_error(
                "row_basis_rank",
                "row-space basis length must equal the flat rank",
            )
        if self.row_space_basis.ambient_dimension != (
            self.annihilator_basis.ambient_dimension
        ):
            raise _validation_error(
                "basis_ambient_dimension",
                "row-space and annihilator bases must use one ambient dimension",
            )
        if self.rank + len(self.annihilator_basis.vectors) != (
            self.row_space_basis.ambient_dimension
        ):
            raise _validation_error(
                "annihilator_dimension",
                "row rank plus annihilator dimension must equal the ambient dimension",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        closed_candidate_indices: tuple[int, ...],
        rank: int,
        row_space_basis: RationalVectorSpaceBasis,
        annihilator_basis: RationalVectorSpaceBasis,
        orbit_size: int,
        stabilizer_order: int,
    ) -> Self:
        return cls.model_construct(
            closed_candidate_indices=closed_candidate_indices,
            rank=rank,
            row_space_basis=row_space_basis,
            annihilator_basis=annihilator_basis,
            orbit_size=orbit_size,
            stabilizer_order=stabilizer_order,
        )


class RationalFlatClassificationComplete(StrictModel):
    """Every satisfying flat, represented once per supplied symmetry orbit."""

    status: Literal["COMPLETE_EXACT"]
    representatives: tuple[RationalFlatOrbitRepresentative, ...] = Field(
        max_length=MAX_RATIONAL_FLAT_RESULT_ORBITS
    )
    orbit_count: StrictInt = Field(ge=0, le=MAX_RATIONAL_FLAT_RESULT_ORBITS)
    solution_flat_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_complete_accounting(self) -> Self:
        if self.orbit_count != len(self.representatives):
            raise _validation_error(
                "orbit_count",
                "orbit_count must equal the representative count",
            )
        if self.solution_flat_count != sum(
            representative.orbit_size for representative in self.representatives
        ):
            raise _validation_error(
                "solution_flat_count",
                "solution_flat_count must equal the sum of representative orbit sizes",
            )
        representative_keys = tuple(
            representative.closed_candidate_indices
            for representative in self.representatives
        )
        if representative_keys != tuple(sorted(set(representative_keys))):
            raise _validation_error(
                "representative_order",
                "orbit representatives must use distinct keys in increasing "
                "canonical candidate-set order",
            )
        return self


RationalFlatIncompleteReason = Literal[
    "STATE_ORBIT_LIMIT",
    "SEARCH_WORK_LIMIT",
    "RESULT_ORBIT_LIMIT",
    "RESULT_OUTPUT_LIMIT",
]


class RationalFlatClassificationIncomplete(StrictModel):
    """The bounded search stopped without making a completeness claim."""

    status: Literal["INCOMPLETE"]
    reason: RationalFlatIncompleteReason
    explored_state_orbit_count: StrictInt = Field(ge=0)
    state_orbit_limit: StrictInt = Field(ge=1)
    result_orbit_limit: StrictInt = Field(
        ge=0,
        le=MAX_RATIONAL_FLAT_RESULT_ORBITS,
        description="Structural maximum number of retained orbit representatives.",
    )
    result_output_byte_limit: StrictInt = Field(
        ge=1,
        description="Maximum canonical byte size of a complete exact result.",
    )
    consumed_search_work: StrictInt = Field(
        ge=0,
        description=(
            "Charged work in the single request ledger across admission, search, "
            "and canonical result projection."
        ),
    )
    search_work_limit: StrictInt = Field(
        ge=1,
        description=(
            "Maximum work admitted for the single preparation, search, and "
            "projection ledger."
        ),
    )

    @model_validator(mode="after")
    def bind_diagnostics_to_limits(self) -> Self:
        if self.explored_state_orbit_count > self.state_orbit_limit:
            raise _validation_error(
                "explored_state_orbit_count",
                "explored state-orbit count cannot exceed its search limit",
            )
        if self.consumed_search_work > self.search_work_limit:
            raise _validation_error(
                "consumed_search_work",
                "consumed request work cannot exceed its work limit",
            )
        return self


RationalFlatClassificationOutcome = Annotated[
    RationalFlatClassificationComplete | RationalFlatClassificationIncomplete,
    Field(discriminator="status"),
]


class ClauseConstrainedRationalFlatClassification(StrictModel):
    """Source-bound complete family or a typed non-conclusive bounded stop."""

    problem: ClauseConstrainedRationalFlatProblem
    symmetry_group_order: StrictInt = Field(
        ge=1,
        le=MAX_RATIONAL_FLAT_GROUP_ORDER,
    )
    outcome: RationalFlatClassificationOutcome

    @model_validator(mode="after")
    def bind_outcome_to_problem(self) -> Self:
        if not isinstance(self.outcome, RationalFlatClassificationComplete):
            return self
        candidate_count = self.problem.candidates.vector_count
        ambient_dimension = len(self.problem.candidates.coordinate_axis)
        for representative in self.outcome.representatives:
            if any(
                index >= candidate_count
                for index in representative.closed_candidate_indices
            ):
                raise _validation_error(
                    "representative_candidate_index",
                    "representative indices must refer to source candidates",
                )
            if (
                not self.problem.minimum_rank
                <= representative.rank
                <= (self.problem.maximum_rank)
            ):
                raise _validation_error(
                    "representative_rank",
                    "representative rank must lie in the requested interval",
                )
            if representative.row_space_basis.ambient_dimension != ambient_dimension:
                raise _validation_error(
                    "representative_ambient_dimension",
                    "representative bases must use the source coordinate dimension",
                )
            if (
                representative.orbit_size * representative.stabilizer_order
                != self.symmetry_group_order
            ):
                raise _validation_error(
                    "orbit_stabilizer",
                    "each orbit size times stabilizer order must equal the symmetry-group order",
                )
        return self

    @classmethod
    def _complete_from_kernel(
        cls,
        *,
        problem: ClauseConstrainedRationalFlatProblem,
        symmetry_group_order: int,
        representatives: tuple[RationalFlatOrbitRepresentative, ...],
        solution_flat_count: int,
    ) -> Self:
        return cls.model_construct(
            problem=problem,
            symmetry_group_order=symmetry_group_order,
            outcome=RationalFlatClassificationComplete.model_construct(
                status="COMPLETE_EXACT",
                representatives=representatives,
                orbit_count=len(representatives),
                solution_flat_count=solution_flat_count,
            ),
        )

    @classmethod
    def _incomplete_from_kernel(
        cls,
        *,
        problem: ClauseConstrainedRationalFlatProblem,
        symmetry_group_order: int,
        reason: RationalFlatIncompleteReason,
        explored_state_orbit_count: int,
        state_orbit_limit: int,
        result_orbit_limit: int,
        result_output_byte_limit: int,
        consumed_search_work: int,
        search_work_limit: int,
    ) -> Self:
        return cls.model_construct(
            problem=problem,
            symmetry_group_order=symmetry_group_order,
            outcome=RationalFlatClassificationIncomplete.model_construct(
                status="INCOMPLETE",
                reason=reason,
                explored_state_orbit_count=explored_state_orbit_count,
                state_orbit_limit=state_orbit_limit,
                result_orbit_limit=result_orbit_limit,
                result_output_byte_limit=result_output_byte_limit,
                consumed_search_work=consumed_search_work,
                search_work_limit=search_work_limit,
            ),
        )


class ClauseConstrainedRationalFlatRequest(StrictModel):
    """Wire request for one clause-constrained rational-flat problem."""

    problem: ClauseConstrainedRationalFlatProblem


__all__ = [
    "MAX_RATIONAL_FLAT_AMBIENT_DIMENSION",
    "MAX_RATIONAL_FLAT_CANDIDATES",
    "MAX_RATIONAL_FLAT_CLAUSES",
    "MAX_RATIONAL_FLAT_CLAUSE_MEMBERSHIPS",
    "MAX_RATIONAL_FLAT_FORBIDDEN_VECTORS",
    "MAX_RATIONAL_FLAT_GROUP_ORDER",
    "MAX_RATIONAL_FLAT_INPUT_COMPONENT_DIGITS",
    "MAX_RATIONAL_FLAT_MATRIX_NONZEROS",
    "MAX_RATIONAL_FLAT_RESULT_BYTES",
    "MAX_RATIONAL_FLAT_RESULT_ORBITS",
    "MAX_RATIONAL_FLAT_SYMMETRY_GENERATORS",
    "ClauseConstrainedRationalFlatClassification",
    "ClauseConstrainedRationalFlatProblem",
    "ClauseConstrainedRationalFlatRequest",
    "RationalFlatClassificationComplete",
    "RationalFlatClassificationIncomplete",
    "RationalFlatClassificationOutcome",
    "RationalFlatOrbitRepresentative",
    "RationalFlatRankInterval",
    "RationalFlatSymmetryGenerator",
    "RationalVectorConfiguration",
]
