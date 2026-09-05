"""Canonical values for clause-constrained prime-field flat classification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import CanonicalizationError
from jacobian.math._labels import OpaqueLabel
from jacobian.math.matrices.finite_fields import PrimeFieldMatrix

MAX_PRIME_FIELD_FLAT_AMBIENT_DIMENSION = 16
MAX_PRIME_FIELD_FLAT_CANDIDATES = 128
MAX_PRIME_FIELD_FLAT_FORBIDDEN_VECTORS = 128
MAX_PRIME_FIELD_FLAT_CLAUSES = 128
MAX_PRIME_FIELD_FLAT_CLAUSE_MEMBERSHIPS = 4_096
MAX_PRIME_FIELD_FLAT_SYMMETRY_GENERATORS = 16
MAX_PRIME_FIELD_FLAT_GROUP_ORDER = 10_000
MAX_PRIME_FIELD_FLAT_RESULT_ORBITS = 100_000
MAX_PRIME_FIELD_FLAT_PRIME = 2_147_483_647

CandidateIndex = Annotated[StrictInt, Field(ge=0, lt=MAX_PRIME_FIELD_FLAT_CANDIDATES)]
CandidateClause = Annotated[
    tuple[CandidateIndex, ...], Field(max_length=MAX_PRIME_FIELD_FLAT_CANDIDATES)
]
FieldResidue = Annotated[StrictInt, Field(ge=0)]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"prime_field_flat.{reason}", message)


def _require_mapping_keys(
    data: Mapping[str, object],
    *,
    allowed: frozenset[str],
    reason: str,
    label: str,
) -> None:
    if any(key not in allowed for key in data):
        raise _validation_error(reason, f"{label} contains unknown fields")


def _canonicalize_or_fail(data: object) -> object:
    try:
        return canonicalize_json_containers(data)
    except CanonicalizationError as exc:
        raise _validation_error(
            "raw_container_structure",
            "raw prime-field-flat containers must be acyclic",
        ) from exc


def _require_raw_envelope(data: object) -> None:
    """Reject oversized JSON containers before canonicalization copies them."""
    if not isinstance(data, Mapping):
        return
    candidates = data.get("candidates")
    if isinstance(candidates, Mapping):
        vectors = candidates.get("vectors")
        if isinstance(vectors, Mapping):
            rows = vectors.get("entries")
            if isinstance(rows, (list, tuple)) and (
                len(rows) > MAX_PRIME_FIELD_FLAT_CANDIDATES
                or any(not isinstance(row, (list, tuple)) or len(row) > MAX_PRIME_FIELD_FLAT_AMBIENT_DIMENSION for row in rows)
            ):
                raise _validation_error("raw_matrix_bound", "candidate matrix exceeds the raw row envelope")
    for key, maximum in (("clauses", MAX_PRIME_FIELD_FLAT_CLAUSES), ("symmetry_generators", MAX_PRIME_FIELD_FLAT_SYMMETRY_GENERATORS)):
        value = data.get(key)
        if isinstance(value, (list, tuple)) and len(value) > maximum:
            raise _validation_error("raw_container_bound", f"{key} exceeds the raw envelope")
    forbidden = data.get("forbidden_vectors")
    if isinstance(forbidden, Mapping):
        rows = forbidden.get("entries")
        if isinstance(rows, (list, tuple)) and (
            len(rows) > MAX_PRIME_FIELD_FLAT_FORBIDDEN_VECTORS
            or any(not isinstance(row, (list, tuple)) or len(row) > MAX_PRIME_FIELD_FLAT_AMBIENT_DIMENSION for row in rows)
        ):
            raise _validation_error("raw_matrix_bound", "forbidden matrix exceeds the raw row envelope")


class PrimeFieldRowMatrix(StrictModel):
    """Dense canonical residue rows in one explicit prime field."""

    prime: StrictInt = Field(ge=2, le=MAX_PRIME_FIELD_FLAT_PRIME)
    entries: tuple[tuple[FieldResidue, ...], ...] = Field(
        max_length=MAX_PRIME_FIELD_FLAT_CANDIDATES
    )
    columns: StrictInt = Field(ge=0, le=MAX_PRIME_FIELD_FLAT_AMBIENT_DIMENSION)

    @model_validator(mode="after")
    def require_canonical_matrix(self) -> Self:
        if any(
            len(row) != self.columns or any(value >= self.prime for value in row)
            for row in self.entries
        ):
            raise _validation_error(
                "matrix_shape",
                "every row must match the column count and hold canonical residues",
            )
        return self

    @property
    def row_count(self) -> int:
        return len(self.entries)

    def as_prime_field_matrix(self) -> PrimeFieldMatrix:
        """Map this labelled-configuration carrier to the canonical matrix value."""
        return PrimeFieldMatrix(self.prime, self.entries, self.columns)

    @classmethod
    def from_prime_field_matrix(cls, matrix: PrimeFieldMatrix) -> Self:
        return cls(prime=matrix.prime, entries=matrix.entries, columns=matrix.columns)


class PrimeFieldVectorConfiguration(StrictModel):
    """A finite labelled row-vector configuration over ``GF(prime)``."""

    prime: StrictInt = Field(
        ge=2,
        le=MAX_PRIME_FIELD_FLAT_PRIME,
        description="Explicit prime field characteristic defining GF(prime).",
    )
    coordinate_axis: tuple[OpaqueLabel, ...] = Field(
        min_length=1,
        max_length=MAX_PRIME_FIELD_FLAT_AMBIENT_DIMENSION,
    )
    vector_labels: tuple[OpaqueLabel, ...] = Field(
        max_length=MAX_PRIME_FIELD_FLAT_CANDIDATES
    )
    vectors: PrimeFieldRowMatrix

    @model_validator(mode="before")
    @classmethod
    def require_raw_configuration(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        _require_mapping_keys(
            data,
            allowed=frozenset({"prime", "coordinate_axis", "vector_labels", "vectors"}),
            reason="configuration_shape",
            label="prime-field vector configuration",
        )
        _require_raw_envelope({"candidates": data})
        return _canonicalize_or_fail(data)

    @model_validator(mode="after")
    def require_prime_and_labels(self) -> Self:
        from sympy import isprime

        if not isprime(self.prime):
            raise _validation_error("prime", "prime must be a prime integer")
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
        if self.vectors.prime != self.prime:
            raise _validation_error(
                "matrix_prime",
                "the candidate matrix must use the configuration prime",
            )
        if self.vectors.row_count != len(self.vector_labels):
            raise _validation_error(
                "vector_label_count",
                "vector_labels must contain one label for every declared row",
            )
        if self.vectors.columns != len(self.coordinate_axis):
            raise _validation_error(
                "coordinate_axis_count",
                "the candidate matrix column count must equal the axis length",
            )
        return self

    @property
    def vector_count(self) -> int:
        return self.vectors.row_count


class PrimeFieldFlatRankInterval(StrictModel):
    """Inclusive admitted ranks for returned prime-field flats."""

    minimum: StrictInt = Field(ge=0, le=MAX_PRIME_FIELD_FLAT_AMBIENT_DIMENSION)
    maximum: StrictInt = Field(ge=0, le=MAX_PRIME_FIELD_FLAT_AMBIENT_DIMENSION)

    @model_validator(mode="after")
    def require_ordered_interval(self) -> Self:
        if self.minimum > self.maximum:
            raise _validation_error(
                "rank_interval_order", "minimum rank must not exceed maximum rank"
            )
        return self


class PrimeFieldFlatSymmetryGenerator(StrictModel):
    """One compatible permutation of coordinates and labelled candidates."""

    coordinate_permutation: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_PRIME_FIELD_FLAT_AMBIENT_DIMENSION,
    )
    candidate_permutation: tuple[StrictInt, ...] = Field(
        max_length=MAX_PRIME_FIELD_FLAT_CANDIDATES
    )

    @model_validator(mode="before")
    @classmethod
    def canonicalize_containers(cls, data: object) -> object:
        return _canonicalize_or_fail(data)


class ClauseConstrainedPrimeFieldFlatProblem(StrictModel):
    """One finite exact prime-field flat classification problem."""

    candidates: PrimeFieldVectorConfiguration
    clauses: tuple[CandidateClause, ...] = Field(
        default=(),
        max_length=MAX_PRIME_FIELD_FLAT_CLAUSES,
    )
    forbidden_vectors: PrimeFieldRowMatrix
    rank_interval: PrimeFieldFlatRankInterval | None = None
    symmetry_generators: tuple[PrimeFieldFlatSymmetryGenerator, ...] = Field(
        default=(),
        max_length=MAX_PRIME_FIELD_FLAT_SYMMETRY_GENERATORS,
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_problem(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        _require_mapping_keys(
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
            label="prime-field-flat problem",
        )
        _require_raw_envelope(data)
        return _canonicalize_or_fail(data)

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
        if sum(map(len, normalized_clauses)) > MAX_PRIME_FIELD_FLAT_CLAUSE_MEMBERSHIPS:
            raise _validation_error(
                "clause_membership_bound",
                "canonical clause memberships exceed the structural bound",
            )
        object.__setattr__(self, "clauses", normalized_clauses)

        ambient_dimension = len(self.candidates.coordinate_axis)
        if self.forbidden_vectors.prime != self.candidates.prime:
            raise _validation_error(
                "forbidden_prime", "forbidden rows must use the candidate prime"
            )
        if self.forbidden_vectors.columns != ambient_dimension:
            raise _validation_error(
                "forbidden_coordinate_axis",
                "forbidden rows must use the candidates' coordinate axis",
            )
        if self.forbidden_vectors.row_count > MAX_PRIME_FIELD_FLAT_FORBIDDEN_VECTORS:
            raise _validation_error(
                "forbidden_count_bound",
                "at most "
                f"{MAX_PRIME_FIELD_FLAT_FORBIDDEN_VECTORS} forbidden rows are admitted",
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
            tuple[tuple[int, ...], tuple[int, ...]],
            PrimeFieldFlatSymmetryGenerator,
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


class PrimeFieldVectorSpaceBasis(StrictModel):
    """A canonical GF(prime) basis with its ambient dimension retained."""

    prime: StrictInt = Field(ge=2, le=MAX_PRIME_FIELD_FLAT_PRIME)
    ambient_dimension: StrictInt = Field(
        ge=1, le=MAX_PRIME_FIELD_FLAT_AMBIENT_DIMENSION
    )
    vectors: tuple[tuple[FieldResidue, ...], ...] = Field(
        default=(),
        max_length=MAX_PRIME_FIELD_FLAT_AMBIENT_DIMENSION,
    )

    @model_validator(mode="after")
    def require_vector_shape(self) -> Self:
        if any(
            len(vector) != self.ambient_dimension
            or any(value >= self.prime for value in vector)
            for vector in self.vectors
        ):
            raise _validation_error(
                "basis_shape",
                "each basis vector must match the ambient dimension and prime",
            )
        return self

    def as_prime_field_matrix(self) -> PrimeFieldMatrix:
        """Expose a returned basis to the canonical finite-field matrix API."""
        return PrimeFieldMatrix(self.prime, self.vectors, self.ambient_dimension)


class PrimeFieldFlatOrbitRepresentative(StrictModel):
    """One canonical closed flat and its exact orbit--stabilizer data."""

    closed_candidate_indices: tuple[CandidateIndex, ...] = Field(
        max_length=MAX_PRIME_FIELD_FLAT_CANDIDATES
    )
    rank: StrictInt = Field(ge=0, le=MAX_PRIME_FIELD_FLAT_AMBIENT_DIMENSION)
    row_space_basis: PrimeFieldVectorSpaceBasis
    annihilator_basis: PrimeFieldVectorSpaceBasis
    orbit_size: StrictInt = Field(ge=1, le=MAX_PRIME_FIELD_FLAT_GROUP_ORDER)
    stabilizer_order: StrictInt = Field(ge=1, le=MAX_PRIME_FIELD_FLAT_GROUP_ORDER)

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
                "row_basis_rank", "row-space basis length must equal the flat rank"
            )
        if self.row_space_basis.prime != self.annihilator_basis.prime:
            raise _validation_error(
                "basis_prime", "row-space and annihilator bases must use one prime"
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
        row_space_basis: PrimeFieldVectorSpaceBasis,
        annihilator_basis: PrimeFieldVectorSpaceBasis,
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


class PrimeFieldFlatClassificationComplete(StrictModel):
    """Every satisfying prime-field flat once per supplied symmetry orbit."""

    status: Literal["COMPLETE_EXACT"]
    representatives: tuple[PrimeFieldFlatOrbitRepresentative, ...] = Field(
        max_length=MAX_PRIME_FIELD_FLAT_RESULT_ORBITS
    )
    orbit_count: StrictInt = Field(ge=0, le=MAX_PRIME_FIELD_FLAT_RESULT_ORBITS)
    solution_flat_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_complete_accounting(self) -> Self:
        if self.orbit_count != len(self.representatives):
            raise _validation_error(
                "orbit_count", "orbit_count must equal the representative count"
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
                "orbit representatives must use distinct keys in increasing order",
            )
        return self


PrimeFieldFlatIncompleteReason = Literal[
    "STATE_ORBIT_LIMIT", "SEARCH_WORK_LIMIT", "RESULT_ORBIT_LIMIT"
]


class PrimeFieldFlatClassificationIncomplete(StrictModel):
    """The bounded search stopped without making a completeness claim."""

    status: Literal["INCOMPLETE"]
    reason: PrimeFieldFlatIncompleteReason
    explored_state_orbit_count: StrictInt = Field(ge=0)
    state_orbit_limit: StrictInt = Field(ge=1)
    result_orbit_limit: StrictInt = Field(ge=0, le=MAX_PRIME_FIELD_FLAT_RESULT_ORBITS)
    consumed_search_work: StrictInt = Field(ge=0)
    search_work_limit: StrictInt = Field(ge=1)

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


PrimeFieldFlatClassificationOutcome = Annotated[
    PrimeFieldFlatClassificationComplete | PrimeFieldFlatClassificationIncomplete,
    Field(discriminator="status"),
]


class ClauseConstrainedPrimeFieldFlatClassification(StrictModel):
    """Source-bound complete family or a typed non-conclusive bounded stop."""

    problem: ClauseConstrainedPrimeFieldFlatProblem
    symmetry_group_order: StrictInt = Field(ge=1, le=MAX_PRIME_FIELD_FLAT_GROUP_ORDER)
    outcome: PrimeFieldFlatClassificationOutcome

    @model_validator(mode="after")
    def bind_outcome_to_problem(self) -> Self:
        if not isinstance(self.outcome, PrimeFieldFlatClassificationComplete):
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
            if representative.row_space_basis.prime != self.problem.candidates.prime:
                raise _validation_error(
                    "representative_prime",
                    "representative bases must use the source prime",
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
        problem: ClauseConstrainedPrimeFieldFlatProblem,
        symmetry_group_order: int,
        representatives: tuple[PrimeFieldFlatOrbitRepresentative, ...],
        solution_flat_count: int,
    ) -> Self:
        return cls.model_construct(
            problem=problem,
            symmetry_group_order=symmetry_group_order,
            outcome=PrimeFieldFlatClassificationComplete.model_construct(
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
        problem: ClauseConstrainedPrimeFieldFlatProblem,
        symmetry_group_order: int,
        reason: PrimeFieldFlatIncompleteReason,
        explored_state_orbit_count: int,
        state_orbit_limit: int,
        result_orbit_limit: int,
        consumed_search_work: int,
        search_work_limit: int,
    ) -> Self:
        return cls.model_construct(
            problem=problem,
            symmetry_group_order=symmetry_group_order,
            outcome=PrimeFieldFlatClassificationIncomplete.model_construct(
                status="INCOMPLETE",
                reason=reason,
                explored_state_orbit_count=explored_state_orbit_count,
                state_orbit_limit=state_orbit_limit,
                result_orbit_limit=result_orbit_limit,
                consumed_search_work=consumed_search_work,
                search_work_limit=search_work_limit,
            ),
        )


class ClauseConstrainedPrimeFieldFlatRequest(StrictModel):
    """Wire request for one clause-constrained prime-field flat problem."""

    problem: ClauseConstrainedPrimeFieldFlatProblem


__all__ = [
    "MAX_PRIME_FIELD_FLAT_AMBIENT_DIMENSION",
    "MAX_PRIME_FIELD_FLAT_CANDIDATES",
    "MAX_PRIME_FIELD_FLAT_CLAUSES",
    "MAX_PRIME_FIELD_FLAT_CLAUSE_MEMBERSHIPS",
    "MAX_PRIME_FIELD_FLAT_FORBIDDEN_VECTORS",
    "MAX_PRIME_FIELD_FLAT_GROUP_ORDER",
    "MAX_PRIME_FIELD_FLAT_PRIME",
    "MAX_PRIME_FIELD_FLAT_RESULT_ORBITS",
    "MAX_PRIME_FIELD_FLAT_SYMMETRY_GENERATORS",
    "ClauseConstrainedPrimeFieldFlatClassification",
    "ClauseConstrainedPrimeFieldFlatProblem",
    "ClauseConstrainedPrimeFieldFlatRequest",
    "PrimeFieldFlatClassificationComplete",
    "PrimeFieldFlatClassificationIncomplete",
    "PrimeFieldFlatClassificationOutcome",
    "PrimeFieldFlatOrbitRepresentative",
    "PrimeFieldFlatRankInterval",
    "PrimeFieldFlatSymmetryGenerator",
    "PrimeFieldRowMatrix",
    "PrimeFieldVectorConfiguration",
    "PrimeFieldVectorSpaceBasis",
]
