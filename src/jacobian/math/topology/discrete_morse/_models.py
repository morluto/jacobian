"""Typed contracts for exact discrete Morse matchings on simplicial complexes.

A discrete Morse matching is a family of codimension-one cover pairs on the
complete face closure of one finite simplicial complex such that each cell
occurs in at most one pair and the directed Hasse graph — unmatched cover
edges directed from the coface down to the face, matched edges reversed and
directed from the face up to the coface — has no directed cycle.  A directed
cycle is exactly a closed V-path and is returned as a concrete mathematical
obstruction, never as an operational failure.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_DIMENSION,
    FiniteSimplicialComplex,
    Simplex,
    SimplicialComplexRequest,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
)

MAX_MORSE_CELLS = 4096
MAX_MORSE_PAIRS = 2048
MAX_MORSE_HASSE_EDGES = 65536

# The Morse-complex and gradient-path outputs enumerate complete path families
# and materialize the reduced boundary, so they admit materially smaller inputs
# than the matching classifier.  Each ceiling follows from the quantity it
# actually bounds: critical cells from the graded basis, gradient paths from the
# complete alternating-path family, search states from the directed Hasse walk,
# and boundary entries from the nonzero reduced incidence rows.
MAX_MORSE_CRITICAL_CELLS = 1024
MAX_MORSE_GRADIENT_PATHS = 4096
MAX_MORSE_GRADIENT_STATES = 200_000
MAX_MORSE_BOUNDARY_ENTRIES = 8192
MAX_MORSE_PATH_STEPS = MAX_MORSE_CELLS
MAX_MORSE_CONTRACTION_CELLS = 32
MAX_MORSE_CONTRACTION_FACE_CANDIDATES = 512
MAX_MORSE_CONTRACTION_PAIRS = 14
MAX_MORSE_CONTRACTION_COEFFICIENT_DIGITS = 64
MorseContractionRow = Annotated[
    tuple[ExactInteger, ...], Field(max_length=MAX_MORSE_CONTRACTION_CELLS)
]
MorseContractionMatrix = Annotated[
    tuple[MorseContractionRow, ...], Field(max_length=MAX_MORSE_CONTRACTION_CELLS)
]
MorseContractionFamily = Annotated[
    tuple[MorseContractionMatrix, ...],
    Field(min_length=1, max_length=MAX_TOPOLOGY_DIMENSION + 1),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"topology.discrete_morse.{reason}", message)


class MorseMatchingOutcome(StrEnum):
    """Discriminated outcome of one supplied matching construction."""

    ACYCLIC_MATCHING = "ACYCLIC_MATCHING"
    CYCLIC_MATCHING = "CYCLIC_MATCHING"
    INVALID_MATCHING = "INVALID_MATCHING"


class MorseMatchingFault(StrEnum):
    """Stable codes for the first matching-input obstruction."""

    UNKNOWN_CELL = "UNKNOWN_CELL"
    NOT_A_COVER_PAIR = "NOT_A_COVER_PAIR"
    DUPLICATE_CELL = "DUPLICATE_CELL"


class MatchingPair(StrictModel):
    """One candidate matching pair ``(face, coface)`` of source cells."""

    face: Simplex
    coface: Simplex


class DiscreteMorseMatchingRequest(StrictModel):
    """One bounded complex and a caller-supplied family of matching pairs.

    ``pairs`` may be empty; the empty matching is admitted and makes every
    face-closure cell critical.
    """

    complex: SimplicialComplexRequest
    pairs: tuple[MatchingPair, ...] = Field(default=())


class CriticalCellProfile(StrictModel):
    """The complete unmatched face family grouped by dimension.

    ``counts_by_dimension`` has one entry per dimension of the source
    complex, ``critical_cells`` lists every critical cell in canonical
    dimension/lexicographic order, and the two Euler characteristics — the
    alternating sum of the critical counts and of the full face counts — are
    both carried so the defining identity is inspectable.
    """

    counts_by_dimension: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_TOPOLOGY_DIMENSION + 1
    )
    critical_cells: tuple[Simplex, ...] = Field(default=(), max_length=MAX_MORSE_CELLS)
    euler_characteristic: StrictInt
    closure_euler_characteristic: StrictInt

    @model_validator(mode="after")
    def require_nonnegative_counts(self) -> Self:
        if any(count < 0 for count in self.counts_by_dimension):
            raise _validation_error(
                "negative_critical_count",
                "critical counts must be nonnegative",
            )
        return self


class DiscreteMorseMatchingResult(StrictModel):
    """Discriminated construction outcome bound to its canonical source.

    ``ACYCLIC_MATCHING`` carries the exact critical-cell profile and one
    topological order of the directed Hasse graph.  ``CYCLIC_MATCHING``
    carries one concrete closed V-path as an ordered tuple of distinct cells
    whose consecutive directed Hasse edges return to the first cell.
    ``INVALID_MATCHING`` carries the first fault with its pair index.
    """

    outcome: MorseMatchingOutcome
    complex: FiniteSimplicialComplex
    pairs: tuple[MatchingPair, ...] = Field(default=(), max_length=MAX_MORSE_PAIRS)
    critical_profile: CriticalCellProfile | None = None
    topological_order: tuple[Simplex, ...] = Field(
        default=(), max_length=MAX_MORSE_CELLS
    )
    hasse_edges: StrictInt | None = Field(default=None, ge=0, le=MAX_MORSE_HASSE_EDGES)
    closed_v_path: tuple[Simplex, ...] = Field(default=(), max_length=MAX_MORSE_CELLS)
    fault: MorseMatchingFault | None = None
    fault_message: str | None = Field(default=None, min_length=1, max_length=512)
    fault_pair_index: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_discriminated_payload(self) -> Self:
        canonical_cells = _canonical_cell_order(self.complex)
        if self.outcome is MorseMatchingOutcome.INVALID_MATCHING:
            self._require_invalid_payload()
        else:
            self._require_valid_matching(canonical_cells)
            if self.outcome is MorseMatchingOutcome.ACYCLIC_MATCHING:
                self._require_acyclic_payload(canonical_cells)
            else:
                self._require_cyclic_payload(canonical_cells)
        return self

    def _require_invalid_payload(self) -> None:
        if self.fault is None or not self.fault_message:
            raise _validation_error(
                "invalid_requires_fault",
                "an invalid matching result requires a fault code and message",
            )
        if self.fault_pair_index is None:
            raise _validation_error(
                "invalid_requires_fault_index",
                "an invalid matching result requires the faulting pair index",
            )
        if (
            self.pairs
            or self.topological_order
            or self.closed_v_path
            or self.critical_profile is not None
            or self.hasse_edges is not None
        ):
            raise _validation_error(
                "invalid_payload_conflict",
                "an invalid matching result carries only its fault",
            )

    def _require_valid_matching(self, canonical_cells: tuple[Simplex, ...]) -> None:
        if (
            self.fault is not None
            or self.fault_message is not None
            or self.fault_pair_index is not None
        ):
            raise _validation_error(
                "payload_conflict",
                "a valid matching result cannot carry an input fault",
            )
        if tuple(sorted((pair.face, pair.coface) for pair in self.pairs)) != tuple(
            (pair.face, pair.coface) for pair in self.pairs
        ):
            raise _validation_error(
                "pairs_not_canonical",
                "accepted matching pairs must be canonical, sorted, and unique",
            )
        cells = {cell for pair in self.pairs for cell in (pair.face, pair.coface)}
        if len(cells) != 2 * len(self.pairs):
            raise _validation_error(
                "pairs_not_disjoint",
                "each cell may occur in at most one matching pair",
            )
        if not cells.issubset(canonical_cells):
            raise _validation_error(
                "pair_cell_outside_closure",
                "matching cells must lie in the source face closure",
            )

    def _require_acyclic_payload(self, canonical_cells: tuple[Simplex, ...]) -> None:
        profile = self.critical_profile
        if profile is None or self.hasse_edges is None:
            raise _validation_error(
                "acyclic_requires_profile",
                "an acyclic result requires its critical-cell profile",
            )
        if self.closed_v_path:
            raise _validation_error(
                "acyclic_rejects_cycle",
                "an acyclic result cannot carry a closed V-path",
            )
        if len(profile.counts_by_dimension) != self.complex.dimension + 1:
            raise _validation_error(
                "profile_dimension_mismatch",
                "critical counts must cover every source dimension",
            )
        if sum(profile.counts_by_dimension) != len(profile.critical_cells):
            raise _validation_error(
                "profile_count_mismatch",
                "critical counts must equal the explicit critical cells",
            )
        if len(profile.critical_cells) != self.complex.closure_size - 2 * len(
            self.pairs
        ):
            raise _validation_error(
                "profile_partition_mismatch",
                "critical and matched cells must partition the face closure",
            )
        if profile.euler_characteristic != profile.closure_euler_characteristic:
            raise _validation_error(
                "euler_identity_violated",
                "critical-count Euler characteristic must equal the closure value",
            )
        if not self.topological_order:
            raise _validation_error(
                "topological_order_required",
                "an acyclic result requires its Hasse topological order",
            )
        if tuple(sorted(self.topological_order)) != tuple(sorted(canonical_cells)):
            raise _validation_error(
                "topological_order_incomplete",
                "a topological order must contain every face-closure cell once",
            )

    def _require_cyclic_payload(self, canonical_cells: tuple[Simplex, ...]) -> None:
        if self.critical_profile is not None or self.hasse_edges is not None:
            raise _validation_error(
                "cyclic_rejects_profile",
                "a cyclic result carries only its closed V-path",
            )
        if self.topological_order:
            raise _validation_error(
                "cyclic_rejects_order",
                "a cyclic result cannot carry a topological order",
            )
        if len(self.closed_v_path) < 2:
            raise _validation_error(
                "cycle_too_short",
                "a closed V-path requires at least two distinct cells",
            )
        if len(set(self.closed_v_path)) != len(self.closed_v_path):
            raise _validation_error(
                "cycle_not_simple",
                "a closed V-path must list distinct cells without repetition",
            )
        if not set(self.closed_v_path).issubset(canonical_cells):
            raise _validation_error(
                "cycle_outside_closure",
                "closed V-path cells must lie in the source face closure",
            )

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build after the admitted matching kernel established the outcome."""

        return cls.model_construct(**values)


class MinimumMorseMatchingRequest(StrictModel):
    """Request a minimum-total-critical-cell acyclic matching."""

    complex: SimplicialComplexRequest


class MinimumMorseMatchingResult(StrictModel):
    """An acyclic matching attaining the minimum total critical-cell count."""

    matching: DiscreteMorseMatchingResult
    minimum_critical_cell_count: StrictInt = Field(ge=0, le=MAX_MORSE_CELLS)

    @model_validator(mode="after")
    def require_matching_objective(self) -> Self:
        profile = self.matching.critical_profile
        if (
            self.matching.outcome is not MorseMatchingOutcome.ACYCLIC_MATCHING
            or profile is None
            or sum(profile.counts_by_dimension) != self.minimum_critical_cell_count
        ):
            raise _validation_error(
                "minimum_result_objective_mismatch",
                "the minimum critical-cell count must equal an acyclic matching profile",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build after the exact search established the optimum."""

        return cls.model_construct(**values)


def _require_canonical_cell(cell: Simplex, reason: str) -> None:
    if len(set(cell)) != len(cell) or tuple(sorted(cell)) != cell:
        raise _validation_error(reason, "cells must be canonical vertex subsets")


class MorseGradientStepKind(StrEnum):
    """Direction of one cover step of a discrete Morse gradient path.

    A ``DOWN`` step traverses an unmatched cover from a coface to a
    codimension-one face; an ``UP`` step traverses a matched cover from a face
    to the coface paired with it.
    """

    DOWN = "DOWN"
    UP = "UP"


class GradientPathStep(StrictModel):
    """One oriented cover step of one gradient path."""

    kind: MorseGradientStepKind
    source: Simplex = Field(min_length=1)
    target: Simplex = Field(min_length=1)

    @model_validator(mode="after")
    def require_canonical_cover_step(self) -> Self:
        _require_canonical_cell(self.source, "non_canonical_gradient_step")
        _require_canonical_cell(self.target, "non_canonical_gradient_step")
        expected = (
            len(self.source) + 1
            if self.kind is MorseGradientStepKind.UP
            else len(self.source) - 1
        )
        if len(self.target) != expected:
            raise _validation_error(
                "gradient_step_dimension",
                "a gradient step must join consecutive dimensions in its direction",
            )
        return self


class GradientPath(StrictModel):
    """One complete alternating gradient path between adjacent critical cells.

    The steps begin and end with a ``DOWN`` step, so ``start`` has dimension
    one above ``target``.  The path is exactly the finite alternating
    matched/unmatched incidence sequence contributing to the Morse boundary
    coefficient from ``start`` to ``target``.
    """

    start: Simplex = Field(min_length=1)
    target: Simplex = Field(min_length=1)
    steps: tuple[GradientPathStep, ...] = Field(
        min_length=1, max_length=MAX_MORSE_PATH_STEPS
    )

    @model_validator(mode="after")
    def require_alternating_replay(self) -> Self:
        kinds = tuple(step.kind for step in self.steps)
        if kinds[0] is not MorseGradientStepKind.DOWN or (
            kinds[-1] is not MorseGradientStepKind.DOWN
        ):
            raise _validation_error(
                "gradient_path_shape",
                "a gradient path begins and ends with a DOWN step",
            )
        for index, kind in enumerate(kinds):
            expected = (
                MorseGradientStepKind.DOWN
                if index % 2 == 0
                else MorseGradientStepKind.UP
            )
            if kind is not expected:
                raise _validation_error(
                    "gradient_path_alternation",
                    "gradient-path steps must alternate DOWN, UP, DOWN, ...",
                )
        if self.steps[0].source != self.start or self.steps[-1].target != self.target:
            raise _validation_error(
                "gradient_path_binding",
                "gradient-path start and target must match its first and last steps",
            )
        for left, right in zip(self.steps, self.steps[1:], strict=False):
            if right.source != left.target:
                raise _validation_error(
                    "gradient_path_chain",
                    "consecutive gradient steps must share their intermediate cell",
                )
        return self


class GradientPathCount(StrictModel):
    """The number of bounded gradient paths ending at one critical cell."""

    target: Simplex = Field(min_length=1)
    count: StrictInt = Field(ge=1, le=MAX_MORSE_GRADIENT_PATHS)


class GradientPathsRequest(StrictModel):
    """One acyclic matching, a critical start cell, and an optional target.

    ``target`` is a critical cell one dimension below ``start``; when omitted
    the result enumerates every gradient path from ``start`` to any critical
    cell of the adjacent lower dimension.
    """

    complex: SimplicialComplexRequest
    pairs: tuple[MatchingPair, ...] = Field(default=(), max_length=MAX_MORSE_PAIRS)
    start: Simplex = Field(min_length=1)
    target: Simplex | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_adjacent_critical_selection(self) -> Self:
        _require_canonical_cell(self.start, "non_canonical_gradient_start")
        if self.target is None:
            return self
        _require_canonical_cell(self.target, "non_canonical_gradient_target")
        if len(self.target) != len(self.start) - 1:
            raise _validation_error(
                "gradient_target_dimension",
                "a gradient-path target must lie one dimension below its start",
            )
        return self


class GradientPathsResult(StrictModel):
    """The complete bounded family of gradient paths from one critical cell.

    Every path is replayed as its explicit alternating down/up step sequence.
    ``counts_by_target`` groups the family by terminal critical cell.
    """

    complex: FiniteSimplicialComplex
    pairs: tuple[MatchingPair, ...] = Field(default=(), max_length=MAX_MORSE_PAIRS)
    critical_profile: CriticalCellProfile
    start: Simplex = Field(min_length=1)
    target: Simplex | None = Field(default=None, min_length=1)
    paths: tuple[GradientPath, ...] = Field(
        default=(), max_length=MAX_MORSE_GRADIENT_PATHS
    )
    counts_by_target: tuple[GradientPathCount, ...] = Field(
        default=(), max_length=MAX_MORSE_CELLS
    )

    @model_validator(mode="after")
    def require_bounded_family(self) -> Self:
        if any(path.start != self.start for path in self.paths):
            raise _validation_error(
                "gradient_path_family_start",
                "every returned gradient path must start at the selected cell",
            )
        if self.target is not None and any(
            path.target != self.target for path in self.paths
        ):
            raise _validation_error(
                "gradient_path_family_target",
                "every returned gradient path must end at the selected target",
            )
        if sum(item.count for item in self.counts_by_target) != len(self.paths):
            raise _validation_error(
                "gradient_path_count_mismatch",
                "gradient-path counts must partition the returned family",
            )
        targets = tuple(item.target for item in self.counts_by_target)
        if targets != tuple(sorted(set(targets))):
            raise _validation_error(
                "gradient_path_counts_order",
                "gradient-path counts must be unique and canonically ordered",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build after the admitted gradient-path kernel established the family."""

        return cls.model_construct(**values)


class CriticalCellBasis(StrictModel):
    """The critical cells of one complex dimension in canonical order."""

    dimension: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)
    cells: tuple[Simplex, ...] = Field(default=(), max_length=MAX_MORSE_CRITICAL_CELLS)

    @model_validator(mode="after")
    def require_canonical_basis(self) -> Self:
        if any(
            len(cell) != self.dimension + 1 or tuple(sorted(cell)) != cell
            for cell in self.cells
        ):
            raise _validation_error(
                "critical_basis_dimension",
                "basis cells must be canonical simplices of the declared dimension",
            )
        if tuple(sorted(set(self.cells))) != self.cells:
            raise _validation_error(
                "critical_basis_order",
                "basis cells must be unique and lexicographically ordered",
            )
        return self


class MorseBoundaryEntry(StrictModel):
    """One nonzero gradient-path contribution to the reduced Morse boundary.

    ``gradient_path_count`` is the exact number of gradient paths from
    ``source`` to ``target``; ``coefficient`` is its GF(2) parity, the signed
    incidence over the characteristic-two field where orientation signs vanish.
    """

    source: Simplex = Field(min_length=2)
    target: Simplex = Field(min_length=1)
    gradient_path_count: StrictInt = Field(ge=1, le=MAX_MORSE_GRADIENT_PATHS)
    coefficient: StrictInt = Field(ge=0, le=1)

    @model_validator(mode="after")
    def require_gf2_incidence(self) -> Self:
        if len(self.target) != len(self.source) - 1:
            raise _validation_error(
                "morse_boundary_dimension",
                "a Morse boundary entry must join adjacent dimensions",
            )
        if self.coefficient != self.gradient_path_count % 2:
            raise _validation_error(
                "morse_boundary_coefficient",
                "the GF(2) coefficient must equal the gradient-path parity",
            )
        return self


class MorseComplexRequest(StrictModel):
    """One acyclic matching over a bounded complex, taken over GF(2)."""

    complex: SimplicialComplexRequest
    pairs: tuple[MatchingPair, ...] = Field(default=(), max_length=MAX_MORSE_PAIRS)
    coefficient_field: Literal["GF(2)"] = "GF(2)"


class IntegerMorseComplexRequest(StrictModel):
    """One simplicial complex and supplied matching for integral reduction."""

    complex: SimplicialComplexRequest
    pairs: tuple[MatchingPair, ...] = Field(default=(), max_length=MAX_MORSE_PAIRS)


class MorseComplexResult(StrictModel):
    """The graded Morse complex of one acyclic matching over GF(2).

    The basis is the critical cells by dimension, the boundary rows are the
    signed GF(2) incidences recovered from complete gradient-path counts, and
    the Euler characteristic and Betti numbers are the reduced complex's own.
    """

    complex: FiniteSimplicialComplex
    pairs: tuple[MatchingPair, ...] = Field(default=(), max_length=MAX_MORSE_PAIRS)
    critical_profile: CriticalCellProfile
    coefficient_field: Literal["GF(2)"] = "GF(2)"
    critical_cells_by_dimension: tuple[CriticalCellBasis, ...] = Field(
        min_length=1, max_length=MAX_TOPOLOGY_DIMENSION + 1
    )
    boundary_entries: tuple[MorseBoundaryEntry, ...] = Field(
        default=(), max_length=MAX_MORSE_BOUNDARY_ENTRIES
    )
    gradient_path_total: StrictInt = Field(ge=0, le=MAX_MORSE_GRADIENT_PATHS)
    boundary_square_zero: bool
    morse_euler_characteristic: StrictInt
    closure_euler_characteristic: StrictInt
    betti_numbers: tuple[StrictInt, ...] = Field(
        default=(), max_length=MAX_TOPOLOGY_DIMENSION + 1
    )

    @model_validator(mode="after")
    def require_graded_morse_complex(self) -> Self:
        dimensions = tuple(
            basis.dimension for basis in self.critical_cells_by_dimension
        )
        if dimensions != tuple(range(len(dimensions))):
            raise _validation_error(
                "morse_basis_dimensions",
                "Morse basis cells must cover contiguous dimensions from zero",
            )
        cells = tuple(
            cell for basis in self.critical_cells_by_dimension for cell in basis.cells
        )
        if tuple(sorted(cells)) != tuple(sorted(self.critical_profile.critical_cells)):
            raise _validation_error(
                "morse_basis_partition",
                "the Morse basis must be exactly the critical-cell family",
            )
        counts = tuple(len(basis.cells) for basis in self.critical_cells_by_dimension)
        if counts != self.critical_profile.counts_by_dimension:
            raise _validation_error(
                "morse_basis_counts",
                "Morse basis sizes must match the critical counts by dimension",
            )
        euler = sum((-1) ** dimension * count for dimension, count in enumerate(counts))
        if (
            self.morse_euler_characteristic != euler
            or self.closure_euler_characteristic != euler
        ):
            raise _validation_error(
                "morse_euler_identity",
                "critical-count and closure Euler characteristics must agree",
            )
        if len(self.betti_numbers) != len(dimensions) or any(
            value < 0 for value in self.betti_numbers
        ):
            raise _validation_error(
                "morse_betti_shape",
                "Betti numbers must be nonnegative and cover every dimension",
            )
        sources = tuple(entry.source for entry in self.boundary_entries)
        if sources != tuple(sorted(sources)):
            raise _validation_error(
                "morse_boundary_order",
                "Morse boundary entries must be canonically ordered by source cell",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build after the admitted Morse-complex kernel established the boundary."""

        return cls.model_construct(**values)


class IntegerMorseComplexResult(StrictModel):
    """The integral Morse chain complex with critical-cell coordinate axes."""

    complex: FiniteSimplicialComplex
    pairs: tuple[MatchingPair, ...] = Field(default=(), max_length=MAX_MORSE_PAIRS)
    critical_profile: CriticalCellProfile
    critical_cells_by_dimension: tuple[CriticalCellBasis, ...] = Field(
        min_length=1, max_length=MAX_TOPOLOGY_DIMENSION + 1
    )
    chain_complex: ChainComplexValue
    gradient_path_total: StrictInt = Field(ge=0, le=MAX_MORSE_GRADIENT_PATHS)

    @model_validator(mode="after")
    def require_integral_morse_axes(self) -> Self:
        if self.chain_complex.coefficient_ring is not CoefficientRing.INTEGER:
            raise _validation_error(
                "integer_morse_coefficient_ring",
                "the integral Morse chain complex must use ZZ coefficients",
            )
        if (
            self.chain_complex.degree_min != 0
            or self.chain_complex.degree_max
            != len(self.critical_cells_by_dimension) - 1
        ):
            raise _validation_error(
                "integer_morse_degree_axes",
                "integer Morse chain degrees must match critical-cell dimensions",
            )
        axis_counts = tuple(
            len(basis.cells) for basis in self.critical_cells_by_dimension
        )
        if self.critical_profile.counts_by_dimension != axis_counts:
            raise _validation_error(
                "integer_morse_profile_counts",
                "critical profile counts must match the labeled critical-cell axes",
            )
        if self.chain_complex.basis_sizes != axis_counts:
            raise _validation_error(
                "integer_morse_basis_sizes",
                "integer Morse chain ranks must match the labeled critical-cell axes",
            )
        if tuple(
            basis.dimension for basis in self.critical_cells_by_dimension
        ) != tuple(range(len(self.critical_cells_by_dimension))):
            raise _validation_error(
                "integer_morse_basis_dimensions",
                "critical-cell axes must cover contiguous dimensions from zero",
            )
        if tuple(
            sorted(
                cell
                for basis in self.critical_cells_by_dimension
                for cell in basis.cells
            )
        ) != tuple(sorted(self.critical_profile.critical_cells)):
            raise _validation_error(
                "integer_morse_basis_partition",
                "critical-cell axes must partition the source critical cells",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build after the bounded signed reduction established its chain value."""

        return cls.model_construct(**values)


class MorseChainContractionRequest(StrictModel):
    """A bounded finite complex and a supplied acyclic matching."""

    complex: SimplicialComplexRequest
    pairs: tuple[MatchingPair, ...] = Field(
        default=(), max_length=MAX_MORSE_CONTRACTION_PAIRS
    )


class MorseChainContractionResult(StrictModel):
    """An integral strong deformation retraction onto the Morse chain complex.

    Inclusion and projection are degree-zero chain maps. The degree-raising
    homotopy uses ``id - inclusion * projection = d * H + H * d``.
    Matrices use the canonical simplex axes carried by the source and critical
    cell bases.
    """

    complex: FiniteSimplicialComplex
    pairs: tuple[MatchingPair, ...] = Field(
        default=(), max_length=MAX_MORSE_CONTRACTION_PAIRS
    )
    source_chain_complex: ChainComplexValue
    critical_cells_by_dimension: tuple[CriticalCellBasis, ...] = Field(
        min_length=1, max_length=MAX_TOPOLOGY_DIMENSION + 1
    )
    morse_chain_complex: ChainComplexValue
    inclusion_matrices: MorseContractionFamily
    projection_matrices: MorseContractionFamily
    homotopy_matrices: MorseContractionFamily

    @model_validator(mode="after")
    def require_canonical_contraction_axes(self) -> Self:
        source = self.source_chain_complex
        target = self.morse_chain_complex
        if (
            source.coefficient_ring is not CoefficientRing.INTEGER
            or target.coefficient_ring is not CoefficientRing.INTEGER
            or source.prime is not None
            or target.prime is not None
            or source.degree_min != 0
            or target.degree_min != 0
            or source.degree_max != self.complex.dimension
            or target.degree_max != self.complex.dimension
            or len(source.basis_sizes) != len(target.basis_sizes)
            or source.basis_sizes != self.complex.f_vector
            or sum(source.basis_sizes) > MAX_MORSE_CONTRACTION_CELLS
            or len(self.pairs) > MAX_MORSE_CONTRACTION_PAIRS
            or tuple(basis.dimension for basis in self.critical_cells_by_dimension)
            != tuple(range(len(target.basis_sizes)))
            or tuple(len(basis.cells) for basis in self.critical_cells_by_dimension)
            != target.basis_sizes
        ):
            raise _validation_error(
                "chain_contraction_axes",
                "contraction endpoints must be ZZ complexes with canonical matching dimensions",
            )
        source_cells = set(_canonical_cell_order(self.complex))
        matched_cells: set[Simplex] = set()
        for pair in self.pairs:
            if (
                pair.face not in source_cells
                or pair.coface not in source_cells
                or len(pair.coface) != len(pair.face) + 1
                or not set(pair.face).issubset(pair.coface)
                or pair.face in matched_cells
                or pair.coface in matched_cells
            ):
                raise _validation_error(
                    "chain_contraction_matching_binding",
                    "matching pairs must be distinct source cover cells",
                )
            matched_cells.update((pair.face, pair.coface))
        critical_cells = tuple(
            cell for basis in self.critical_cells_by_dimension for cell in basis.cells
        )
        if (
            len(set(critical_cells)) != len(critical_cells)
            or not set(critical_cells).issubset(source_cells)
            or set(critical_cells) != source_cells - matched_cells
        ):
            raise _validation_error(
                "chain_contraction_critical_binding",
                "critical bases must partition exactly the unmatched source cells",
            )
        dimensions = len(source.basis_sizes)
        if any(
            len(matrices) != dimensions
            for matrices in (
                self.inclusion_matrices,
                self.projection_matrices,
                self.homotopy_matrices,
            )
        ):
            raise _validation_error(
                "chain_contraction_degree_count",
                "inclusion, projection, and homotopy must carry every degree",
            )
        map_entry_count = sum(
            len(row)
            for matrices in (
                self.inclusion_matrices,
                self.projection_matrices,
                self.homotopy_matrices,
            )
            for matrix in matrices
            for row in matrix
        )
        if map_entry_count > 3 * MAX_MORSE_CONTRACTION_CELLS**2:
            raise _validation_error(
                "chain_contraction_output_bound",
                "chain-contraction maps exceed their admitted matrix-entry bound",
            )
        map_entries = (
            entry
            for matrices in (
                self.inclusion_matrices,
                self.projection_matrices,
                self.homotopy_matrices,
            )
            for matrix in matrices
            for row in matrix
            for entry in row
        )
        if any(
            len(str(abs(entry))) > MAX_MORSE_CONTRACTION_COEFFICIENT_DIGITS
            for entry in map_entries
        ):
            raise _validation_error(
                "chain_contraction_coefficient_bound",
                "chain-contraction map coefficients exceed the 64-digit bound",
            )
        for degree, (source_rank, target_rank) in enumerate(
            zip(source.basis_sizes, target.basis_sizes, strict=True)
        ):
            inclusion = self.inclusion_matrices[degree]
            projection = self.projection_matrices[degree]
            homotopy = self.homotopy_matrices[degree]
            if len(inclusion) != source_rank or any(
                len(row) != target_rank for row in inclusion
            ):
                raise _validation_error(
                    "chain_contraction_inclusion_shape",
                    "inclusion matrix axes must be source-by-critical in each degree",
                )
            if len(projection) != target_rank or any(
                len(row) != source_rank for row in projection
            ):
                raise _validation_error(
                    "chain_contraction_projection_shape",
                    "projection matrix axes must be critical-by-source in each degree",
                )
            next_rank = source.basis_sizes[degree + 1] if degree + 1 < dimensions else 0
            if len(homotopy) != next_rank or any(
                len(row) != source_rank for row in homotopy
            ):
                raise _validation_error(
                    "chain_contraction_homotopy_shape",
                    "homotopy matrix axes must map source degree n to source degree n+1",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build after exact contraction identities have been checked once."""

        return cls.model_construct(**values)


def _canonical_cell_order(complex_: FiniteSimplicialComplex) -> tuple[Simplex, ...]:
    return tuple(face for group in complex_.faces_by_dimension for face in group.faces)


__all__ = [
    "MAX_MORSE_BOUNDARY_ENTRIES",
    "MAX_MORSE_CELLS",
    "MAX_MORSE_CONTRACTION_CELLS",
    "MAX_MORSE_CONTRACTION_COEFFICIENT_DIGITS",
    "MAX_MORSE_CONTRACTION_FACE_CANDIDATES",
    "MAX_MORSE_CONTRACTION_PAIRS",
    "MAX_MORSE_CRITICAL_CELLS",
    "MAX_MORSE_GRADIENT_PATHS",
    "MAX_MORSE_GRADIENT_STATES",
    "MAX_MORSE_HASSE_EDGES",
    "MAX_MORSE_PAIRS",
    "MAX_MORSE_PATH_STEPS",
    "CriticalCellBasis",
    "CriticalCellProfile",
    "DiscreteMorseMatchingRequest",
    "DiscreteMorseMatchingResult",
    "GradientPath",
    "GradientPathCount",
    "GradientPathStep",
    "GradientPathsRequest",
    "GradientPathsResult",
    "IntegerMorseComplexRequest",
    "IntegerMorseComplexResult",
    "MatchingPair",
    "MorseBoundaryEntry",
    "MorseComplexRequest",
    "MorseComplexResult",
    "MorseGradientStepKind",
    "MorseMatchingFault",
    "MorseMatchingOutcome",
]
