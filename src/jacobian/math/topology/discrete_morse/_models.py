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
from typing import Any, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_DIMENSION,
    FiniteSimplicialComplex,
    Simplex,
    SimplicialComplexRequest,
)

MAX_MORSE_CELLS = 4096
MAX_MORSE_PAIRS = 2048
MAX_MORSE_HASSE_EDGES = 65536


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


def _canonical_cell_order(complex_: FiniteSimplicialComplex) -> tuple[Simplex, ...]:
    return tuple(face for group in complex_.faces_by_dimension for face in group.faces)


__all__ = [
    "MAX_MORSE_CELLS",
    "MAX_MORSE_HASSE_EDGES",
    "MAX_MORSE_PAIRS",
    "CriticalCellProfile",
    "DiscreteMorseMatchingRequest",
    "DiscreteMorseMatchingResult",
    "MatchingPair",
    "MorseMatchingFault",
    "MorseMatchingOutcome",
]
