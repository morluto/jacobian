"""Canonical finite delta-matroid values and exact exchange kernels."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import canonicalize_json
from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem

MAX_DELTA_MEMBERSHIPS = 1_024
MAX_DELTA_LABEL_BYTES = 2_048
MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS = 250_000
MAX_DELTA_RESULT_BYTES = 65_536


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"delta_matroid.{reason}", message)


class DeltaMatroidAdmissionError(ValueError):
    """Native admission failure for delta-matroid operations."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class DeltaMatroidObstruction(StrictModel):
    """The first exact obstruction to a complete feasible family."""

    kind: Literal["EMPTY_FEASIBLE_FAMILY", "SYMMETRIC_EXCHANGE"]
    left_feasible: tuple[int, ...] | None = None
    right_feasible: tuple[int, ...] | None = None
    element: int | None = Field(default=None, ge=0)
    symmetric_difference: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def require_kind_specific_witness(self) -> Self:
        witness = (
            self.left_feasible,
            self.right_feasible,
            self.element,
            self.symmetric_difference,
        )
        if self.kind == "EMPTY_FEASIBLE_FAMILY":
            if any(item is not None for item in witness):
                raise _validation_error(
                    "obstruction_witness_present",
                    "empty-family obstruction must not carry an exchange row",
                )
            return self
        if any(item is None for item in witness):
            raise _validation_error(
                "obstruction_witness_missing",
                "symmetric-exchange obstruction must carry its full row",
            )
        return self


def canonical_feasible_rows(
    system: FiniteFeasibleSetSystem,
) -> tuple[tuple[int, ...], ...]:
    """Return the one wire order for a complete feasible family."""

    return tuple(sorted(system.feasible))


def _exchange_work(
    rows: tuple[tuple[int, ...], ...],
) -> tuple[int, int]:
    """Count the complete exchange-instance and candidate-work envelope."""

    sets = tuple(frozenset(row) for row in rows)
    instances = 0
    candidate_space = 0
    for left in sets:
        for right in sets:
            difference_size = len(left ^ right)
            instances += difference_size
            candidate_space += difference_size * difference_size
    return instances, candidate_space


def _wire_size(system: FiniteFeasibleSetSystem) -> int:
    """Exact size of the source portion repeated by recognition results."""

    return len(
        canonicalize_json(
            {
                "ground": list(system.ground),
                "feasible": [list(row) for row in system.feasible],
            }
        )
    )


def require_delta_matroid_admission(system: FiniteFeasibleSetSystem) -> None:
    """Bound all work and the canonical recognition result before replay."""

    memberships = sum(len(row) for row in system.feasible)
    if memberships > MAX_DELTA_MEMBERSHIPS:
        raise DeltaMatroidAdmissionError(
            "memberships_exceeded",
            "delta-matroid feasible-family memberships exceed the "
            f"{MAX_DELTA_MEMBERSHIPS}-entry envelope",
        )
    try:
        label_bytes = sum(len(label.encode("utf-8")) for label in system.ground)
    except UnicodeEncodeError:
        raise DeltaMatroidAdmissionError(
            "labels_not_utf8",
            "delta-matroid ground labels must be UTF-8-representable",
        ) from None
    if label_bytes > MAX_DELTA_LABEL_BYTES:
        raise DeltaMatroidAdmissionError(
            "label_bytes_exceeded",
            "delta-matroid ground labels exceed the "
            f"{MAX_DELTA_LABEL_BYTES}-byte envelope",
        )
    # Every nonempty row carries at least one membership, so the membership
    # envelope bounds the row count and keeps this ordered-pair scan bounded.
    _, candidate_space = _exchange_work(canonical_feasible_rows(system))
    if candidate_space > MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS:
        raise DeltaMatroidAdmissionError(
            "candidate_work_exceeded",
            "delta-matroid symmetric-exchange candidate checks exceed the "
            f"{MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS}-check envelope",
        )
    # A valid result carries the source and canonical value, while an invalid
    # result carries the source and one bounded obstruction.  The factor and
    # fixed headroom cover both public shapes without expanding a new family.
    if 2 * _wire_size(system) + 4_096 > MAX_DELTA_RESULT_BYTES:
        raise DeltaMatroidAdmissionError(
            "result_bytes_exceeded",
            "delta-matroid recognition result exceeds the "
            f"{MAX_DELTA_RESULT_BYTES}-byte envelope",
        )


def first_symmetric_exchange_obstruction(
    system: FiniteFeasibleSetSystem,
) -> DeltaMatroidObstruction | None:
    """Return the deterministic first symmetric-exchange obstruction.

    The checked rows are ordered lexicographically by their canonical index
    tuples.  For every ordered feasible pair and every element of its
    symmetric difference, every candidate exchange element is tried in
    increasing index order until the required feasible set is found.
    """

    rows = canonical_feasible_rows(system)
    if not rows:
        return DeltaMatroidObstruction(kind="EMPTY_FEASIBLE_FAMILY")
    feasible = set(rows)
    for left_row in rows:
        left = frozenset(left_row)
        for right_row in rows:
            difference = tuple(sorted(left ^ frozenset(right_row)))
            for element in difference:
                if any(
                    tuple(sorted(left ^ frozenset((element, candidate)))) in feasible
                    for candidate in difference
                ):
                    continue
                return DeltaMatroidObstruction(
                    kind="SYMMETRIC_EXCHANGE",
                    left_feasible=left_row,
                    right_feasible=right_row,
                    element=element,
                    symmetric_difference=difference,
                )
    return None


class FiniteDeltaMatroid(StrictModel):
    """A complete, canonical finite feasible family satisfying symmetric exchange."""

    ground: tuple[str, ...] = Field()
    feasible: tuple[tuple[int, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_complete_canonical_delta_matroid(self) -> Self:
        system = FiniteFeasibleSetSystem(ground=self.ground, feasible=self.feasible)
        try:
            require_delta_matroid_admission(system)
        except DeltaMatroidAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
        expected_rows = canonical_feasible_rows(system)
        if self.feasible != expected_rows:
            raise _validation_error(
                "rows_not_canonical",
                "delta-matroid feasible rows must be lexicographically ordered",
            )
        obstruction = first_symmetric_exchange_obstruction(system)
        if obstruction is not None:
            raise _validation_error(
                "exchange_axiom_failed",
                "delta-matroid feasible rows violate symmetric exchange",
            )
        return self

    @classmethod
    def _from_kernel(cls, system: FiniteFeasibleSetSystem) -> FiniteDeltaMatroid:
        """Construct after the exhaustive recognition kernel passed."""

        return cls.model_construct(
            ground=system.ground,
            feasible=canonical_feasible_rows(system),
        )


__all__ = [
    "MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS",
    "MAX_DELTA_LABEL_BYTES",
    "MAX_DELTA_MEMBERSHIPS",
    "MAX_DELTA_RESULT_BYTES",
    "DeltaMatroidObstruction",
    "FiniteDeltaMatroid",
    "canonical_feasible_rows",
    "first_symmetric_exchange_obstruction",
    "require_delta_matroid_admission",
]
