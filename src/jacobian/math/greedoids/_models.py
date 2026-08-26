"""Typed wire contracts for greedoid operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.greedoids.values import FiniteFeasibleSetSystem

MAX_GROUND_SIZE = 64
"""Schema-visible cap on ground-set cardinality for greedoid requests."""

MAX_FEASIBLE_COUNT = 4096
"""Schema-visible cap on feasible-row count for greedoid requests."""

MAX_EXCHANGE_WORK = 2_000_000
"""Maximum ordered exchange candidate-membership checks per recognition."""

MAX_GROUND_LABEL_UTF8_BYTES = 1_024
"""Maximum UTF-8 size of one retained ground label."""

MAX_GROUND_LABEL_TOTAL_UTF8_BYTES = 65_536
"""Maximum UTF-8 storage for all retained ground labels."""

MAX_INTERMEDIATE_MEMBERSHIPS = 262_144
"""Maximum feasible-row membership storage inspected by a kernel."""

MAX_RESULT_BYTES = 2_000_000
"""Conservative bound for complete index-family results."""


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by greedoid contracts."""

    return PydanticCustomError(f"greedoid.{reason}", message)


class GreedoidAdmissionError(ValueError):
    """Native admission failure for greedoid operations."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def require_bounded_carrier(system: FiniteFeasibleSetSystem) -> None:
    """Bound the greedoid execution envelope before any kernel expands.

    The shared carrier is structural only; these operation-owned ceilings
    control the ordered-pair and family-scan work of the greedoid kernels.
    """

    if len(system.ground) > MAX_GROUND_SIZE:
        raise GreedoidAdmissionError(
            "ground_size_exceeds_budget",
            f"ground size exceeds the bounded budget of {MAX_GROUND_SIZE} elements",
        )
    if len(system.feasible) > MAX_FEASIBLE_COUNT:
        raise GreedoidAdmissionError(
            "feasible_count_exceeds_budget",
            f"feasible-set count exceeds the bounded budget of "
            f"{MAX_FEASIBLE_COUNT} rows",
        )
    label_bytes = 0
    for label in system.ground:
        try:
            encoded_length = len(label.encode("utf-8"))
        except UnicodeError as exc:
            raise GreedoidAdmissionError(
                "ground_label_invalid_utf8",
                "ground labels must be UTF-8 encodable",
            ) from exc
        if encoded_length > MAX_GROUND_LABEL_UTF8_BYTES:
            raise GreedoidAdmissionError(
                "ground_label_exceeds_budget",
                "a ground label exceeds the bounded UTF-8 size budget",
            )
        label_bytes += encoded_length
    if label_bytes > MAX_GROUND_LABEL_TOTAL_UTF8_BYTES:
        raise GreedoidAdmissionError(
            "ground_labels_exceed_budget",
            "ground labels exceed the bounded UTF-8 storage budget",
        )
    by_size: dict[int, int] = {}
    memberships = 0
    for row in system.feasible:
        by_size[len(row)] = by_size.get(len(row), 0) + 1
        memberships += len(row)
    if memberships > MAX_INTERMEDIATE_MEMBERSHIPS:
        raise GreedoidAdmissionError(
            "intermediate_memberships_exceed_budget",
            "feasible-set memberships exceed the bounded intermediate-storage budget",
        )
    exchange_pairs = sum(
        larger_count * smaller_count
        for larger_size, larger_count in by_size.items()
        for smaller_size, smaller_count in by_size.items()
        if larger_size > smaller_size
    )
    candidate_membership_probes = exchange_pairs * len(system.ground)
    accessibility_probes = memberships
    total_probes = candidate_membership_probes + accessibility_probes
    if total_probes > MAX_EXCHANGE_WORK:
        raise GreedoidAdmissionError(
            "exchange_work_exceeds_budget",
            "exhaustive exchange and accessibility membership work exceeds the bounded budget",
        )
    # Recognition and bases can retain every feasible row. Charge the exact
    # index-family shape, including tuple delimiters and the source labels,
    # before any kernel allocates its working index.
    result_bytes = label_bytes + memberships * 3 + len(system.feasible) * 16
    if result_bytes > MAX_RESULT_BYTES:
        raise GreedoidAdmissionError(
            "result_exceeds_budget",
            "the complete greedoid result exceeds the bounded result budget",
        )


class RecognizeRequest(StrictModel):
    """Recognize a feasible-set family as a greedoid."""

    system: FiniteFeasibleSetSystem

    @model_validator(mode="after")
    def require_bounded_system(self) -> Self:
        try:
            require_bounded_carrier(self.system)
        except GreedoidAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
        return self


class RecognizeResult(StrictModel):
    """``GREEDOID`` with rank/bases, or ``NOT_A_GREEDOID`` with the first obstruction."""

    status: str
    obstruction: str | None = None
    larger_set: tuple[int, ...] | None = None
    smaller_set: tuple[int, ...] | None = None
    feasible_set: tuple[int, ...] | None = None
    rank: int | None = None
    bases: tuple[tuple[int, ...], ...] = ()
    ground_size: int | None = None

    @model_validator(mode="after")
    def bind_status(self) -> Self:
        if self.status not in ("GREEDOID", "NOT_A_GREEDOID"):
            raise _validation_error(
                "recognize_status_invalid", "status must be GREEDOID or NOT_A_GREEDOID"
            )
        if self.status == "GREEDOID":
            if self.obstruction is not None:
                raise _validation_error(
                    "greedoid_has_obstruction", "a GREEDOID result has no obstruction"
                )
        else:
            if self.obstruction is None:
                raise _validation_error(
                    "non_greedoid_missing_obstruction",
                    "a NOT_A_GREEDOID result must name an obstruction",
                )
        return self


class RankRequest(StrictModel):
    """Compute greedoid rank for an optional ground subset."""

    system: FiniteFeasibleSetSystem
    subset: tuple[int, ...] | None = Field(default=None, max_length=MAX_GROUND_SIZE)

    @model_validator(mode="after")
    def require_valid_subset(self) -> Self:
        try:
            require_bounded_carrier(self.system)
        except GreedoidAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
        if self.subset is not None:
            n = len(self.system.ground)
            if len(set(self.subset)) != len(self.subset):
                raise _validation_error(
                    "subset_duplicate", "subset must not contain duplicates"
                )
            if any(not 0 <= i < n for i in self.subset):
                raise _validation_error(
                    "subset_index_out_of_range", "subset indices must be in range"
                )
        return self


class RankResult(StrictModel):
    """The greedoid rank of the supplied subset."""

    status: Literal["GREEDOID", "NOT_A_GREEDOID"] = "GREEDOID"
    rank: int | None = Field(default=None, ge=0)
    subset: tuple[int, ...] | None = Field(default=None)
    obstruction: str | None = None

    @model_validator(mode="after")
    def bind_status(self) -> Self:
        if self.status == "GREEDOID":
            if self.rank is None or self.obstruction is not None:
                raise _validation_error(
                    "rank_greedoid_branch_invalid",
                    "a GREEDOID result requires rank and has no obstruction",
                )
        elif self.rank is not None or self.obstruction is None:
            raise _validation_error(
                "rank_non_greedoid_branch_invalid",
                "a NOT_A_GREEDOID result requires an obstruction and no rank",
            )
        return self


class BasesRequest(StrictModel):
    """Compute the maximal feasible subsets (bases)."""

    system: FiniteFeasibleSetSystem
    subset: tuple[int, ...] | None = Field(default=None, max_length=MAX_GROUND_SIZE)

    @model_validator(mode="after")
    def require_valid_subset(self) -> Self:
        try:
            require_bounded_carrier(self.system)
        except GreedoidAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
        if self.subset is not None:
            n = len(self.system.ground)
            if len(set(self.subset)) != len(self.subset):
                raise _validation_error(
                    "subset_duplicate", "subset must not contain duplicates"
                )
            if any(not 0 <= i < n for i in self.subset):
                raise _validation_error(
                    "subset_index_out_of_range", "subset indices must be in range"
                )
        return self


class BasesResult(StrictModel):
    """The basis family and common rank."""

    status: Literal["GREEDOID", "NOT_A_GREEDOID"] = "GREEDOID"
    rank: int | None = Field(default=None, ge=0)
    bases: tuple[tuple[int, ...], ...]
    obstruction: str | None = None

    @model_validator(mode="after")
    def bind_status(self) -> Self:
        if self.status == "GREEDOID":
            if self.rank is None or self.obstruction is not None:
                raise _validation_error(
                    "bases_greedoid_branch_invalid",
                    "a GREEDOID result requires rank and has no obstruction",
                )
        elif self.rank is not None or self.bases or self.obstruction is None:
            raise _validation_error(
                "bases_non_greedoid_branch_invalid",
                "a NOT_A_GREEDOID result requires an obstruction with no rank or bases",
            )
        return self


class BasicWordProfileRequest(StrictModel):
    """Profile a candidate basic word."""

    system: FiniteFeasibleSetSystem
    word: tuple[int, ...] = Field(default=(), max_length=MAX_GROUND_SIZE)

    @model_validator(mode="after")
    def require_bounded_system(self) -> Self:
        try:
            require_bounded_carrier(self.system)
        except GreedoidAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
        return self


class BasicWordProfileResult(StrictModel):
    """Whether the word is a basic word, with first obstruction if not."""

    status: str
    obstruction: str | None = None
    prefix_index: int | None = None
    prefix_set: tuple[int, ...] | None = None
    prefix_length: int | None = None
    is_full: bool | None = None
    rank: int | None = None

    @model_validator(mode="after")
    def bind_status(self) -> Self:
        if self.status not in ("BASIC_WORD", "NOT_A_BASIC_WORD"):
            raise _validation_error(
                "basic_word_status_invalid",
                "status must be BASIC_WORD or NOT_A_BASIC_WORD",
            )
        return self


class ConvexGeometryRequest(StrictModel):
    """Compute the complementary closed-set family of a full-support antimatroid."""

    system: FiniteFeasibleSetSystem

    @model_validator(mode="after")
    def require_bounded_system(self) -> Self:
        try:
            require_bounded_carrier(self.system)
        except GreedoidAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
        return self


class ConvexGeometryResult(StrictModel):
    """The closed-set family and the feasible->closed complement map.

    ``complement_map`` is an ordered list of ``(feasible, closed)`` pairs so
    the wire representation stays JSON-safe.
    """

    status: Literal["ANTIMATROID", "NOT_AN_ANTIMATROID"] = "ANTIMATROID"
    closed_family: tuple[tuple[int, ...], ...] = ()
    complement_map: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
    obstruction: str | None = None

    @model_validator(mode="after")
    def bind_status(self) -> Self:
        if self.status == "ANTIMATROID":
            if self.obstruction is not None:
                raise _validation_error(
                    "antimatroid_has_obstruction",
                    "an ANTIMATROID result has no obstruction",
                )
        elif self.closed_family or self.complement_map or self.obstruction is None:
            raise _validation_error(
                "non_antimatroid_branch_invalid",
                "a NOT_AN_ANTIMATROID result requires an obstruction and no closed-family data",
            )
        return self


__all__ = [
    "BasesRequest",
    "BasesResult",
    "BasicWordProfileRequest",
    "BasicWordProfileResult",
    "ConvexGeometryRequest",
    "ConvexGeometryResult",
    "RankRequest",
    "RankResult",
    "RecognizeRequest",
    "RecognizeResult",
]
