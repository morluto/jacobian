"""Canonical finite matroids represented by their complete basis family."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Self

from pydantic import ConfigDict, Field, StrictInt, StrictStr, model_validator
from pydantic_core import PydanticCustomError

from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel, canonicalize_json_containers

MAX_FINITE_BASIS_GROUND_SIZE = 64
MAX_FINITE_BASIS_COUNT = 4_096
MAX_FINITE_BASIS_MEMBERSHIPS = 65_536
MAX_FINITE_BASIS_LABEL_BYTES = 1_024
MAX_FINITE_BASIS_TOTAL_LABEL_BYTES = 16_384
MAX_FINITE_BASIS_EXCHANGE_CHECKS = 2_000_000
_BASIS_CHECKPOINT_STRIDE = 4_096

_BasisIndex = Annotated[
    StrictInt,
    Field(ge=0, le=MAX_FINITE_BASIS_GROUND_SIZE - 1),
]
_GroundLabel = Annotated[StrictStr, Field(max_length=MAX_FINITE_BASIS_LABEL_BYTES)]
_BasisRow = Annotated[
    tuple[_BasisIndex, ...],
    Field(max_length=MAX_FINITE_BASIS_GROUND_SIZE),
]
_BasisFamily = Annotated[
    tuple[_BasisRow, ...],
    Field(min_length=1, max_length=MAX_FINITE_BASIS_COUNT),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"finite_basis_matroid.{reason}", message)


def _preflight_ground_axis(ground: object) -> None:
    if not isinstance(ground, (list, tuple)):
        return
    if len(ground) > MAX_FINITE_BASIS_GROUND_SIZE:
        raise _validation_error(
            "ground_size_bound",
            f"ground has more than {MAX_FINITE_BASIS_GROUND_SIZE} labels",
        )
    label_bytes = 0
    for label in ground:
        if not isinstance(label, str):
            continue
        if len(label) > MAX_FINITE_BASIS_LABEL_BYTES:
            raise _validation_error(
                "ground_label_bound",
                f"a ground label exceeds {MAX_FINITE_BASIS_LABEL_BYTES} characters",
            )
        try:
            encoded_length = len(label.encode("utf-8"))
        except UnicodeEncodeError:
            raise _validation_error(
                "ground_label_utf8", "ground labels must be UTF-8 encodable"
            ) from None
        if encoded_length > MAX_FINITE_BASIS_LABEL_BYTES:
            raise _validation_error(
                "ground_label_bound",
                "a ground label exceeds the per-label UTF-8 storage bound",
            )
        label_bytes += encoded_length
        if label_bytes > MAX_FINITE_BASIS_TOTAL_LABEL_BYTES:
            raise _validation_error(
                "ground_label_total_bound",
                "ground labels exceed the total UTF-8 storage bound",
            )


def _preflight_basis_family(bases: object) -> None:
    if not isinstance(bases, (list, tuple)):
        return
    if len(bases) > MAX_FINITE_BASIS_COUNT:
        raise _validation_error(
            "basis_count_bound",
            f"basis family has more than {MAX_FINITE_BASIS_COUNT} rows",
        )
    memberships = 0
    for row in bases:
        if not isinstance(row, (list, tuple)):
            continue
        if len(row) > MAX_FINITE_BASIS_GROUND_SIZE:
            raise _validation_error(
                "basis_size_bound",
                "a basis has more entries than the maximum ground size",
            )
        memberships += len(row)
        if memberships > MAX_FINITE_BASIS_MEMBERSHIPS:
            raise _validation_error(
                "membership_bound",
                "basis-family memberships exceed the admitted storage bound",
            )


def _preflight_raw_envelope(data: object) -> None:
    """Bound nested containers before Pydantic materializes tuple fields."""

    if not isinstance(data, Mapping):
        return
    _preflight_ground_axis(data.get("ground"))
    _preflight_basis_family(data.get("bases"))


class FiniteBasisMatroid(StrictModel):
    """A finite matroid bound to its complete, canonical basis family.

    The ground axis is retained even when all elements are loops. Bases are
    sorted ground-index tuples, sorted lexicographically as a family. The
    empty-ground matroid and every rank-zero matroid are represented by the
    single basis ``()``. Construction validates canonical structure only. Call
    ``require_basis_exchange`` before relying on the matroid claim; it does not
    claim field representability.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A finite matroid represented by its complete canonical basis "
                "family. Basis exchange must be checked explicitly under the "
                "published work limit before relying on the matroid claim. "
                "This carrier does not assert representability over a field."
            ),
            "admission_limits": {
                "max_ground_elements": MAX_FINITE_BASIS_GROUND_SIZE,
                "max_basis_rows": MAX_FINITE_BASIS_COUNT,
                "max_basis_memberships": MAX_FINITE_BASIS_MEMBERSHIPS,
                "max_ground_label_utf8_bytes_each": MAX_FINITE_BASIS_LABEL_BYTES,
                "max_ground_label_utf8_bytes_total": MAX_FINITE_BASIS_TOTAL_LABEL_BYTES,
                "max_basis_exchange_candidate_checks": MAX_FINITE_BASIS_EXCHANGE_CHECKS,
            },
        }
    )

    ground: tuple[_GroundLabel, ...] = Field(
        max_length=MAX_FINITE_BASIS_GROUND_SIZE,
        description=(
            "Unique labels retaining the matroid ground axis, including loops "
            f"and rank-zero elements; at most {MAX_FINITE_BASIS_GROUND_SIZE}."
        ),
    )
    bases: _BasisFamily = Field(
        description=(
            "Complete nonempty family of bases as lexicographically ordered, "
            "sorted tuples of distinct ground indices. Every basis has common "
            "cardinality equal to the matroid rank."
        )
    )

    @classmethod
    def _from_kernel(
        cls,
        *,
        ground: tuple[str, ...],
        bases: tuple[tuple[int, ...], ...],
    ) -> FiniteBasisMatroid:
        """Construct a basis carrier after its producing theorem was checked."""
        return cls.model_construct(ground=ground, bases=bases)

    @model_validator(mode="before")
    @classmethod
    def preflight_raw_envelope(cls, data: object) -> object:
        _preflight_raw_envelope(data)
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_canonical_basis_matroid(self) -> Self:
        ground_size = len(self.ground)
        if len(set(self.ground)) != ground_size:
            raise _validation_error("ground_duplicate", "ground labels must be unique")

        label_bytes = 0
        for label in self.ground:
            try:
                encoded_length = len(label.encode("utf-8"))
            except UnicodeEncodeError:
                raise _validation_error(
                    "ground_label_utf8", "ground labels must be UTF-8 encodable"
                ) from None
            if encoded_length > MAX_FINITE_BASIS_LABEL_BYTES:
                raise _validation_error(
                    "ground_label_bound",
                    "a ground label exceeds the per-label UTF-8 storage bound",
                )
            label_bytes += encoded_length
            if label_bytes > MAX_FINITE_BASIS_TOTAL_LABEL_BYTES:
                raise _validation_error(
                    "ground_label_total_bound",
                    "ground labels exceed the total UTF-8 storage bound",
                )

        if self.bases != tuple(sorted(set(self.bases))):
            raise _validation_error(
                "basis_family_canonical",
                "basis rows must be sorted, unique, and lexicographically ordered",
            )
        for basis in self.bases:
            if basis != tuple(sorted(set(basis))):
                raise _validation_error(
                    "basis_canonical", "each basis must be sorted and duplicate-free"
                )
            if any(index >= ground_size for index in basis):
                raise _validation_error(
                    "basis_index", "basis indices must lie on the retained ground axis"
                )

        rank = len(self.bases[0])
        if any(len(basis) != rank for basis in self.bases):
            raise _validation_error(
                "basis_cardinality", "all bases must have the same cardinality"
            )
        memberships = sum(len(basis) for basis in self.bases)
        if memberships > MAX_FINITE_BASIS_MEMBERSHIPS:
            raise _validation_error(
                "membership_bound",
                "basis-family memberships exceed the admitted storage bound",
            )

        return self

    def require_basis_exchange(self) -> None:
        maximum_difference = min(self.rank, self.ground_size - self.rank)
        work_bound = len(self.bases) ** 2 * maximum_difference**2
        if work_bound > MAX_FINITE_BASIS_EXCHANGE_CHECKS:
            raise _validation_error(
                "exchange_work_bound",
                "worst-case complete basis-exchange work exceeds the admitted bound",
            )
        basis_family = set(self.bases)
        basis_sets = tuple(frozenset(basis) for basis in self.bases)
        checks = 0
        for left, left_set in zip(self.bases, basis_sets, strict=True):
            for right, right_set in zip(self.bases, basis_sets, strict=True):
                left_only = left_set - right_set
                if not left_only:
                    continue
                right_only = right_set - left_set
                for removed in left_only:
                    for added in right_only:
                        checks += 1
                        if checks % _BASIS_CHECKPOINT_STRIDE == 0:
                            request_checkpoint(
                                "during finite basis exchange validation"
                            )
                        exchanged = tuple(sorted((left_set - {removed}) | {added}))
                        if exchanged in basis_family:
                            break
                    else:
                        raise _validation_error(
                            "basis_exchange",
                            f"basis exchange fails for {left}, {right}, and {removed}",
                        )

    @property
    def rank(self) -> int:
        """The common cardinality of the complete basis family."""

        return len(self.bases[0])

    @property
    def ground_size(self) -> int:
        return len(self.ground)


__all__ = [
    "MAX_FINITE_BASIS_COUNT",
    "MAX_FINITE_BASIS_EXCHANGE_CHECKS",
    "MAX_FINITE_BASIS_GROUND_SIZE",
    "MAX_FINITE_BASIS_LABEL_BYTES",
    "MAX_FINITE_BASIS_MEMBERSHIPS",
    "MAX_FINITE_BASIS_TOTAL_LABEL_BYTES",
    "FiniteBasisMatroid",
]
