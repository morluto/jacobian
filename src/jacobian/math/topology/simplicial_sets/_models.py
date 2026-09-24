"""Bounded wire contracts for finite truncated simplicial sets."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_SIMPLICIAL_SET_DEGREE = 4
MAX_SIMPLICES_PER_DEGREE = 32
MAX_TOTAL_SIMPLICES = 96
MAX_SIMPLEX_LABEL_LENGTH = 32

SimplexLabel = str
IndexRow = tuple[int, ...]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"simplicial_set.{reason}", message)


def _require_label_table(
    sets: tuple[tuple[str, ...], ...], max_degree: int
) -> tuple[int, ...]:
    if len(sets) != max_degree + 1:
        raise _validation_error(
            "degree_coverage_invalid",
            f"sets must cover degrees 0..{max_degree} exactly once",
        )
    sizes: list[int] = []
    for degree, level in enumerate(sets):
        if not 0 <= len(level) <= MAX_SIMPLICES_PER_DEGREE:
            raise _validation_error(
                "degree_size_out_of_bounds",
                f"degree {degree} must hold between 0 and "
                f"{MAX_SIMPLICES_PER_DEGREE} simplices",
            )
        if len(set(level)) != len(level):
            raise _validation_error(
                "simplex_labels_not_unique",
                f"degree-{degree} simplex labels must be unique",
            )
        for label in level:
            if not isinstance(label, str) or not 1 <= len(label) <= (
                MAX_SIMPLEX_LABEL_LENGTH
            ):
                raise _validation_error(
                    "simplex_label_invalid",
                    f"degree-{degree} labels must be nonempty strings of at most "
                    f"{MAX_SIMPLEX_LABEL_LENGTH} characters",
                )
        sizes.append(len(level))
    if sum(sizes) > MAX_TOTAL_SIMPLICES:
        raise _validation_error(
            "total_simplex_budget_exceeded",
            f"{sum(sizes)} simplices exceed the {MAX_TOTAL_SIMPLICES}-cell bound",
        )
    return tuple(sizes)


def _require_index_table(
    table: tuple[tuple[tuple[int, ...], ...], ...],
    sizes: tuple[int, ...],
    *,
    kind: str,
) -> None:
    """Check face/degeneracy table shape and index ranges structurally."""
    degree_count = len(sizes) - 1
    if kind == "face":
        if len(table) != degree_count:
            raise _validation_error(
                "face_table_degree_mismatch",
                "face maps must provide one entry per degree 1..N",
            )
        for offset, level in enumerate(table):
            degree = offset + 1
            if len(level) != degree + 1:
                raise _validation_error(
                    "face_map_count_mismatch",
                    f"degree {degree} must carry exactly {degree + 1} face maps",
                )
            for index, row in enumerate(level):
                if len(row) != sizes[degree] or any(
                    not 0 <= target < sizes[degree - 1] for target in row
                ):
                    raise _validation_error(
                        "face_map_axis_invalid",
                        f"face map d_{index} in degree {degree} must send every "
                        "simplex to a degree-(n-1) index",
                    )
    else:
        if len(table) != degree_count:
            raise _validation_error(
                "degeneracy_table_degree_mismatch",
                "degeneracy maps must provide one entry per degree 0..N-1",
            )
        for degree, level in enumerate(table):
            if len(level) != degree + 1:
                raise _validation_error(
                    "degeneracy_map_count_mismatch",
                    f"degree {degree} must carry exactly {degree + 1} degeneracy maps",
                )
            for index, row in enumerate(level):
                if len(row) != sizes[degree] or any(
                    not 0 <= target < sizes[degree + 1] for target in row
                ):
                    raise _validation_error(
                        "degeneracy_map_axis_invalid",
                        f"degeneracy map s_{index} in degree {degree} must send "
                        "every simplex to a degree-(n+1) index",
                    )


class SimplicialSetTablesRequest(StrictModel):
    """Complete finite degree tables X_0..X_N with every in-range map.

    ``face_maps[n - 1][i]`` is the index row of ``d_i : X_n -> X_(n-1)`` and
    ``degeneracy_maps[n][i]`` is the index row of ``s_i : X_n -> X_(n+1)``.
    """

    max_degree: int = Field(ge=0, le=MAX_SIMPLICIAL_SET_DEGREE)
    sets: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=8)
    face_maps: tuple[tuple[tuple[int, ...], ...], ...] = ()
    degeneracy_maps: tuple[tuple[tuple[int, ...], ...], ...] = ()

    @model_validator(mode="after")
    def require_structural_tables(self) -> Self:
        sizes = _require_label_table(self.sets, self.max_degree)
        _require_index_table(self.face_maps, sizes, kind="face")
        _require_index_table(self.degeneracy_maps, sizes, kind="degeneracy")
        return self


class SimplicialIdentityObstruction(StrictModel):
    """The first unequal simplicial-identity row, with its full comparison."""

    identity_family: Literal["FACE_FACE", "FACE_DEGENERACY", "DEGENERACY_DEGENERACY"]
    degree: int = Field(ge=0, le=MAX_SIMPLICIAL_SET_DEGREE)
    left_description: str = Field(min_length=1, max_length=128)
    right_description: str = Field(min_length=1, max_length=128)
    row: int = Field(ge=0)
    left_row: tuple[int, ...] = Field(min_length=1)
    right_row: tuple[int, ...] = Field(min_length=1)


class FiniteTruncatedSimplicialSet(StrictModel):
    """Canonical checked prefix with every simplicial identity exhausted."""

    max_degree: int = Field(ge=0, le=MAX_SIMPLICIAL_SET_DEGREE)
    sets: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=8)
    face_maps: tuple[tuple[tuple[int, ...], ...], ...] = ()
    degeneracy_maps: tuple[tuple[tuple[int, ...], ...], ...] = ()
    total_simplices: int = Field(ge=0, le=MAX_TOTAL_SIMPLICES)
    checked_identities: int = Field(ge=0)

    @model_validator(mode="after")
    def require_structural_prefix(self) -> Self:
        sizes = _require_label_table(self.sets, self.max_degree)
        _require_index_table(self.face_maps, sizes, kind="face")
        _require_index_table(self.degeneracy_maps, sizes, kind="degeneracy")
        if self.total_simplices != sum(sizes):
            raise _validation_error(
                "total_simplex_count_mismatch",
                "total_simplices must equal the sum of the degree sizes",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class SimplicialSetTablesResult(StrictModel):
    """Checked simplicial set or the first unequal identity row."""

    status: Literal["SIMPLICIAL_SET", "NOT_A_SIMPLICIAL_SET"]
    simplicial_set: FiniteTruncatedSimplicialSet | None = None
    checked_identities: int = Field(ge=0)
    obstruction: SimplicialIdentityObstruction | None = None

    @model_validator(mode="after")
    def require_status_binding(self) -> Self:
        if self.status == "SIMPLICIAL_SET":
            if self.simplicial_set is None or self.obstruction is not None:
                raise _validation_error(
                    "result_status_binding_invalid",
                    "a SIMPLICIAL_SET result carries the set and no obstruction",
                )
            if self.checked_identities != self.simplicial_set.checked_identities:
                raise _validation_error(
                    "identity_count_binding_invalid",
                    "checked_identities must match the retained set",
                )
        elif self.simplicial_set is not None or self.obstruction is None:
            raise _validation_error(
                "result_status_binding_invalid",
                "a NOT_A_SIMPLICIAL_SET result carries the obstruction and no set",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


__all__ = [
    "MAX_SIMPLICES_PER_DEGREE",
    "MAX_SIMPLICIAL_SET_DEGREE",
    "MAX_TOTAL_SIMPLICES",
    "FiniteTruncatedSimplicialSet",
    "SimplicialIdentityObstruction",
    "SimplicialSetTablesRequest",
    "SimplicialSetTablesResult",
]
