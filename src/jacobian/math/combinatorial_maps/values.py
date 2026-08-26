"""Provider-independent values for exact combinatorial-map operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_MAP_VERTICES = 256
MAX_MAP_DARTS = 1024
MAX_ROTATION_LENGTH = 64
MAX_LABEL_BYTES = 1024
# The dual map has one vertex per primal face and one rotation entry per
# primal facial-walk element.  These bounds keep the dual of every accepted
# map inside the same value budgets: the dual's faces correspond one-for-one
# with primal vertices and its facial walks with primal rotation rows.
MAX_FACES = MAX_MAP_VERTICES
MAX_FACIAL_WALK_LENGTH = MAX_ROTATION_LENGTH


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"combinatorial_map.{reason}", message)


def _validate_dart(
    dart: tuple[int, int, int], vertex_count: int, dart_count: int
) -> None:
    if len(dart) != 3:
        raise _validation_error(
            "dart_shape", "each dart must carry (tail, head, reverse)"
        )
    tail, head, reverse = dart
    if not 0 <= tail < vertex_count:
        raise _validation_error("dart_tail_out_of_range", "dart tail out of range")
    if not 0 <= head < vertex_count:
        raise _validation_error("dart_head_out_of_range", "dart head out of range")
    if not 0 <= reverse < dart_count:
        raise _validation_error(
            "dart_reverse_out_of_range", "dart reverse out of range"
        )


def _validate_involution(darts: tuple[tuple[int, int, int], ...]) -> None:
    for dart_index, dart in enumerate(darts):
        tail, head, reverse = dart
        if reverse == dart_index:
            raise _validation_error(
                "reverse_fixed_point", "dart reversal must be fixed-point-free"
            )
        r_tail, r_head, r_reverse = darts[reverse]
        if r_reverse != dart_index:
            raise _validation_error(
                "reverse_not_involution", "reverse must be an involution"
            )
        if r_tail != head or r_head != tail:
            raise _validation_error(
                "reverse_endpoints", "reverse must exchange the endpoints of its dart"
            )


def _build_outgoing(
    darts: tuple[tuple[int, int, int], ...], vertex_count: int
) -> dict[int, list[int]]:
    outgoing: dict[int, list[int]] = {v: [] for v in range(vertex_count)}
    for dart_index, dart in enumerate(darts):
        outgoing[dart[0]].append(dart_index)
    return outgoing


def _validate_rotation(
    rotations: tuple[tuple[int, ...], ...],
    outgoing: dict[int, list[int]],
    darts: tuple[tuple[int, int, int], ...],
) -> None:
    for vertex, row in enumerate(rotations):
        if not outgoing[vertex]:
            raise _validation_error(
                "vertex_without_dart",
                "every vertex must be incident to at least one dart",
            )
        if not row:
            raise _validation_error(
                "empty_rotation",
                "an outgoing-dart vertex must declare a nonempty rotation",
            )
        if len(row) > MAX_ROTATION_LENGTH:
            raise _validation_error(
                "rotation_too_long",
                "a local rotation exceeds the bounded rotation-length budget",
            )
        if len(row) != len(outgoing[vertex]):
            raise _validation_error(
                "rotation_length", "rotation length must equal the outgoing dart count"
            )
        seen: set[int] = set()
        for dart_index in row:
            if not 0 <= dart_index < len(darts):
                raise _validation_error(
                    "rotation_dart_out_of_range", "rotation dart index out of range"
                )
            dart = darts[dart_index]
            if dart[0] != vertex:
                raise _validation_error(
                    "foreign_rotation_dart",
                    "rotation must list only outgoing darts of its vertex",
                )
            if dart_index in seen:
                raise _validation_error(
                    "duplicate_rotation_dart", "rotation must not repeat a dart"
                )
            seen.add(dart_index)


def _validate_facial_budgets(map_: FiniteCombinatorialMap) -> None:
    """Reject maps whose facial structure would overflow the dual budgets.

    The dual of an accepted map must itself be an accepted value: it has one
    vertex per primal face and one rotation entry per primal facial-walk
    element, so the primal face count and facial-walk lengths are bounded
    here.  The constraint set is closed under duality (dual faces correspond
    to primal vertices and dual facial walks to primal rotation rows).
    """
    from jacobian.math.combinatorial_maps._face_orbits import face_orbit_data

    walks, _, _ = face_orbit_data(map_)
    if len(walks) > MAX_FACES:
        raise _validation_error(
            "face_count_too_large", "face count exceeds the bounded dual-vertex budget"
        )
    for walk in walks:
        if len(walk) > MAX_FACIAL_WALK_LENGTH:
            raise _validation_error(
                "facial_walk_too_long",
                "a facial walk exceeds the bounded dual-rotation budget",
            )


class FiniteCombinatorialMap(StrictModel):
    """An immutable well-formed finite combinatorial map.

    ``darts`` is a tuple of ``dart_count`` dart records, each carrying its
    ``tail`` and ``head`` vertex index and the index of the opposite dart
    (``reverse``).  ``rotations`` is a tuple of ``vertex_count`` rows; each
    row is the cyclic order of outgoing dart indices at that vertex.
    """

    vertex_count: int = Field(ge=1, le=MAX_MAP_VERTICES)
    darts: tuple[tuple[int, int, int], ...]
    rotations: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def require_well_formed(self) -> Self:
        if len(self.rotations) != self.vertex_count:
            raise _validation_error(
                "rotation_row_count", "rotations must have vertex_count rows"
            )
        if not self.darts:
            raise _validation_error(
                "empty_darts", "a combinatorial map needs at least one dart"
            )
        if len(self.darts) > MAX_MAP_DARTS:
            raise _validation_error(
                "dart_count_too_large", "dart count exceeds the bounded map budget"
            )
        for dart in self.darts:
            _validate_dart(dart, self.vertex_count, len(self.darts))
        _validate_involution(self.darts)
        outgoing = _build_outgoing(self.darts, self.vertex_count)
        _validate_rotation(self.rotations, outgoing, self.darts)
        _validate_facial_budgets(self)
        return self


class FacialWalk(StrictModel):
    """One cyclic facial walk, serialized as an ordered dart sequence."""

    darts: tuple[int, ...] = Field(min_length=1)


__all__ = [
    "MAX_FACES",
    "MAX_FACIAL_WALK_LENGTH",
    "MAX_LABEL_BYTES",
    "MAX_MAP_DARTS",
    "MAX_MAP_VERTICES",
    "MAX_ROTATION_LENGTH",
    "FacialWalk",
    "FiniteCombinatorialMap",
]
