"""Typed wire contracts for combinatorial-map operations."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.multigraph._models import LooplessMultigraph
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.matrices.values import SparseRationalMatrix
from jacobian.math.topology.combinatorial_maps.values import (
    FiniteCombinatorialMap,
    _build_outgoing,
    _validate_facial_budgets,
    _validate_involution,
    _validate_rotation,
)

MAX_EMBEDDING_VERTICES = 64
MAX_EMBEDDING_EDGES = 256
MAX_EMBEDDING_DEGREE = 64
MAX_EMBEDDING_ROTATION_ENTRIES = 4 * MAX_EMBEDDING_EDGES
MAX_ROTATION_SYSTEM_CANDIDATES = 100_000
MAX_MINIMUM_GENUS_CANDIDATES = 100_000
MAX_MULTIGRAPH_EMBEDDING_VERTICES = 64
MAX_MULTIGRAPH_EMBEDDING_EDGES = 256

EmbeddingCheckStatus = Literal["ORIENTABLE_CELLULAR_EMBEDDING", "INVALID_EMBEDDING"]

EmbeddingObstructionCode = Literal[
    "ROTATION_ROW_COUNT",
    "ROTATION_DEGREE_MISMATCH",
    "EDGE_INDEX_OUT_OF_RANGE",
    "FOREIGN_INCIDENCE",
    "DUPLICATE_INCIDENCE",
    "GRAPH_DISCONNECTED",
]

SignedEmbeddingCheckStatus = Literal[
    "ORIENTABLE_EMBEDDING",
    "NONORIENTABLE_EMBEDDING",
    "INVALID_EMBEDDING",
]

SignedEmbeddingObstructionCode = Literal[
    "ROTATION_ROW_COUNT",
    "ROTATION_DEGREE_MISMATCH",
    "EDGE_INDEX_OUT_OF_RANGE",
    "FOREIGN_INCIDENCE",
    "DUPLICATE_INCIDENCE",
    "SIGN_INDEX_OUT_OF_RANGE",
    "GRAPH_DISCONNECTED",
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"combinatorial_map.{reason}", message)


def _graph_is_connected(graph: SimpleUndirectedGraph) -> bool:
    if not graph.vertices:
        return False
    index = {label: position for position, label in enumerate(graph.vertices)}
    reached = {0}
    changed = True
    while changed:
        changed = False
        for left, right in graph.edges:
            if index[left] in reached or index[right] in reached:
                before = len(reached)
                reached.update((index[left], index[right]))
                changed |= len(reached) != before
    return len(reached) == len(graph.vertices)


def _rotation_system_total(graph: SimpleUndirectedGraph) -> int:
    from math import factorial

    index = {label: position for position, label in enumerate(graph.vertices)}
    degrees = [0] * len(graph.vertices)
    for left, right in graph.edges:
        degrees[index[left]] += 1
        degrees[index[right]] += 1
    total = 1
    for degree in degrees:
        if degree:
            total *= factorial(degree - 1)
    return total


def _canonical_cycle(row: tuple[int, ...]) -> tuple[int, ...]:
    if not row:
        return ()
    pivot = row.index(min(row))
    return row[pivot:] + row[:pivot]


def _source_edge_rotations(
    graph: SimpleUndirectedGraph, rotations: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...] | None:
    """Return the canonical edge-index rows, or ``None`` for bad incidence."""
    index = {label: position for position, label in enumerate(graph.vertices)}
    incident: list[set[int]] = [set() for _ in graph.vertices]
    for edge_index, (left, right) in enumerate(graph.edges):
        incident[index[left]].add(edge_index)
        incident[index[right]].add(edge_index)
    if len(rotations) != len(graph.vertices):
        return None
    if any(
        len(row) != len(incident[vertex]) or set(row) != incident[vertex]
        for vertex, row in enumerate(rotations)
    ):
        return None
    canonical = tuple(_canonical_cycle(tuple(row)) for row in rotations)
    return canonical if canonical == rotations else None


def _face_ledger(
    darts: tuple[tuple[int, int, int], ...],
    dart_rotations: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Derive the deterministic ``alpha . sigma`` face ledger."""
    dart_count = len(darts)
    alpha = tuple(dart[2] for dart in darts)
    sigma = [0] * dart_count
    for row in dart_rotations:
        for offset, dart in enumerate(row):
            sigma[dart] = row[(offset + 1) % len(row)]
    phi = tuple(alpha[sigma[dart]] for dart in range(dart_count))
    walks: list[tuple[int, ...]] = []
    face_of = [0] * dart_count
    seen: set[int] = set()
    for start in range(dart_count):
        if start in seen:
            continue
        current = start
        walk: list[int] = []
        while current not in seen:
            seen.add(current)
            face_of[current] = len(walks)
            walk.append(current)
            current = phi[current]
        walks.append(tuple(walk))
    return tuple(walks), tuple(face_of)


class OrientableEmbeddingCheckRequest(StrictModel):
    """Check one supplied rotation system of a connected bounded simple graph.

    ``rotations[i]`` is the cyclic order at ``graph.vertices[i]`` of incident
    edge indices into ``graph.edges``.  Every unsigned cyclic order describes a
    cellular embedding on a closed orientable surface, so this checker admits
    only the orientable convention and has no nonorientable candidate to reject.
    """

    graph: SimpleUndirectedGraph
    rotations: tuple[tuple[int, ...], ...]


class OrientableEmbeddingCheckResult(StrictModel):
    """A checked orientable cellular embedding or the first rotation obstruction.

    For an admitted rotation system the result carries the canonical local
    rotations, the complete dart permutations alpha (edge reversal), sigma
    (rotation successor), phi = alpha . sigma (face successor), the face-walk
    family, ``chi = V - E + F``, and the exact orientable genus
    ``g = (2 - chi) / 2``.  Face cycles partition every dart exactly once.
    """

    graph: SimpleUndirectedGraph
    status: EmbeddingCheckStatus
    rotations: tuple[tuple[int, ...], ...] = ()
    dart_rotations: tuple[tuple[int, ...], ...] = ()
    darts: tuple[tuple[int, int, int], ...] = ()
    alpha: tuple[int, ...] = ()
    sigma: tuple[int, ...] = ()
    phi: tuple[int, ...] = ()
    face_walks: tuple[tuple[int, ...], ...] = ()
    face_of_dart: tuple[int, ...] = ()
    vertices: int = Field(default=0, ge=0)
    edges: int = Field(default=0, ge=0)
    faces: int = Field(default=0, ge=0)
    euler_characteristic: int = 0
    genus: int = Field(default=0, ge=0)
    obstruction_code: EmbeddingObstructionCode | None = None
    obstruction_detail: str | None = None

    @model_validator(mode="after")
    def require_checked_embedding(self) -> Self:  # noqa: C901
        if (self.obstruction_code is None) != (self.obstruction_detail is None):
            raise _validation_error(
                "embedding_obstruction_payload",
                "obstruction code and detail must agree",
            )
        if self.status == "INVALID_EMBEDDING":
            if self.obstruction_code is None:
                raise _validation_error(
                    "embedding_invalid_payload",
                    "an invalid embedding carries its first obstruction",
                )
            if (
                self.vertices != len(self.graph.vertices)
                or self.edges != len(self.graph.edges)
                or self.darts
                or self.dart_rotations
                or self.alpha
                or self.sigma
                or self.phi
                or self.face_walks
                or self.face_of_dart
                or self.faces
                or self.euler_characteristic
                or self.genus
            ):
                raise _validation_error(
                    "embedding_invalid_ledger",
                    "an invalid embedding carries no derived cell ledger",
                )
            from jacobian.math.topology.combinatorial_maps.operations import (
                check_orientable_embedding,
            )

            expected = check_orientable_embedding(self.graph, self.rotations)
            if expected != self:
                raise _validation_error(
                    "embedding_invalid_obstruction",
                    "invalid status must carry the actual first source obstruction",
                )
            return self
        if self.obstruction_code is not None:
            raise _validation_error(
                "embedding_valid_payload",
                "a checked embedding carries no obstruction",
            )
        if not _graph_is_connected(self.graph):
            raise _validation_error(
                "embedding_source_connectivity",
                "a checked embedding must bind a connected source graph",
            )
        dart_count = len(self.darts)
        if not (
            len(self.alpha) == len(self.sigma) == len(self.phi) == dart_count
            and len(self.face_of_dart) == dart_count
            and self.vertices == len(self.graph.vertices)
            and self.edges == len(self.graph.edges)
        ):
            raise _validation_error(
                "embedding_ledger_axes",
                "embedding ledgers must cover the retained source cell axes",
            )
        index = {label: position for position, label in enumerate(self.graph.vertices)}
        expected_darts = tuple(
            dart
            for edge_index, (left, right) in enumerate(self.graph.edges)
            for dart in (
                (index[left], index[right], 2 * edge_index + 1),
                (index[right], index[left], 2 * edge_index),
            )
        )
        if self.darts != expected_darts:
            raise _validation_error(
                "embedding_dart_source",
                "dart endpoints and reversals must bind the retained source edges",
            )
        if self.alpha != tuple(dart[2] for dart in self.darts):
            raise _validation_error(
                "embedding_alpha_source",
                "alpha must be the source edge-reversal permutation",
            )
        canonical = _source_edge_rotations(self.graph, self.rotations)
        if canonical is None:
            raise _validation_error(
                "embedding_rotation_source",
                "rotations must be the canonical source edge rotations",
            )
        expected_dart_rotations = tuple(
            tuple(
                2 * edge_index
                if self.graph.edges[edge_index][0] == self.graph.vertices[vertex]
                else 2 * edge_index + 1
                for edge_index in row
            )
            for vertex, row in enumerate(canonical)
        )
        if self.dart_rotations != expected_dart_rotations:
            raise _validation_error(
                "embedding_rotation_source",
                "dart rotations must bind the canonical source rotations",
            )
        outgoing = {dart for row in self.dart_rotations for dart in row}
        if outgoing != set(range(dart_count)):
            raise _validation_error(
                "embedding_rotation_source",
                "dart rotations must partition source outgoing darts by vertex",
            )
        sigma = [0] * dart_count
        for row in self.dart_rotations:
            for offset, dart in enumerate(row):
                sigma[dart] = row[(offset + 1) % len(row)]
        expected_phi = tuple(self.alpha[sigma[dart]] for dart in range(dart_count))
        if self.sigma != tuple(sigma) or self.phi != expected_phi:
            raise _validation_error(
                "embedding_permutation_source",
                "sigma and phi must bind the source rotation permutation",
            )
        expected_faces, expected_face_of = _face_ledger(self.darts, self.dart_rotations)
        expected_count = len(expected_faces) if dart_count else 1
        if self.face_walks != expected_faces or self.face_of_dart != expected_face_of:
            raise _validation_error(
                "embedding_face_source",
                "face walks and assignments must equal the source face permutation",
            )
        if (
            self.faces != expected_count
            or self.euler_characteristic != self.vertices - self.edges + self.faces
        ):
            raise _validation_error(
                "embedding_face_count",
                "face count and Euler characteristic must bind the source ledger",
            )
        excess = 2 - self.euler_characteristic
        if excess < 0 or excess % 2 != 0 or self.genus != excess // 2:
            raise _validation_error(
                "embedding_genus",
                "orientable genus requires a nonnegative even 2 - chi",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class SignedEmbeddingCheckResult(StrictModel):
    """A checked signed embedding with its orientability decision.

    A signed rotation system gives every edge a sign: 1 (untwisted) or 0
    (twisted, orientation-reversing).  The checker decides orientability
    through the orientable double cover and an exact balance test rather
    than from face parities: a connected signed system is orientable
    exactly when every cycle carries an even number of twisted edges
    (equivalently, when its orientable double cover is disconnected).

    For an admitted signed rotation system the result carries the canonical
    local rotations, the base dart data (``darts`` with ``(tail, head,
    reverse)``, the edge-reversal permutation ``alpha`` and the rotation
    successor ``sigma``, exactly as in the unsigned checker), the complete
    base face-walk family, ``chi = V - E + F``, and the surface
    classification:

    - ``ORIENTABLE_EMBEDDING``: the signature is balanced; the closed
      surface is the orientable one of genus ``g = (2 - chi) / 2`` and no
      witness is carried.
    - ``NONORIENTABLE_EMBEDDING``: the signature is unbalanced; the closed
      surface is the nonorientable one of genus ``h = 2 - chi`` and
      ``witness_dart_walk`` records a concrete orientation-reversing closed
      walk (an odd-twist cycle, hence closed and head-to-tail).
    - ``INVALID_EMBEDDING``: the rotation system does not describe a
      cellular embedding of the graph; ``obstruction_code`` and
      ``obstruction_detail`` carry the first reason.

    ``signs`` holds the resolved 0/1 signing for an admitted system, and is
    ``None`` only when a ``twisted_edges`` encoding failed before resolution.
    ``twisted_edges`` retains the supplied twisted-edge list for such failed
    encodings, so the verifier can replay the exact input.

    Face walks are dart cycles in the checker's ``alpha . sigma``
    convention, matching ``graph.embedding.orientable.check``: consecutive
    entries need not be head-to-tail darts, and a walk may repeat darts.
    Every graph edge occurs in exactly two face-walk positions in total.
    The faces are projected from the orientable double cover, so each one
    crosses an even number of twisted edges; orientability is decided by
    the balance test, never by face parity.
    """

    graph: SimpleUndirectedGraph
    status: SignedEmbeddingCheckStatus
    signs: tuple[int, ...] | None = None
    twisted_edges: tuple[int, ...] | None = None
    rotations: tuple[tuple[int, ...], ...] = ()
    dart_rotations: tuple[tuple[int, ...], ...] = ()
    darts: tuple[tuple[int, int, int], ...] = ()
    alpha: tuple[int, ...] = ()
    sigma: tuple[int, ...] = ()
    face_walks: tuple[tuple[int, ...], ...] = ()
    vertices: int = Field(default=0, ge=0)
    edges: int = Field(default=0, ge=0)
    faces: int = Field(default=0, ge=0)
    euler_characteristic: int = 0
    genus: int = Field(default=0, ge=0)
    orientable: bool = False
    witness_dart_walk: tuple[int, ...] | None = None
    obstruction_code: SignedEmbeddingObstructionCode | None = None
    obstruction_detail: str | None = None

    @model_validator(mode="after")
    def require_signed_embedding_payload(self) -> Self:
        if (self.obstruction_code is None) != (self.obstruction_detail is None):
            raise _validation_error(
                "signed_embedding_obstruction_payload",
                "obstruction code and detail must agree",
            )
        if self.status == "INVALID_EMBEDDING":
            if self.obstruction_code is None:
                raise _validation_error(
                    "signed_embedding_invalid_payload",
                    "an invalid embedding carries its first obstruction",
                )
            if (
                self.vertices != len(self.graph.vertices)
                or self.edges != len(self.graph.edges)
                or self.dart_rotations
                or self.darts
                or self.alpha
                or self.sigma
                or self.face_walks
                or self.faces
                or self.euler_characteristic
                or self.genus
                or self.orientable
                or self.witness_dart_walk is not None
            ):
                raise _validation_error(
                    "signed_embedding_invalid_ledger",
                    "an invalid embedding carries no derived cell ledger",
                )
            from jacobian.math.topology.combinatorial_maps.operations import (
                check_signed_embedding,
            )

            expected = check_signed_embedding(
                self.graph,
                self.rotations,
                signs=self.signs,
                twisted_edges=self.twisted_edges,
            )
            if expected != self:
                raise _validation_error(
                    "signed_embedding_invalid_obstruction",
                    "invalid status must carry the actual first source obstruction",
                )
            return self
        if self.obstruction_code is not None:
            raise _validation_error(
                "signed_embedding_valid_payload",
                "a checked embedding carries no obstruction",
            )
        if self.status == "ORIENTABLE_EMBEDDING":
            if not self.orientable or self.witness_dart_walk is not None:
                raise _validation_error(
                    "signed_embedding_orientable_payload",
                    "an orientable embedding carries no orientation-reversing witness",
                )
        elif self.orientable or self.witness_dart_walk is None:
            raise _validation_error(
                "signed_embedding_witness_payload",
                "a nonorientable embedding carries an orientation-reversing witness",
            )
        return self

    @model_validator(mode="after")
    def require_signed_embedding_cells(self) -> Self:
        if self.status == "INVALID_EMBEDDING":
            return self
        if not _graph_is_connected(self.graph):
            raise _validation_error(
                "signed_embedding_source_connectivity",
                "a checked embedding must bind a connected source graph",
            )
        dart_count = len(self.darts)
        if not (len(self.alpha) == len(self.sigma) == dart_count) or any(
            len(dart) != 3
            or not 0 <= dart[0] < len(self.graph.vertices)
            or not 0 <= dart[1] < len(self.graph.vertices)
            or not 0 <= dart[2] < dart_count
            for dart in self.darts
        ):
            raise _validation_error(
                "signed_embedding_permutation_axes",
                "alpha and sigma must cover every dart and darts must reference "
                "declared vertices and darts",
            )
        if (
            self.signs is None
            or len(self.signs) != len(self.graph.edges)
            or any(sign not in (0, 1) for sign in self.signs)
        ):
            raise _validation_error(
                "signed_embedding_cell_counts",
                "signs must carry one 0/1 entry per edge of the bound graph",
            )
        index = {label: position for position, label in enumerate(self.graph.vertices)}
        expected_darts = tuple(
            dart
            for edge_index, (left, right) in enumerate(self.graph.edges)
            for dart in (
                (index[left], index[right], 2 * edge_index + 1),
                (index[right], index[left], 2 * edge_index),
            )
        )
        if self.darts != expected_darts or self.alpha != tuple(
            dart[2] for dart in self.darts
        ):
            raise _validation_error(
                "signed_embedding_dart_source",
                "darts and alpha must bind the retained source edges",
            )
        canonical = _source_edge_rotations(self.graph, self.rotations)
        if canonical is None:
            raise _validation_error(
                "signed_embedding_rotation_source",
                "rotations must be the canonical source edge rotations",
            )
        expected_dart_rotations = tuple(
            tuple(
                2 * edge_index
                if self.graph.edges[edge_index][0] == self.graph.vertices[vertex]
                else 2 * edge_index + 1
                for edge_index in row
            )
            for vertex, row in enumerate(canonical)
        )
        if self.dart_rotations != expected_dart_rotations:
            raise _validation_error(
                "signed_embedding_rotation_source",
                "dart rotations must bind the canonical source rotations",
            )
        outgoing = {dart for row in self.dart_rotations for dart in row}
        if outgoing != set(range(dart_count)):
            raise _validation_error(
                "signed_embedding_rotation_source",
                "dart rotations must partition source outgoing darts by vertex",
            )
        sigma = [0] * dart_count
        for row in self.dart_rotations:
            for offset, dart in enumerate(row):
                sigma[dart] = row[(offset + 1) % len(row)]
        if self.sigma != tuple(sigma):
            raise _validation_error(
                "signed_embedding_sigma_source",
                "sigma must bind the retained source rotation permutation",
            )
        if self.vertices != len(self.graph.vertices) or self.edges != len(
            self.graph.edges
        ):
            raise _validation_error(
                "signed_embedding_graph_counts",
                "vertex and edge counts must match the bound graph",
            )
        if self.euler_characteristic != self.vertices - self.edges + self.faces:
            raise _validation_error(
                "signed_embedding_euler_characteristic",
                "characteristic must equal vertices - edges + faces",
            )
        return self

    @model_validator(mode="after")
    def require_signed_embedding_source_ledger(self) -> Self:
        if self.status == "INVALID_EMBEDDING":
            return self
        if self.twisted_edges is not None:
            if tuple(sorted(set(self.twisted_edges))) != self.twisted_edges:
                raise _validation_error(
                    "signed_embedding_twisted_axis",
                    "twisted edge indices must be unique and sorted",
                )
            if any(
                index < 0 or index >= len(self.graph.edges)
                for index in self.twisted_edges
            ):
                raise _validation_error(
                    "signed_embedding_twisted_axis",
                    "twisted edge indices must lie on the source edge axis",
                )
            expected = tuple(
                0 if index in self.twisted_edges else 1
                for index in range(len(self.graph.edges))
            )
            if self.signs != expected:
                raise _validation_error(
                    "signed_embedding_sign_axis",
                    "resolved signs must agree with the retained twisted-edge encoding",
                )
        return self

    @model_validator(mode="after")
    def require_signed_embedding_faces(self) -> Self:
        if self.status == "INVALID_EMBEDDING":
            return self
        dart_count = len(self.darts)
        if dart_count == 0:
            if self.faces != 1 or self.face_walks:
                raise _validation_error(
                    "signed_embedding_edgeless_faces",
                    "the edgeless single-vertex embedding has one empty face",
                )
            if self.euler_characteristic != 2 or self.genus != 0 or not self.orientable:
                raise _validation_error(
                    "signed_embedding_edgeless_surface",
                    "the edgeless single-vertex embedding is the trivial sphere",
                )
            return self
        if any(not 0 <= dart < dart_count for walk in self.face_walks for dart in walk):
            raise _validation_error(
                "signed_embedding_face_walk_shape",
                "face walks must reference declared darts",
            )
        occurrences = [0] * len(self.graph.edges)
        for walk in self.face_walks:
            for dart in walk:
                occurrences[dart // 2] += 1
        if any(count != 2 for count in occurrences):
            raise _validation_error(
                "signed_embedding_face_partition",
                "every graph edge must occur in exactly two face-walk positions",
            )
        if len(self.face_walks) != self.faces:
            raise _validation_error(
                "signed_embedding_face_count",
                "the face ledger must carry one walk per face",
            )
        # Signed faces are the deterministic projection of the orientable
        # double-cover ledger, not merely any edge-counting partition.  Keep
        # this replay local to the value boundary so serialized claims cannot
        # replace the retained rotation/sign source with a forged face family.
        from ._signed_faces import signed_face_walks

        canonical = _source_edge_rotations(self.graph, self.rotations)
        if canonical is None or self.signs is None:
            raise _validation_error(
                "signed_embedding_face_source",
                "signed face derivation requires canonical rotations and signs",
            )
        index = {label: position for position, label in enumerate(self.graph.vertices)}
        endpoints = [(index[left], index[right]) for left, right in self.graph.edges]
        expected_faces = (
            signed_face_walks(endpoints, canonical, self.signs) if dart_count else ()
        )
        if self.face_walks != expected_faces:
            raise _validation_error(
                "signed_embedding_face_source",
                "face walks must equal the deterministic signed face ledger",
            )
        if (
            self.faces != (len(expected_faces) if dart_count else 1)
            or self.euler_characteristic != self.vertices - self.edges + self.faces
        ):
            raise _validation_error(
                "signed_embedding_face_count",
                "face count and Euler characteristic must bind the signed ledger",
            )
        return self

    @model_validator(mode="after")
    def require_signed_embedding_surface(self) -> Self:
        if self.status == "INVALID_EMBEDDING" or not self.darts:
            return self
        if self.status == "ORIENTABLE_EMBEDDING":
            excess = 2 - self.euler_characteristic
            if excess < 0 or excess % 2 != 0 or self.genus != excess // 2:
                raise _validation_error(
                    "signed_embedding_orientable_genus",
                    "orientable genus requires a nonnegative even 2 - chi",
                )
            return self
        if self.euler_characteristic > 1 or self.genus != 2 - self.euler_characteristic:
            raise _validation_error(
                "signed_embedding_nonorientable_genus",
                "nonorientable genus requires chi <= 1 and h = 2 - chi",
            )
        witness = self.witness_dart_walk
        if witness is None:
            raise _validation_error(
                "signed_embedding_witness_shape",
                "a nonorientable embedding must carry its witness dart walk",
            )
        dart_count = len(self.darts)
        if not witness or any(not 0 <= dart < dart_count for dart in witness):
            raise _validation_error(
                "signed_embedding_witness_shape",
                "the witness must be a nonempty closed dart walk",
            )
        tails = [self.darts[dart][0] for dart in witness]
        heads = [self.darts[dart][1] for dart in witness]
        if any(
            heads[position] != tails[(position + 1) % len(witness)]
            for position in range(len(witness))
        ):
            raise _validation_error(
                "signed_embedding_witness_binding",
                "the witness must be a closed head-to-tail dart walk",
            )
        if self.signs is None:
            raise _validation_error(
                "signed_embedding_witness_signs",
                "a nonorientable embedding carries its resolved signing",
            )
        if sum(1 for dart in witness if self.signs[dart // 2] == 0) % 2 != 1:
            raise _validation_error(
                "signed_embedding_witness_parity",
                "the witness must cross an odd number of twisted edges",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class SignedEmbeddingCheckRequest(StrictModel):
    """Check one supplied signed rotation system of a connected simple graph.

    ``rotations[i]`` is the cyclic order at ``graph.vertices[i]`` of incident
    edge indices into ``graph.edges``.  ``signs[j]`` is 1 when edge ``j``
    preserves the local orientation at both endpoints and 0 when traversing
    the edge reverses the surface orientation (a twisted, crosscap-carrying
    edge).  Edge signs may be transported either as one integer per edge
    (``signs``) or as a list of twisted edge indices (``twisted_edges``);
    at most one form may be supplied, and supplying neither means every
    edge is untwisted.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A connected `SimpleUndirectedGraph` with one cyclic rotation "
                "row per vertex listing incident edge indices, plus edge "
                "signs. Provide either `signs` (one 0/1 integer per edge, in "
                "edge order) or `twisted_edges` (indices of edges whose sign "
                "is 0, i.e. orientation-reversing); unset means every edge "
                "is untwisted."
            )
        }
    )

    graph: SimpleUndirectedGraph = Field(
        description=(
            "The connected simple graph being embedded. Vertices keep their "
            "labels; edges are canonical label-ordered pairs."
        ),
    )
    rotations: tuple[tuple[int, ...], ...] = Field(
        description=(
            "One cyclic order of incident edge indices per graph vertex, "
            "aligned with `graph.vertices`. Each incident edge appears "
            "exactly once in its endpoint's row."
        ),
    )
    signs: tuple[int, ...] | None = Field(
        default=None,
        description=(
            "Exactly one sign per graph edge in edge order: 1 (untwisted) or "
            "0 (twisted, orientation-reversing). Omit when using "
            "`twisted_edges`."
        ),
    )
    twisted_edges: tuple[int, ...] | None = Field(
        default=None,
        description=(
            "Indices of edges whose sign is 0 (twisted). Omit when using "
            "`signs`; an omitted `twisted_edges` with no `signs` means all "
            "edges are untwisted."
        ),
    )

    @model_validator(mode="after")
    def require_one_sign_encoding(self) -> Self:
        if self.signs is not None and self.twisted_edges is not None:
            raise _validation_error(
                "signed_embedding_sign_encoding",
                "provide either signs or twisted_edges, not both",
            )
        return self


RotationSystemFindStatus = Literal["FOUND", "EXHAUSTED", "UNKNOWN"]

RotationSystemFindReason = Literal[
    "CANDIDATE_BUDGET_EXCEEDED",
    "GRAPH_DISCONNECTED",
]


class RotationSystemFindRequest(StrictModel):
    """Find a rotation system of genus at most ``max_genus`` by bounded search.

    Rotation systems are enumerated in deterministic vertex order with each
    local row ranging over cyclic orders (first entry fixed), so every
    distinct cellular embedding appears exactly once up to cyclic shifts.
    At most ``max_candidates`` systems are examined; a negative conclusion
    follows only from completed search.  The graph must satisfy the shared
    embedding-check envelope.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A connected `SimpleUndirectedGraph` with a genus bound and "
                "a candidate budget. Enumeration visits rotation systems in "
                "deterministic vertex order; at most `max_candidates` systems "
                "are checked, and a negative conclusion follows only from "
                "completed search."
            )
        }
    )

    graph: SimpleUndirectedGraph = Field(
        description="The connected simple graph to embed.",
    )
    max_genus: int = Field(
        ge=0,
        description="Accept the first rotation system of genus at most this bound.",
    )
    max_candidates: int = Field(
        ge=1,
        le=MAX_ROTATION_SYSTEM_CANDIDATES,
        description=(
            "Examine at most this many rotation systems before reporting "
            "UNKNOWN with a budget reason."
        ),
    )


class RotationSystemFindResult(StrictModel):
    """A bounded genus-search outcome with its exhaustion receipt.

    - ``FOUND``: ``rotations`` is the first enumerated rotation system of
      genus at most ``max_genus`` and ``certificate`` is its checker result;
      ``candidates_examined`` counts systems examined up to and including
      the witness.
    - ``EXHAUSTED``: every one of the ``total_candidates`` rotation systems
      was examined and none has genus at most ``max_genus``; the receipt is
      ``candidates_examined == total_candidates``.
    - ``UNKNOWN``: no conclusion; ``reason`` distinguishes the spent
      candidate budget from a disconnected graph (which the connected
      Euler convention does not cover).

    ``total_candidates`` is the exact product over vertices of
    ``(degree - 1)!`` (one row for an isolated or degree-one vertex).
    """

    graph: SimpleUndirectedGraph
    max_genus: int = Field(default=0, ge=0)
    max_candidates: int = Field(default=1, ge=1, le=MAX_ROTATION_SYSTEM_CANDIDATES)
    status: RotationSystemFindStatus
    rotations: tuple[tuple[int, ...], ...] = ()
    certificate: OrientableEmbeddingCheckResult | None = None
    candidates_examined: int = Field(default=0, ge=0)
    total_candidates: int = Field(default=0, ge=0)
    reason: RotationSystemFindReason | None = None
    reason_detail: str | None = None

    @model_validator(mode="after")
    def require_find_payload(self) -> Self:
        if (self.reason is None) != (self.reason_detail is None):
            raise _validation_error(
                "genus_search_reason_payload",
                "reason and detail must agree",
            )
        if self.status == "UNKNOWN":
            if self.reason is None:
                raise _validation_error(
                    "genus_search_unknown_payload",
                    "an unknown search carries its bounded reason",
                )
            if self.rotations or self.certificate is not None:
                raise _validation_error(
                    "genus_search_unknown_witness",
                    "an unknown search carries no rotation system",
                )
            return self
        if self.reason is not None:
            raise _validation_error(
                "genus_search_decided_payload",
                "a decided search carries no reason",
            )
        if self.status == "FOUND":
            if not self.rotations or self.certificate is None:
                raise _validation_error(
                    "genus_search_found_payload",
                    "a found search carries its rotation system and certificate",
                )
        elif self.rotations or self.certificate is not None:
            raise _validation_error(
                "genus_search_exhausted_payload",
                "an exhausted search carries no rotation system",
            )
        return self

    @model_validator(mode="after")
    def require_find_certificate(self) -> Self:
        if self.status != "FOUND":
            return self
        certificate = self.certificate
        if certificate is None:
            raise _validation_error(
                "genus_search_found_payload",
                "a found search carries its rotation system and certificate",
            )
        if certificate.graph != self.graph:
            raise _validation_error(
                "genus_search_certificate_graph",
                "the certificate must check the searched graph",
            )
        if certificate.status != "ORIENTABLE_CELLULAR_EMBEDDING":
            raise _validation_error(
                "genus_search_certificate_status",
                "the certificate must be an admitted cellular embedding",
            )
        if certificate.rotations != self.rotations:
            raise _validation_error(
                "genus_search_certificate_rotations",
                "the certificate must check the found rotation system",
            )
        if certificate.genus > self.max_genus:
            raise _validation_error(
                "genus_search_certificate_genus",
                "the certificate genus must respect the search bound",
            )
        return self

    @model_validator(mode="after")
    def require_find_receipt(self) -> Self:
        if (
            type(self.max_genus) is not int
            or self.max_genus < 0
            or type(self.max_candidates) is not int
            or not 1 <= self.max_candidates <= MAX_ROTATION_SYSTEM_CANDIDATES
        ):
            raise _validation_error(
                "genus_search_bounds",
                "search genus and candidate limits must be admitted integers",
            )
        expected_total = _rotation_system_total(self.graph)
        if self.total_candidates != expected_total or expected_total < 1:
            raise _validation_error(
                "genus_search_total_candidates",
                "the receipt total must equal the source rotation-system count",
            )
        if not 0 <= self.candidates_examined <= self.total_candidates:
            raise _validation_error(
                "genus_search_examined_bounds",
                "examined systems must lie between zero and the total",
            )
        if self.candidates_examined > self.max_candidates:
            raise _validation_error(
                "genus_search_budget_receipt",
                "examined systems cannot exceed the declared candidate budget",
            )
        if self.status == "EXHAUSTED":
            if not _graph_is_connected(self.graph):
                raise _validation_error(
                    "genus_search_status",
                    "an exhausted search must bind a connected source graph",
                )
            if self.total_candidates > self.max_candidates:
                raise _validation_error(
                    "genus_search_exhaustion_budget",
                    "an exhausted search must fit its candidate budget",
                )
            if self.candidates_examined != self.total_candidates:
                raise _validation_error(
                    "genus_search_exhaustion_receipt",
                    "an exhausted search examined every rotation system",
                )
        elif self.status == "FOUND":
            if self.candidates_examined < 1:
                raise _validation_error(
                    "genus_search_found_receipt",
                    "a found search examined at least its witness",
                )
        elif self.reason == "CANDIDATE_BUDGET_EXCEEDED":
            if (
                not _graph_is_connected(self.graph)
                or self.total_candidates <= self.max_candidates
                or self.candidates_examined != self.max_candidates
            ):
                raise _validation_error(
                    "genus_search_budget_receipt",
                    "a budget-exhausted search spent its full budget before exhaustion",
                )
        elif self.candidates_examined != 0 or _graph_is_connected(self.graph):
            raise _validation_error(
                "genus_search_disconnected_receipt",
                "a disconnected search examined no rotation system",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class MinimumGenusRequest(StrictModel):
    """Search all admitted rotation systems for the exact minimum genus."""

    graph: SimpleUndirectedGraph = Field(
        description=(
            "Connected simple graph; the exact search envelope admits at most "
            f"{MAX_EMBEDDING_VERTICES} vertices and {MAX_EMBEDDING_EDGES} edges."
        )
    )
    max_candidates: int = Field(
        default=MAX_MINIMUM_GENUS_CANDIDATES,
        ge=1,
        le=MAX_MINIMUM_GENUS_CANDIDATES,
        description=(
            "Maximum complete rotation systems examined before UNKNOWN; at most "
            f"{MAX_MINIMUM_GENUS_CANDIDATES} candidates are admitted."
        ),
    )


class MinimumGenusResult(StrictModel):
    """Exact minimum genus, or UNKNOWN when exhaustive search was bounded."""

    graph: SimpleUndirectedGraph
    status: Literal["EXACT", "UNKNOWN"]
    minimum_genus: int | None = Field(default=None, ge=0)
    rotations: tuple[tuple[int, ...], ...] = ()
    certificate: OrientableEmbeddingCheckResult | None = None
    candidates_examined: int = Field(default=0, ge=0)
    total_candidates: int = Field(default=1, ge=1)
    max_candidates: int = Field(
        default=MAX_MINIMUM_GENUS_CANDIDATES, ge=1, le=MAX_MINIMUM_GENUS_CANDIDATES
    )
    reason: Literal["CANDIDATE_BUDGET_EXCEEDED", "GRAPH_DISCONNECTED"] | None = None

    @model_validator(mode="after")
    def require_minimum_payload(self) -> Self:
        if (
            type(self.max_candidates) is not int
            or not 1 <= self.max_candidates <= MAX_MINIMUM_GENUS_CANDIDATES
        ):
            raise _validation_error(
                "minimum_genus_bounds",
                "candidate limits must be admitted integers",
            )
        expected_total = _rotation_system_total(self.graph)
        if self.total_candidates != expected_total:
            raise _validation_error(
                "minimum_genus_total_candidates",
                "the receipt total must equal the source rotation-system count",
            )
        if not 0 <= self.candidates_examined <= self.total_candidates:
            raise _validation_error(
                "minimum_genus_examined_bounds",
                "examined systems must lie between zero and the total",
            )
        if self.candidates_examined > self.max_candidates:
            raise _validation_error(
                "minimum_genus_budget_receipt",
                "examined systems cannot exceed the declared candidate budget",
            )
        if self.status == "EXACT":
            if not _graph_is_connected(self.graph):
                raise _validation_error(
                    "minimum_genus_status",
                    "EXACT must bind a connected source graph",
                )
            if (
                self.reason is not None
                or self.minimum_genus is None
                or self.certificate is None
            ):
                raise _validation_error(
                    "minimum_genus_exact_payload",
                    "EXACT requires a genus and certificate",
                )
            if (
                self.certificate.graph != self.graph
                or self.certificate.genus != self.minimum_genus
            ):
                raise _validation_error(
                    "minimum_genus_certificate",
                    "certificate must realize the reported minimum",
                )
            if self.certificate.rotations != self.rotations:
                raise _validation_error(
                    "minimum_genus_rotations",
                    "certificate rotations must match the result",
                )
            if self.candidates_examined != self.total_candidates:
                raise _validation_error(
                    "minimum_genus_receipt", "EXACT examines every candidate"
                )
        else:
            if (
                self.reason is None
                or self.minimum_genus is not None
                or self.certificate is not None
                or self.rotations
            ):
                raise _validation_error(
                    "minimum_genus_unknown_payload",
                    "UNKNOWN carries no minimum witness",
                )
            if self.reason == "CANDIDATE_BUDGET_EXCEEDED":
                if (
                    not _graph_is_connected(self.graph)
                    or self.total_candidates <= self.max_candidates
                    or self.candidates_examined != self.max_candidates
                ):
                    raise _validation_error(
                        "minimum_genus_budget_receipt",
                        "UNKNOWN spent the declared candidate budget before exhaustion",
                    )
            elif self.reason == "GRAPH_DISCONNECTED" and (
                self.candidates_examined != 0 or _graph_is_connected(self.graph)
            ):
                raise _validation_error(
                    "minimum_genus_disconnected_receipt",
                    "a disconnected search examined no rotation system",
                )
        return self

    @property
    def genus(self) -> int | None:
        return self.minimum_genus

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class MultigraphEmbeddingRequest(StrictModel):
    """Check a rotation system on an edge-ID-bound loopless multigraph."""

    graph: LooplessMultigraph = Field(
        description=(
            "Connected loopless multigraph with at most "
            f"{MAX_MULTIGRAPH_EMBEDDING_VERTICES} vertices and "
            f"{MAX_MULTIGRAPH_EMBEDDING_EDGES} edges for this exact checker."
        )
    )
    rotations: tuple[tuple[str, ...], ...] = Field(
        description=(
            "Exactly one tuple of edge IDs per source vertex; each row must list "
            "that vertex's incident edge IDs exactly once."
        )
    )


class MultigraphEmbeddingResult(StrictModel):
    """Source-bound embedding ledger retaining edge IDs and dual incidence."""

    graph: LooplessMultigraph
    status: Literal["EMBEDDED", "INVALID"]
    rotations: tuple[tuple[str, ...], ...] = ()
    darts: tuple[tuple[int, int, int], ...] = ()
    edge_dart_ids: tuple[str, ...] = ()
    face_walks: tuple[tuple[int, ...], ...] = ()
    dual_edge_ids: tuple[str, ...] = ()
    embedding_map: FiniteCombinatorialMap | None = None
    dual_map: FiniteCombinatorialMap | None = None
    obstruction: str | None = None

    @model_validator(mode="after")
    def require_embedding_ledger(self) -> Self:  # noqa: C901
        edge_count = len(self.graph.edges)
        dart_count = 2 * edge_count
        if self.status == "INVALID":
            if self.obstruction is None:
                raise _validation_error(
                    "multigraph_invalid_obstruction", "INVALID requires an obstruction"
                )
            if self.darts or self.edge_dart_ids or self.face_walks:
                raise _validation_error(
                    "multigraph_invalid_ledger", "INVALID carries no embedding ledger"
                )
            if self.embedding_map is not None or self.dual_map is not None:
                raise _validation_error(
                    "multigraph_invalid_maps", "INVALID carries no embedding maps"
                )
            return self
        if self.obstruction is not None:
            raise _validation_error(
                "multigraph_embedded_obstruction", "EMBEDDED carries no obstruction"
            )
        if self.graph.vertex_count == 0:
            raise _validation_error(
                "multigraph_zero_vertices", "a multigraph embedding needs a vertex"
            )
        expected_ids = tuple(edge.edge_id for edge in self.graph.edges)
        if edge_count == 0:
            if self.rotations != tuple(() for _ in range(self.graph.vertex_count)):
                raise _validation_error(
                    "multigraph_edgeless_rotations",
                    "edgeless rotations must be empty rows",
                )
            if any(
                (self.darts, self.edge_dart_ids, self.face_walks, self.dual_edge_ids)
            ):
                raise _validation_error(
                    "multigraph_edgeless_ledger",
                    "the sphere-point ledger has no darts or faces",
                )
            if self.embedding_map is not None or self.dual_map is not None:
                raise _validation_error(
                    "multigraph_edgeless_maps", "the sphere-point ledger has no maps"
                )
            return self
        if len(self.rotations) != self.graph.vertex_count:
            raise _validation_error(
                "multigraph_rotation_axis", "rotations must cover every source vertex"
            )
        incidence = [
            {
                edge.edge_id
                for edge in self.graph.edges
                if vertex in (edge.left, edge.right)
            }
            for vertex in range(self.graph.vertex_count)
        ]
        if any(
            set(row) != expected or len(row) != len(expected)
            for row, expected in zip(self.rotations, incidence, strict=True)
        ):
            raise _validation_error(
                "multigraph_rotation_incidence",
                "rotations must match source edge incidences",
            )
        if len(self.darts) != dart_count or len(self.edge_dart_ids) != dart_count:
            raise _validation_error(
                "multigraph_dart_axis",
                "dart and edge-ID ledgers must contain two entries per edge",
            )
        expected_darts = tuple(
            dart
            for edge_index, edge in enumerate(self.graph.edges)
            for dart in (
                (edge.left, edge.right, 2 * edge_index + 1),
                (edge.right, edge.left, 2 * edge_index),
            )
        )
        edge_index = {
            edge.edge_id: index for index, edge in enumerate(self.graph.edges)
        }
        expected_dart_rotations = tuple(
            tuple(
                2 * edge_index[edge_id]
                + (0 if self.graph.edges[edge_index[edge_id]].left == vertex else 1)
                for edge_id in row
            )
            for vertex, row in enumerate(self.rotations)
        )
        if self.darts != expected_darts:
            raise _validation_error(
                "multigraph_dart_source",
                "dart endpoints and reversals must bind source edge order",
            )
        if tuple(self.edge_dart_ids) != tuple(
            edge_id for edge_id in expected_ids for _ in (0, 1)
        ):
            raise _validation_error(
                "multigraph_edge_axis",
                "edge dart IDs must retain source edge order twice",
            )
        if len(self.face_walks) < 1 or sorted(
            dart for walk in self.face_walks for dart in walk
        ) != list(range(dart_count)):
            raise _validation_error(
                "multigraph_face_partition", "face walks must partition every dart"
            )
        if (
            self.embedding_map is not None
            and self.embedding_map.rotations != expected_dart_rotations
        ):
            raise _validation_error(
                "multigraph_embedding_rotation",
                "the embedding map rotations must bind every source rotation row",
            )
        sigma = [0] * dart_count
        for row in expected_dart_rotations:
            for offset, dart in enumerate(row):
                sigma[dart] = row[(offset + 1) % len(row)]
        alpha = tuple(dart[2] for dart in expected_darts)
        phi = tuple(alpha[sigma[dart]] for dart in range(dart_count))
        expected_faces: list[tuple[int, ...]] = []
        seen: set[int] = set()
        for start in range(dart_count):
            if start in seen:
                continue
            walk: list[int] = []
            current = start
            while current not in seen:
                seen.add(current)
                walk.append(current)
                current = phi[current]
            expected_faces.append(tuple(walk))
        if self.face_walks != tuple(expected_faces):
            raise _validation_error(
                "multigraph_face_ledger",
                "face walks must equal the source rotation face permutation",
            )
        if tuple(self.dual_edge_ids) != expected_ids:
            raise _validation_error(
                "multigraph_dual_edge_axis",
                "dual edge IDs must retain the source edge axis",
            )
        if self.embedding_map is None or self.dual_map is None:
            raise _validation_error(
                "multigraph_maps",
                "an edge-bearing embedding carries primal and dual maps",
            )
        if (
            self.embedding_map.vertex_count != self.graph.vertex_count
            or self.embedding_map.darts != self.darts
            or len(self.embedding_map.rotations) != self.graph.vertex_count
            or any(
                set(row)
                != {
                    dart
                    for dart, (tail, _head, _reverse) in enumerate(self.darts)
                    if tail == vertex
                }
                for vertex, row in enumerate(self.embedding_map.rotations)
            )
        ):
            raise _validation_error(
                "multigraph_embedding_map",
                "the embedding map must bind source vertices, darts, and rotations",
            )
        face_of_dart = {
            dart: face_index
            for face_index, walk in enumerate(self.face_walks)
            for dart in walk
        }
        expected_dual_darts = tuple(
            (face_of_dart[dart], face_of_dart[reverse], reverse)
            for dart, (_tail, _head, reverse) in enumerate(self.darts)
        )
        if (
            self.dual_map.vertex_count != len(self.face_walks)
            or self.dual_map.darts != expected_dual_darts
            or self.dual_map.rotations != self.face_walks
        ):
            raise _validation_error(
                "multigraph_dual_map", "the dual map must bind the complete face ledger"
            )
        return self

    @property
    def map(self) -> FiniteCombinatorialMap | None:
        return self.embedding_map

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class FacesRequest(StrictModel):
    """Compute the complete face-orbit family of a combinatorial map."""

    map: FiniteCombinatorialMap


class FacesResult(StrictModel):
    """The complete face-orbit family of the supplied map.

    The retained map binds the carrier and result envelope. Parsing this wire
    value never re-enters an operation.
    """

    map: FiniteCombinatorialMap
    face_walks: tuple[tuple[int, ...], ...]
    face_of_dart: tuple[int, ...]
    successor: tuple[int, ...]

    @model_validator(mode="after")
    def require_face_result_shape(self) -> Self:
        _validate_involution(self.map.darts)
        _validate_rotation(
            self.map.rotations,
            _build_outgoing(self.map.darts, self.map.vertex_count),
            self.map.darts,
        )
        dart_count = len(self.map.darts)
        if len(self.face_of_dart) != dart_count or len(self.successor) != dart_count:
            raise _validation_error(
                "face_result_dart_count",
                "face assignment and successor must cover the map's darts",
            )
        if (
            any(not walk for walk in self.face_walks)
            or sum(len(walk) for walk in self.face_walks) != dart_count
        ):
            raise _validation_error(
                "face_walk_shape",
                "face walks must be nonempty and have one entry per dart in total",
            )
        if any(
            dart < 0 or dart >= dart_count for walk in self.face_walks for dart in walk
        ) or any(
            dart < 0 or dart >= len(self.face_walks) for dart in self.face_of_dart
        ):
            raise _validation_error(
                "face_result_index_out_of_range",
                "face-result indices must lie in their declared carriers",
            )
        if any(dart < 0 or dart >= dart_count for dart in self.successor):
            raise _validation_error(
                "successor_out_of_range",
                "successor entries must be dart indices",
            )
        outgoing = {dart for row in self.map.rotations for dart in row}
        if outgoing != set(range(dart_count)):
            raise _validation_error(
                "face_result_map_shape",
                "the retained map rotations must cover every dart exactly once",
            )
        alpha = tuple(dart[2] for dart in self.map.darts)
        sigma = [0] * dart_count
        for row in self.map.rotations:
            for offset, dart in enumerate(row):
                sigma[dart] = row[(offset + 1) % len(row)]
        expected_successor = tuple(alpha[sigma[dart]] for dart in range(dart_count))
        if tuple(self.successor) != expected_successor:
            raise _validation_error(
                "face_result_successor_binding",
                "the face successor must equal reverse after rotation",
            )
        expected_faces, expected_face_of = _face_ledger(
            self.map.darts, self.map.rotations
        )
        _validate_facial_budgets([list(walk) for walk in expected_faces])
        if self.face_walks != expected_faces or self.face_of_dart != expected_face_of:
            raise _validation_error(
                "face_result_ledger_binding",
                "face walks and assignments must equal the deterministic face orbits",
            )
        return self


class EulerCharacteristicRequest(StrictModel):
    """Compute per-component and total Euler characteristic."""

    map: FiniteCombinatorialMap


class EulerCharacteristicCounts(StrictModel):
    """Cell counts and their Euler characteristic for one component family."""

    vertices: int = Field(ge=0)
    edges: int = Field(ge=0)
    faces: int = Field(ge=0)
    characteristic: int

    @model_validator(mode="after")
    def bind_characteristic(self) -> Self:
        if self.characteristic != self.vertices - self.edges + self.faces:
            raise _validation_error(
                "euler_characteristic_mismatch",
                "characteristic must equal vertices - edges + faces",
            )
        return self


class EulerCharacteristicResult(StrictModel):
    """Per-component and total Euler characteristic."""

    per_component: tuple[EulerCharacteristicCounts, ...]
    total: EulerCharacteristicCounts


class OrientableGenusRequest(StrictModel):
    """Compute per-component and total orientable genus."""

    map: FiniteCombinatorialMap


class OrientableGenusResult(StrictModel):
    """Per-component and total orientable genus."""

    per_component: tuple[int, ...]
    total: int = Field(ge=0)


class OrientationReverseRequest(StrictModel):
    """Reverse every local cyclic order of a combinatorial map."""

    map: FiniteCombinatorialMap


class CombinatorialMapBijection(StrictModel):
    """A bijection of explicitly declared dart or face axes of two maps."""

    source: FiniteCombinatorialMap
    target: FiniteCombinatorialMap
    kind: Literal["DART", "FACE"]
    source_axis: tuple[int, ...] = Field(max_length=1024)
    target_axis: tuple[int, ...] = Field(max_length=1024)
    images: tuple[int, ...] = Field(max_length=1024)

    @model_validator(mode="after")
    def require_bijection_shape(self) -> Self:
        size = len(self.source_axis)
        if (
            self.source_axis != tuple(range(size))
            or self.target_axis != tuple(range(size))
            or tuple(sorted(self.images)) != self.target_axis
        ):
            raise _validation_error(
                "bijection_axis", "images must biject the declared canonical axes"
            )
        if self.kind == "DART" and (
            size != len(self.source.darts) or size != len(self.target.darts)
        ):
            raise _validation_error(
                "bijection_darts",
                "dart axes must equal their source and target map axes",
            )
        return self


class OrientationReverseResult(StrictModel):
    """The reversal relation, with a source-target face-axis bijection."""

    bijection: CombinatorialMapBijection

    @model_validator(mode="after")
    def require_face_kind(self) -> Self:
        if self.bijection.kind != "FACE":
            raise _validation_error(
                "bijection_kind", "orientation reversal carries a face bijection"
            )
        return self

    @property
    def map(self) -> FiniteCombinatorialMap:
        return self.bijection.source

    @property
    def reversed_map(self) -> FiniteCombinatorialMap:
        return self.bijection.target

    @property
    def face_bijection(self) -> dict[int, int]:
        return dict(zip(self.bijection.source_axis, self.bijection.images, strict=True))


class ConnectedComponentsRequest(StrictModel):
    """Return the component partition of vertices, darts, and faces."""

    map: FiniteCombinatorialMap


class ConnectedComponentsResult(StrictModel):
    """``vertex -> component``, ``dart -> component``, ``face -> component``."""

    vertex_component: tuple[int, ...]
    dart_component: tuple[int, ...]
    face_component: tuple[int, ...]


class DualRequest(StrictModel):
    """Compute the exact embedded dual of a combinatorial map."""

    map: FiniteCombinatorialMap


class DualResult(StrictModel):
    """A source-target dart-axis bijection for the embedded dual."""

    bijection: CombinatorialMapBijection

    @model_validator(mode="after")
    def require_dart_kind(self) -> Self:
        if self.bijection.kind != "DART":
            raise _validation_error(
                "bijection_kind", "duality carries a dart bijection"
            )
        return self

    @property
    def dual(self) -> FiniteCombinatorialMap:
        return self.bijection.target

    @property
    def primal_to_dual(self) -> dict[int, int]:
        return dict(zip(self.bijection.source_axis, self.bijection.images, strict=True))


class VertexFaceIncidenceRequest(StrictModel):
    """Return the exact incidence structure between vertices and faces."""

    map: FiniteCombinatorialMap


class VertexFaceIncidenceResult(StrictModel):
    """Exact sparse multiplicities on source vertices by the retained face axis."""

    source: FacesResult
    multiplicity: SparseRationalMatrix

    @model_validator(mode="after")
    def require_incidence_axes(self) -> Self:
        if (
            self.multiplicity.row_count != self.source.map.vertex_count
            or self.multiplicity.column_count != len(self.source.face_walks)
        ):
            raise _validation_error(
                "incidence_shape", "incidence must use the source vertex and face axes"
            )
        return self

    @property
    def boolean_incidence(self) -> dict[int, tuple[int, ...]]:
        """Project nonzero multiplicities to incident face indices on the same axes."""
        return {
            vertex: tuple(
                entry.column
                for entry in self.multiplicity.entries
                if entry.row == vertex
            )
            for vertex in range(self.multiplicity.row_count)
        }


__all__ = [
    "CombinatorialMapBijection",
    "ConnectedComponentsRequest",
    "ConnectedComponentsResult",
    "DualRequest",
    "DualResult",
    "EmbeddingCheckStatus",
    "EmbeddingObstructionCode",
    "EulerCharacteristicCounts",
    "EulerCharacteristicRequest",
    "EulerCharacteristicResult",
    "FacesRequest",
    "FacesResult",
    "MinimumGenusRequest",
    "MinimumGenusResult",
    "MultigraphEmbeddingRequest",
    "MultigraphEmbeddingResult",
    "OrientableEmbeddingCheckRequest",
    "OrientableEmbeddingCheckResult",
    "OrientableGenusRequest",
    "OrientableGenusResult",
    "OrientationReverseRequest",
    "OrientationReverseResult",
    "RotationSystemFindReason",
    "RotationSystemFindRequest",
    "RotationSystemFindResult",
    "RotationSystemFindStatus",
    "SignedEmbeddingCheckRequest",
    "SignedEmbeddingCheckResult",
    "SignedEmbeddingCheckStatus",
    "SignedEmbeddingObstructionCode",
    "VertexFaceIncidenceRequest",
    "VertexFaceIncidenceResult",
]
