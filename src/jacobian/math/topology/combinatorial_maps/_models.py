"""Typed wire contracts for combinatorial-map operations."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.matrices.values import SparseRationalMatrix
from jacobian.math.topology.combinatorial_maps.values import (
    FiniteCombinatorialMap,
)

MAX_EMBEDDING_VERTICES = 64
MAX_EMBEDDING_EDGES = 256
MAX_EMBEDDING_DEGREE = 64
MAX_EMBEDDING_ROTATION_ENTRIES = 4 * MAX_EMBEDDING_EDGES

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
    def require_checked_embedding(self) -> Self:
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
            return self
        if self.obstruction_code is not None:
            raise _validation_error(
                "embedding_valid_payload",
                "a checked embedding carries no obstruction",
            )
        dart_count = len(self.darts)
        if not (
            len(self.alpha) == len(self.sigma) == len(self.phi) == dart_count
            and len(self.face_of_dart) == dart_count
        ):
            raise _validation_error(
                "embedding_permutation_axes",
                "alpha, sigma, phi, and the face assignment must cover every dart",
            )
        if dart_count == 0:
            if self.faces != 1 or self.face_walks:
                raise _validation_error(
                    "embedding_edgeless_faces",
                    "the edgeless single-vertex embedding has one empty face",
                )
        else:
            if any(not walk for walk in self.face_walks):
                raise _validation_error(
                    "embedding_face_walk_shape",
                    "face walks must be nonempty for a map with darts",
                )
            covered = [dart for walk in self.face_walks for dart in walk]
            if sorted(covered) != list(range(dart_count)):
                raise _validation_error(
                    "embedding_face_partition",
                    "face walks must partition every dart exactly once",
                )
        if self.vertices != len(self.graph.vertices) or self.edges != len(
            self.graph.edges
        ):
            raise _validation_error(
                "embedding_cell_counts",
                "vertex and edge counts must match the bound graph",
            )
        if self.euler_characteristic != self.vertices - self.edges + self.faces:
            raise _validation_error(
                "embedding_euler_characteristic",
                "characteristic must equal vertices - edges + faces",
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
        assert witness is not None
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
    "OrientableEmbeddingCheckRequest",
    "OrientableEmbeddingCheckResult",
    "OrientableGenusRequest",
    "OrientableGenusResult",
    "OrientationReverseRequest",
    "OrientationReverseResult",
    "SignedEmbeddingCheckRequest",
    "SignedEmbeddingCheckResult",
    "SignedEmbeddingCheckStatus",
    "SignedEmbeddingObstructionCode",
    "VertexFaceIncidenceRequest",
    "VertexFaceIncidenceResult",
]
