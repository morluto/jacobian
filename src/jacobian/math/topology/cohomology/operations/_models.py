"""Typed wire contracts for cohomology operations."""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_CHAIN_GROUP,
    MAX_TOPOLOGY_DIMENSION,
    MAX_TOPOLOGY_PRIME,
    FiniteSimplicialComplex,
    HomologyConvention,
)
from jacobian.math.topology._request_admission import (
    require_canonical_complex_admission,
)
from jacobian.math.topology.cohomology.operations._simplicial import (
    SimplicialCohomologyResult,
)

MAX_AMBIENT_SIMPLEX_VERTICES = 64
"""Cap on the vertex count of one supplied ambient simplex.

Supported cochain degrees stop at 16, so top-square targets carry at most
``2*16 + 1 = 33`` vertices; 64 keeps validation work linear and bounded with
headroom over every simplex dimension these operations can target. The cap
is encoded in the ``BoundedAmbientSimplex`` schema type, so an oversized
inner array is rejected during request parsing before any vertex traversal,
hashing, or sorting work.
"""

MAX_RESULT_COCHAIN_DEGREE = 128
"""Cap on the cohomological degree of any returned cochain.

Instability squares ``Sq^k(x) = 0`` for ``k > deg(x)`` return the empty
degree-``deg(x) + k`` cochain at constant work, so ``square_degree`` is
bounded output-sensitively: every request whose returned degree stays within
this budget is admitted, however far ``k`` lies above ``deg(x)``. Top squares
return degree ``2*cochain_degree <= 32``, always inside this budget.
"""

MAX_COCHAIN_DEGREE = 2 * MAX_TOPOLOGY_DIMENSION
"""Cap on the degree of a bare ``SimplicialCochain``.

A cochain above the complex dimension is the empty (zero) cochain, so the
cap must cover the product of two bounded input cochains.  The cup product of
degree-``p`` and degree-``q`` factors has degree ``p + q``, and both factors
are admitted up to ``MAX_TOPOLOGY_DIMENSION``, so the product degree reaches
``2 * MAX_TOPOLOGY_DIMENSION``; the kernel rejects any request whose product
degree leaves this envelope.
"""

MAX_VERTEX_LABEL_DIGITS = 6
"""Bound on decimal digits per vertex label (abs value < 10**6).

With at most 4096 ambient simplices each carrying up to 64 vertices and
up to 1024 support simplices, the worst-case JSON payload for labels stays
under ~2 MB when each label is at most 6 digits, keeping transport,
hashing/sorting, retained-source serialization, and exact result size
bounded. A single label such as ``10**N`` for arbitrary ``N`` would
otherwise inflate modular reduction, hashing, and retained integers without
a declared envelope and can fail during JSON encoding.
"""

MAX_COEFFICIENT_DIGITS = 6
"""Bound on decimal digits per coefficient before modular reduction.

Coefficients are reduced modulo 2 (Steenrod) or prime (Bockstein), but the
retained source stores the original integers for source binding. Bounding
each coefficient to 6 digits keeps retained-source size, hashing, and
sorting bounded while still admitting every residue class via bounded
representatives (``-999999..999999`` covers all residues for the admitted
primes).
"""

MAX_VERTEX_LABEL_MAGNITUDE = 10**MAX_VERTEX_LABEL_DIGITS - 1
"""Largest admissible absolute vertex label (at most 6 decimal digits)."""

BoundedVertexLabel = Annotated[
    int,
    Field(ge=-MAX_VERTEX_LABEL_MAGNITUDE, le=MAX_VERTEX_LABEL_MAGNITUDE),
]
"""One ambient-simplex vertex label, magnitude-bounded at the schema layer."""

BoundedAmbientSimplex = Annotated[
    tuple[BoundedVertexLabel, ...],
    Field(min_length=1, max_length=MAX_AMBIENT_SIMPLEX_VERTICES),
]
"""One ambient simplex with a schema-level per-simplex vertex cap.

The advertised per-simplex cap is enforced while parsing each inner array,
so a malformed request with one extremely large simplex is rejected before
``_require_bounded_vertex_labels`` traversal or ``_validate_simplex_entries``
hashing and sorting can run on it.
"""


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"cohomology_operation.{code}", message)


def _require_bounded_vertex_labels(
    entries: tuple[tuple[int, ...], ...],
    label: str,
) -> None:
    for simplex in entries:
        for vertex in simplex:
            if len(format_canonical_integer(abs(vertex))) > MAX_VERTEX_LABEL_DIGITS:
                raise _validation_error(
                    "vertex_label_bound",
                    f"{label} vertex label {vertex} exceeds the "
                    f"{MAX_VERTEX_LABEL_DIGITS}-digit bound",
                )


def _require_bounded_coefficients(
    coefficients: tuple[int, ...],
    label: str,
) -> None:
    for coefficient in coefficients:
        if len(format_canonical_integer(abs(coefficient))) > MAX_COEFFICIENT_DIGITS:
            raise _validation_error(
                "coefficient_bound",
                f"{label} coefficient {coefficient} exceeds the "
                f"{MAX_COEFFICIENT_DIGITS}-digit bound",
            )


def _validate_simplex_entries(
    entries: tuple[tuple[int, ...], ...],
    label: str,
) -> None:
    _require_bounded_vertex_labels(entries, label)
    for simplex in entries:
        if not simplex:
            raise _validation_error(
                "simplex_empty", f"{label} must have at least one vertex"
            )
        if len(set(simplex)) != len(simplex):
            raise _validation_error(
                "simplex_vertices_not_distinct", f"{label} vertices must be distinct"
            )
        if tuple(sorted(simplex)) != simplex:
            raise _validation_error(
                "simplex_vertices_not_canonical",
                f"{label} vertices must be sorted canonical",
            )


def _int_tuples_from_complex(
    simplicial_complex: FiniteSimplicialComplex,
) -> tuple[tuple[int, ...], ...]:
    """Convert a canonical ``FiniteSimplicialComplex`` into integer ambient tuples.

    The cohomology operation uses integer vertex labels ``0..n-1``.  The
    topology domain's canonical complex carries string labels in
    lexicographic order; the deterministic mapping ``label -> index`` via
    ``simplicial_complex.vertices`` preserves the canonical order so every face
    ``("a","b")`` becomes ``(idx_a, idx_b)`` with sorted vertices.
    """
    vertex_to_idx = {
        label: idx for idx, label in enumerate(simplicial_complex.vertices)
    }
    int_faces: list[tuple[int, ...]] = []
    for group in simplicial_complex.faces_by_dimension:
        for face in group.faces:
            int_face = tuple(vertex_to_idx[v] for v in face)
            # Faces are already canonical in string order; the index mapping
            # preserves that order because ``vertices`` is lexicographically
            # sorted, so the resulting int tuple remains sorted canonical.
            int_faces.append(int_face)
    return tuple(sorted(int_faces))


def _effective_ambient(
    ambient_simplices: tuple[tuple[int, ...], ...],
    ambient_complex: FiniteSimplicialComplex | None,
) -> tuple[tuple[int, ...], ...]:
    """Return the effective ambient set from either integer simplices or a canonical complex.

    When a ``FiniteSimplicialComplex`` is supplied its complete face closure
    is materialized as integer tuples; otherwise the raw ``ambient_simplices``
    are used.  If both are provided the union is used, so callers that
    supply both do not need to keep them perfectly synchronized.
    """
    if ambient_complex is not None:
        require_canonical_complex_admission(ambient_complex)
        derived = _int_tuples_from_complex(ambient_complex)
        if ambient_simplices:
            # Union keeps validation permissive while still requiring closure
            # and containment on the combined set.
            combined = tuple(sorted(set(ambient_simplices) | set(derived)))
            return combined
        return derived
    return ambient_simplices


class SteenrodSquareRequest(StrictModel):
    """Compute Steenrod squares Sq^k(x) for a cocycle over GF(2).

    The input is a simplicial cochain over GF(2): a list of simplex
    vertices with coefficients modulo 2.  Only three families are
    supported: ``Sq^0`` is the identity, ``Sq^{deg} = cup product``
    (the top square) requires the ambient simplicial complex to locate
    its ``(2*deg)``-simplex targets, and ``Sq^k = 0`` for ``k > deg``
    (instability).  Instability squares perform constant work and return
    the empty degree-``deg+k`` cochain, so they are admitted whenever the
    returned degree stays within the declared result budget rather than
    under a fixed ceiling on ``k``.  Intermediate squares ``0 < k < deg``
    require cup-``i`` structure and are rejected as unsupported.

    Nonzero cochains require ``ambient_simplices`` or ``ambient_complex``
    for cocycle verification (``d x = 0``) and the result is then an
    exact cohomology operation; only the zero cochain is admissible
    without an ambient complex, where ``Sq^0`` and ``k>deg`` are computed
    at the chain level without a cohomology claim. Top squares always
    require an ambient complex to locate their targets. ``ambient_complex``
    accepts the canonical value produced by
    ``topology.simplicial_complex.canonicalize`` (``FiniteSimplicialComplex``)
    and is materialized deterministically to integer simplices via the
    sorted vertex order, so ``topology`` outputs can be composed without
    relabeling. Each vertex label and coefficient is bounded to 6 decimal
    digits.
    """

    cochain_degree: int = Field(ge=0, le=16)
    simplex_values: tuple[tuple[int, ...], ...] = Field(min_length=0, max_length=1024)
    simplex_coefficients: tuple[int, ...] = Field(min_length=0, max_length=1024)
    square_degree: int = Field(ge=0, le=MAX_RESULT_COCHAIN_DEGREE)
    ambient_simplices: tuple[BoundedAmbientSimplex, ...] = Field(
        default=(),
        max_length=4096,
        description=(
            "Simplices of the ambient complex, each a canonically sorted "
            "vertex tuple with at most 64 vertices; the collection must be "
            "downward closed (every face of every listed simplex is listed). "
            "Top squares (square_degree == cochain_degree > 0) require an "
            "ambient complex; when ambient_complex is not supplied this field "
            "carries the complex.  Intermediate squares are unsupported."
        ),
    )
    ambient_complex: FiniteSimplicialComplex | None = Field(
        default=None,
        description=(
            "Canonical simplicial complex from topology.simplicial_complex.canonicalize. "
            "When supplied its complete face closure is materialized to integer "
            "simplices via the sorted vertex order and used as the ambient complex "
            "for cocycle verification and cup-product targeting, so topology outputs "
            "compose without relabeling.  Supply either ambient_simplices or "
            "ambient_complex; if both are given their union is used."
        ),
    )

    @model_validator(mode="after")
    def require_matching_lengths(self) -> Self:
        if len(self.simplex_values) != len(self.simplex_coefficients):
            raise _validation_error(
                "length_mismatch",
                "simplex_values and simplex_coefficients must have the same length",
            )
        # Validate simplex dimensions: each simplex must have exactly cochain_degree+1 distinct vertices.
        expected_dim = self.cochain_degree + 1
        for simplex in self.simplex_values:
            if len(simplex) != expected_dim:
                raise _validation_error(
                    "simplex_dimension",
                    f"each simplex must have exactly cochain_degree+1={expected_dim} vertices",
                )
            _validate_simplex_entries((simplex,), "simplex")
        _require_bounded_coefficients(self.simplex_coefficients, "simplex_coefficient")
        return self


class SteenrodSquareResult(SteenrodSquareRequest):
    """A canonical Steenrod-square result bound structurally to its source.

    Deserialization checks only the source-derived degree and canonical
    cochain representation. It does not recompute the cup product.
    """

    result_degree: int = Field(ge=0)
    result_simplex_values: tuple[tuple[int, ...], ...] = Field(default=())
    result_simplex_coefficients: tuple[int, ...] = Field(default=())
    is_zero: bool

    @model_validator(mode="after")
    def require_canonical_result_shape(self) -> Self:
        if len(self.result_simplex_values) != len(self.result_simplex_coefficients):
            raise _validation_error(
                "result_length_mismatch",
                "result_simplex_values and result_simplex_coefficients must have the same length",
            )
        if len(set(self.result_simplex_values)) != len(self.result_simplex_values):
            raise _validation_error(
                "result_duplicate_simplex",
                "canonical result support must not repeat a simplex",
            )
        for simplex in self.result_simplex_values:
            if len(simplex) != self.result_degree + 1:
                raise _validation_error(
                    "result_simplex_dimension",
                    "each result simplex must have result_degree plus one vertices",
                )
            _validate_simplex_entries((simplex,), "result simplex")
        _require_bounded_coefficients(
            self.result_simplex_coefficients, "result_simplex_coefficient"
        )
        return self

    @classmethod
    def _from_kernel(
        cls,
        cochain_degree: int,
        simplex_values: tuple[tuple[int, ...], ...],
        simplex_coefficients: tuple[int, ...],
        square_degree: int,
        ambient_simplices: tuple[tuple[int, ...], ...],
        ambient_complex: FiniteSimplicialComplex | None,
        result_degree: int,
        result_simplex_values: tuple[tuple[int, ...], ...],
        result_simplex_coefficients: tuple[int, ...],
        is_zero: bool,
    ) -> Self:
        """Construct a trusted result emitted by the owner-local kernel."""

        return cls.model_construct(
            cochain_degree=cochain_degree,
            simplex_values=simplex_values,
            simplex_coefficients=simplex_coefficients,
            square_degree=square_degree,
            ambient_simplices=ambient_simplices,
            ambient_complex=ambient_complex,
            result_degree=result_degree,
            result_simplex_values=result_simplex_values,
            result_simplex_coefficients=result_simplex_coefficients,
            is_zero=is_zero,
        )


class BocksteinRequest(StrictModel):
    """Compute the Bockstein homomorphism beta: H^n(Z/p) -> H^{n+1}(Z/p).

    The Bockstein for the short exact sequence 0 -> Z/p -> Z/p^2 -> Z/p -> 0
    requires the ambient simplicial complex to compute the coboundary of a
    lift. This operation currently only supports the trivial case where the
    input cocycle is zero modulo p (hence Bockstein is zero); non-zero
    cocycles are rejected as unsupported until the complex is provided.
    Duplicate simplex keys are summed modulo ``prime`` before the zero test,
    so a cochain whose sparse support cancels to zero is accepted as the
    zero cocycle.
    """

    prime: int = Field(ge=2, le=10_000)
    cochain_degree: int = Field(ge=0, le=16)
    simplex_values: tuple[tuple[int, ...], ...] = Field(min_length=0, max_length=1024)
    simplex_coefficients: tuple[int, ...] = Field(min_length=0, max_length=1024)
    ambient_simplices: tuple[BoundedAmbientSimplex, ...] = Field(
        default=(),
        max_length=4096,
        description=(
            "Optional ambient complex for future non-zero Bockstein; currently "
            "only the zero cocycle is supported and this field is accepted for "
            "composeability with topology outputs but not required."
        ),
    )
    ambient_complex: FiniteSimplicialComplex | None = Field(
        default=None,
        description=(
            "Canonical simplicial complex from topology.simplicial_complex.canonicalize. "
            "Accepted for composeability; currently only the zero cocycle is "
            "supported, so the complex is not required for the exact zero result "
            "but is validated for downward closure and support containment when supplied."
        ),
    )

    @model_validator(mode="after")
    def require_matching_lengths(self) -> Self:
        if len(self.simplex_values) != len(self.simplex_coefficients):
            raise _validation_error(
                "length_mismatch",
                "simplex_values and simplex_coefficients must have the same length",
            )
        expected_dim = self.cochain_degree + 1
        for simplex in self.simplex_values:
            if len(simplex) != expected_dim:
                raise _validation_error(
                    "simplex_dimension",
                    f"each simplex must have exactly cochain_degree+1={expected_dim} vertices",
                )
            _validate_simplex_entries((simplex,), "simplex")
        _require_bounded_coefficients(self.simplex_coefficients, "simplex_coefficient")
        return self


class BocksteinResult(BocksteinRequest):
    """The structurally canonical result of the supported Bockstein branch."""

    result_degree: int = Field(ge=0)
    result_simplex_values: tuple[tuple[int, ...], ...] = Field(default=())
    result_simplex_coefficients: tuple[int, ...] = Field(default=())
    is_zero: bool

    @classmethod
    def _from_kernel(
        cls,
        prime: int,
        cochain_degree: int,
        simplex_values: tuple[tuple[int, ...], ...],
        simplex_coefficients: tuple[int, ...],
        ambient_simplices: tuple[tuple[int, ...], ...],
        ambient_complex: FiniteSimplicialComplex | None,
        result_degree: int,
        result_simplex_values: tuple[tuple[int, ...], ...],
        result_simplex_coefficients: tuple[int, ...],
        is_zero: bool,
    ) -> Self:
        """Construct a trusted result emitted by the owner-local kernel."""

        return cls.model_construct(
            prime=prime,
            cochain_degree=cochain_degree,
            simplex_values=simplex_values,
            simplex_coefficients=simplex_coefficients,
            ambient_simplices=ambient_simplices,
            ambient_complex=ambient_complex,
            result_degree=result_degree,
            result_simplex_values=result_simplex_values,
            result_simplex_coefficients=result_simplex_coefficients,
            is_zero=is_zero,
        )


__all__ = [
    "BocksteinRequest",
    "BocksteinResult",
    "CohomologyRingRequest",
    "CohomologyRingResult",
    "CupProductEntry",
    "CupProductRequest",
    "CupProductResult",
    "InducedCohomologyMapRequest",
    "InducedCohomologyMapResult",
    "InducedCohomologyMatrix",
    "SimplicialCochain",
    "SimplicialMap",
    "SteenrodSquareRequest",
    "SteenrodSquareResult",
]


class SimplicialCochain(StrictModel):
    """One prime-field simplicial cochain bound to its complex and degree.

    Coefficients align with ``complex.faces_by_dimension[degree].faces`` in
    canonical order and lie in ``0..prime-1``.  Degrees above the complex
    dimension carry the empty (zero) cochain; the degree envelope
    ``MAX_COCHAIN_DEGREE`` covers the product of two bounded input cochains.
    """

    complex: FiniteSimplicialComplex
    prime: StrictInt = Field(ge=2, le=MAX_TOPOLOGY_PRIME)
    degree: StrictInt = Field(ge=0, le=MAX_COCHAIN_DEGREE)
    coefficients: tuple[StrictInt, ...] = Field(max_length=MAX_TOPOLOGY_CHAIN_GROUP)

    @model_validator(mode="after")
    def require_cochain_axis(self) -> Self:
        dimensions = tuple(entry.dimension for entry in self.complex.faces_by_dimension)
        if self.degree in dimensions:
            faces = self.complex.faces_by_dimension[dimensions.index(self.degree)].faces
        else:
            faces = ()
        if len(self.coefficients) != len(faces):
            raise _validation_error(
                "simplicial_cochain_axis",
                "cochain coefficients must cover the degree faces exactly once",
            )
        if any(
            coefficient < 0 or coefficient >= self.prime
            for coefficient in self.coefficients
        ):
            raise _validation_error(
                "simplicial_cochain_coefficients",
                "cochain coefficients must lie in the prime field",
            )
        return self


class SimplicialMap(StrictModel):
    """One simplicial vertex map between canonical complexes.

    ``vertex_map`` lists the target label of each source vertex in source
    vertex order.  Every source face must map onto a target face (degenerate
    images onto lower-dimensional faces are admitted); this simpliciality is
    the value's defining invariant and is checked on construction.
    """

    source: FiniteSimplicialComplex
    target: FiniteSimplicialComplex
    vertex_map: tuple[str, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def require_simplicial_map(self) -> Self:
        if len(self.vertex_map) != len(self.source.vertices):
            raise _validation_error(
                "simplicial_map_vertex_axis",
                "the vertex map covers every source vertex exactly once",
            )
        target_vertices = set(self.target.vertices)
        if any(label not in target_vertices for label in self.vertex_map):
            raise _validation_error(
                "simplicial_map_target_labels",
                "every vertex image must be a declared target vertex",
            )
        target_faces = {
            face for entry in self.target.faces_by_dimension for face in entry.faces
        }
        source_index = {
            label: position for position, label in enumerate(self.source.vertices)
        }
        for entry in self.source.faces_by_dimension:
            for face in entry.faces:
                image = tuple(
                    sorted({self.vertex_map[source_index[vertex]] for vertex in face})
                )
                if image not in target_faces:
                    raise _validation_error(
                        "simplicial_map_face_image",
                        "every source face must map onto a target face",
                    )
        return self


class CupProductRequest(StrictModel):
    """Multiply two simplicial cochains by Alexander-Whitney."""

    complex: FiniteSimplicialComplex
    prime: StrictInt = Field(ge=2, le=MAX_TOPOLOGY_PRIME)
    left: SimplicialCochain
    right: SimplicialCochain


class CupProductResult(StrictModel):
    """The Alexander-Whitney product cochain with its source factors."""

    complex: FiniteSimplicialComplex
    prime: StrictInt = Field(ge=2, le=MAX_TOPOLOGY_PRIME)
    left: SimplicialCochain
    right: SimplicialCochain
    product: SimplicialCochain

    @model_validator(mode="after")
    def require_cup_product_shape(self) -> Self:
        for name, cochain in (("left", self.left), ("right", self.right)):
            if cochain.complex != self.complex or cochain.prime != self.prime:
                raise _validation_error(
                    "cup_product_source_binding",
                    f"the {name} cochain must bind the source complex and prime",
                )
        if self.product.complex != self.complex or self.product.prime != self.prime:
            raise _validation_error(
                "cup_product_source_binding",
                "the product cochain must bind the source complex and prime",
            )
        if self.product.degree != self.left.degree + self.right.degree:
            raise _validation_error(
                "cup_product_degree",
                "the product degree is the sum of the factor degrees",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class CupProductEntry(StrictModel):
    """One cohomology product in class and coboundary coordinates.

    ``class_components`` are the coordinates of the left-cup-right product
    in the product-degree cohomology basis; ``coboundary_components``
    complete the exact cochain equation products from the basis product.
    """

    left_degree: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)
    left_index: StrictInt = Field(ge=0)
    right_degree: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)
    right_index: StrictInt = Field(ge=0)
    class_components: tuple[StrictInt, ...]
    coboundary_components: tuple[StrictInt, ...]


class CohomologyRingRequest(StrictModel):
    """Compute the prime-field cohomology ring multiplication table."""

    complex: FiniteSimplicialComplex
    prime: StrictInt = Field(ge=2, le=MAX_TOPOLOGY_PRIME)
    convention: HomologyConvention = HomologyConvention.UNREDUCED


class CohomologyRingResult(StrictModel):
    """Cohomology with its cup-product structure constants.

    ``products`` holds one entry per cohomology-basis pair whose degrees
    fit in the complex dimension, in degree-major order.  Together with the
    retained cohomology bases, the entries determine the full graded ring.
    """

    complex: FiniteSimplicialComplex
    prime: StrictInt = Field(ge=2, le=MAX_TOPOLOGY_PRIME)
    convention: HomologyConvention
    cohomology: SimplicialCohomologyResult
    products: tuple[CupProductEntry, ...]

    @model_validator(mode="after")
    def require_ring_shape(self) -> Self:
        if (
            self.cohomology.complex != self.complex
            or self.cohomology.prime != self.prime
            or self.cohomology.convention != self.convention
        ):
            raise _validation_error(
                "cohomology_ring_source_binding",
                "retained cohomology must bind the source complex, prime, "
                "and convention",
            )
        betti = {
            group.dimension: group.betti_number for group in self.cohomology.groups
        }
        coboundary_ranks = {
            group.dimension: len(group.coboundary_basis)
            for group in self.cohomology.groups
        }
        expected: list[tuple[int, int, int, int]] = []
        for left_degree, left_betti in betti.items():
            for left_index in range(left_betti):
                for right_degree, right_betti in betti.items():
                    total = left_degree + right_degree
                    if total > self.complex.dimension:
                        continue
                    for right_index in range(right_betti):
                        expected.append(
                            (left_degree, left_index, right_degree, right_index)
                        )
        actual = tuple(
            (
                entry.left_degree,
                entry.left_index,
                entry.right_degree,
                entry.right_index,
            )
            for entry in self.products
        )
        if actual != tuple(expected):
            raise _validation_error(
                "cohomology_ring_table_coverage",
                "products must cover every in-dimension basis pair exactly once",
            )
        for entry in self.products:
            total = entry.left_degree + entry.right_degree
            if len(entry.class_components) != betti.get(total, 0):
                raise _validation_error(
                    "cohomology_ring_class_axis",
                    "class components must cover the product-degree basis",
                )
            if len(entry.coboundary_components) != coboundary_ranks.get(total, 0):
                raise _validation_error(
                    "cohomology_ring_coboundary_axis",
                    "coboundary components must cover the product-degree basis",
                )
            for coefficient in (*entry.class_components, *entry.coboundary_components):
                if coefficient < 0 or coefficient >= self.prime:
                    raise _validation_error(
                        "cohomology_ring_coefficients",
                        "structure constants must lie in the prime field",
                    )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class InducedCohomologyMapRequest(StrictModel):
    """Pull cohomology classes back along a simplicial map."""

    map: SimplicialMap
    prime: StrictInt = Field(ge=2, le=MAX_TOPOLOGY_PRIME)
    convention: HomologyConvention = HomologyConvention.UNREDUCED


class InducedCohomologyMatrix(StrictModel):
    """Pullback ``H^k(target) -> H^k(source)`` in the retained bases.

    Rows index the source cohomology basis, columns the target basis;
    either side is empty when the degree exceeds its complex dimension.
    """

    degree: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)
    rows: tuple[tuple[StrictInt, ...], ...]


class InducedCohomologyMapResult(StrictModel):
    """Source and target cohomology with every pullback matrix."""

    map: SimplicialMap
    prime: StrictInt = Field(ge=2, le=MAX_TOPOLOGY_PRIME)
    convention: HomologyConvention
    source_cohomology: SimplicialCohomologyResult
    target_cohomology: SimplicialCohomologyResult
    matrices: tuple[InducedCohomologyMatrix, ...]

    @model_validator(mode="after")
    def require_induced_map_shape(self) -> Self:
        if (
            self.source_cohomology.complex != self.map.source
            or self.target_cohomology.complex != self.map.target
        ):
            raise _validation_error(
                "induced_map_cohomology_binding",
                "retained cohomologies must bind the map source and target",
            )
        for cohomology in (self.source_cohomology, self.target_cohomology):
            if (
                cohomology.prime != self.prime
                or cohomology.convention != self.convention
            ):
                raise _validation_error(
                    "induced_map_coefficient_binding",
                    "retained cohomologies must bind the prime and convention",
                )
        top = max(self.map.source.dimension, self.map.target.dimension)
        if tuple(matrix.degree for matrix in self.matrices) != tuple(range(top + 1)):
            raise _validation_error(
                "induced_map_degree_coverage",
                "matrices must cover every degree through the top dimension",
            )
        for matrix in self.matrices:
            source_betti = next(
                (
                    group.betti_number
                    for group in self.source_cohomology.groups
                    if group.dimension == matrix.degree
                ),
                0,
            )
            target_betti = next(
                (
                    group.betti_number
                    for group in self.target_cohomology.groups
                    if group.dimension == matrix.degree
                ),
                0,
            )
            if len(matrix.rows) != source_betti or any(
                len(row) != target_betti for row in matrix.rows
            ):
                raise _validation_error(
                    "induced_map_matrix_shape",
                    "each matrix must map target classes to source classes",
                )
            for row in matrix.rows:
                if any(
                    coefficient < 0 or coefficient >= self.prime for coefficient in row
                ):
                    raise _validation_error(
                        "induced_map_coefficients",
                        "pullback entries must lie in the prime field",
                    )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)
