"""Typed wire contracts for cohomology operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.topology._models import FiniteSimplicialComplex

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
            if len(str(abs(vertex))) > MAX_VERTEX_LABEL_DIGITS:
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
        if len(str(abs(coefficient))) > MAX_COEFFICIENT_DIGITS:
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
        expected_degree = self.cochain_degree + self.square_degree
        if self.result_degree != expected_degree:
            raise _validation_error(
                "result_degree",
                "result_degree must equal cochain_degree plus square_degree",
            )
        if len(self.result_simplex_values) != len(self.result_simplex_coefficients):
            raise _validation_error(
                "result_length_mismatch",
                "result_simplex_values and result_simplex_coefficients must have the same length",
            )
        if self.is_zero != (not self.result_simplex_values):
            raise _validation_error(
                "result_zero_shape",
                "is_zero must agree with whether the canonical result support is empty",
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
        if any(
            coefficient % 2 != 1 for coefficient in self.result_simplex_coefficients
        ):
            raise _validation_error(
                "result_coefficient",
                "canonical GF(2) result coefficients must equal one",
            )
        effective_ambient = _effective_ambient(
            self.ambient_simplices, self.ambient_complex
        )
        ambient_faces = set(effective_ambient)
        if effective_ambient and any(
            simplex not in ambient_faces for simplex in self.result_simplex_values
        ):
            raise _validation_error(
                "result_outside_ambient",
                "result support must lie inside the ambient complex",
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

    @model_validator(mode="after")
    def require_supported_zero_shape(self) -> Self:
        if (
            self.result_degree != self.cochain_degree + 1
            or self.result_simplex_values
            or self.result_simplex_coefficients
            or not self.is_zero
        ):
            raise _validation_error(
                "result_shape",
                "the supported Bockstein branch returns the empty degree-(cochain_degree + 1) cochain",
            )
        return self

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
    "SteenrodSquareRequest",
    "SteenrodSquareResult",
]
