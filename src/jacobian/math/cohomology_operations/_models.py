"""Typed wire contracts for cohomology operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.topology._models import FiniteSimplicialComplex

MAX_AMBIENT_SIMPLEX_VERTICES = 64
"""Cap on the vertex count of one supplied ambient simplex.

Supported cochain degrees stop at 16, so top-square targets carry at most
``2*16 + 1 = 33`` vertices; 64 keeps validation work linear and bounded with
headroom over every simplex dimension these operations can target.
"""


def _validate_simplex_entries(
    entries: tuple[tuple[int, ...], ...],
    label: str,
) -> None:
    for simplex in entries:
        if not simplex:
            raise ValueError(f"{label} must have at least one vertex")
        if len(set(simplex)) != len(simplex):
            raise ValueError(f"{label} vertices must be distinct")
        if tuple(sorted(simplex)) != simplex:
            raise ValueError(f"{label} vertices must be sorted canonical")


def _require_downward_closed(simplices: tuple[tuple[int, ...], ...]) -> None:
    """Require every codimension-one face of every supplied simplex.

    A finite simplicial complex is determined by its listed simplices only
    when it is closed under taking faces. Without closure, a tetrahedron
    entry would silently imply triangles and edges that the coboundary test
    and top-square targeting never see.
    """
    known = set(simplices)
    for simplex in simplices:
        for index in range(len(simplex)):
            face = simplex[:index] + simplex[index + 1 :]
            if face and face not in known:
                raise ValueError(
                    "ambient_simplices must be downward closed: the face "
                    f"{face} of {simplex} is absent"
                )


def _require_cocycle(
    cochain_degree: int,
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
    ambient_simplices: tuple[tuple[int, ...], ...],
) -> None:
    """Require the GF(2) coboundary to vanish on ambient (degree+1)-simplices.

    Steenrod squares are cohomology operations: the supplied cochain must be
    a cocycle. Over GF(2) the coboundary of a degree-p cochain on a
    (p+1)-simplex is the GF(2) sum of its values on the p+1
    codimension-one faces.
    """
    values_by_face: dict[tuple[int, ...], int] = {}
    for simplex, coefficient in zip(simplex_values, simplex_coefficients, strict=True):
        key = tuple(simplex)
        values_by_face[key] = (values_by_face.get(key, 0) + coefficient) % 2
    for sigma in ambient_simplices:
        if len(sigma) != cochain_degree + 2:
            continue
        coboundary = 0
        for index in range(len(sigma)):
            face = sigma[:index] + sigma[index + 1 :]
            coboundary = (coboundary + values_by_face.get(face, 0)) % 2
        if coboundary:
            raise ValueError(
                "the supplied cochain is not a cocycle: its "
                "coboundary does not vanish on the ambient complex"
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


def _validate_ambient_complex(
    cochain_degree: int,
    support: tuple[tuple[int, ...], ...],
    coefficients: tuple[int, ...],
    ambient: tuple[tuple[int, ...], ...],
) -> None:
    """Validate shape, closure, support containment, and cocyclicity.

    The complex must be closed under taking faces before any of the other
    checks run: an omitted implied face would hide a nonzero coboundary or
    let a cup product target a simplex that does not exist.
    """
    _validate_simplex_entries(ambient, "ambient simplex")
    if any(len(simplex) > MAX_AMBIENT_SIMPLEX_VERTICES for simplex in ambient):
        raise ValueError(
            "each ambient simplex may carry at most "
            f"{MAX_AMBIENT_SIMPLEX_VERTICES} vertices"
        )
    known = set(ambient)
    for simplex in support:
        if simplex not in known:
            raise ValueError("cochain support must lie inside the ambient complex")
    _require_downward_closed(ambient)
    _require_cocycle(cochain_degree, support, coefficients, ambient)


def _is_zero_mod2_cochain(
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
) -> bool:
    """Return whether the GF(2) cochain represented by sparse support is zero."""
    merged: dict[tuple[int, ...], int] = {}
    for simplex, coefficient in zip(simplex_values, simplex_coefficients, strict=True):
        key = tuple(simplex)
        merged[key] = (merged.get(key, 0) + coefficient) % 2
    return not any(value != 0 for value in merged.values())


def _is_zero_mod_prime_cochain(
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
    prime: int,
) -> bool:
    """Return whether the Z/p cochain represented by sparse support is zero."""
    merged: dict[tuple[int, ...], int] = {}
    for simplex, coefficient in zip(simplex_values, simplex_coefficients, strict=True):
        key = tuple(simplex)
        merged[key] = (merged.get(key, 0) + coefficient) % prime
    return not any(value != 0 for value in merged.values())


class SteenrodSquareRequest(StrictModel):
    """Compute Steenrod squares Sq^k(x) for a cocycle over GF(2).

    The input is a simplicial cochain over GF(2): a list of simplex
    vertices with coefficients modulo 2.  Only three families are
    supported: ``Sq^0`` is the (co)chain-level identity, ``Sq^{deg} =
    cup product`` (the top square) requires the ambient simplicial complex
    to locate its ``(2*deg)``-simplex targets, and ``Sq^k = 0`` for
    ``k > deg`` (instability).  Intermediate squares ``0 < k < deg``
    require cup-``i`` structure and are rejected as unsupported.

    When ``ambient_simplices`` or ``ambient_complex`` is supplied the
    cochain is verified to be a cocycle (``d x = 0``) and the result is
    an exact cohomology operation; without an ambient complex the
    ``k=0`` and ``k>deg`` branches are computed at the cochain level
    and do not claim to represent a cohomology class.  ``ambient_complex``
    accepts the canonical value produced by
    ``topology.simplicial_complex.canonicalize`` (``FiniteSimplicialComplex``)
    and is materialized deterministically to integer simplices via the
    sorted vertex order, so ``topology`` outputs can be composed without
    relabeling.
    """

    cochain_degree: int = Field(ge=0, le=16)
    simplex_values: tuple[tuple[int, ...], ...] = Field(min_length=0, max_length=1024)
    simplex_coefficients: tuple[int, ...] = Field(min_length=0, max_length=1024)
    square_degree: int = Field(ge=0, le=16)
    ambient_simplices: tuple[tuple[int, ...], ...] = Field(
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
            raise ValueError(
                "simplex_values and simplex_coefficients must have the same length"
            )
        # Validate simplex dimensions: each simplex must have exactly cochain_degree+1 distinct vertices.
        expected_dim = self.cochain_degree + 1
        for simplex in self.simplex_values:
            if len(simplex) != expected_dim:
                raise ValueError(
                    f"each simplex must have exactly cochain_degree+1={expected_dim} vertices"
                )
            _validate_simplex_entries((simplex,), "simplex")
        # Intermediate squares 0<k<deg are not implemented; reject to avoid false zero.
        if 0 < self.square_degree < self.cochain_degree:
            raise ValueError(
                "intermediate Steenrod squares 0<k<deg require cup-i products and are not supported"
            )
        # Effective ambient from either integer tuples or the canonical complex.
        effective_ambient = _effective_ambient(
            self.ambient_simplices, self.ambient_complex
        )
        has_nonzero = not _is_zero_mod2_cochain(
            self.simplex_values, self.simplex_coefficients
        )
        if has_nonzero and not effective_ambient:
            raise ValueError(
                "Steenrod squares are cohomology operations: the supplied "
                "cochain must be verified as a cocycle against an ambient "
                "simplicial complex; supply ambient_simplices or ambient_complex"
            )
        # Top squares need the ambient to locate (2*deg)-simplex targets.
        if (
            self.square_degree == self.cochain_degree
            and self.cochain_degree >= 1
            and not effective_ambient
        ):
            raise ValueError(
                "the top Steenrod square requires the ambient simplicial "
                "complex; supply ambient_simplices or ambient_complex"
            )
        if effective_ambient:
            _validate_ambient_complex(
                self.cochain_degree,
                self.simplex_values,
                self.simplex_coefficients,
                effective_ambient,
            )
        return self


class SteenrodSquareResult(SteenrodSquareRequest):
    """The result of a Steenrod square operation, bound to its source."""

    result_degree: int = Field(ge=0)
    result_simplex_values: tuple[tuple[int, ...], ...] = Field(default=())
    result_simplex_coefficients: tuple[int, ...] = Field(default=())
    is_zero: bool

    @model_validator(mode="after")
    def bind_to_source_cochain(self) -> Self:
        from jacobian.math.cohomology_operations._operations import (
            steenrod_square_fields,
        )

        effective = _effective_ambient(self.ambient_simplices, self.ambient_complex)
        expected = steenrod_square_fields(
            self.cochain_degree,
            self.simplex_values,
            self.simplex_coefficients,
            self.square_degree,
            effective,
        )
        actual = (
            self.result_degree,
            self.result_simplex_values,
            self.result_simplex_coefficients,
            self.is_zero,
        )
        if actual != expected:
            raise ValueError(
                "result must equal the exact Steenrod-square replay of the "
                "retained source cochain"
            )
        return self


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
    ambient_simplices: tuple[tuple[int, ...], ...] = Field(
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
            raise ValueError(
                "simplex_values and simplex_coefficients must have the same length"
            )
        from sympy import isprime

        if not isprime(self.prime):
            raise ValueError("prime must be a prime integer")
        expected_dim = self.cochain_degree + 1
        for simplex in self.simplex_values:
            if len(simplex) != expected_dim:
                raise ValueError(
                    f"each simplex must have exactly cochain_degree+1={expected_dim} vertices"
                )
            _validate_simplex_entries((simplex,), "simplex")
        has_nonzero = not _is_zero_mod_prime_cochain(
            self.simplex_values, self.simplex_coefficients, self.prime
        )
        # Only zero cocycles are supported without the ambient complex.
        # Non-zero coefficients would require computing (1/p) d(lift), which needs
        # the simplicial coboundary and hence the full complex. Reject as unsupported
        # to avoid returning a false exact zero.
        if has_nonzero:
            raise ValueError(
                "non-zero Bockstein requires the ambient simplicial complex; unsupported in this bounded operation"
            )
        # When an ambient complex is supplied, validate its shape and that the
        # (possibly duplicate) support lies inside it.  Merged zero cochains
        # are still validated for containment so a caller cannot claim a zero
        # on simplices outside the complex.
        effective_ambient = _effective_ambient(
            self.ambient_simplices, self.ambient_complex
        )
        if effective_ambient:
            _validate_simplex_entries(effective_ambient, "ambient simplex")
            if any(len(s) > MAX_AMBIENT_SIMPLEX_VERTICES for s in effective_ambient):
                raise ValueError(
                    "each ambient simplex may carry at most "
                    f"{MAX_AMBIENT_SIMPLEX_VERTICES} vertices"
                )
            known = set(effective_ambient)
            for simplex in self.simplex_values:
                if simplex not in known:
                    raise ValueError(
                        "cochain support must lie inside the ambient complex"
                    )
            _require_downward_closed(effective_ambient)
        return self


class BocksteinResult(BocksteinRequest):
    """The result of the Bockstein homomorphism, bound to its source."""

    result_degree: int = Field(ge=0)
    result_simplex_values: tuple[tuple[int, ...], ...] = Field(default=())
    result_simplex_coefficients: tuple[int, ...] = Field(default=())
    is_zero: bool

    @model_validator(mode="after")
    def bind_to_source_cochain(self) -> Self:
        from jacobian.math.cohomology_operations._operations import (
            bockstein_fields,
        )

        expected = bockstein_fields(
            self.prime,
            self.cochain_degree,
            self.simplex_coefficients,
            self.simplex_values,
        )
        actual = (
            self.result_degree,
            self.result_simplex_values,
            self.result_simplex_coefficients,
            self.is_zero,
        )
        if actual != expected:
            raise ValueError(
                "result must equal the exact Bockstein replay of the "
                "retained source cochain"
            )
        return self


__all__ = [
    "BocksteinRequest",
    "BocksteinResult",
    "SteenrodSquareRequest",
    "SteenrodSquareResult",
]
