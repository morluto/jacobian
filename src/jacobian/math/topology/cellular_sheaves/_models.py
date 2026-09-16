"""Typed contracts for exact finite cellular sheaves on simplicial complexes.

A cellular sheaf assigns one finite-dimensional based exact vector space to
every nonempty simplex and one linear restriction map ``rho: F(sigma) ->
F(tau)`` to every face inclusion, so that identities hold and saturated-chain
compositions are path independent.  This module owns the request and result
contracts for :func:`from_cover_maps`, which accepts a complete candidate
cover diagram and either returns the canonical sheaf with every derived
comparable-face restriction or rejects the first missing, wrong-axis, or
noncommuting entry as a typed mathematical negative.
"""

from __future__ import annotations

from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Any, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.topology._models import FiniteSimplicialComplex, Simplex

MAX_SHEAF_SIMPLICES = 64
MAX_SHEAF_STALK_RANK = 8
MAX_SHEAF_TOTAL_STALK_RANK = 512
MAX_SHEAF_COVER_MAPS = 512
MAX_SHEAF_ENTRY_DIGITS = 64
MAX_SHEAF_PRIME = 1000003
MAX_SHEAF_DERIVED_RESTRICTIONS = 2048
MAX_SHEAF_RESTRICTION_CELLS = 65536
MAX_SHEAF_DIAMONDS = 65536

BasisLabel = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,31}$",
        strict=True,
    ),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"topology.cellular_sheaf.{reason}", message)


def _require_canonical_simplex(simplex: Simplex, *, label: str) -> None:
    if not simplex:
        raise _validation_error("empty_simplex", f"{label} must be a nonempty simplex")
    if tuple(sorted(simplex)) != simplex:
        raise _validation_error(
            "simplex_not_canonical",
            f"{label} must use canonical increasing vertex order",
        )
    if len(set(simplex)) != len(simplex):
        raise _validation_error(
            "simplex_not_canonical", f"{label} must have distinct vertices"
        )


class SheafField(StrEnum):
    """Exact coefficient fields supported by cellular sheaves."""

    RATIONAL = "QQ"
    PRIME_FIELD = "GF_p"


class SheafOutcome(StrEnum):
    """Discriminated outcome of one cover-diagram construction."""

    CELLULAR_SHEAF = "CELLULAR_SHEAF"
    NOT_A_SHEAF = "NOT_A_SHEAF"


class SheafObstructionCode(StrEnum):
    """Stable codes for the first diagram obstruction."""

    MISSING_COVER_MAP = "MISSING_COVER_MAP"
    COVER_MAP_WRONG_AXIS = "COVER_MAP_WRONG_AXIS"
    NONCOMMUTING_DIAMOND = "NONCOMMUTING_DIAMOND"


class SheafStalk(StrictModel):
    """A based finite-dimensional stalk on one nonempty simplex.

    ``basis`` is an ordered tuple of unique basis labels; its length is the
    stalk rank.  The zero-dimensional stalk is represented by an empty basis
    and is a first-class value.
    """

    simplex: Simplex
    basis: tuple[BasisLabel, ...] = Field(default=())

    @model_validator(mode="after")
    def require_based_stalk(self) -> Self:
        _require_canonical_simplex(self.simplex, label="stalk simplex")
        if len(set(self.basis)) != len(self.basis):
            raise _validation_error(
                "basis_labels_not_unique",
                "stalk basis labels must be unique",
            )
        return self


class CoverRestrictionMatrix(StrictModel):
    """One candidate restriction matrix on a face-poset cover inclusion.

    ``source`` must be a codimension-one face of ``target``.  ``entries`` is
    a dense row-major matrix whose rows index the target stalk basis and whose
    columns index the source stalk basis, matching ``rho: F(source) ->
    F(target)``.  Entries are exact scalar strings: integers or ``num/den``
    rationals over ``QQ`` and integer representatives over ``GF(p)``.
    """

    source: Simplex
    target: Simplex
    entries: tuple[tuple[str, ...], ...] = Field(default=())

    @model_validator(mode="after")
    def require_rectangular_matrix(self) -> Self:
        _require_canonical_simplex(self.source, label="cover source")
        _require_canonical_simplex(self.target, label="cover target")
        widths = {len(row) for row in self.entries}
        if len(widths) > 1:
            raise _validation_error(
                "matrix_not_rectangular",
                "cover restriction entries must be a rectangular matrix",
            )
        return self


class FromCoverMapsRequest(StrictModel):
    """Build a cellular sheaf from a complete candidate cover diagram."""

    complex: FiniteSimplicialComplex
    coefficient_field: SheafField = SheafField.RATIONAL
    prime: StrictInt | None = Field(default=None, ge=2, le=MAX_SHEAF_PRIME)
    stalks: tuple[SheafStalk, ...] = Field(default=())
    cover_maps: tuple[CoverRestrictionMatrix, ...] = Field(default=())


class SheafRestriction(StrictModel):
    """One exact restriction ``rho: F(source) -> F(target)``.

    Rows are indexed by ``row_basis`` (the target stalk basis) and columns by
    ``column_basis`` (the source stalk basis).  ``cover_path`` records the
    saturated chain of simplices along which a derived composite was taken;
    for a cover inclusion it is exactly ``(source, target)``.
    """

    source: Simplex
    target: Simplex
    row_basis: tuple[BasisLabel, ...] = Field(default=())
    column_basis: tuple[BasisLabel, ...] = Field(default=())
    entries: tuple[tuple[str, ...], ...] = Field(default=())
    cover_path: tuple[Simplex, ...] = Field(default=())

    @model_validator(mode="after")
    def require_bound_matrix(self) -> Self:
        _require_canonical_simplex(self.source, label="restriction source")
        _require_canonical_simplex(self.target, label="restriction target")
        if len(self.entries) != len(self.row_basis) or any(
            len(row) != len(self.column_basis) for row in self.entries
        ):
            raise _validation_error(
                "restriction_axis_mismatch",
                "restriction entries must match the declared row and column bases",
            )
        if self.cover_path:
            if self.cover_path[0] != self.source or self.cover_path[-1] != self.target:
                raise _validation_error(
                    "cover_path_endpoints",
                    "cover path must start at the source and end at the target",
                )
            for earlier, later in pairwise(self.cover_path):
                if not set(earlier) < set(later) or len(later) != len(earlier) + 1:
                    raise _validation_error(
                        "cover_path_not_saturated",
                        "cover path steps must be codimension-one face inclusions",
                    )
        return self


class FiniteCellularSheaf(StrictModel):
    """A checked cellular sheaf with its complete derived restriction diagram.

    ``stalks`` binds one based stalk to every nonempty simplex in canonical
    order.  ``cover_restrictions`` covers every codimension-one inclusion and
    ``derived_restrictions`` every longer comparable inclusion.  ``diamonds``
    counts the elementary commuting squares that were replayed and
    ``comparable_pairs`` counts every ordered comparable face pair; both are
    inspectable accounting, never re-derived by a validator.
    """

    complex: FiniteSimplicialComplex
    coefficient_field: SheafField
    prime: StrictInt | None = Field(default=None, ge=2, le=MAX_SHEAF_PRIME)
    stalks: tuple[SheafStalk, ...] = Field(default=())
    cover_restrictions: tuple[SheafRestriction, ...] = Field(default=())
    derived_restrictions: tuple[SheafRestriction, ...] = Field(default=())
    diamonds: StrictInt = Field(default=0, ge=0)
    comparable_pairs: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def require_bound_diagram(self) -> Self:
        if self.coefficient_field is SheafField.PRIME_FIELD:
            if self.prime is None:
                raise _validation_error(
                    "prime_required", "GF(p) sheaves must carry their prime modulus"
                )
        elif self.prime is not None:
            raise _validation_error(
                "prime_forbidden", "QQ sheaves must not carry a prime modulus"
            )
        canonical = self.canonical_face_order
        if tuple(stalk.simplex for stalk in self.stalks) != canonical:
            raise _validation_error(
                "stalk_order_invalid",
                "stalks must bind every face in canonical dimension/lexical order",
            )
        basis_for = {stalk.simplex: stalk.basis for stalk in self.stalks}
        seen_pairs: set[tuple[Simplex, Simplex]] = set()
        for restriction in (
            *self.cover_restrictions,
            *self.derived_restrictions,
        ):
            if (
                basis_for.get(restriction.target) != restriction.row_basis
                or basis_for.get(restriction.source) != restriction.column_basis
            ):
                raise _validation_error(
                    "restriction_basis_unbound",
                    "restriction axes must equal the declared stalk bases",
                )
            key = (restriction.source, restriction.target)
            if key in seen_pairs:
                raise _validation_error(
                    "duplicate_restriction",
                    "each comparable face pair carries exactly one restriction",
                )
            seen_pairs.add(key)
        return self

    @property
    def canonical_face_order(self) -> tuple[Simplex, ...]:
        return tuple(
            face for group in self.complex.faces_by_dimension for face in group.faces
        )

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class DiamondCounterexample(StrictModel):
    """Two saturated cover paths with unequal composed restriction matrices.

    Both paths run from ``source`` to ``target`` through distinct intermediate
    cells; the matrices are the exact composites in canonical scalar text.
    """

    source: Simplex
    target: Simplex
    first_path: tuple[Simplex, ...]
    second_path: tuple[Simplex, ...]
    first_matrix: tuple[tuple[str, ...], ...]
    second_matrix: tuple[tuple[str, ...], ...]

    @model_validator(mode="after")
    def require_parallel_paths(self) -> Self:
        for path in (self.first_path, self.second_path):
            if len(path) < 3 or path[0] != self.source or path[-1] != self.target:
                raise _validation_error(
                    "diamond_path_endpoints",
                    "diamond paths must run from the source to the target",
                )
        if self.first_path == self.second_path:
            raise _validation_error(
                "diamond_paths_identical",
                "a noncommuting diamond requires two distinct paths",
            )
        if tuple(len(row) for row in self.first_matrix) != tuple(
            len(row) for row in self.second_matrix
        ) or len(self.first_matrix) != len(self.second_matrix):
            raise _validation_error(
                "diamond_matrix_shape",
                "both composites must share the source/target stalk shape",
            )
        return self


class SheafObstruction(StrictModel):
    """The first obstruction that keeps a candidate diagram from being a sheaf."""

    code: SheafObstructionCode
    message: str = Field(min_length=1, max_length=512)
    source: Simplex | None = None
    target: Simplex | None = None
    diamond: DiamondCounterexample | None = None

    @model_validator(mode="after")
    def require_obstruction_payload(self) -> Self:
        if self.code is SheafObstructionCode.NONCOMMUTING_DIAMOND:
            if self.diamond is None:
                raise _validation_error(
                    "diamond_required",
                    "a noncommuting obstruction requires both concrete paths",
                )
        elif self.diamond is not None:
            raise _validation_error(
                "diamond_forbidden",
                "only a noncommuting obstruction carries a diamond",
            )
        if self.code is not SheafObstructionCode.NONCOMMUTING_DIAMOND and (
            self.source is None or self.target is None
        ):
            raise _validation_error(
                "obstruction_pair_required",
                "a missing or wrong-axis obstruction names its cover pair",
            )
        return self


class FromCoverMapsResult(StrictModel):
    """Discriminated construction outcome bound to its source diagram."""

    outcome: SheafOutcome
    sheaf: FiniteCellularSheaf | None = None
    obstruction: SheafObstruction | None = None

    @model_validator(mode="after")
    def require_discriminated_payload(self) -> Self:
        if self.outcome is SheafOutcome.CELLULAR_SHEAF:
            if self.sheaf is None or self.obstruction is not None:
                raise _validation_error(
                    "sheaf_payload_conflict",
                    "a sheaf outcome carries the canonical sheaf only",
                )
        elif self.sheaf is not None or self.obstruction is None:
            raise _validation_error(
                "obstruction_payload_conflict",
                "a negative outcome carries its obstruction only",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


__all__ = [
    "MAX_SHEAF_COVER_MAPS",
    "MAX_SHEAF_DERIVED_RESTRICTIONS",
    "MAX_SHEAF_DIAMONDS",
    "MAX_SHEAF_ENTRY_DIGITS",
    "MAX_SHEAF_PRIME",
    "MAX_SHEAF_RESTRICTION_CELLS",
    "MAX_SHEAF_SIMPLICES",
    "MAX_SHEAF_STALK_RANK",
    "MAX_SHEAF_TOTAL_STALK_RANK",
    "BasisLabel",
    "CoverRestrictionMatrix",
    "DiamondCounterexample",
    "FiniteCellularSheaf",
    "FromCoverMapsRequest",
    "FromCoverMapsResult",
    "SheafField",
    "SheafObstruction",
    "SheafObstructionCode",
    "SheafOutcome",
    "SheafRestriction",
    "SheafStalk",
]
