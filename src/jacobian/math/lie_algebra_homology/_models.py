"""Typed wire contracts for Lie algebra homology operations."""

from __future__ import annotations

from math import comb
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

_MAX_PRIME_FIELD_AXIS = 256
"""Shared PrimeFieldMatrix row/column cap."""

MAX_LIE_ALGEBRA_DIMENSION = max(
    n
    for n in range(1, _MAX_PRIME_FIELD_AXIS + 1)
    if comb(n, n // 2) <= _MAX_PRIME_FIELD_AXIS
)
"""Largest admitted Lie-algebra dimension.

Derived from the execution envelope rather than hard-coded: every CE
differential d_p has shape C(n, p - 1) x C(n, p), so the widest chain group
C(n, floor(n / 2)) must fit one prime-field matrix row/column axis. The
complete dense complex then carries sum_p C(n, p - 1) * C(n, p) field entries
(167,960 at the admitted maximum), inside the kernel and transport budgets.
"""

MAX_LIE_ALGEBRA_PRIME = 2_147_483_647
"""Conservative characteristic envelope shared with the GF(p) linear-algebra domain.

CE construction performs no prime-dependent expansion: each differential keeps
its C(n, degree - 1) x C(n, degree) residue shape for every characteristic,
each stored value is one canonical residue, and elimination work scales only
polynomially in log(prime) through the shared DomainMatrix kernel. The dense
complex admits at most 167,960 residues in total, so per-entry bit growth
never approaches any budget before the shared backend characteristic cap.
Until a sharper characteristic/bit-length budget is established for this
domain, this documented conservative fallback admits exactly the
characteristics the shared ``PrimeFieldMatrix`` kernel accepts, keeping one
GF(p) envelope across domains rather than an operation-local ceiling.
"""


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"lie_algebra_homology.{reason}", message)


def _require_prime(prime: int) -> None:
    from sympy import isprime

    if not isprime(prime):
        raise _validation_error("prime_not_prime", "prime must be a prime integer")


def _require_canonical_residues(
    c: tuple[tuple[tuple[int, ...], ...], ...], n: int, p: int
) -> None:
    for i in range(n):
        for j in range(n):
            if any(not 0 <= value < p for value in c[i][j]):
                raise _validation_error(
                    "canonical_residues",
                    "structure constant entries must be canonical GF(prime) "
                    f"residues: 0 <= value < {p}",
                )


def _require_alternating(
    c: tuple[tuple[tuple[int, ...], ...], ...], n: int, p: int
) -> None:
    for i in range(n):
        if any(value % p != 0 for value in c[i][i]):
            raise _validation_error(
                "alternating",
                "structure constants must define an alternating bracket: [e_i, e_i] = 0",
            )


def _require_antisymmetric(
    c: tuple[tuple[tuple[int, ...], ...], ...], n: int, p: int
) -> None:
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if c[i][j][k] % p != (-c[j][i][k]) % p:
                    raise _validation_error(
                        "antisymmetric",
                        "structure constants must define an antisymmetric "
                        "bracket: [e_i, e_j] = -[e_j, e_i]",
                    )


def _require_jacobi(c: tuple[tuple[tuple[int, ...], ...], ...], n: int, p: int) -> None:
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for component in range(n):
                    jacobi_sum = sum(
                        c[i][j][m] * c[m][k][component]
                        + c[j][k][m] * c[m][i][component]
                        + c[k][i][m] * c[m][j][component]
                        for m in range(n)
                    )
                    if jacobi_sum % p != 0:
                        raise _validation_error(
                            "jacobi",
                            "structure constants must satisfy the Jacobi identity",
                        )


class LieAlgebra(StrictModel):
    """A finite-dimensional Lie algebra over a prime field GF(p).

    The Lie bracket is specified by structure constants: for basis
    elements e_i, e_j, the bracket [e_i, e_j] = sum_k c_{ij}^k * e_k.
    Every coefficient must be a canonical GF(p) residue (0 <= c < p), so one
    GF(p) Lie algebra has exactly one serialized source value; the tensor
    must also define a genuine Lie bracket: it is alternating,
    antisymmetric modulo p, and satisfies the Jacobi identity exactly;
    all are established at this request boundary because the
    Chevalley-Eilenberg differential squares to zero only for such
    brackets.
    """

    prime: int = Field(ge=2, le=MAX_LIE_ALGEBRA_PRIME)
    dimension: int = Field(ge=1, le=MAX_LIE_ALGEBRA_DIMENSION)
    structure_constants: tuple[tuple[tuple[int, ...], ...], ...] = Field(
        min_length=1, max_length=MAX_LIE_ALGEBRA_DIMENSION
    )

    @model_validator(mode="after")
    def require_valid_structure(self) -> Self:
        n = self.dimension
        if len(self.structure_constants) != n:
            raise _validation_error(
                "structure_shape",
                "structure_constants must be dimension x dimension x dimension",
            )
        for i in range(n):
            if len(self.structure_constants[i]) != n:
                raise _validation_error(
                    "structure_shape",
                    "each structure_constants[i] must have dimension rows",
                )
        for i in range(n):
            for j in range(n):
                if len(self.structure_constants[i][j]) != n:
                    raise _validation_error(
                        "structure_shape",
                        "structure constant entry must have dimension components",
                    )

        p = self.prime
        c = self.structure_constants
        _require_prime(p)
        # Canonical residues first: identities hold only modulo p, so an
        # unreduced representative (e.g. 2 over GF(2)) would otherwise define
        # the same GF(p) algebra while serializing a distinct source value,
        # and both result types would retain that noncanonical coefficient.
        _require_canonical_residues(c, n, p)
        _require_alternating(c, n, p)
        _require_antisymmetric(c, n, p)
        _require_jacobi(c, n, p)
        return self


class ChevalleyEilenbergComplexRequest(StrictModel):
    """Compute the Chevalley-Eilenberg chain complex for a Lie algebra with trivial coefficients."""

    lie_algebra: LieAlgebra


class DifferentialMatrix(StrictModel):
    """One differential matrix in the Chevalley-Eilenberg complex.

    The differential is retained as the canonical prime-field matrix value so
    it serializes with the result and composes unchanged with GF(p) matrix
    operations.

    Exterior-basis axes: with ``n`` the retained Lie-algebra dimension and
    ``p`` the differential's ``degree``, row ``i`` is the lexicographically
    ``i``-th element of ``combinations(range(n), p - 1)`` (target wedge basis
    of Lambda^{p-1}) and column ``j`` is the lexicographically ``j``-th
    element of ``combinations(range(n), p)`` (source wedge basis of
    Lambda^p). Both axes are therefore reconstructible from the retained
    algebra dimension and this degree.
    """

    degree: int = Field(ge=0)
    matrix: PrimeFieldMatrix


class ChevalleyEilenbergComplexResult(StrictModel):
    """The Chevalley-Eilenberg chain complex with trivial coefficients.

    The result retains its defining ``LieAlgebra`` and every authoritative
    claim (binomial chain-group dimensions, complete degree coverage, matrix
    shapes, GF(p) residues, and the exact differentials of the bracket) is
    replayed against that source at validation, so a malformed or relayed
    complex cannot validate.
    """

    lie_algebra: LieAlgebra
    dimension: int = Field(ge=1)
    group_dimensions: tuple[int, ...] = Field(min_length=1)
    differentials: tuple[DifferentialMatrix, ...]
    prime: int = Field(ge=2, le=MAX_LIE_ALGEBRA_PRIME)

    @model_validator(mode="after")
    def bind_complex_to_lie_algebra(self) -> Self:
        from math import comb

        n = self.lie_algebra.dimension
        p = self.lie_algebra.prime
        if self.prime != p:
            raise _validation_error(
                "complex_prime_mismatch", "prime must match the source Lie algebra"
            )
        if self.dimension != n:
            raise _validation_error(
                "complex_dimension_mismatch",
                "dimension must match the source Lie algebra",
            )
        expected_dims = tuple(comb(n, k) for k in range(n + 1))
        if self.group_dimensions != expected_dims:
            raise _validation_error(
                "complex_group_dimensions",
                "group_dimensions must be the binomial sequence of the "
                "source Lie algebra dimension",
            )
        degrees = [differential.degree for differential in self.differentials]
        if degrees != list(range(1, n + 1)):
            raise _validation_error(
                "complex_degrees",
                "the complete complex must carry one differential for each "
                f"degree 1..{n} in order",
            )
        for differential in self.differentials:
            if (
                differential.matrix.columns != expected_dims[differential.degree]
                or len(differential.matrix.entries)
                != expected_dims[differential.degree - 1]
            ):
                raise _validation_error(
                    "complex_differential_shape",
                    "differential dimensions must match the chain groups of "
                    "the source Lie algebra",
                )
            # Canonical GF(prime) residues are enforced by the retained
            # PrimeFieldMatrix value itself.
        from jacobian.math.lie_algebra_homology._operations import (
            _ce_differentials,
        )

        if tuple(self.differentials) != _ce_differentials(self.lie_algebra):
            raise _validation_error(
                "complex_replay",
                "differentials must be the exact Chevalley-Eilenberg complex "
                "reconstructed from the retained Lie algebra bracket",
            )
        return self


class LieHomologyRequest(StrictModel):
    """Compute Lie algebra homology with trivial coefficients."""

    lie_algebra: LieAlgebra


class LieHomologyGroup(StrictModel):
    """One Lie homology group.

    ``betti`` is dim(H_k); ``chain_dimension`` is the dimension of the
    chain group C_k feeding it, which the binomial sequence also carries.
    """

    degree: int = Field(ge=0)
    betti: int = Field(ge=0)
    chain_dimension: int = Field(ge=1)


class LieHomologyResult(StrictModel):
    """Lie algebra homology groups with trivial coefficients."""

    lie_algebra: LieAlgebra
    groups: tuple[LieHomologyGroup, ...] = Field(min_length=1)
    dimension: int = Field(ge=1)
    prime: int = Field(ge=2, le=MAX_LIE_ALGEBRA_PRIME)

    @model_validator(mode="after")
    def bind_to_source_lie_algebra(self) -> Self:
        from jacobian.math.lie_algebra_homology._operations import (
            lie_homology_groups,
        )

        if len(self.groups) != self.dimension + 1:
            raise _validation_error(
                "homology_group_coverage",
                "homology groups must cover degrees 0..dimension",
            )
        if (
            self.dimension != self.lie_algebra.dimension
            or self.prime != self.lie_algebra.prime
        ):
            raise _validation_error(
                "homology_source_mismatch",
                "dimension and prime must match the retained Lie algebra",
            )
        expected = lie_homology_groups(self.lie_algebra)
        if self.groups != expected:
            raise _validation_error(
                "homology_replay",
                "groups must equal the exact Lie-homology replay of the "
                "retained Lie algebra",
            )
        return self


__all__ = [
    "MAX_LIE_ALGEBRA_DIMENSION",
    "MAX_LIE_ALGEBRA_PRIME",
    "ChevalleyEilenbergComplexRequest",
    "ChevalleyEilenbergComplexResult",
    "DifferentialMatrix",
    "LieAlgebra",
    "LieHomologyGroup",
    "LieHomologyRequest",
    "LieHomologyResult",
]
