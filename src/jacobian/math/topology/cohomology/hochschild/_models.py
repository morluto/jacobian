"""Typed wire contracts for Hochschild complex operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.matrices.finite_fields.linear_algebra import (
    PrimeFieldMatrix,
)

MAX_HOCHSCHILD_DEGREE = 4
MAX_PRIME = 10_000
MAX_HOCHSCHILD_TENSOR_ELEMENTS = 20_000
# A dense boundary matrix d_k holds n^(k-1) * n^k entries; Gaussian elimination
# copies it and performs O(pivots * entries) field work. The entry budget keeps
# both the matrix and its elimination copy small alongside the tensor budget.
MAX_HOCHSCHILD_MATRIX_ENTRIES = 131_072
# Algebra admission derives every bound from measured work instead of a fixed
# dimension ceiling:
#
# - Structure input: the multiplication table carries dimension^3 constant
#   entries plus ``dimension`` augmentation scalars -- a dense GF(p)-entry
#   payload of the same size class as the densest admitted max_degree=1
#   homology boundary d_2 : A^tensor2 -> A, which has exactly
#   dimension x dimension^2 = dimension^3 entries. The table therefore shares
#   MAX_HOCHSCHILD_MATRIX_ENTRIES (dimension <= 50).
# - Associativity admission: the defining walk visits all dimension^4 basis
#   quadruples and evaluates two dimension-term dot products per quadruple,
#   i.e. 2*dimension^5 modular multiply-adds (~38M steps/second measured).
#   MAX_ASSOCIATIVITY_DOT_STEPS bounds that admission work directly
#   (2*n^5 <= 20M admits n <= 25 at sub-second cost). Larger dimensions are a
#   backend gap -- admitting them requires an accelerated associativity
#   kernel, not a change of mathematical domain.
# - Tensor elements, boundary-matrix entries, matrix dimensions, and result
#   sizes stay bounded per request by the owner admission operation and the
#   request/result structural validators below, which bound each
#   (dimension, max_degree) combination separately.
MAX_STRUCTURE_CONSTANT_ENTRIES = MAX_HOCHSCHILD_MATRIX_ENTRIES
MAX_ASSOCIATIVITY_DOT_STEPS = 20_000_000


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


class AlgebraStructure(StrictModel):
    """A finite-dimensional associative algebra over GF(p) with an augmentation.

    The multiplication is specified by structure constants:
    e_i * e_j = sum_k c_{ij}^k * e_k. The augmentation epsilon maps each basis
    element to a GF(p) scalar and must be an algebra homomorphism,
    ``epsilon(e_i e_j) == epsilon(e_i) * epsilon(e_j) mod p``; it defines the
    trivial coefficient module K on which A acts through epsilon.
    """

    prime: int = Field(ge=2, le=MAX_PRIME)
    dimension: int = Field(ge=1)
    structure_constants: tuple[tuple[tuple[int, ...], ...], ...] = Field(min_length=1)
    augmentation: tuple[int, ...] = Field(min_length=1)

    def _require_canonical_residues(self) -> None:
        """Reject noncanonical constants and augmentation values: each must
        already lie in 0..p-1 to avoid implicit field coercion."""
        prime = self.prime
        dimension = self.dimension
        for i in range(dimension):
            for j in range(dimension):
                if any(not 0 <= c < prime for c in self.structure_constants[i][j]):
                    raise _validation_error(
                        "hochschild_complex.canonical_residues",
                        "structure constants must be canonical residues in 0..p-1",
                    )
        if any(not 0 <= value < prime for value in self.augmentation):
            raise _validation_error(
                "hochschild_complex.canonical_residues",
                "augmentation values must be canonical residues in 0..p-1",
            )

    def _require_multiplicative_augmentation(self) -> None:
        """The trivial module via epsilon is an A-bimodule only when epsilon is
        multiplicative; this is exactly what makes the Hochschild differential
        square to zero."""
        prime = self.prime
        epsilon = self.augmentation
        constants = self.structure_constants
        dimension = self.dimension
        for i in range(dimension):
            for j in range(dimension):
                product_epsilon = (
                    sum(
                        coefficient * epsilon[coefficient_index]
                        for coefficient_index, coefficient in enumerate(constants[i][j])
                    )
                    % prime
                )
                if product_epsilon != (epsilon[i] * epsilon[j]) % prime:
                    raise _validation_error(
                        "hochschild_complex.augmentation_homomorphism",
                        "augmentation must be an algebra homomorphism: "
                        "epsilon(e_i e_j) must equal epsilon(e_i)*epsilon(e_j) mod p",
                    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.structure_constants) != self.dimension:
            raise _validation_error(
                "hochschild_complex.structure_shape",
                "structure_constants must be dimension x dimension",
            )
        for row in self.structure_constants:
            if len(row) != self.dimension:
                raise _validation_error(
                    "hochschild_complex.structure_shape",
                    "structure_constants must be square",
                )
            for v in row:
                if len(v) != self.dimension:
                    raise _validation_error(
                        "hochschild_complex.structure_shape",
                        "structure_constants must be 3D",
                    )
        if len(self.augmentation) != self.dimension:
            raise _validation_error(
                "hochschild_complex.augmentation_shape",
                "augmentation must have one entry per basis element",
            )
        from sympy import isprime

        if not isprime(self.prime):
            raise _validation_error(
                "hochschild_complex.prime", "prime must be a prime integer"
            )
        self._require_canonical_residues()
        return self

    def _require_associative(self) -> None:
        """Hochschild differentials square to zero only over associative algebras.

        The defining walk costs 2*dimension^5 modular multiply-adds over all
        basis quadruples; require_valid gates it with MAX_ASSOCIATIVITY_DOT_STEPS.
        """
        p = self.prime
        c = self.structure_constants
        n = self.dimension
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for ell in range(n):
                        left = sum(c[i][j][t] * c[t][k][ell] for t in range(n)) % p
                        right = sum(c[j][k][t] * c[i][t][ell] for t in range(n)) % p
                        if left != right:
                            raise _validation_error(
                                "hochschild_complex.associativity",
                                "multiplication must be associative modulo p",
                            )


class HochschildChainComplexRequest(StrictModel):
    """Compute the Hochschild chain complex with trivial coefficients.

    The coefficient module is ``K = GF(p)`` on which A acts through the
    retained augmentation epsilon. The chain groups are C_n = A^tensor-n and
    the differential is the full Hochschild boundary: interior adjacent
    multiplications plus the two augmentation-dependent endpoint faces
    (epsilon(a_1) a_2 ox ... and (-1)^n epsilon(a_n) a_1 ox ...). It squares
    to zero because epsilon is an algebra homomorphism and the multiplication
    is associative. The operation computes exact HH(A, K) chains; no cyclic
    wraparound beyond the Hochschild endpoint face is applied.
    """

    algebra: AlgebraStructure
    max_degree: int = Field(ge=1, le=MAX_HOCHSCHILD_DEGREE)


class HochschildDifferential(StrictModel):
    """One Hochschild differential as the canonical prime-field matrix value.

    ``matrix`` is the domain-owned ``PrimeFieldMatrix`` carrying its source
    prime, entries, and declared column axis, so a serialized boundary feeds
    the GF(p) rank/RREF/nullspace consumers unchanged; ``degree`` stays
    separate chain-complex metadata.
    """

    degree: int = Field(ge=1)
    matrix: PrimeFieldMatrix

    @property
    def source_dim(self) -> int:
        return self.matrix.columns

    @property
    def target_dim(self) -> int:
        return len(self.matrix.entries)


class HochschildChainComplexResult(StrictModel):
    """A structurally coherent Hochschild chain-complex prefix.

    Kernel-produced boundaries use :meth:`_from_kernel`. Deserializing a
    separately supplied result validates only source binding, field,
    dimensions, and the admitted envelope; the bar differentials are produced
    by the owner operation.
    """

    algebra: AlgebraStructure
    algebra_dimension: int = Field(ge=1)
    group_dimensions: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_HOCHSCHILD_DEGREE + 1
    )
    differentials: tuple[HochschildDifferential, ...] = Field(
        min_length=0, max_length=MAX_HOCHSCHILD_DEGREE
    )
    prime: int = Field(ge=2, le=MAX_PRIME)

    @model_validator(mode="after")
    def require_bound_to_algebra(self) -> Self:
        algebra = self.algebra
        if self.algebra_dimension != algebra.dimension or self.prime != algebra.prime:
            raise _validation_error(
                "hochschild_complex.algebra_binding",
                "algebra_dimension and prime must match the retained algebra",
            )
        dimension = algebra.dimension
        expected_groups = tuple(
            [1] + [dimension**k for k in range(1, len(self.group_dimensions))]
        )
        if self.group_dimensions != expected_groups:
            raise _validation_error(
                "hochschild_complex.group_dimensions",
                "group_dimensions must be the bar complex dimensions "
                "C_k = A^tensor-k of the retained algebra",
            )
        if len(self.differentials) != len(self.group_dimensions) - 1:
            raise _validation_error(
                "hochschild_complex.differential_count",
                "one differential per positive degree is required",
            )
        for degree, differential in enumerate(self.differentials, start=1):
            if differential.degree != degree:
                raise _validation_error(
                    "hochschild_complex.differential_degree",
                    "differential degrees must be consecutive from 1",
                )
            # Bind each differential to the algebra's field: a matrix over a
            # different prime could pass every shape and entry comparison and
            # then silently change fields inside rank/RREF/nullspace consumers.
            if differential.matrix.prime != algebra.prime:
                raise _validation_error(
                    "hochschild_complex.differential_prime",
                    "differential matrices must carry the retained algebra prime",
                )
            if (
                differential.source_dim != dimension** degree
                or differential.target_dim != dimension ** (degree - 1)
            ):
                raise _validation_error(
                    "hochschild_complex.differential_dimensions",
                    "differential dimensions must match the retained algebra",
                )
            if dimension ** (2 * degree - 1) > MAX_HOCHSCHILD_MATRIX_ENTRIES:
                raise _validation_error(
                    "hochschild_complex.matrix_budget",
                    "differential exceeds the supported boundary-matrix entry budget",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        algebra: AlgebraStructure,
        group_dimensions: tuple[int, ...],
        differentials: tuple[HochschildDifferential, ...],
    ) -> Self:
        """Construct a result emitted by the owner-local kernel."""

        return cls.model_construct(
            algebra=algebra,
            algebra_dimension=algebra.dimension,
            group_dimensions=group_dimensions,
            differentials=differentials,
            prime=algebra.prime,
        )


class HochschildHomologyRequest(StrictModel):
    """Compute exact Hochschild homology HH_n(A, K) with trivial coefficients.

    K = GF(p) carries the trivial A-bimodule structure defined by the retained
    augmentation epsilon; the operation computes the full Hochschild boundary
    (interior multiplications plus both augmentation endpoint faces) and
    returns the exact Betti numbers of the retained augmented algebra.
    """

    algebra: AlgebraStructure
    max_degree: int = Field(ge=1, le=MAX_HOCHSCHILD_DEGREE)


class HochschildHomologyGroup(StrictModel):
    """One Hochschild homology group."""

    degree: int = Field(ge=0)
    betti: int = Field(ge=0)


class HochschildHomologyResult(StrictModel):
    """Structurally bounded Hochschild homology groups with trivial coefficients.

    The exact rank computation is performed by the owner operation, never as a
    result validation side effect.
    """

    algebra: AlgebraStructure
    max_degree: int = Field(ge=1, le=MAX_HOCHSCHILD_DEGREE)
    groups: tuple[HochschildHomologyGroup, ...] = Field(min_length=1)
    prime: int = Field(ge=2, le=MAX_PRIME)

    @model_validator(mode="after")
    def bind_to_source_algebra(self) -> Self:
        if self.prime != self.algebra.prime:
            raise _validation_error(
                "hochschild_complex.prime_binding",
                "prime must match the retained algebra",
            )
        if len(self.groups) != self.max_degree + 1 or tuple(
            group.degree for group in self.groups
        ) != tuple(range(self.max_degree + 1)):
            raise _validation_error(
                "hochschild_complex.group_degrees",
                "groups must cover degrees 0..max_degree exactly once in order",
            )
        for group in self.groups:
            if group.betti > self.algebra.dimension**group.degree:
                raise _validation_error(
                    "hochschild_complex.betti_bound",
                    "betti cannot exceed its Hochschild chain-group dimension",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        algebra: AlgebraStructure,
        max_degree: int,
        groups: tuple[HochschildHomologyGroup, ...],
    ) -> Self:
        """Construct a result emitted by the owner-local kernel."""

        return cls.model_construct(
            algebra=algebra,
            max_degree=max_degree,
            groups=groups,
            prime=algebra.prime,
        )


__all__ = [
    "AlgebraStructure",
    "HochschildChainComplexRequest",
    "HochschildChainComplexResult",
    "HochschildDifferential",
    "HochschildHomologyGroup",
    "HochschildHomologyRequest",
    "HochschildHomologyResult",
]
