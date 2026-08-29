"""Domain-owned Hochschild complex operations."""

from __future__ import annotations

from collections.abc import Callable

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix
from jacobian.math.topology.cohomology.hochschild._bar import bar_differential_entries
from jacobian.math.topology.cohomology.hochschild._models import (
    MAX_HOCHSCHILD_TENSOR_ELEMENTS,
    AlgebraStructure,
    HochschildChainComplexResult,
    HochschildDifferential,
    HochschildHomologyGroup,
    HochschildHomologyResult,
)


def _admit(operation: Callable[[], None], *, location: tuple[str, ...]) -> None:
    try:
        operation()
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=location,
            code="hochschild.invalid_domain",
            message=str(exc),
        ) from exc


def _boundary_rank(
    algebra: AlgebraStructure,
    degree: int,
    element_budget: int = MAX_HOCHSCHILD_TENSOR_ELEMENTS,
) -> int:
    """GF(p)-rank of the Hochschild boundary C_degree -> C_{degree-1}.

    Uses the maintained prime-field rank backend (python-flint ``nmod_mat``)
    via a thin adapter so the operation does not duplicate the kernel that
    ``finite_fields._flint.matrix_rank`` already owns for ``PrimeFieldMatrix``.
    """
    n = algebra.dimension
    if n**degree > element_budget:
        raise ValueError("requested degree exceeds the supported tensor-element budget")
    if degree == 0:
        return 0
    entries = bar_differential_entries(
        algebra.structure_constants,
        algebra.prime,
        degree,
        algebra.augmentation,
    )
    if not entries or not entries[0]:
        return 0
    # Prefer the maintained FLINT backend; it handles the largest admitted
    # homology boundary (125x625) without Python-level modular updates and
    # without reimplementing the kernel. For matrices within the canonical
    # PrimeFieldMatrix dimension bound we reuse the existing adapter, otherwise
    # we call nmod_mat directly so homology can still be decided without
    # storing a PrimeFieldMatrix.
    try:
        from jacobian.math.finite_fields._flint import matrix_rank

        # Fast path when the boundary fits the canonical PrimeFieldMatrix
        # representation that chain-complex differentials use.
        if len(entries) <= 256 and len(entries[0]) <= 256:
            matrix = PrimeFieldMatrix(
                prime=algebra.prime,
                entries=entries,
                columns=n**degree,
            )
            return matrix_rank(matrix)
        from flint import nmod_mat

        backend = nmod_mat([list(row) for row in entries], algebra.prime)
        return int(backend.rank())
    except ImportError:
        # Fallback when python-flint is unavailable: compute rank via
        # SymPy's DomainMatrix without the PrimeFieldMatrix dimension check
        # so the largest admitted homology boundary (125x625) is still decided.
        import sympy
        from sympy.polys.matrices import DomainMatrix

        domain_matrix = DomainMatrix(
            [list(row) for row in entries],
            (len(entries), len(entries[0])),
            sympy.GF(algebra.prime),
        )
        return int(domain_matrix.rank())


def hochschild_chain_complex(
    algebra: AlgebraStructure,
    max_degree: int,
) -> HochschildChainComplexResult:
    """Compute the exact Hochschild chain complex with trivial coefficients.

    K = GF(p) carries the trivial A-bimodule structure defined by the retained
    augmentation epsilon. The differential on C_k = A^tensor-k is the full
    Hochschild boundary: interior adjacent multiplications plus both
    augmentation-dependent endpoint faces. It squares to zero because epsilon
    is an algebra homomorphism and the multiplication is associative.
    """
    from jacobian.math.topology.cohomology.hochschild._bounds import (
        require_algebra_admission,
        require_hochschild_budget,
    )

    _admit(lambda: require_algebra_admission(algebra), location=("algebra",))
    _admit(
        lambda: require_hochschild_budget(algebra.dimension, max_degree),
        location=("max_degree",),
    )
    alg = algebra
    n = alg.dimension
    p = alg.prime
    max_deg = max_degree

    group_dims = [1]
    for k in range(1, max_deg + 1):
        group_dims.append(n**k)

    differentials = []

    for degree in range(1, max_deg + 1):
        differentials.append(
            HochschildDifferential(
                degree=degree,
                matrix=PrimeFieldMatrix(
                    prime=p,
                    entries=tuple(
                        bar_differential_entries(
                            alg.structure_constants, p, degree, alg.augmentation
                        )
                    ),
                    columns=n**degree,
                ),
            )
        )

    return HochschildChainComplexResult._from_kernel(
        alg, tuple(group_dims), tuple(differentials)
    )


def hochschild_homology_groups(
    algebra: AlgebraStructure,
    max_degree: int,
) -> tuple[HochschildHomologyGroup, ...]:
    """Pure Hochschild-homology core returning the exact groups for one algebra.

    Kept free of result-model construction so it remains a reusable exact
    rank computation.
    """
    n = algebra.dimension

    group_dims = [1]
    for k in range(1, max_degree + 1):
        group_dims.append(n**k)

    ranks = [_boundary_rank(algebra, degree) for degree in range(1, max_degree + 2)]

    groups = []
    for k in range(max_degree + 1):
        rank_d_out = ranks[k]
        rank_d_in = ranks[k - 1] if k > 0 else 0
        dim = group_dims[k]
        betti = dim - rank_d_out - rank_d_in
        if betti < 0:
            raise ValueError(
                "internal inconsistency: differentials do not square to zero"
            )
        groups.append(
            HochschildHomologyGroup(
                degree=k,
                betti=betti,
            )
        )
    return tuple(groups)


def hochschild_homology(
    algebra: AlgebraStructure,
    max_degree: int,
) -> HochschildHomologyResult:
    """Compute exact Hochschild homology HH_n(A, K) for trivial coefficients."""
    from jacobian.math.topology.cohomology.hochschild._bounds import (
        require_algebra_admission,
        require_hochschild_budget,
    )

    _admit(lambda: require_algebra_admission(algebra), location=("algebra",))
    _admit(
        lambda: require_hochschild_budget(algebra.dimension, max_degree),
        location=("max_degree",),
    )
    alg = algebra

    return HochschildHomologyResult._from_kernel(
        alg,
        max_degree,
        hochschild_homology_groups(alg, max_degree),
    )


__all__ = [
    "hochschild_chain_complex",
    "hochschild_homology",
    "hochschild_homology_groups",
]
