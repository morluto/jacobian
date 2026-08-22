"""Tests for Lie algebra homology operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.lie_algebra_homology._models import (
    ChevalleyEilenbergComplexRequest,
    LieAlgebra,
    LieHomologyRequest,
)
from jacobian.math.lie_algebra_homology._operations import (
    compute_chevalley_eilenberg_complex,
    compute_lie_homology,
)


def _sl2_gf5() -> LieAlgebra:
    """The standard three-dimensional simple Lie algebra sl(2) over GF(5).

    Basis (E, F, H) with [E, F] = H, [H, E] = 2E, [H, F] = -2F.
    """
    return LieAlgebra(
        prime=5,
        dimension=3,
        structure_constants=(
            ((0, 0, 0), (0, 0, 1), (3, 0, 0)),
            ((0, 0, 4), (0, 0, 0), (0, 2, 0)),
            ((2, 0, 0), (0, 3, 0), (0, 0, 0)),
        ),
    )


def _compose(d1, d2, prime: int) -> list[list[int]]:
    """Return the composition matrix d1 after d2 (target x source)."""
    rows, inner = len(d1), len(d1[0])
    assert inner == len(d2)
    cols = len(d2[0])
    return [
        [
            sum(d1[r][k] * d2[k][c] for k in range(inner)) % prime
            for c in range(cols)
        ]
        for r in range(rows)
    ]


class TestChevalleyEilenbergComplex:
    """Test Chevalley-Eilenberg complex computation."""

    def test_abelian_2d(self):
        """CE complex of a 2D abelian Lie algebra has zero differentials."""
        g = LieAlgebra(
            prime=2,
            dimension=2,
            structure_constants=(((0, 0), (0, 0)), ((0, 0), (0, 0))),
        )
        result = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=g)
        )
        assert result.dimension == 2
        assert result.group_dimensions == (1, 2, 1)

    def test_3d_abelian(self):
        """CE complex of a 3D abelian Lie algebra."""
        g = LieAlgebra(
            prime=5,
            dimension=3,
            structure_constants=(
                ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
                ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
                ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
            ),
        )
        result = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=g)
        )
        assert result.group_dimensions == (1, 3, 3, 1)

    def test_sl2_differentials_square_to_zero(self):
        """Consecutive CE differentials of sl(2)/GF(5) compose to exactly zero."""
        result = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=_sl2_gf5())
        )
        matrices = {
            d.degree: [list(row) for row in d.entries]
            for d in result.differentials
        }
        # d_1 is identically zero and d_3 composes against d_2.
        assert matrices[1] == [[0, 0, 0]]
        for degree in range(1, 3):
            composed = _compose(
                matrices[degree], matrices[degree + 1], result.prime
            )
            assert all(value == 0 for row in composed for value in row)

    def test_sl2_d2_matrix_entries(self):
        """The d_2 matrix of sl(2) encodes the bracket columns exactly."""
        result = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=_sl2_gf5())
        )
        d2 = next(d for d in result.differentials if d.degree == 2)
        # targets (E, F, H) x sources (EF, EH, FH); each single-pair column
        # carries the standard (-1)^(a+b+pi) CE sign:
        # d(E^F) = -H, d(E^H) = -(-2E) = 2E... encoded as exact GF(5) residues.
        assert d2.entries == ((0, 2, 0), (0, 0, 3), (4, 0, 0))


class TestLieAlgebraValidation:
    """Adversarial rejection of tensors that are not Lie brackets."""

    def test_composite_prime_rejected(self):
        with pytest.raises(ValidationError, match="prime must be a prime integer"):
            LieAlgebra(
                prime=4,
                dimension=1,
                structure_constants=(((0,),),),
            )

    def test_alternation_violation_rejected(self):
        constants = (
            ((0, 0), (1, 0)),
            ((4, 0), (0, 1)),
        )  # [e_1, e_1] = e_0 != 0
        with pytest.raises(ValidationError, match="alternating"):
            LieAlgebra(prime=5, dimension=2, structure_constants=constants)

    def test_antisymmetry_violation_rejected(self):
        constants = (
            ((0, 0), (1, 0)),
            ((1, 0), (0, 0)),
        )  # c[0][1] = (1,0) is not -c[1][0] mod 5
        with pytest.raises(ValidationError, match="antisymmetric"):
            LieAlgebra(prime=5, dimension=2, structure_constants=constants)

    def test_jacobi_violation_rejected(self):
        # [e0, e1] = e0 and [e1, e2] = e1 violate the Jacobi identity:
        # [[e0, e1], e2] + [[e1, e2], e0] + [[e2, e0], e1] = -e0 != 0.
        constants = (
            ((0, 0, 0), (1, 0, 0), (0, 0, 0)),
            ((4, 0, 0), (0, 0, 0), (0, 1, 0)),
            ((0, 0, 0), (0, 4, 0), (0, 0, 0)),
        )
        with pytest.raises(ValidationError, match="Jacobi"):
            LieAlgebra(prime=5, dimension=3, structure_constants=constants)


class TestLieHomology:
    """Test Lie algebra homology computation."""

    def test_abelian_2d(self):
        """Homology of a 2D abelian Lie algebra over GF(2)."""
        g = LieAlgebra(
            prime=2,
            dimension=2,
            structure_constants=(((0, 0), (0, 0)), ((0, 0), (0, 0))),
        )
        result = compute_lie_homology(LieHomologyRequest(lie_algebra=g))
        assert result.groups[0].betti == 1
        assert result.groups[1].betti == 2
        assert result.groups[2].betti == 1

    def test_abelian_1d(self):
        """Homology of a 1D abelian Lie algebra."""
        g = LieAlgebra(
            prime=7,
            dimension=1,
            structure_constants=(((0,),),),
        )
        result = compute_lie_homology(LieHomologyRequest(lie_algebra=g))
        assert result.groups[0].betti == 1
        assert result.groups[1].betti == 1

    def test_abelian_3d(self):
        """Homology of a 3D abelian Lie algebra over GF(5)."""
        g = LieAlgebra(
            prime=5,
            dimension=3,
            structure_constants=(
                ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
                ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
                ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
            ),
        )
        result = compute_lie_homology(LieHomologyRequest(lie_algebra=g))
        # H_0 = 1, H_1 = 3, H_2 = 3, H_3 = 1
        assert result.groups[0].betti == 1
        assert result.groups[1].betti == 3
        assert result.groups[2].betti == 3
        assert result.groups[3].betti == 1

    def test_sl2_gf5(self):
        """Homology of sl(2) over GF(5): trivial below the top degree.

        d_1 = 0, rank d_2 = 3, d_3 = 0, so the Betti numbers are
        (1, 0, 0, 1); the top class is the unimodular volume class.
        """
        result = compute_lie_homology(LieHomologyRequest(lie_algebra=_sl2_gf5()))
        assert tuple(group.betti for group in result.groups) == (1, 0, 0, 1)

    def test_affine_algebra_gf5(self):
        """Homology of the 2D non-abelian algebra [x, y] = x over GF(5)."""
        g = LieAlgebra(
            prime=5,
            dimension=2,
            structure_constants=(
                ((0, 0), (1, 0)),
                ((4, 0), (0, 0)),
            ),
        )
        result = compute_lie_homology(LieHomologyRequest(lie_algebra=g))
        # d_2: e0 ^ e1 -> e0 has rank 1; Betti numbers are (1, 1, 0).
        assert tuple(group.betti for group in result.groups) == (1, 1, 0)


def test_malformed_middle_axis_rejected():
    """structure_constants rows shorter than dimension are rejected, not indexed."""
    with pytest.raises(ValidationError):
        LieAlgebra(
            prime=5,
            dimension=2,
            structure_constants=(((0,),), ((0,),)),
        )
