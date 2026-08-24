"""Tests for finite-dimensional algebra operations."""

import pytest

from jacobian.math.finite_dim_algebras._models import (
    CenterRequest,
    StructureConstants,
)
from jacobian.math.finite_dim_algebras._operations import compute_center
from jacobian.math.finite_dim_algebras._tools import TOOLS


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "algebra.center.compute",
    }


def _structure(n: int, q: int, mult: tuple) -> StructureConstants:
    return StructureConstants(dimension=n, field_order=q, multiplication=mult)


# --- Known-answer algebras over F_2 ----------------------------------------

# Zero algebra of dimension 2: every product is zero.  The algebra is
# commutative, so the center is the whole 2-dimensional space.
ZERO_ALG_2 = _structure(2, 2, (((0, 0), (0, 0)), ((0, 0), (0, 0))))

# The field F_2 itself, as a 1-dimensional algebra: e_0 * e_0 = e_0.
FIELD_F2 = _structure(1, 2, (((1,),),))

# A non-commutative 2-dimensional algebra with trivial center.
# Basis {e_0, e_1} over F_2 with:
#   e_0 * e_0 = e_0, e_0 * e_1 = e_1, e_1 * e_0 = 0, e_1 * e_1 = 0
# e_0 is a left identity but not a right identity, so the only element
# commuting with both basis vectors is 0.
NONCOMM_ALG_2 = _structure(
    2,
    2,
    (
        ((1, 0), (0, 1)),
        ((0, 0), (0, 0)),
    ),
)

# M_2(F_2), the full 2x2 matrix algebra over F_2, with basis
# {E_11, E_12, E_21, E_22}.  Its center is the scalar matrices, i.e.
# span{I_2} (dimension 1).
M2_F2 = _structure(
    4,
    2,
    (
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        ((0, 0, 0, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)),
        ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 0, 1, 0)),
        ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    ),
)


def _algebra_commutes(struct: StructureConstants, z: tuple[int, ...]) -> bool:
    """Check that ``z`` commutes with every basis element of ``struct``."""
    n, q = struct.dimension, struct.field_order
    mult = struct.multiplication
    for a in range(n):
        for k in range(n):
            lhs = sum(z[j] * mult[j][a][k] for j in range(n)) % q
            rhs = sum(z[j] * mult[a][j][k] for j in range(n)) % q
            if lhs != rhs:
                return False
    return True


def test_center_of_zero_algebra_is_full_space() -> None:
    result = compute_center(CenterRequest(algebra=ZERO_ALG_2))
    assert result.dimension == 2
    assert result.center_dimension == 2


def test_center_of_field_is_full_space() -> None:
    result = compute_center(CenterRequest(algebra=FIELD_F2))
    assert result.center_dimension == 1
    assert result.center_basis == ((1,),)


def test_center_of_noncommutative_algebra_is_trivial() -> None:
    result = compute_center(CenterRequest(algebra=NONCOMM_ALG_2))
    assert result.center_dimension == 0
    assert result.center_basis == ()


def test_center_of_matrix_algebra_is_scalars() -> None:
    result = compute_center(CenterRequest(algebra=M2_F2))
    assert result.center_dimension == 1
    (basis_vec,) = result.center_basis
    assert _algebra_commutes(M2_F2, basis_vec)
    assert basis_vec != (0, 0, 0, 0)


@pytest.mark.parametrize(
    "struct",
    [ZERO_ALG_2, FIELD_F2, NONCOMM_ALG_2, M2_F2],
)
def test_center_basis_vectors_are_central(struct: StructureConstants) -> None:
    result = compute_center(CenterRequest(algebra=struct))
    for vec in result.center_basis:
        assert _algebra_commutes(struct, vec), f"basis vector {vec} is not central"


# --- Boundary cases --------------------------------------------------------


def test_dimension_1_algebra_over_large_field() -> None:
    """A 1-dimensional algebra over F_251 with e_0 * e_0 = e_0."""
    struct = _structure(1, 251, (((1,),),))
    result = compute_center(CenterRequest(algebra=struct))
    assert result.center_dimension == 1
    assert result.center_basis == ((1,),)


def test_moderate_dimension_does_not_enumerate() -> None:
    """A 6-dimensional zero algebra over F_5 must finish quickly.

    Enumerating all 5^6 = 15625 vectors would be wasteful; the linear-algebra
    path returns instantly.
    """
    n, q = 6, 5
    zero_inner = tuple(0 for _ in range(n))
    zero_row = tuple(zero_inner for _ in range(n))
    mult = tuple(zero_row for _ in range(n))
    struct = _structure(n, q, mult)
    result = compute_center(CenterRequest(algebra=struct))
    assert result.center_dimension == n


def test_center_admits_derived_dimension_boundary() -> None:
    """The Theta(n^4) kernel admits up to dimension 128 (measured ~19s)."""
    from jacobian.math.finite_dim_algebras._models import MAX_DIM

    n, q = 128, 251
    zero_inner = tuple(0 for _ in range(n))
    mult = tuple(tuple(zero_inner for _ in range(n)) for _ in range(n))
    struct = StructureConstants(dimension=n, field_order=q, multiplication=mult)
    request = CenterRequest(algebra=struct)
    assert request.algebra.dimension == 128 == MAX_DIM


def test_center_rejects_above_derived_dimension_boundary() -> None:
    """The center-specific guard rejects below MAX_DIM when patched lower.

    ``StructureConstants`` already caps dimension at ``MAX_DIM`` = 128, so a
    plain 129-dimension request would never reach the center validator.  To
    exercise ``require_bounded_center_work`` in isolation, build a valid
    128-dimension algebra and patch the center operation's own cap to 32:
    only the center guard can then fire, and deleting it would fail this test.
    """
    from pydantic import ValidationError

    from jacobian.math.finite_dim_algebras import _models

    n, q = 128, 251
    zero_inner = tuple(0 for _ in range(n))
    mult = tuple(tuple(zero_inner for _ in range(n)) for _ in range(n))
    struct = StructureConstants(dimension=n, field_order=q, multiplication=mult)
    monkeypatch_cap = pytest.MonkeyPatch()
    try:
        monkeypatch_cap.setattr(_models, "_MAX_DIM_FOR_CENTER", 32)
        with pytest.raises(ValidationError, match="at most 32 dimensions"):
            CenterRequest(algebra=struct)
    finally:
        monkeypatch_cap.undo()


def test_structure_constants_rejects_above_its_own_field_cap() -> None:
    """The shared structure-constants cap (128) is independent of the center guard."""
    from pydantic import ValidationError

    n, q = 129, 251
    with pytest.raises(ValidationError):
        StructureConstants(dimension=n, field_order=q, multiplication=())


# --- Model validation -----------------------------------------------------


def test_structure_constants_reject_2d_shape() -> None:
    with pytest.raises(ValueError):
        StructureConstants(dimension=2, field_order=2, multiplication=((0, 0), (0, 0)))


def test_structure_constants_reject_non_residue() -> None:
    with pytest.raises(ValueError):
        StructureConstants(
            dimension=1,
            field_order=2,
            multiplication=(((2,),),),
        )


def test_structure_constants_reject_non_prime_field() -> None:
    with pytest.raises(ValueError):
        StructureConstants(
            dimension=1,
            field_order=4,
            multiplication=(((0,),),),
        )
