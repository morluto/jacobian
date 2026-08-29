"""Tests for finite-dimensional algebra operations."""

import pytest

from jacobian.math.finite_dim_algebras import center_basis
from jacobian.math.finite_dim_algebras._models import (
    MAX_DIM,
    MAX_STRUCTURE_CONSTANT_ENTRIES,
    CenterRequest,
    StructureConstants,
)
from jacobian.math.finite_dim_algebras._tools import TOOLS, compute_center


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "algebra.center.compute",
    }


def _structure(
    n: int, q: int, mult: tuple[tuple[tuple[int, ...], ...], ...]
) -> StructureConstants:
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
    assert center_basis(ZERO_ALG_2) == result.center_basis


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


def test_flint_center_executes_above_previous_dimension_limit() -> None:
    """FLINT computes the exact center beyond the former dimension-128 cap."""
    n, q = 129, 251
    zero_inner = (0,) * n
    zero_row = (zero_inner,) * n
    mult = (zero_row,) * n
    struct = StructureConstants(dimension=n, field_order=q, multiplication=mult)
    result = compute_center(CenterRequest(algebra=struct))
    assert result.dimension == n
    assert result.center_dimension == n
    assert result.center_basis == tuple(
        tuple(int(row == column) for row in range(n)) for column in range(n)
    )


def test_dimension_limit_is_derived_from_structure_constant_budget() -> None:
    assert MAX_DIM**3 == MAX_STRUCTURE_CONSTANT_ENTRIES


def test_published_dimension_fits_canonical_request_byte_limit() -> None:
    """Worst-case valid tensors must encode under the 10 MiB transport envelope."""
    from jacobian.canonical import (
        CanonicalizationError,
        CanonicalLimits,
        encode_strict_json,
    )

    n = MAX_DIM
    residue = 250
    inner = [residue] * n
    row = [inner] * n
    payload = {
        "algebra": {
            "dimension": n,
            "field_order": 251,
            "multiplication": [row] * n,
        }
    }
    encoded = encode_strict_json(payload)
    assert len(encoded) <= CanonicalLimits().max_input_bytes

    oversized_inner = [residue] * (n + 1)
    oversized_row = [oversized_inner] * (n + 1)
    oversized = {
        "algebra": {
            "dimension": n + 1,
            "field_order": 251,
            "multiplication": [oversized_row] * (n + 1),
        }
    }
    with pytest.raises(CanonicalizationError):
        encode_strict_json(oversized)


def test_structure_constants_rejects_above_its_own_field_cap() -> None:
    """The structure tensor cannot exceed its materialization budget."""
    from pydantic import ValidationError

    n, q = MAX_DIM + 1, 251
    with pytest.raises(ValidationError):
        StructureConstants(dimension=n, field_order=q, multiplication=())


# --- Model validation -----------------------------------------------------


def test_structure_constants_reject_2d_shape() -> None:
    with pytest.raises(ValueError):
        StructureConstants.model_validate(
            {"dimension": 2, "field_order": 2, "multiplication": ((0, 0), (0, 0))}
        )


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
