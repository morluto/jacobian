from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    StructureConstant,
)
from jacobian.math.lie_algebras.basis_transport._models import LieBasisChangeResult
from jacobian.math.lie_algebras.basis_transport._tools import TOOLS
from jacobian.math.lie_algebras.basis_transport.operations import (
    lie_algebra_change_basis,
)
from jacobian.math.matrices.values import rational_matrix_from_fractions

SL2 = FiniteDimensionalLieAlgebra.model_validate(
    {
        "basis": ["e", "f", "h"],
        "structure_constants": [
            {"i": 0, "j": 1, "k": 2, "coefficient": {"num": 1, "den": 1}},
            {"i": 0, "j": 2, "k": 0, "coefficient": {"num": -2, "den": 1}},
            {"i": 1, "j": 2, "k": 1, "coefficient": {"num": 2, "den": 1}},
        ],
    }
)


def _matrix(entries: list[list[tuple[int, int]]]):
    return rational_matrix_from_fractions(
        tuple(
            tuple(Fraction(numerator, denominator) for numerator, denominator in row)
            for row in entries
        )
    )


def _multiply(left, right):
    return tuple(
        tuple(
            sum(
                (left[i][k] * right[k][j] for k in range(len(right))),
                Fraction(0),
            )
            for j in range(len(right[0]))
        )
        for i in range(len(left))
    )


def _commutator(left, right):
    def product(a, b):
        return tuple(
            tuple(
                sum((a[i][k] * b[k][j] for k in range(2)), Fraction(0))
                for j in range(2)
            )
            for i in range(2)
        )

    ab = product(left, right)
    ba = product(right, left)
    return tuple(tuple(ab[i][j] - ba[i][j] for j in range(2)) for i in range(2))


def test_rational_basis_transport_matches_matrix_commutators_and_retains_inverse_maps() -> (
    None
):
    # Columns are u=e+h, v=f/2, w=h in source coordinates.
    p = (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1, 2), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(1)),
    )
    result = lie_algebra_change_basis(
        SL2,
        ("u", "v", "w"),
        _matrix(
            [
                [(1, 1), (0, 1), (0, 1)],
                [(0, 1), (1, 2), (0, 1)],
                [(1, 1), (0, 1), (1, 1)],
            ]
        ),
    )
    assert [
        (c.i, c.j, c.k, c.coefficient.as_fraction())
        for c in result.target.structure_constants
    ] == [
        (0, 1, 1, Fraction(-2)),
        (0, 1, 2, Fraction(1, 2)),
        (0, 2, 0, Fraction(-2)),
        (0, 2, 2, Fraction(2)),
        (1, 2, 1, Fraction(2)),
    ]

    # Independent 2x2 matrix realization: source basis matrices are E,F,H.
    source_basis = (
        ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(0))),
        ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))),
        ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(-1))),
    )
    target_basis = tuple(
        tuple(
            tuple(
                sum((p[k][j] * source_basis[k][r][c] for k in range(3)), Fraction(0))
                for c in range(2)
            )
            for r in range(2)
        )
        for j in range(3)
    )
    expected_inverse = (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(2), Fraction(0)),
        (Fraction(-1), Fraction(0), Fraction(1)),
    )
    inverse = tuple(
        tuple(value.as_fraction() for value in row)
        for row in result.source_to_target.entries
    )
    assert inverse == expected_inverse
    for i in range(3):
        for j in range(i + 1, 3):
            matrix_bracket = _commutator(target_basis[i], target_basis[j])
            # Recover coordinates directly by the independent linear equations
            # for the (a,b;c,d) entries in the E,F,H matrix basis.
            e_coefficient = matrix_bracket[0][1]
            f_coefficient = matrix_bracket[1][0]
            h_coefficient = matrix_bracket[0][0]
            source_coordinates = (e_coefficient, f_coefficient, h_coefficient)
            target_coordinates = tuple(
                sum(
                    (inverse[row][k] * source_coordinates[k] for k in range(3)),
                    Fraction(0),
                )
                for row in range(3)
            )
            expected = tuple(
                sum(
                    (
                        c.coefficient.as_fraction()
                        for c in result.target.structure_constants
                        if (c.i, c.j, c.k) == (i, j, k)
                    ),
                    Fraction(0),
                )
                for k in range(3)
            )
            assert target_coordinates == expected

    forward = tuple(
        tuple(value.as_fraction() for value in row)
        for row in result.target_to_source.entries
    )
    identity = tuple(tuple(Fraction(i == j) for j in range(3)) for i in range(3))
    assert _multiply(forward, inverse) == identity
    assert _multiply(inverse, forward) == identity
    assert result.source == SL2
    assert (
        LieBasisChangeResult.model_validate_json(result.model_dump_json(), strict=True)
        == result
    )


def test_identity_basis_change_preserves_source_algebra() -> None:
    result = lie_algebra_change_basis(
        SL2,
        SL2.basis,
        _matrix(
            [
                [(1, 1), (0, 1), (0, 1)],
                [(0, 1), (1, 1), (0, 1)],
                [(0, 1), (0, 1), (1, 1)],
            ]
        ),
    )
    assert result.target == SL2
    assert result.target_to_source == result.source_to_target


def test_basis_change_revalidates_trusted_model_construct_input() -> None:
    unsafe = FiniteDimensionalLieAlgebra.model_construct(
        basis=("e", "f", "h"),
        _jacobi_admitted=True,
        structure_constants=tuple(
            StructureConstant.model_construct(
                i=i,
                j=j,
                k=k,
                coefficient=CanonicalRational.model_construct(num=value, den=1),
            )
            for i, j, k, value in (
                (0, 1, 2, 1),
                (0, 2, 0, 1),
                (1, 2, 0, 1),
            )
        ),
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        lie_algebra_change_basis(
            unsafe,
            SL2.basis,
            _matrix(
                [
                    [(1, 1), (0, 1), (0, 1)],
                    [(0, 1), (1, 1), (0, 1)],
                    [(0, 1), (0, 1), (1, 1)],
                ]
            ),
        )
    assert exc_info.value.errors()[0]["type"] == "lie_algebra.jacobi_identity"


def test_singular_basis_change_is_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="must be invertible"):
        lie_algebra_change_basis(
            SL2,
            ("u", "v", "w"),
            _matrix(
                [
                    [(1, 1), (0, 1), (0, 1)],
                    [(1, 1), (0, 1), (0, 1)],
                    [(0, 1), (0, 1), (1, 1)],
                ]
            ),
        )


def test_public_manifest_declares_basis_transport() -> None:
    tool = TOOLS[0]
    assert tool.operation_id == "lie_algebra.basis_change.compute"
    assert tool.request_type.__name__ == "LieBasisChangeRequest"
    assert tool.result_type is LieBasisChangeResult
