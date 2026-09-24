from itertools import product

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.affine_semigroups.graver import graver_basis
from jacobian.math.matrices.values import IntegerMatrix


def _matrix(entries):
    return IntegerMatrix.model_validate({"entries": [list(entries)]})


def test_known_partition_identity_graver_basis():
    result = graver_basis(_matrix((1, 2, 3)))
    assert result.vectors == (
        (0, 3, -2),
        (1, -2, 1),
        (1, 1, -1),
        (2, -1, 0),
        (3, 0, -1),
    )
    assert all(
        sum(a * z for a, z in zip((1, 2, 3), vector, strict=True)) == 0
        for vector in result.vectors
    )


def test_zero_and_unit_coefficients_have_complete_moves():
    assert graver_basis(_matrix((0, 0, 1))).vectors == (
        (0, 1, 0),
        (1, 0, 0),
    )
    assert graver_basis(_matrix((1, 1, 1))).vectors == (
        (0, 1, -1),
        (1, -1, 0),
        (1, 0, -1),
    )


def test_basis_is_covariant_under_row_sign_and_scale():
    expected = graver_basis(_matrix((2, -3, 5))).vectors
    assert graver_basis(_matrix((-2, 3, -5))).vectors == expected
    assert graver_basis(_matrix((4, -6, 10))).vectors == expected


def test_small_matrices_against_independent_finite_conformal_check():
    for entries in product((-1, 0, 1), repeat=3):
        actual = graver_basis(_matrix(entries)).vectors
        bound = 2
        complete_kernel = [
            vector
            for vector in product(range(-bound, bound + 1), repeat=3)
            if any(vector)
            and sum(a * z for a, z in zip(entries, vector, strict=True)) == 0
        ]
        minimal = []
        for vector in sorted(
            complete_kernel, key=lambda item: (sum(map(abs, item)), item)
        ):
            if any(
                all(
                    left == 0 or (left > 0) == (right > 0)
                    for left, right in zip(other, vector, strict=True)
                )
                and all(
                    abs(left) <= abs(right)
                    for left, right in zip(other, vector, strict=True)
                )
                for other in minimal
            ):
                continue
            minimal.append(vector)
        expected = tuple(
            sorted(
                {
                    vector
                    if next(value for value in vector if value) > 0
                    else tuple(-value for value in vector)
                    for vector in minimal
                }
            )
        )
        assert actual == expected


def test_exact_work_envelope_rejects_before_enumeration():
    with pytest.raises(
        OperationResourceAdmissionError, match="complete Graver enumeration"
    ):
        graver_basis(_matrix((100, 99, 98, 97, 96)))
