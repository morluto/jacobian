"""General coordinate-permutation equivalence fixtures (#3721)."""

from __future__ import annotations

from itertools import permutations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.codes.linear._canonicalization import (
    LinearCodeCanonicalizationRequest,
    canonicalize_linear_code,
)
from jacobian.math.combinatorics.codes.linear._models import (
    DualCodeRequest,
    GeneratorMatrixRequest,
)
from jacobian.math.combinatorics.codes.linear._tools import (
    compute_dual_code,
    compute_from_generator,
)
from jacobian.math.combinatorics.codes.linear.values import PrimeFieldLinearEncoder


def _encoder() -> PrimeFieldLinearEncoder:
    return PrimeFieldLinearEncoder(
        field_order=2,
        message_axis=("m",),
        coordinate_axis=("x", "y", "z"),
        generator_matrix=((1, 0, 1),),
    )


def _relabel(
    encoder: PrimeFieldLinearEncoder, perm: tuple[int, ...]
) -> PrimeFieldLinearEncoder:
    return PrimeFieldLinearEncoder(
        field_order=encoder.field_order,
        message_axis=encoder.message_axis,
        coordinate_axis=tuple(encoder.coordinate_axis[i] for i in perm),
        generator_matrix=tuple(
            tuple(row[i] for i in perm) for row in encoder.generator_matrix
        ),
    )


def _independent_rref(
    matrix: tuple[tuple[int, ...], ...], prime: int
) -> tuple[tuple[int, ...], ...]:
    """Plain-Python mod-p RREF, independent of the FLINT-backed carrier."""

    rows = [list(row) for row in matrix]
    pivots: list[int] = []
    row = 0
    for column in range(len(rows[0])):
        pivot = next(
            (r for r in range(row, len(rows)) if rows[r][column] % prime), None
        )
        if pivot is None:
            continue
        rows[row], rows[pivot] = rows[pivot], rows[row]
        inverse = pow(rows[row][column], -1, prime)
        rows[row] = [(value * inverse) % prime for value in rows[row]]
        for r in range(len(rows)):
            if r != row and rows[r][column] % prime:
                factor = rows[r][column]
                rows[r] = [
                    (a - factor * b) % prime
                    for a, b in zip(rows[r], rows[row], strict=True)
                ]
        pivots.append(column)
        row += 1
    return tuple(tuple(r) for r in rows)


def _canonical_key(result) -> tuple:
    encoder = result.canonical_encoder
    return (encoder.field_order, len(encoder.coordinate_axis), encoder.generator_matrix)


def test_relabelled_code_shares_canonical_representative() -> None:
    """Permutation-equivalent codes agree exactly on canonical representatives."""
    source = _encoder()
    relabelled = _relabel(source, (2, 0, 1))
    left = canonicalize_linear_code(LinearCodeCanonicalizationRequest(encoder=source))
    right = canonicalize_linear_code(
        LinearCodeCanonicalizationRequest(encoder=relabelled)
    )
    assert _canonical_key(left) == _canonical_key(right)


def test_different_bases_of_one_row_space_share_canonical() -> None:
    """Equal row spaces with different bases canonicalize identically."""
    first = PrimeFieldLinearEncoder(
        field_order=2,
        message_axis=("m0", "m1"),
        coordinate_axis=("x", "y", "z"),
        generator_matrix=((1, 0, 1), (0, 1, 1)),
    )
    # Add the first row into the second: same row space, different basis.
    second = PrimeFieldLinearEncoder(
        field_order=2,
        message_axis=("m0", "m1"),
        coordinate_axis=("x", "y", "z"),
        generator_matrix=((1, 0, 1), (1, 1, 0)),
    )
    left = canonicalize_linear_code(LinearCodeCanonicalizationRequest(encoder=first))
    right = canonicalize_linear_code(LinearCodeCanonicalizationRequest(encoder=second))
    assert _canonical_key(left) == _canonical_key(right)


def test_inequivalent_same_parameter_codes_differ() -> None:
    """Same [n, k] does not imply permutation equivalence."""
    repetition = PrimeFieldLinearEncoder(
        field_order=2,
        message_axis=("m",),
        coordinate_axis=("x", "y", "z"),
        generator_matrix=((1, 1, 1),),
    )
    single_parity = _encoder()
    left = canonicalize_linear_code(
        LinearCodeCanonicalizationRequest(encoder=repetition)
    )
    right = canonicalize_linear_code(
        LinearCodeCanonicalizationRequest(encoder=single_parity)
    )
    assert _canonical_key(left) != _canonical_key(right)


def test_repeated_coordinate_columns() -> None:
    """Duplicate columns keep canonicalization exact with a smaller orbit."""
    encoder = PrimeFieldLinearEncoder(
        field_order=2,
        message_axis=("m",),
        coordinate_axis=("x", "y", "z"),
        generator_matrix=((1, 1, 0),),
    )
    result = canonicalize_linear_code(
        LinearCodeCanonicalizationRequest(encoder=encoder)
    )
    assert result.orbit_size == 3
    assert result.orbit_size * result.stabilizer_size == 6
    replayed = compute_from_generator(
        GeneratorMatrixRequest(
            field_order=encoder.field_order,
            generator_matrix=tuple(
                tuple(row[i] for i in result.transporter)
                for row in encoder.generator_matrix
            ),
            coordinate_axis=result.transported_axis,
        )
    )
    assert replayed.encoder == result.canonical_encoder


def test_dual_code_covariance() -> None:
    """Duality commutes with coordinate relabelling on canonical forms."""
    source = PrimeFieldLinearEncoder(
        field_order=2,
        message_axis=("m0", "m1"),
        coordinate_axis=("x", "y", "z"),
        generator_matrix=((1, 0, 1), (0, 1, 1)),
    )
    relabelled = _relabel(source, (1, 2, 0))
    dual_left = compute_dual_code(DualCodeRequest(encoder=source)).encoder
    dual_right = compute_dual_code(DualCodeRequest(encoder=relabelled)).encoder
    canonical_left = canonicalize_linear_code(
        LinearCodeCanonicalizationRequest(encoder=dual_left)
    )
    canonical_right = canonicalize_linear_code(
        LinearCodeCanonicalizationRequest(encoder=dual_right)
    )
    assert _canonical_key(canonical_left) == _canonical_key(canonical_right)
    assert len(canonical_left.canonical_encoder.generator_matrix) == 1


def test_brute_force_orbit_matches_independent_rref_oracle() -> None:
    """A test-side mod-p RREF replays the full S_3 orbit independently."""
    source = _encoder()
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for perm in permutations(range(3)):
        permuted = tuple(tuple(row[i] for i in perm) for row in source.generator_matrix)
        seen.add(_independent_rref(permuted, source.field_order))
    result = canonicalize_linear_code(LinearCodeCanonicalizationRequest(encoder=source))
    assert result.canonical_encoder.generator_matrix == min(seen)
    assert result.orbit_size == len(seen)


def test_exhaustion_is_operational_never_inequivalence() -> None:
    """An over-budget action raises; it never reports inequivalence."""
    source = PrimeFieldLinearEncoder(
        field_order=2,
        message_axis=("m",),
        coordinate_axis=tuple(f"x{index}" for index in range(10)),
        generator_matrix=((1, 0, 0, 0, 0, 0, 0, 0, 0, 0),),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        canonicalize_linear_code(LinearCodeCanonicalizationRequest(encoder=source))
    assert "factorial" in error.value.errors()[0]["type"]
