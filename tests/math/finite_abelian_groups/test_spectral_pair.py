"""Exact finite-Abelian spectral-pair contract tests."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations, product
from typing import Any

import pytest
from pydantic import ValidationError
from sympy import Poly, Symbol, cyclotomic_poly
from sympy.polys.domains import ZZ

from jacobian.canonical import encode_strict_json
from jacobian.math import finite_abelian_groups as domain
from jacobian.math.finite_abelian_groups import (
    FiniteAbelianProductGroup,
    FiniteAbelianSpectralPairRequest,
    FiniteAbelianSpectralPairResult,
    FiniteAbelianSpectralPairSource,
    decide_finite_abelian_spectral_pair,
)
from jacobian.math.number_theory._finite_abelian_groups import (
    FINITE_ABELIAN_SPECTRAL_PAIR_OPERATION,
)


def _source(
    moduli: tuple[int, ...],
    points: tuple[tuple[int, ...], ...],
    frequencies: tuple[tuple[int, ...], ...],
) -> FiniteAbelianSpectralPairSource:
    return FiniteAbelianSpectralPairSource(
        group=FiniteAbelianProductGroup(moduli=moduli),
        points=points,
        frequencies=frequencies,
    )


def test_known_spectral_pair_in_z4() -> None:
    result = decide_finite_abelian_spectral_pair(
        _source((4,), ((0,), (2,)), ((0,), (1,)))
    )

    assert result.is_spectral is True
    assert result.reason == "SPECTRAL"
    assert result.source.group.exponent == 4
    assert result.first_nonorthogonal_pair is None


def test_nonorthogonal_pair_returns_first_exact_remainder() -> None:
    result = decide_finite_abelian_spectral_pair(
        _source((4,), ((0,), (1,)), ((0,), (1,)))
    )

    assert result.is_spectral is False
    assert result.reason == "NONORTHOGONAL_FREQUENCIES"
    witness = result.first_nonorthogonal_pair
    assert witness is not None
    assert witness.left_frequency == (0,)
    assert witness.right_frequency == (1,)
    # The fixed inner product uses lambda - mu. Here 0 - 1 = 3 mod 4,
    # so 1 + X^3 has remainder 1 - X modulo Phi_4 = X^2 + 1.
    assert witness.difference == (3,)
    assert witness.remainder_coefficients == ("1", "-1")


def test_failure_witness_is_first_in_canonical_pair_order() -> None:
    result = decide_finite_abelian_spectral_pair(
        _source((6,), ((0,), (1,), (2,)), ((0,), (2,), (3,)))
    )

    assert result.reason == "NONORTHOGONAL_FREQUENCIES"
    witness = result.first_nonorthogonal_pair
    assert witness is not None
    assert (witness.left_frequency, witness.right_frequency) == ((0,), (3,))
    assert witness.difference == (3,)
    assert witness.remainder_coefficients == ("1", "0")


def test_failure_remainder_reconstructs_character_polynomial_division() -> None:
    source = _source((4,), ((0,), (1,)), ((0,), (1,)))
    result = decide_finite_abelian_spectral_pair(source)
    witness = result.first_nonorthogonal_pair
    assert witness is not None
    generator = Symbol("X")
    exponent = source.group.exponent
    character_polynomial = Poly(
        sum(
            generator
            ** (
                sum(
                    (exponent // modulus) * point_coordinate * difference_coordinate
                    for point_coordinate, difference_coordinate, modulus in zip(
                        point,
                        witness.difference,
                        source.group.moduli,
                        strict=True,
                    )
                )
                % exponent
            )
            for point in source.points
        ),
        generator,
        domain=ZZ,
    )
    returned_remainder = Poly(
        sum(
            int(coefficient) * generator**degree
            for degree, coefficient in enumerate(witness.remainder_coefficients)
        ),
        generator,
        domain=ZZ,
    )
    cyclotomic = cyclotomic_poly(exponent, generator, polys=True)
    quotient, remainder = character_polynomial.div(cyclotomic, auto=False)

    assert remainder == returned_remainder
    assert character_polynomial == quotient * cyclotomic + returned_remainder
    assert returned_remainder.is_zero is False
    assert returned_remainder.degree() < len(witness.remainder_coefficients)


def test_unequal_cardinality_is_an_exact_negative_without_reduction() -> None:
    source = _source((65,), ((0,),), ((0,), (1,)))
    request = FiniteAbelianSpectralPairRequest(source=source)

    result = FINITE_ABELIAN_SPECTRAL_PAIR_OPERATION.run(request)

    assert result.is_spectral is False
    assert result.reason == "CARDINALITY_MISMATCH"
    assert result.source.group.exponent == 65
    assert result.first_nonorthogonal_pair is None


def test_singleton_pair_is_spectral_without_reduction() -> None:
    result = decide_finite_abelian_spectral_pair(
        _source((4_096,), ((3_999,),), ((1_337,),))
    )

    assert result.is_spectral is True
    assert result.reason == "SPECTRAL"
    assert result.source.group.exponent == 4_096


def test_source_normalizes_coordinates_and_row_order() -> None:
    source = _source(
        (4, 3),
        ((5, -1), (0, 0), (3, 4)),
        ((-1, 1), (4, 3), (2, -1)),
    )

    assert source.points == ((0, 0), (1, 2), (3, 1))
    assert source.frequencies == ((0, 0), (2, 2), (3, 1))


@pytest.mark.parametrize(
    ("points", "frequencies", "message"),
    [
        (((0,), (4,)), ((0,), (1,)), "point rows must be distinct"),
        (((0,), (1,)), ((0,), (4,)), "frequency rows must be distinct"),
        (((0, 0),), ((0,),), "point row must match"),
    ],
)
def test_source_rejects_noncanonical_set_degeneracies(
    points: tuple[tuple[int, ...], ...],
    frequencies: tuple[tuple[int, ...], ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _source((4,), points, frequencies)


def test_source_requires_both_sets_nonempty() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        _source((4,), (), ((0,),))


def test_product_pairing_weights_unequal_moduli() -> None:
    group = tuple(product(range(2), range(4)))

    result = decide_finite_abelian_spectral_pair(_source((2, 4), group, group))

    assert result.is_spectral is True
    assert result.character_convention == "POSITIVE_PRODUCT_DUAL_PAIRING"
    # With N=4 the first coordinate contributes 2*a_1*(lambda_1-mu_1),
    # not a_1*(lambda_1-mu_1). Omitting N/m_1 would fail this full-group basis.


@pytest.mark.parametrize("modulus", [2, 3])
def test_complete_cyclic_period_has_exact_character_orthogonality(
    modulus: int,
) -> None:
    complete_period = tuple((value,) for value in range(modulus))

    result = decide_finite_abelian_spectral_pair(
        _source((modulus,), complete_period, complete_period)
    )

    assert result.is_spectral is True


def _z2_squared_oracle(
    points: tuple[tuple[int, ...], ...],
    frequencies: tuple[tuple[int, ...], ...],
) -> bool:
    for left, right in combinations(frequencies, 2):
        character_sum = sum(
            -1
            if (((left[0] - right[0]) * point[0]) + ((left[1] - right[1]) * point[1]))
            % 2
            else 1
            for point in points
        )
        if character_sum != 0:
            return False
    return True


def test_every_equal_size_pair_in_z2_squared_matches_integer_oracle() -> None:
    group = tuple(product(range(2), repeat=2))
    for size in range(1, len(group) + 1):
        for points in combinations(group, size):
            for frequencies in combinations(group, size):
                expected = _z2_squared_oracle(points, frequencies)
                result = decide_finite_abelian_spectral_pair(
                    _source((2, 2), points, frequencies)
                )
                assert result.is_spectral is expected


def test_frontier_fixture_at_sixty_points_is_spectral() -> None:
    # Tao Zhang, arXiv:2607.15632v1, Theorem 4.1 and equations (29)-(30):
    # the explicit A_+ and Lambda_+ in Z_60 x Z_12.
    f = (0, 5, 0, 7, 9, 2, 10, 4, 7, 8, 10, 10)
    subgroup = {((4 * index) % 12, (2 * index) % 12) for index in range(6)}
    v2 = subgroup | {((1 + x) % 12, (2 + y) % 12) for x, y in subgroup}
    points = {
        ((36 * layer) % 60, coordinate)
        for layer in (0, 1, 2)
        for coordinate in range(12)
    }
    points |= {((25 * coordinate + 24) % 60, coordinate) for coordinate in range(12)}
    points |= {((25 * x + 48) % 60, y) for x, y in v2}
    frequencies = {
        ((5 * f[y] + 12 * residue) % 60, y) for y in range(12) for residue in range(5)
    }
    assert len(points) == len(frequencies) == 60

    source = _source((60, 12), tuple(points), tuple(frequencies))
    result = decide_finite_abelian_spectral_pair(source)

    assert result.is_spectral is True
    assert result.source.group.exponent == 60
    assert domain._spectral_pair_work(source).cyclotomic_degree == 16


def test_maximum_equal_size_work_boundary_is_admitted() -> None:
    complete_group = tuple((value,) for value in range(64))
    source = _source((64,), complete_group, complete_group)

    request = FiniteAbelianSpectralPairRequest(source=source)
    work = domain._spectral_pair_work(source)
    result = decide_finite_abelian_spectral_pair(request.source)

    assert work.character_terms == domain.MAX_SPECTRAL_CHARACTER_TERMS
    assert work.cyclotomic_reductions == domain.MAX_SPECTRAL_CYCLOTOMIC_REDUCTIONS
    assert work.cyclotomic_degree == 32
    assert work.cyclotomic_degree <= domain.MAX_SPECTRAL_CYCLOTOMIC_DEGREE
    assert work.cyclotomic_dense_ops == 295_750
    assert work.cyclotomic_dense_ops <= domain.MAX_SPECTRAL_CYCLOTOMIC_DENSE_OPS
    assert (
        work.cyclotomic_coefficient_bits
        <= domain.MAX_SPECTRAL_CYCLOTOMIC_COEFFICIENT_BITS
    )
    assert (
        work.cyclotomic_intermediate_bits
        <= domain.MAX_SPECTRAL_CYCLOTOMIC_INTERMEDIATE_BITS
    )
    assert (
        work.remainder_coefficient_bits
        <= domain.MAX_SPECTRAL_REMAINDER_COEFFICIENT_BITS
    )
    assert work.predicted_result_bytes <= domain.MAX_SPECTRAL_RESULT_BYTES
    assert len(encode_strict_json(result.model_dump(mode="json"))) < (
        domain.MAX_SPECTRAL_RESULT_BYTES
    )
    assert result.is_spectral is True


def test_exponent_immediately_above_reduction_boundary_is_rejected() -> None:
    source = _source((65,), ((0,), (1,)), ((0,), (1,)))

    with pytest.raises(ValidationError, match="exponent at most 64"):
        FiniteAbelianSpectralPairRequest(source=source)
    with pytest.raises(ValueError, match="exponent at most 64"):
        decide_finite_abelian_spectral_pair(source)


def test_cyclotomic_degree_boundary_at_prime_exponent_is_admitted() -> None:
    source = _source((61,), ((0,), (1,)), ((0,), (1,)))
    result = decide_finite_abelian_spectral_pair(source)

    assert (
        domain._spectral_pair_work(source).cyclotomic_degree
        == domain.MAX_SPECTRAL_CYCLOTOMIC_DEGREE
    )
    assert result.first_nonorthogonal_pair is not None
    assert len(result.first_nonorthogonal_pair.remainder_coefficients) == 60


def test_group_rank_and_order_boundaries() -> None:
    assert FiniteAbelianProductGroup(moduli=(2, 2, 2, 2, 2, 2)).order == 64
    assert FiniteAbelianProductGroup(moduli=(64, 64)).order == 4_096

    with pytest.raises(ValidationError, match="at most 6 items"):
        FiniteAbelianProductGroup(moduli=(2, 2, 2, 2, 2, 2, 2))
    with pytest.raises(ValidationError, match="4,096-element bound"):
        FiniteAbelianProductGroup(moduli=(4_096, 2))


def test_set_size_and_coordinate_boundaries() -> None:
    maximum_rows = tuple((value,) for value in range(64))
    source = _source((64,), maximum_rows, ((1_000_000,),))
    assert source.frequencies == ((0,),)

    with pytest.raises(ValidationError, match="at most 64 items"):
        _source((65,), tuple((value,) for value in range(65)), ((0,),))
    with pytest.raises(ValidationError, match="less than or equal to 1000000"):
        _source((2,), ((1_000_001,),), ((0,),))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(is_spectral=True),
        lambda payload: payload.update(reason="SPECTRAL"),
        lambda payload: payload["first_nonorthogonal_pair"].update(
            remainder_coefficients=["1", "1"]
        ),
        lambda payload: payload["source"].update(points=[[0], [2]]),
    ],
)
def test_result_replay_rejects_conclusion_witness_and_source_mutations(
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    result = decide_finite_abelian_spectral_pair(
        _source((4,), ((0,), (1,)), ((0,), (1,)))
    )
    payload = result.model_dump(mode="json")
    mutation(payload)

    with pytest.raises(ValidationError, match="replayed exact decision"):
        FiniteAbelianSpectralPairResult.model_validate(payload)


def test_canonical_source_round_trips_unchanged_through_catalog_request() -> None:
    source = _source((4,), ((6,), (0,)), ((5,), (0,)))
    source_payload = source.model_dump(mode="json")
    request = FINITE_ABELIAN_SPECTRAL_PAIR_OPERATION.request_type.model_validate(
        {"source": source_payload}
    )

    result = FINITE_ABELIAN_SPECTRAL_PAIR_OPERATION.run(request)

    assert result.source.model_dump(mode="json") == source_payload
    assert result.is_spectral is True
