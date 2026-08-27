"""Exact finite-Abelian spectral-pair contract tests."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations, product
from typing import Any

import pytest
from sympy import Poly, Symbol, cyclotomic_poly
from sympy.polys.domains import ZZ
from tests.math.finite_abelian_groups._support import finite_abelian_validation_error

from jacobian.canonical import encode_strict_json
from jacobian.math import finite_abelian_groups as domain
from jacobian.math.finite_abelian_groups import (
    FiniteAbelianGroupFactorizationRequest,
    FiniteAbelianProductGroup,
    FiniteAbelianSpectralPairRequest,
    FiniteAbelianSpectralPairResult,
    FiniteAbelianSpectralPairSource,
    decide_finite_abelian_spectral_pair,
    finite_abelian_group_factorization,
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
    with finite_abelian_validation_error():
        _source((4,), points, frequencies)


def test_degenerate_empty_sets_route_through_trivial_decisions() -> None:
    equal_empty = _source((12,), (), ())
    work = domain._spectral_pair_work(equal_empty)

    assert work.cyclotomic_degree is None
    assert work.character_terms == 0

    result = decide_finite_abelian_spectral_pair(equal_empty)

    assert result.is_spectral is True
    assert result.reason == "SPECTRAL"
    assert result.first_nonorthogonal_pair is None

    empty_points = decide_finite_abelian_spectral_pair(_source((12,), (), ((0,),)))

    assert empty_points.is_spectral is False
    assert empty_points.reason == "CARDINALITY_MISMATCH"

    empty_frequencies = decide_finite_abelian_spectral_pair(_source((12,), ((5,),), ()))

    assert empty_frequencies.is_spectral is False
    assert empty_frequencies.reason == "CARDINALITY_MISMATCH"


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


def test_derived_budgets_admit_exponent_above_the_former_fixed_cap() -> None:
    source = _source((65,), ((0,), (1,)), ((0,), (1,)))
    work = domain._spectral_pair_work(source)

    assert work.cyclotomic_degree == 48
    assert work.character_terms == 4
    assert work.cyclotomic_reductions == 2
    assert work.cyclotomic_dense_ops == 304_920
    assert work.cyclotomic_intermediate_bits == 138
    assert work.remainder_coefficient_bits == 835

    result = decide_finite_abelian_spectral_pair(source)

    assert result.is_spectral is False
    assert result.reason == "NONORTHOGONAL_FREQUENCIES"
    witness = result.first_nonorthogonal_pair
    assert witness is not None
    assert (witness.left_frequency, witness.right_frequency) == ((0,), (1,))
    assert witness.difference == (64,)
    assert len(witness.remainder_coefficients) == 48


def test_equal_size_pair_over_derived_dense_op_budget_is_rejected() -> None:
    source = _source((128,), ((0,), (1,)), ((0,), (1,)))

    with finite_abelian_validation_error():
        FiniteAbelianSpectralPairRequest(source=source)
    with pytest.raises(
        ValueError, match="cyclotomic construction work exceeds its dense-op bound"
    ):
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
    assert FiniteAbelianProductGroup(moduli=(4_096, 2)).order == 8_192
    assert FiniteAbelianProductGroup(moduli=(2,) * 7).order == 128

    # The reusable group value carries no axis ceiling; consuming operations
    # derive their rank envelope from serialized output size and work.
    wide_group = FiniteAbelianProductGroup(moduli=(2,) * 65)
    assert wide_group.order == 2**65
    assert wide_group.exponent == 2

    # The rank and order ceilings stay operation-specific on the exhaustive
    # factorization path, whose kernel materializes the ambient group.
    with finite_abelian_validation_error():
        FiniteAbelianGroupFactorizationRequest(
            moduli=(2,) * 7,
            left=((0,) * 7,),
            right=((0,) * 7,),
        )

    with finite_abelian_validation_error():
        FiniteAbelianGroupFactorizationRequest(
            moduli=(4_096, 2),
            left=((0, 0),),
            right=((0, 0),),
        )


def test_native_factorization_accepts_canonical_group_values() -> None:
    result = finite_abelian_group_factorization(
        FiniteAbelianProductGroup(moduli=(2, 2)),
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1)),
    )

    assert result.is_exact_factorization
    assert result.group_order == 4
    assert result.pair_count == 4


def test_singleton_pair_beyond_the_group_order_cap_is_admitted() -> None:
    result = decide_finite_abelian_spectral_pair(
        _source((4_097,), ((3_999,),), ((1_337,),))
    )

    assert result.is_spectral is True
    assert result.reason == "SPECTRAL"
    assert result.source.group.order == 4_097


def test_singleton_beyond_the_former_rank_cap_is_admitted() -> None:
    zero = (0, 0, 0, 0, 0, 0, 0)
    unit = (1, 0, 0, 0, 0, 0, 0)
    result = decide_finite_abelian_spectral_pair(_source((2,) * 7, (zero,), (unit,)))

    assert result.is_spectral is True
    assert result.reason == "SPECTRAL"
    assert result.source.group.exponent == 2


def test_equal_size_pair_beyond_the_former_rank_cap_fits_budgets() -> None:
    zero = (0, 0, 0, 0, 0, 0, 0)
    unit = (1, 0, 0, 0, 0, 0, 0)
    source = _source((2,) * 7, (zero, unit), (zero, unit))
    work = domain._spectral_pair_work(source)

    assert work.character_terms == 4
    assert work.predicted_result_bytes <= domain.MAX_SPECTRAL_RESULT_BYTES

    result = decide_finite_abelian_spectral_pair(source)

    # 1 + X modulo Phi_2 = X + 1 vanishes exactly.
    assert result.is_spectral is True
    assert result.reason == "SPECTRAL"


def test_singleton_pair_in_z2_power_65_is_decided_without_reduction() -> None:
    zero = (0,) * 65
    unit = (1,) + (0,) * 64
    source = _source((2,) * 65, (zero,), (unit,))
    work = domain._spectral_pair_work(source)

    assert work.cyclotomic_degree is None
    assert work.character_terms == 0
    assert work.predicted_result_bytes == 4_128
    assert work.predicted_result_bytes <= domain.MAX_SPECTRAL_RESULT_BYTES

    request = FiniteAbelianSpectralPairRequest(source=source)
    result = FINITE_ABELIAN_SPECTRAL_PAIR_OPERATION.run(request)

    assert result.is_spectral is True
    assert result.reason == "SPECTRAL"
    assert result.source.group.exponent == 2
    assert result.first_nonorthogonal_pair is None


def test_nonorthogonality_witness_admits_axes_beyond_the_former_rank_cap() -> None:
    zero = (0,) * 65
    shifted = (0, 1) + (0,) * 63
    points = (zero, (1,) + (0,) * 64)
    result = decide_finite_abelian_spectral_pair(
        _source((2,) * 65, points, (zero, shifted))
    )

    assert result.is_spectral is False
    assert result.reason == "NONORTHOGONAL_FREQUENCIES"
    witness = result.first_nonorthogonal_pair
    assert witness is not None
    assert witness.left_frequency == zero
    assert witness.right_frequency == shifted
    assert witness.difference == shifted
    assert witness.remainder_coefficients == ("2",)


def test_singleton_axis_envelope_boundary_is_admitted() -> None:
    # Singleton envelope: 2,568 + 24 * rank <= 32,768 serialized bytes.
    rank = 1_258
    source = _source(
        (2,) * rank,
        ((0,) * rank,),
        ((1,) + (0,) * (rank - 1),),
    )

    result = decide_finite_abelian_spectral_pair(source)

    assert result.is_spectral is True
    assert domain._spectral_pair_work(source).predicted_result_bytes == 32_760


def test_singleton_axis_envelope_boundary_is_rejected() -> None:
    rank = 1_259
    source = _source(
        (2,) * rank,
        ((0,) * rank,),
        ((1,) + (0,) * (rank - 1),),
    )

    with finite_abelian_validation_error():
        FiniteAbelianSpectralPairRequest(source=source)


def test_singleton_source_over_the_source_byte_bound_is_rejected_before_lcm() -> None:
    # At this rank the serialized source alone exceeds the byte budget, so
    # rejection precedes any exponent arithmetic on the declared moduli.
    rank = 4_000
    source = _source(
        (2,) * rank,
        ((0,) * rank,),
        ((1,) + (0,) * (rank - 1),),
    )

    with finite_abelian_validation_error():
        FiniteAbelianSpectralPairRequest(source=source)
    with pytest.raises(
        ValueError,
        match="spectral-pair result exceeds its serialized byte bound",
    ):
        decide_finite_abelian_spectral_pair(source)


def test_forged_witness_elements_beyond_source_rank_are_rejected() -> None:
    result = decide_finite_abelian_spectral_pair(
        _source((4,), ((0,), (1,)), ((0,), (1,)))
    )
    payload = result.model_dump(mode="json")
    payload["first_nonorthogonal_pair"]["left_frequency"] = [0] * 200

    with finite_abelian_validation_error():
        FiniteAbelianSpectralPairResult.model_validate(payload)


def test_singleton_beyond_the_former_modulus_ceiling_is_admitted() -> None:
    result = decide_finite_abelian_spectral_pair(
        _source((1_000_001,), ((1_000_000,),), ((5_000,),))
    )

    assert result.is_spectral is True
    assert result.reason == "SPECTRAL"


def test_modulus_bound_tracks_canonical_json_safe_integers() -> None:
    limit = domain.MAX_FINITE_GROUP_MODULUS

    assert limit == (1 << 53) - 1
    assert FiniteAbelianProductGroup(moduli=(limit,)).order == limit
    with finite_abelian_validation_error():
        FiniteAbelianProductGroup(moduli=(limit + 1,))


def test_singleton_at_modulus_bound_serializes_within_budget() -> None:
    source = _source((domain.MAX_FINITE_GROUP_MODULUS,), ((7,),), ((11,),))
    work = domain._spectral_pair_work(source)

    assert work.cyclotomic_degree is None
    assert work.predicted_result_bytes <= domain.MAX_SPECTRAL_RESULT_BYTES

    result = decide_finite_abelian_spectral_pair(source)

    assert result.is_spectral is True
    assert len(encode_strict_json(result.model_dump(mode="json"))) < (
        domain.MAX_SPECTRAL_RESULT_BYTES
    )


def test_equal_size_pair_in_group_above_order_cap_fits_derived_budgets() -> None:
    source = _source((64, 64, 2), ((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 0, 1)))
    work = domain._spectral_pair_work(source)

    assert source.group.order == 8_192
    assert work.cyclotomic_degree == 32
    assert work.cyclotomic_dense_ops == 295_750

    result = decide_finite_abelian_spectral_pair(source)

    assert result.is_spectral is False
    assert result.reason == "NONORTHOGONAL_FREQUENCIES"
    witness = result.first_nonorthogonal_pair
    assert witness is not None
    assert witness.difference == (0, 0, 1)
    assert witness.remainder_coefficients[0] == "2"


def test_coordinate_and_materialization_fallback_boundaries() -> None:
    fallback_rows = tuple((value,) for value in range(domain.MAX_SPECTRAL_SET_SIZE))
    source = _source((1_000_000,), fallback_rows, ((1_000_000,),))
    assert source.frequencies == ((0,),)

    limit = domain.MAX_FINITE_GROUP_MODULUS
    wide = _source((limit,), ((limit - 1,),), ((0,),))
    assert wide.points == ((limit - 1,),)
    assert wide.frequencies == ((0,),)

    with finite_abelian_validation_error():
        _source((64,), tuple((value,) for value in range(4_097)), ((0,),))
    with finite_abelian_validation_error():
        _source(
            (2,),
            ((domain.MAX_FINITE_GROUP_COORDINATE + 1,),),
            ((0,),),
        )


def test_cardinality_mismatch_admits_point_sets_above_the_row_fallback() -> None:
    source = _source((128,), tuple((value,) for value in range(65)), ((0,),))

    result = FINITE_ABELIAN_SPECTRAL_PAIR_OPERATION.run(
        FiniteAbelianSpectralPairRequest(source=source)
    )

    assert result.is_spectral is False
    assert result.reason == "CARDINALITY_MISMATCH"
    assert len(result.source.points) == 65
    assert result.first_nonorthogonal_pair is None


def test_equal_size_sets_over_character_term_budget_are_still_rejected() -> None:
    rows = tuple((value,) for value in range(65))
    source = _source((80,), rows, rows)

    with finite_abelian_validation_error():
        FiniteAbelianSpectralPairRequest(source=source)


def test_serialized_source_bytes_reject_oversized_mismatch_sets() -> None:
    source = _source(
        (1_000_000,),
        tuple((value,) for value in range(2_400)),
        ((999_983,),),
    )

    assert len(source.points) < domain.MAX_SPECTRAL_SET_SIZE
    with finite_abelian_validation_error():
        FiniteAbelianSpectralPairRequest(source=source)
    with pytest.raises(ValueError, match="serialized byte bound"):
        decide_finite_abelian_spectral_pair(source)


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

    with finite_abelian_validation_error():
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
