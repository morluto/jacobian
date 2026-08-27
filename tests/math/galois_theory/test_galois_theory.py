"""Contract and mathematical-property tests for Galois operations."""

from typing import Any

import pytest
from pydantic import ValidationError
from sympy.combinatorics import Permutation, PermutationGroup

from jacobian.math.galois_theory._models import (
    FiniteFieldFactor,
    FrobeniusCycleRequest,
    GaloisFactorRequest,
    GaloisFactorResult,
    GaloisGroupRequest,
    GaloisGroupResult,
    SolvableRequest,
    SolvableResult,
)
from jacobian.math.galois_theory._operations import (
    _verify_galois_factor_result,
    compute_frobenius_cycle,
    compute_galois_factor,
    compute_galois_group,
    compute_solvable,
    verify_galois_group_result,
    verify_solvable_result,
)
from jacobian.math.galois_theory._tools import TOOLS


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "polynomial.galois.factor_mod_p.compute",
        "polynomial.galois.frobenius_cycle.compute",
        "polynomial.galois_group.compute",
        "polynomial.solvable_by_radicals.decide",
    }


def _multiply_mod(
    left: tuple[int, ...], right: tuple[int, ...], prime: int
) -> tuple[int, ...]:
    result = [0] * (len(left) + len(right) - 1)
    for i, left_coefficient in enumerate(left):
        for j, right_coefficient in enumerate(right):
            result[i + j] = (
                result[i + j] + left_coefficient * right_coefficient
            ) % prime
    return tuple(result)


def _reconstruct(result: GaloisFactorResult) -> tuple[int, ...]:
    polynomial: tuple[int, ...] = (result.unit,)
    for factor in result.factors:
        for _ in range(factor.multiplicity):
            polynomial = _multiply_mod(
                polynomial, factor.coefficients, result.field_order
            )
    return polynomial


@pytest.mark.parametrize(
    ("prime", "coefficients"),
    [
        (5, (1, 0, 1)),
        (3, (0, 0, 1)),
        (3, (2, 2)),
        (7, (6, 0, 0, 1)),
        (2, (1, 1, 1, 1, 1)),
    ],
)
def test_factorization_reconstructs_source(
    prime: int, coefficients: tuple[int, ...]
) -> None:
    result = compute_galois_factor(
        GaloisFactorRequest(field_order=prime, coefficients=coefficients)
    )
    assert _reconstruct(result) == coefficients
    assert result.factor_count == sum(factor.multiplicity for factor in result.factors)
    assert all(
        0 <= coefficient < prime
        for factor in result.factors
        for coefficient in factor.coefficients
    )


def test_repeated_factor_is_not_reported_irreducible() -> None:
    result = compute_galois_factor(
        GaloisFactorRequest(field_order=3, coefficients=(0, 0, 1))
    )
    assert result.unit == 1
    assert result.factors[0].coefficients == (0, 1)
    assert result.factors[0].multiplicity == 2
    assert result.distinct_factor_count == 1
    assert result.factor_count == 2
    assert result.is_irreducible is False


def test_nonmonic_factorization_retains_unit() -> None:
    result = compute_galois_factor(
        GaloisFactorRequest(field_order=3, coefficients=(2, 2))
    )
    assert result.unit == 2
    assert result.factors[0].coefficients == (1, 1)
    assert _reconstruct(result) == (2, 2)


@pytest.mark.parametrize(
    ("prime", "coefficients"),
    [
        (3, (1, 0, 1)),
        (5, (2, 1, 0, 1)),
        (7, (6, 0, 1, 1)),
    ],
)
def test_multiplying_by_every_field_unit_preserves_monic_factors(
    prime: int, coefficients: tuple[int, ...]
) -> None:
    base = compute_galois_factor(
        GaloisFactorRequest(field_order=prime, coefficients=coefficients)
    )

    for scalar in range(1, prime):
        scaled_coefficients = tuple(
            scalar * coefficient % prime for coefficient in coefficients
        )
        scaled = compute_galois_factor(
            GaloisFactorRequest(
                field_order=prime,
                coefficients=scaled_coefficients,
            )
        )

        assert scaled.factors == base.factors
        assert scaled.unit == scalar * base.unit % prime
        assert _reconstruct(scaled) == scaled_coefficients


def test_zero_and_noncanonical_degree_are_rejected_before_sympy() -> None:
    with pytest.raises(ValidationError) as exc:
        GaloisFactorRequest(field_order=3, coefficients=(0, 0))
    assert exc.value.errors()[0]["type"] == "galois_theory.polynomial_zero"
    with pytest.raises(ValidationError) as exc:
        GaloisFactorRequest(field_order=3, coefficients=(1, 0))
    assert exc.value.errors()[0]["type"] == "galois_theory.polynomial_zero"


def test_factorization_result_parses_structurally_and_private_verification_rejects_forgery() -> (
    None
):
    result = compute_galois_factor(
        GaloisFactorRequest(field_order=5, coefficients=(1, 0, 1))
    )
    payload = result.model_dump()
    assert GaloisFactorResult.model_validate(payload) == result

    payload["unit"] = 2
    altered_unit = GaloisFactorResult.model_validate(payload)
    assert altered_unit.unit == 2
    assert not _verify_galois_factor_result(altered_unit)

    payload = result.model_dump()
    payload["field_order"] = 4
    with pytest.raises(ValidationError, match="field_order must be prime"):
        GaloisFactorResult.model_validate(payload)

    forged = GaloisFactorResult(
        field_order=3,
        source_coefficients=(2, 0, 1),
        unit=1,
        factors=(FiniteFieldFactor(coefficients=(2, 0, 1), multiplicity=1),),
        distinct_factor_count=1,
        factor_count=1,
        is_irreducible=True,
    )
    assert not _verify_galois_factor_result(forged)


def test_factor_producer_runs_the_backend_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from sympy import Poly

    original = Poly.factor_list
    calls = 0

    def counted(self: Poly, *args: object, **kwargs: object) -> Any:
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Poly, "factor_list", counted)
    result = compute_galois_factor(
        GaloisFactorRequest(field_order=5, coefficients=(1, 0, 1))
    )

    assert result.is_irreducible is False
    assert calls == 1


def test_frobenius_cycle_is_canonical_positive_partition() -> None:
    result = compute_frobenius_cycle(
        FrobeniusCycleRequest(
            field_order=5,
            polynomial_degree=4,
            factorization_degrees=(1, 3),
        )
    )
    assert result.cycle_type == (3, 1)
    assert result.is_irreducible is False
    irreducible = compute_frobenius_cycle(
        FrobeniusCycleRequest(
            field_order=3,
            polynomial_degree=2,
            factorization_degrees=(2,),
        )
    )
    assert irreducible.cycle_type == (2,)
    assert irreducible.is_irreducible is True


def test_frobenius_cycle_rejects_unrealizable_distinct_factor_pattern() -> None:
    with pytest.raises(ValidationError) as exc:
        FrobeniusCycleRequest(
            field_order=2,
            polynomial_degree=3,
            factorization_degrees=(1, 1, 1),
        )
    assert exc.value.errors()[0]["type"] == "galois_theory.partition_unrealizable"


@pytest.mark.parametrize("degrees", [(2, 0), (3, -1)])
def test_frobenius_rejects_nonpositive_factor_degrees(
    degrees: tuple[int, ...],
) -> None:
    with pytest.raises(ValidationError):
        FrobeniusCycleRequest(
            field_order=3,
            polynomial_degree=2,
            factorization_degrees=degrees,
        )


@pytest.mark.parametrize("request_type", [GaloisGroupRequest, SolvableRequest])
def test_galois_backend_domain_is_enforced_before_execution(request_type: type) -> None:
    with pytest.raises(ValidationError) as exc:
        request_type(coefficients=(0, 0, 0, 0, 1))
    assert exc.value.errors()[0]["type"] == "galois_theory.polynomial_not_irreducible"
    with pytest.raises(ValidationError) as exc:
        request_type(coefficients=(-2, 0, 0, 0, 0, 0, 0, 1))
    assert exc.value.errors()[0]["type"] == "too_long"


def _group_from_result(result: object) -> PermutationGroup:
    group = result.group  # type: ignore[attr-defined]
    return PermutationGroup(
        *(Permutation(list(generator)) for generator in group.generators)
    )


def test_galois_group_returns_composable_generators() -> None:
    result = compute_galois_group(GaloisGroupRequest(coefficients=(-1, -1, 0, 0, 0, 1)))
    assert result.degree == 5
    assert result.order == 120
    assert result.is_solvable is False
    assert result.group.root_axis == tuple(f"root_{index}" for index in range(5))
    assert _group_from_result(result).order() == result.order

    forged = GaloisGroupResult.model_validate(
        {**result.model_dump(), "order": result.order + 1}
    )
    assert not verify_galois_group_result(forged)


def test_solvable_quintic_returns_the_group_certificate() -> None:
    result = compute_solvable(SolvableRequest(coefficients=(-2, 0, 0, 0, 0, 1)))
    assert result.solvable_by_radicals is True
    assert _group_from_result(result).order() == 20

    forged = SolvableResult.model_validate(
        {**result.model_dump(), "solvable_by_radicals": False}
    )
    assert not verify_solvable_result(forged)


def test_unsolvable_quintic_uses_actual_group() -> None:
    result = compute_solvable(SolvableRequest(coefficients=(-1, -1, 0, 0, 0, 1)))
    assert result.solvable_by_radicals is False
    assert _group_from_result(result).order() == 120
