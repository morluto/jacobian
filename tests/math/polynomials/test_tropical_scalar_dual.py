"""Exact scalar duality between min-plus and max-plus parents."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.tropical import TropicalScalar, TropicalSemiring
from jacobian.math.polynomials.tropical._models import (
    ScalarDualRequest,
    ScalarDualResult,
)
from jacobian.math.polynomials.tropical._tools import TOOLS, compute_scalar_dual


def _finite(convention: str, base: str, value: Fraction) -> TropicalScalar:
    semiring = TropicalSemiring(convention=convention, base=base)  # type: ignore[arg-type]
    return TropicalScalar(
        semiring=semiring,
        kind="FINITE",
        value=CanonicalRational.from_fraction(value),
    )


def _infinity(convention: str, base: str) -> TropicalScalar:
    semiring = TropicalSemiring(convention=convention, base=base)  # type: ignore[arg-type]
    kind = "POSITIVE_INFINITY" if convention == "MIN_PLUS" else "NEGATIVE_INFINITY"
    return TropicalScalar(semiring=semiring, kind=kind)  # type: ignore[arg-type]


def test_dual_negates_finite_values_and_is_an_involution() -> None:
    cases = (
        ("MIN_PLUS", "QQ", Fraction(7, 12), "MAX_PLUS"),
        ("MAX_PLUS", "ZZ", Fraction(-9), "MIN_PLUS"),
        ("MIN_PLUS", "ZZ", Fraction(0), "MAX_PLUS"),
    )
    for source_convention, base, value, target_convention in cases:
        source = _finite(source_convention, base, value)
        mapped = compute_scalar_dual(ScalarDualRequest(scalar=source))
        assert mapped.source_semiring == source.semiring
        assert mapped.target_semiring == TropicalSemiring(
            convention=target_convention,
            base=base,  # type: ignore[arg-type]
        )
        assert mapped.result.value == CanonicalRational.from_fraction(-value)

        restored = compute_scalar_dual(ScalarDualRequest(scalar=mapped.result))
        assert restored.result == source


def test_dual_swaps_only_the_licensed_additive_infinity() -> None:
    for convention, opposite, source_kind, target_kind in (
        ("MIN_PLUS", "MAX_PLUS", "POSITIVE_INFINITY", "NEGATIVE_INFINITY"),
        ("MAX_PLUS", "MIN_PLUS", "NEGATIVE_INFINITY", "POSITIVE_INFINITY"),
    ):
        source = _infinity(convention, "QQ")
        result = compute_scalar_dual(ScalarDualRequest(scalar=source))
        assert source.kind == source_kind
        assert result.result.kind == target_kind
        assert result.target_semiring == TropicalSemiring(
            convention=opposite,
            base="QQ",  # type: ignore[arg-type]
        )
        assert (
            compute_scalar_dual(ScalarDualRequest(scalar=result.result)).result
            == source
        )


def test_dual_respects_min_max_and_tropical_multiplication_oracle() -> None:
    # The order-reversing negation is checked against ordinary exact arithmetic:
    # -min(a,b)=max(-a,-b), while -(a+b)=(-a)+(-b).
    for a, b in ((Fraction(-5, 4), Fraction(3, 7)), (Fraction(2), Fraction(2))):
        left = _finite("MIN_PLUS", "QQ", a)
        right = _finite("MIN_PLUS", "QQ", b)
        dleft = compute_scalar_dual(ScalarDualRequest(scalar=left)).result
        dright = compute_scalar_dual(ScalarDualRequest(scalar=right)).result
        assert dleft.value is not None and dright.value is not None
        assert -min(a, b) == max(dleft.value.as_fraction(), dright.value.as_fraction())
        assert -(a + b) == dleft.value.as_fraction() + dright.value.as_fraction()


def test_manifest_operation_serializes_both_parent_identities() -> None:
    source = _finite("MIN_PLUS", "QQ", Fraction(11, 5))
    operation = next(
        tool for tool in TOOLS if tool.operation_id == "tropical.scalar.dual.compute"
    )
    assert operation is not None
    result = operation.run(ScalarDualRequest(scalar=source))
    restored = ScalarDualResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.source_semiring.convention == "MIN_PLUS"
    assert restored.target_semiring.convention == "MAX_PLUS"
    assert restored.result.value == CanonicalRational.from_fraction(Fraction(-11, 5))
