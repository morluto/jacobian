from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory._kempner_models import KempnerDigitSet
from jacobian.math.number_theory.kempner.operations import enclose_kempner_series
from jacobian.math.number_theory.modular_forms.operations import (
    hecke,
    named_q_expansion,
    sturm_bound,
    u_operator,
    v_operator,
)
from jacobian.math.number_theory.modular_forms.values import (
    MAX_Q_TRANSFORM_OUTPUT_PRECISION,
    MAX_Q_TRANSFORM_SOURCE_ORDER,
    ModularFormSpace,
    ModularQExpansion,
)
from jacobian.math.polynomials.series._models import TruncatedSeries


def test_dense_kempner_prefix_recurrence_is_exact() -> None:
    family = KempnerDigitSet(base=10, allowed_digits=tuple(range(9)))
    result = enclose_kempner_series(family, 4)
    expected = sum(
        (Fraction(1, n) for n in range(1, 10_000) if "9" not in str(n)), Fraction(0)
    )
    assert result.partial_sum.as_fraction() == expected


def test_gamma0_sturm_and_hecke_metadata() -> None:
    space = ModularFormSpace(level=1, weight=4, kind="M")
    assert sturm_bound(space).bound == 0
    expansion = named_q_expansion(space, "E4", 8)
    transformed = hecke(expansion, 1, 4)
    assert transformed.space == space
    assert transformed.q_expansion.truncation_order == 4


def test_q_transforms_accept_the_exact_last_source_index_boundary() -> None:
    space = ModularFormSpace(level=1, weight=4, kind="M")
    short = named_q_expansion(space, "E4", 3)
    longer = named_q_expansion(space, "E4", 4)

    assert u_operator(short, 2, 2) == u_operator(longer, 2, 2)
    assert hecke(short, 2, 2) == hecke(longer, 2, 2)


def test_u_and_v_bind_their_gamma0_prime_codomain() -> None:
    source_space = ModularFormSpace(level=1, weight=4, kind="M")
    source = named_q_expansion(source_space, "E4", 8)
    for transform in (u_operator, v_operator):
        result = transform(source, 2, 2)
        assert result.space == ModularFormSpace(level=2, weight=4, kind="M")
        restored = type(result).model_validate(result.model_dump())
        assert restored.space == result.space
        assert restored.q_expansion == result.q_expansion


def test_u_and_v_reject_unadmitted_source_levels() -> None:
    source = ModularQExpansion(
        space=ModularFormSpace(level=2, weight=4, kind="M"),
        weight=4,
        q_expansion=named_q_expansion(
            ModularFormSpace(level=1, weight=4, kind="M"), "E4", 2
        ).q_expansion,
    )
    for transform in (u_operator, v_operator):
        with pytest.raises(OperationDomainValidationError):
            transform(source, 2, 1)


def test_named_form_native_admission_is_stable() -> None:
    with pytest.raises(OperationDomainValidationError):
        named_q_expansion("bad", "E4", 2)  # type: ignore[arg-type]
    with pytest.raises(OperationDomainValidationError):
        named_q_expansion(ModularFormSpace(level=1, weight=6, kind="M"), "E4", 2)


def test_sturm_level_one_boundary_remains_explicit() -> None:
    assert sturm_bound(ModularFormSpace(level=1, weight=12, kind="M")).bound == 1


def test_q_transforms_reject_forged_scalar_carriers_before_arithmetic() -> None:
    source = named_q_expansion(ModularFormSpace(level=1, weight=4, kind="M"), "E4", 2)
    forged_scalar = CanonicalRational.model_construct(num=1, den=0)
    forged_series = TruncatedSeries.model_construct(
        variable="q", truncation_order=2, coefficients=(forged_scalar, forged_scalar)
    )
    forged = ModularQExpansion.model_construct(
        space=source.space, weight=4, q_expansion=forged_series, basis_id="forged"
    )
    for transform, args in (
        (hecke, (1, 1)),
        (u_operator, (2, 1)),
        (v_operator, (2, 1)),
    ):
        with pytest.raises(OperationDomainValidationError):
            transform(forged, *args)


def test_q_transforms_reject_unadmitted_prefix_and_output_before_indexing() -> None:
    source = named_q_expansion(ModularFormSpace(level=1, weight=4, kind="M"), "E4", 2)
    with pytest.raises(OperationResourceAdmissionError):
        u_operator(source, 2, MAX_Q_TRANSFORM_OUTPUT_PRECISION + 1)
    forged_series = TruncatedSeries.model_construct(
        variable="q",
        truncation_order=MAX_Q_TRANSFORM_SOURCE_ORDER + 1,
        coefficients=(),
    )
    forged = ModularQExpansion.model_construct(
        space=source.space,
        weight=4,
        q_expansion=forged_series,
        basis_id="forged",
    )
    with pytest.raises(OperationResourceAdmissionError):
        v_operator(forged, 2, 1)


def test_operator_prime_bound_precedes_primality_and_target_construction() -> None:
    source = named_q_expansion(ModularFormSpace(level=1, weight=4, kind="M"), "E4", 2)
    with pytest.raises(OperationResourceAdmissionError):
        v_operator(source, 10**100, 1)


def test_hecke_rejects_accumulated_denominator_growth_before_kernel() -> None:
    index = 25_279
    huge = CanonicalRational(num=1, den=10**4092 + 1)
    zero = CanonicalRational(num=0, den=1)
    source = ModularQExpansion(
        space=ModularFormSpace(level=1, weight=0, kind="M"),
        weight=0,
        q_expansion=TruncatedSeries(
            variable="q",
            truncation_order=index,
            coefficients=(huge,) + (zero,) * (index - 1),
        ),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        hecke(source, index, 1)
    assert error.value.errors()[0]["type"] == ("modular_form.coefficient_growth_bound")


def test_hecke_weight_zero_uses_exact_inverse_divisor_factor() -> None:
    space = ModularFormSpace(level=1, weight=0, kind="M")
    source = ModularQExpansion(
        space=space,
        weight=0,
        q_expansion=TruncatedSeries(
            variable="q",
            truncation_order=13,
            coefficients=(
                CanonicalRational(num=0, den=1),
                CanonicalRational(num=2, den=1),
                *tuple(CanonicalRational(num=0, den=1) for _ in range(11)),
            ),
        ),
    )
    result = hecke(source, 3, 4)
    assert result.q_expansion.coefficients[3].as_fraction() == Fraction(2, 3)
