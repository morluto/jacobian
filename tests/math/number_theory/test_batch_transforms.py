from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
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
    ModularFormSpace,
    ModularQExpansion,
)


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
