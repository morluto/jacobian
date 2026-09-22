from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._kempner_models import KempnerDigitSet
from jacobian.math.number_theory.kempner.operations import enclose_kempner_series
from jacobian.math.number_theory.modular_forms.operations import (
    hecke,
    named_q_expansion,
    sturm_bound,
)
from jacobian.math.number_theory.modular_forms.values import ModularFormSpace


def test_dense_kempner_prefix_recurrence_is_exact() -> None:
    family = KempnerDigitSet(base=10, allowed_digits=tuple(range(9)))
    result = enclose_kempner_series(family, 4)
    expected = sum(
        (Fraction(1, n) for n in range(1, 10_000) if "9" not in str(n)),
        Fraction(0),
    )
    assert result.partial_sum.as_fraction() == expected


def test_gamma0_sturm_and_hecke_metadata() -> None:
    space = ModularFormSpace(level=1, weight=4, kind="M")
    assert sturm_bound(space).bound == 0
    expansion = named_q_expansion(space, "E4", 8)
    transformed = hecke(expansion, 1, 4)
    assert transformed.space == space
    assert transformed.q_expansion.truncation_order == 4


def test_native_transform_rejects_malformed_parent_and_form() -> None:
    space = ModularFormSpace(level=1, weight=4, kind="M")
    with pytest.raises(OperationDomainValidationError):
        named_q_expansion(space, "bad", 2)
    with pytest.raises(OperationDomainValidationError):
        named_q_expansion(ModularFormSpace(level=1, weight=6, kind="M"), "E4", 2)
