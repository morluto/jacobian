"""The top-homology result envelope must bound what the result retains."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleKoszulRequest,
    ModuleKoszulTopHomology,
    ModuleKoszulTopHomologyRequest,
)
from jacobian.math.koszul.module_operations import (
    module_koszul_complex,
    module_koszul_top_homology,
)


def _q(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _algebra_and_module(
    label: str,
) -> tuple[FiniteCommutativeAlgebra, BasedFiniteModule]:
    """The dual-number regular module, with a caller-chosen unit basis label."""
    algebra = FiniteCommutativeAlgebra(
        basis=(label, "e"),
        multiplication=(
            ((_q(1), _q(0)), (_q(0), _q(1))),
            ((_q(0), _q(1)), (_q(0), _q(0))),
        ),
        unit=(_q(1), _q(0)),
    )
    module = BasedFiniteModule(
        algebra=algebra,
        basis=(label, "e"),
        action=(
            ((_q(1), _q(0)), (_q(0), _q(1))),
            ((_q(0), _q(0)), (_q(1), _q(0))),
        ),
    )
    return algebra, module


def _compute(
    module: BasedFiniteModule,
    sequence: tuple[tuple[CanonicalRational, ...], ...],
) -> ModuleKoszulTopHomology:
    complex_value = module_koszul_complex(
        ModuleKoszulRequest(algebra=module.algebra, module=module, sequence=sequence)
    )
    return module_koszul_top_homology(
        ModuleKoszulTopHomologyRequest(complex=complex_value)
    )


def test_an_ordinary_label_is_still_admitted() -> None:
    """Negative control: the retained-size charge must not refuse ordinary input."""
    _, module = _algebra_and_module("1")
    result = _compute(module, ((_q(0), _q(1)),))
    assert result.top_homology_basis == ((_q(0), _q(1)),)


def test_a_very_long_basis_label_is_refused_by_the_result_envelope() -> None:
    """The result envelope counts retained materialization, not just basis count.

    The admission measured the retained payload's serialized size and discarded
    it, so the result check saw only the number of basis elements. The carriers
    impose no string-length limit, so a basis with a very long label could
    return an arbitrarily large result under a finite envelope.
    """
    _, module = _algebra_and_module("L" * 3_000_000)
    with pytest.raises(OperationResourceAdmissionError) as refusal:
        _compute(module, ((_q(0), _q(1)),))
    assert (
        refusal.value.errors()[0]["type"] == "koszul.module.top_homology_result_bound"
    )
