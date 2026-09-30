"""The top-homology result envelope must bound what the result retains."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.koszul import module_operations
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleDifferential,
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

    Each retained occurrence contributes its label characters to the intrinsic
    allocation estimate. The carriers impose no string-length limit, so counting
    only the number of basis elements would leave retained label data unbounded.
    """
    _, module = _algebra_and_module("L" * 3_000_000)
    with pytest.raises(OperationResourceAdmissionError) as refusal:
        _compute(module, ((_q(0), _q(1)),))
    assert (
        refusal.value.errors()[0]["type"] == "koszul.module.top_homology_result_bound"
    )


@pytest.mark.parametrize(
    "character",
    ("L", "é", "😀", '"', "\\", "\n"),
    ids=("ascii", "accent", "astral", "quote", "slash", "newline"),
)
@pytest.mark.parametrize("mapping", (False, True))
def test_equal_length_labels_have_the_same_intrinsic_admission(
    monkeypatch: pytest.MonkeyPatch, character: str, mapping: bool
) -> None:
    label = character * 1000
    _, module = _algebra_and_module(label)
    value = module_koszul_complex(
        ModuleKoszulRequest(
            algebra=module.algebra, module=module, sequence=((_q(0), _q(1)),)
        )
    )
    request = ModuleKoszulTopHomologyRequest(complex=value)
    monkeypatch.setattr(module_operations, "MAX_KOSZUL_TOP_HOMOLOGY_RESULT_CELLS", 6000)
    result = module_koszul_top_homology(request.model_dump() if mapping else request)
    assert result.module.basis[0] == label
    assert result.annihilator_basis == result.top_homology_basis == ((_q(0), _q(1)),)
    assert (
        ModuleKoszulTopHomology.model_validate_json(result.model_dump_json()) == result
    )


def test_native_admission_never_uses_json_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, module = _algebra_and_module("1")
    value = module_koszul_complex(
        ModuleKoszulRequest(
            algebra=module.algebra, module=module, sequence=((_q(0), _q(1)),)
        )
    )
    serializations: list[object] = []

    def forbidden_json(self: object) -> str:
        serializations.append(self)
        raise AssertionError("native admission attempted JSON serialization")

    for model in (FiniteCommutativeAlgebra, BasedFiniteModule, ModuleDifferential):
        monkeypatch.setattr(model, "model_dump_json", forbidden_json)
    result = module_koszul_top_homology(ModuleKoszulTopHomologyRequest(complex=value))
    assert result.top_homology_basis == ((_q(0), _q(1)),)
    assert serializations == []


def test_large_unicode_labels_remain_admitted_and_compose() -> None:
    label = "😀" * 100_000
    _, module = _algebra_and_module(label)
    result = _compute(module, ((_q(0), _q(1)),))
    decoded = ModuleKoszulTopHomology.model_validate_json(result.model_dump_json())
    assert decoded == result
    assert decoded.annihilator_basis == ((_q(0), _q(1)),)
