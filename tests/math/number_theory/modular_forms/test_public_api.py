"""Exact public native API contract for level-one modular forms."""

from jacobian.math.number_theory import modular_forms


def test_exact_public_api_symbols() -> None:
    assert tuple(modular_forms.__all__) == (
        "LevelOneModularQExpansion",
        "level_one_named_q_expansion",
    )
    assert all(hasattr(modular_forms, name) for name in modular_forms.__all__)
