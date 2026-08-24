"""Public native API tests for exact Dirichlet characters."""

from __future__ import annotations

from jacobian.math.dirichlet_characters import __all__


def test_public_api_exports_only_canonical_value_and_native_operations() -> None:
    assert __all__ == [
        "PrincipalDirichletCharacter",
        "principal_dirichlet_character",
        "principal_dirichlet_character_value",
    ]
