"""Keep the native character-degree convenience out of public discovery."""

from jacobian.catalog.catalog import Catalog


def test_character_degree_remains_native_only() -> None:
    assert Catalog.open().operation("character.degree.compute") is None
