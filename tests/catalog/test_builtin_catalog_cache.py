"""The immutable built-in operation catalog is compiled once per process."""

from jacobian.catalog.catalog import Catalog


def test_open_reuses_the_compiled_builtin_catalog() -> None:
    assert Catalog.open() is Catalog.open()
