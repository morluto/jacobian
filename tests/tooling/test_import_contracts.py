"""Regression tests for the exhaustive public-math import contract."""

from __future__ import annotations

import tomllib
from pathlib import Path


def _load_pyproject() -> dict:
    with Path("pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_public_math_contract_is_package_level() -> None:
    """The forbidden contract source is the jacobian.math package, not a per-module allowlist."""
    contracts = _load_pyproject()["tool"]["importlinter"]["contracts"]
    math_contract = next(
        c
        for c in contracts
        if c["name"] == "The public math library does not depend on product layers"
    )
    assert math_contract["source_modules"] == ["jacobian.math"]


def test_public_math_contract_forbids_all_product_layers() -> None:
    contracts = _load_pyproject()["tool"]["importlinter"]["contracts"]
    math_contract = next(
        c
        for c in contracts
        if c["name"] == "The public math library does not depend on product layers"
    )
    forbidden = set(math_contract["forbidden_modules"])
    assert forbidden == {
        "jacobian.catalog",
        "jacobian.cli",
        "jacobian.dispatch",
        "jacobian.mcp",
        "jacobian.process",
    }


def test_public_math_contract_has_no_per_domain_source_list() -> None:
    """No individual math domain modules are listed in source_modules."""
    contracts = _load_pyproject()["tool"]["importlinter"]["contracts"]
    math_contract = next(
        c
        for c in contracts
        if c["name"] == "The public math library does not depend on product layers"
    )
    sources = math_contract["source_modules"]
    assert len(sources) == 1
    assert "jacobian.math" in sources[0]
    # No per-domain module paths should appear
    for src in sources:
        assert src.count(".") <= 2, f"{src} looks like a per-domain allowlist entry"
