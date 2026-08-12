from __future__ import annotations

import tomllib
from pathlib import Path

from jacobian.provider_runtime import (
    NETWORKX_VERSION,
    PYTHON_FLINT_VERSION,
    SYMPY_VERSION,
    Z3_SOLVER_VERSION,
)
from jacobian.providers.external_solver_runtime import CVC5_VERSION

ROOT = Path(__file__).parents[3]


def test_required_provider_dependencies_match_exact_runtime_pins() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    dependencies = set(project["dependencies"])

    assert {
        f"networkx=={NETWORKX_VERSION}",
        f"python-flint=={PYTHON_FLINT_VERSION}",
        f"sympy=={SYMPY_VERSION}",
        f"z3-solver=={Z3_SOLVER_VERSION}",
        f"cvc5=={CVC5_VERSION}",
    } <= dependencies
    assert "optional-dependencies" not in project
