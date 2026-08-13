from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _imported_modules(target: str) -> set[str]:
    completed = subprocess.run(
        [sys.executable, "-c", f"import {target}, sys; print('\\n'.join(sys.modules))"],
        check=True,
        capture_output=True,
        env={**os.environ, "SYMPY_GROUND_TYPES": "python"},
        text=True,
    )
    return set(completed.stdout.splitlines())


def _assert_not_imported(imported: set[str], prefixes: tuple[str, ...]) -> None:
    leaked = {
        name
        for name in imported
        for prefix in prefixes
        if name == prefix or name.startswith(f"{prefix}.")
    }
    assert not leaked, f"forbidden modules imported: {sorted(leaked)}"


def test_native_namespace_does_not_import_runtime_or_transport() -> None:
    forbidden = (
        "jacobian.runtime",
        "jacobian.mcp",
        "jacobian.operation_installation",
        "jacobian.providers",
    )
    _assert_not_imported(_imported_modules("jacobian.math"), forbidden)


def test_native_namespace_does_not_eagerly_import_packaged_backends() -> None:
    _assert_not_imported(
        _imported_modules("jacobian.math"),
        ("networkx", "sympy", "flint"),
    )


def test_runtime_assembly_does_not_import_packaged_backends(tmp_path: Path) -> None:
    script = """
import sys
from pathlib import Path
from jacobian.runtime import create_runtime

runtime = create_runtime(Path(sys.argv[1]))
try:
    forbidden = {"networkx", "sympy", "flint", "z3", "cvc5"}
    imported = sorted(forbidden.intersection(sys.modules))
    if imported:
        raise AssertionError(f"packaged backends imported during assembly: {imported}")
finally:
    runtime.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "runtime")],
        check=False,
        capture_output=True,
        env={**os.environ, "SYMPY_GROUND_TYPES": "python"},
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr


def test_portfolio_leaf_import_does_not_load_assembly_or_domains() -> None:
    _assert_not_imported(
        _imported_modules("jacobian.portfolio.provider_resolution"),
        (
            "jacobian.domains",
            "jacobian.portfolio.assembler",
            "jacobian.portfolio.builtin",
            "jacobian.runtime",
        ),
    )


def test_portfolio_root_is_an_import_transparent_internal_namespace() -> None:
    script = """
import jacobian.portfolio as portfolio
import sys

assert portfolio.__all__ == ()
assert not hasattr(portfolio, "PortfolioPlan")
assert not hasattr(portfolio, "install_portfolio")
children = sorted(
    name for name in sys.modules if name.startswith("jacobian.portfolio.")
)
if children:
    raise AssertionError(f"portfolio root imported child modules: {children}")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env={**os.environ, "SYMPY_GROUND_TYPES": "python"},
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_native_matrices_does_not_import_capabilities_or_provider_loading() -> None:
    _assert_not_imported(
        _imported_modules("jacobian.math.matrices"),
        (
            "jacobian.adapters",
            "jacobian.artifact_repository",
            "jacobian.capability_dispatch",
            "jacobian.capability_service",
            "jacobian.operation_installation",
            "jacobian.provider_runtime",
            "jacobian.providers",
            "jacobian.runtime",
            "jacobian.store",
        ),
    )


def test_native_finite_fields_does_not_eagerly_import_flint() -> None:
    imported = _imported_modules("jacobian.math.finite_fields")
    _assert_not_imported(imported, ("flint",))
    _assert_not_imported(
        imported,
        (
            "jacobian.adapters",
            "jacobian.operation_installation",
            "jacobian.providers",
            "jacobian.runtime",
            "jacobian.storage",
        ),
    )


def test_sympy_finite_field_construction_and_projective_line_do_not_need_flint() -> (
    None
):
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from jacobian.math.finite_fields import Axis, finite_field, "
                "projective_line; "
                "field = finite_field(2, (1, 1, 1)); "
                "line = projective_line(field, Axis(name='p', labels=('x', 'y'))); "
                "assert len(line.points) == 5; "
                "assert 'flint' not in sys.modules"
            ),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "SYMPY_GROUND_TYPES": "python"},
        text=True,
    )

    assert completed.stderr == ""
