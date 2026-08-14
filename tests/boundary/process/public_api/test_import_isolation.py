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
        "jacobian.operation_binding",
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
from tests.support.catalog_build_runtime import create_catalog_build_runtime

runtime = create_catalog_build_runtime(Path(sys.argv[1]))
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


def test_backend_check_import_does_not_load_assembly_or_domains() -> None:
    _assert_not_imported(
        _imported_modules("jacobian.maintained_backends"),
        (
            "jacobian.domains",
            "jacobian.catalog_build",
            "jacobian.builtin_operation_modules",
            "jacobian.runtime",
        ),
    )


def test_native_matrices_does_not_import_operations_or_provider_loading() -> None:
    _assert_not_imported(
        _imported_modules("jacobian.math.matrices"),
        (
            "jacobian.adapters",
            "jacobian.artifact_repository",
            "jacobian.operation_dispatch",
            "jacobian.catalog_operation_collector",
            "jacobian.operation_binding",
            "jacobian.provider_runtime",
            "jacobian.providers",
            "jacobian.runtime",
            "jacobian.store",
        ),
    )


def test_native_probability_import_does_not_load_wire_or_runtime_owner() -> None:
    imported = _imported_modules("jacobian.math.probability")
    _assert_not_imported(
        imported,
        (
            "jacobian.domains.probability",
            "jacobian.operation_bindings",
            "jacobian.operation_binding",
            "jacobian.provider_runtime",
            "jacobian.providers",
            "jacobian.runtime",
            "jacobian_checkers",
        ),
    )


def test_native_finite_fields_does_not_eagerly_import_flint() -> None:
    imported = _imported_modules("jacobian.math.finite_fields")
    _assert_not_imported(imported, ("flint",))
    _assert_not_imported(
        imported,
        (
            "jacobian.adapters",
            "jacobian.operation_binding",
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
