from __future__ import annotations

import subprocess
import sys


def _imported_modules(target: str) -> set[str]:
    completed = subprocess.run(
        [sys.executable, "-c", f"import {target}, sys; print('\\n'.join(sys.modules))"],
        check=True,
        capture_output=True,
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


def test_native_namespace_does_not_eagerly_import_optional_backends() -> None:
    _assert_not_imported(
        _imported_modules("jacobian.math"),
        ("networkx", "sympy", "flint"),
    )


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
