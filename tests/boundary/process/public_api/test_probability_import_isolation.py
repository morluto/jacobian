from __future__ import annotations

import os
import subprocess
import sys


def test_native_probability_import_does_not_load_its_wire_or_runtime_owner() -> None:
    target = "jacobian.math.probability"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {target}, sys; print('\\n'.join(sys.modules))",
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "SYMPY_GROUND_TYPES": "python"},
        text=True,
    )
    imported = set(completed.stdout.splitlines())
    forbidden = (
        "jacobian.domains.probability",
        "jacobian.operation_bindings",
        "jacobian.operation_installation",
        "jacobian.provider_runtime",
        "jacobian.providers",
        "jacobian.runtime",
        "jacobian_checkers",
    )
    leaked = {
        name
        for name in imported
        for prefix in forbidden
        if name == prefix or name.startswith(f"{prefix}.")
    }

    assert not leaked, (
        f"native probability imported wire/runtime modules: {sorted(leaked)}"
    )
