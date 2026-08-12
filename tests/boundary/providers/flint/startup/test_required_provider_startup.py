from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("import_name", "provider"),
    (
        ("networkx", "jacobian.networkx"),
        ("sympy", "jacobian.sympy"),
        ("flint", "python-flint"),
        ("z3", "jacobian.z3"),
        ("cvc5", "cvc5"),
    ),
)
def test_runtime_rejects_a_base_installation_without_required_provider(
    tmp_path: Path,
    import_name: str,
    provider: str,
) -> None:
    script = """
import importlib.abc
import sys
from pathlib import Path

blocked = sys.argv[1]

class BlockProvider(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == blocked or fullname.startswith(f"{blocked}."):
            raise ImportError(f"{blocked} intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockProvider())

from jacobian.runtime import create_runtime

create_runtime(Path(sys.argv[2]))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            import_name,
            str(tmp_path / "runtime"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode != 0
    assert "required Python providers are unavailable" in completed.stderr
    assert f"{provider}:" in completed.stderr
