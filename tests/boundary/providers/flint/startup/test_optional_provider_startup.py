from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_runtime_starts_and_exposes_unrelated_capabilities_without_flint(
    tmp_path: Path,
) -> None:
    script = """
import importlib.abc
import sys
from pathlib import Path

class BlockFlint(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "flint" or fullname.startswith("flint."):
            raise ImportError("python-flint intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockFlint())

from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.runtime.model import JacobianRuntime

runtime = create_runtime(
    Path(sys.argv[1]),
    checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
)
ids = {
    descriptor.capability_id
    for descriptor in runtime.core.capabilities.catalog().capabilities
}
assert "integer.compute.gcd" in ids
assert "finite_abelian_group.exact_factorization.verify" in ids
assert "probability.finite_distribution.raw_moment.compute" not in ids
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "runtime")],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr


def test_runtime_starts_and_exposes_unrelated_capabilities_without_z3(
    tmp_path: Path,
) -> None:
    script = """
import importlib.abc
import sys
from pathlib import Path

class BlockZ3(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "z3" or fullname.startswith("z3."):
            raise ImportError("z3-solver intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockZ3())

from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.runtime.model import JacobianRuntime

runtime = create_runtime(Path(sys.argv[1]))
ids = {
    descriptor.capability_id
    for descriptor in runtime.core.capabilities.catalog().capabilities
}
assert "integer.compute.gcd" in ids
assert "graph.invariant.girth.compute" in ids
assert "graph.domination.minimum.compute" not in ids
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "runtime")],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
