"""Repeatable local measurements for exact installed provider identities."""

from __future__ import annotations

import importlib
import importlib.metadata
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from jacobian.contracts.operations import ProviderObservation
from jacobian.contracts.provider_measurements import (
    ProviderInstalledSize,
    ProviderMeasurement,
    ProviderMeasurementSample,
    ProviderMeasurementStatus,
)
from jacobian.process_policy import (
    ProcessRequest,
    ProcessTermination,
    execute_process,
)
from jacobian.worker_environment import worker_environment

_PROBE_TIMEOUT_SECONDS = 120
_COLD_INSTALL_TIMEOUT_SECONDS = 600
_MAX_DIAGNOSTIC_BYTES = 64 * 1024
_PYTHON_PROBE = r"""
import sys

provider = sys.argv[1]
operation = sys.argv[2]

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

if provider == "jacobian.networkx":
    import networkx as backend
    if operation == "reproduction":
        graph = backend.path_graph(32)
        require(backend.is_connected(graph), "networkx connectivity reproduction failed")
elif provider == "jacobian.sympy":
    import sympy as backend
    if operation == "reproduction":
        x, y = backend.symbols("x y")
        matrix = backend.Matrix([x**2 + y, x * y])
        require(
            matrix.jacobian((x, y)).shape == (2, 2),
            "sympy Jacobian reproduction failed",
        )
elif provider == "jacobian.z3":
    import z3 as backend
    if operation == "reproduction":
        x = backend.Real("x")
        solver = backend.Solver()
        solver.add(x == 1)
        require(solver.check() == backend.sat, "z3 satisfiability reproduction failed")
elif provider == "cvc5":
    import cvc5 as backend
    if operation == "reproduction":
        solver = backend.Solver()
        solver.setOption("produce-proofs", "true")
        solver.setOption("proof-format-mode", "alethe")
        parser = backend.InputParser(solver)
        parser.setStringInput(
            backend.InputLanguage.SMT_LIB_2_6,
            "(set-logic QF_UF)\n"
            "(declare-fun p () Bool)\n"
            "(assert p)\n"
            "(assert (not p))\n"
            "(check-sat)\n",
            "provider-measure.smt2",
        )
        result = None
        while True:
            command = parser.nextCommand()
            if command.isNull():
                break
            output = command.invoke(solver, parser.getSymbolManager())
            if command.getCommandName() == "check-sat":
                result = output.strip()
        require(result == "unsat", "cvc5 satisfiability reproduction failed")
        proofs = solver.getProof(backend.ProofComponent.FULL)
        require(len(proofs) == 1, "cvc5 proof count reproduction failed")
        require(
            solver.proofToString(proofs[0], backend.ProofFormat.ALETHE),
            "cvc5 Alethe proof reproduction failed",
        )
elif provider == "python-flint":
    import flint as backend
    if operation == "reproduction":
        augmented = backend.fmpq_mat([[2, 1, 5], [1, -1, 1]])
        reduced, rank = augmented.rref()
        require(rank == 2, "python-flint rank reproduction failed")
        require(
            reduced == backend.fmpq_mat([[1, 0, 2], [0, 1, 1]]),
            "python-flint reduction reproduction failed",
        )
else:
    import jacobian.canonical as backend
    if operation == "reproduction":
        require(
            backend.canonicalize_json({"value": 1}),
            "Jacobian canonicalization reproduction failed",
        )

# Report this child's own peak resident set so a short probe that exits
# before the engine's procfs sampler can poll it still yields a trustworthy
# positive RSS.  RUSAGE_SELF is the child's own high-water mark, not the
# parent's cumulative prior-child rusage, so it never blends siblings.
try:
    import resource as _resource
    _rss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    print(
        "JACOBIAN_MEASUREMENT_RSS_BYTES="
        + str(_rss * 1024 if sys.platform.startswith("linux") else _rss)
    )
except (ImportError, OSError, ValueError):
    pass
"""

# Marker emitted by the Python probe child carrying its own peak RSS, so a
# short probe that exits before the engine's procfs sampler polls it still
# reports a trustworthy positive value.  The child reads RUSAGE_SELF, which is
# its own high-water mark and never the parent's cumulative prior-child rusage.
_RSS_MARKER = re.compile(rb"JACOBIAN_MEASUREMENT_RSS_BYTES=(\d+)")


def _process_environment(*, toolchain_path: str | None = None) -> dict[str, str]:
    return worker_environment(path_prefix=toolchain_path)


def _child_peak_rss_bytes(stdout: bytes, sampled: int | None) -> int | None:
    """Return a trustworthy peak RSS for one completed probe.

    The Python probe child prints its own ``RUSAGE_SELF`` high-water mark as a
    final marker.  That value is the child's own peak, captured at exit, so it
    is trustworthy for short probes that complete before the engine's procfs
    sampler can poll them and it never blends prior siblings.  When the marker
    is absent (e.g. a Lean executable probe, or a platform without
    ``resource``) we fall back to the engine's sampled ``peak_rss_bytes``.
    """
    matches = _RSS_MARKER.findall(stdout)
    if matches:
        return int(matches[-1])
    return sampled


def _measure_command(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
) -> ProviderMeasurementSample:
    started = time.perf_counter()
    try:
        result = execute_process(
            ProcessRequest(
                executable=command[0],
                arguments=tuple(command[1:]),
                environment=environment or _process_environment(),
                cwd=str(Path.cwd()),
                timeout_seconds=_PROBE_TIMEOUT_SECONDS,
                stdin_bytes=b"",
                stdout_limit_bytes=_MAX_DIAGNOSTIC_BYTES,
                stderr_limit_bytes=_MAX_DIAGNOSTIC_BYTES,
            )
        )
        if result.termination is not ProcessTermination.EXITED:
            return ProviderMeasurementSample(
                status=ProviderMeasurementStatus.ERROR,
                detail="The provider measurement failed.",
            )
        if result.returncode != 0:
            return ProviderMeasurementSample(
                status=ProviderMeasurementStatus.ERROR,
                detail="The provider measurement failed.",
            )
        return ProviderMeasurementSample(
            status=ProviderMeasurementStatus.COMPLETED,
            seconds=time.perf_counter() - started,
            peak_rss_bytes=_child_peak_rss_bytes(result.stdout, result.peak_rss_bytes),
            output_bytes=len(result.stdout) + len(result.stderr),
        )
    except (OSError, RuntimeError, ValueError):
        return ProviderMeasurementSample(
            status=ProviderMeasurementStatus.ERROR,
            detail="The provider measurement failed.",
        )


def _python_probe(
    runtime: ProviderObservation,
    operation: str,
) -> ProviderMeasurementSample:
    return _measure_command(
        [sys.executable, "-c", _PYTHON_PROBE, runtime.provider, operation]
    )


def _lean_probe(*, reproduction: bool) -> ProviderMeasurementSample:
    from jacobian_checkers import lean4

    try:
        executable, _ = lean4.inspect_runtime(require_mathlib=False)
    except RuntimeError:
        return ProviderMeasurementSample(
            status=ProviderMeasurementStatus.ERROR,
            detail="The pinned Lean runtime is unavailable.",
        )
    if not reproduction:
        command = [str(executable), "-V"]
        return _measure_command(
            command,
            environment=_process_environment(toolchain_path=str(executable.parent)),
        )
    with tempfile.TemporaryDirectory(prefix="jacobian-lean-measure-") as directory:
        source = Path(directory) / "Main.lean"
        source.write_text(
            "theorem jacobian_provider_probe : (1 : Nat) = 1 := by rfl\n",
            encoding="utf-8",
        )
        command = [str(executable), str(source)]
        return _measure_command(
            command,
            environment=_process_environment(toolchain_path=str(executable.parent)),
        )


def _file_size(path: Path) -> int:
    if path.is_file() and not path.is_symlink():
        return path.stat().st_size
    return 0


def _tree_size(root: Path) -> int:
    return sum(_file_size(path) for path in root.rglob("*"))


def _installed_size(runtime: ProviderObservation) -> ProviderInstalledSize:
    distribution_name = runtime.configuration.get("distribution")
    try:
        if isinstance(distribution_name, str):
            distribution = importlib.metadata.distribution(distribution_name)
            files = distribution.files
            if files is None:
                return ProviderInstalledSize(
                    status=ProviderMeasurementStatus.ERROR,
                    detail="The provider distribution file manifest is unavailable.",
                )
            total = 0
            for package_path in files:
                total += _file_size(Path(str(distribution.locate_file(package_path))))
            return ProviderInstalledSize(
                status=ProviderMeasurementStatus.COMPLETED,
                bytes=total,
            )
        if runtime.provider == "jacobian.lean4":
            from jacobian_checkers import lean4

            executable, mathlib = lean4.inspect_runtime(require_mathlib=True)
            roots = {executable.parent.parent.resolve()}
            if mathlib is not None:
                roots.add(mathlib.resolve())
            return ProviderInstalledSize(
                status=ProviderMeasurementStatus.COMPLETED,
                bytes=sum(_tree_size(root) for root in roots),
            )
        module = importlib.import_module("jacobian")
        module_path = Path(str(module.__file__)).resolve().parent
        return ProviderInstalledSize(
            status=ProviderMeasurementStatus.COMPLETED,
            bytes=_tree_size(module_path),
        )
    except importlib.metadata.PackageNotFoundError:
        return ProviderInstalledSize(
            status=ProviderMeasurementStatus.ERROR,
            detail="The provider distribution metadata is unavailable.",
        )
    except (OSError, RuntimeError):
        return ProviderInstalledSize(
            status=ProviderMeasurementStatus.ERROR,
            detail="The provider installed-size measurement failed.",
        )


def _cold_install_spec(runtime: ProviderObservation) -> str | None:
    distribution_name = runtime.configuration.get("distribution")
    if isinstance(distribution_name, str) and runtime.version is not None:
        return f"{distribution_name}=={runtime.version}"
    return None


def _measure_cold_install(
    runtime: ProviderObservation,
    *,
    enabled: bool,
) -> ProviderMeasurementSample:
    if not enabled:
        return ProviderMeasurementSample(
            status=ProviderMeasurementStatus.SKIPPED,
            detail="Cold install was not requested.",
        )
    spec = _cold_install_spec(runtime)
    uv = shutil.which("uv")
    if spec is None or uv is None:
        return ProviderMeasurementSample(
            status=ProviderMeasurementStatus.SKIPPED,
            detail="This provider has no automated cold-install probe.",
        )
    with tempfile.TemporaryDirectory(prefix="jacobian-provider-install-") as directory:
        root = Path(directory)
        target = root / "target"
        environment = worker_environment(
            overrides={"UV_CACHE_DIR": str(root / "cache")},
        )
        started = time.perf_counter()
        try:
            result = execute_process(
                ProcessRequest(
                    executable=str(Path(uv).resolve(strict=True)),
                    arguments=(
                        "pip",
                        "install",
                        "--python",
                        sys.executable,
                        "--target",
                        str(target),
                        "--no-deps",
                        spec,
                    ),
                    environment=environment,
                    cwd=str(Path.cwd()),
                    timeout_seconds=_COLD_INSTALL_TIMEOUT_SECONDS,
                    stdin_bytes=b"",
                    stdout_limit_bytes=1024,
                    stderr_limit_bytes=_MAX_DIAGNOSTIC_BYTES,
                )
            )
        except (OSError, ValueError):
            return ProviderMeasurementSample(
                status=ProviderMeasurementStatus.ERROR,
                detail="The cold install measurement failed.",
            )
        if (
            result.termination is not ProcessTermination.EXITED
            or result.returncode != 0
        ):
            return ProviderMeasurementSample(
                status=ProviderMeasurementStatus.ERROR,
                detail="The cold install measurement failed.",
            )
        return ProviderMeasurementSample(
            status=ProviderMeasurementStatus.COMPLETED,
            seconds=time.perf_counter() - started,
            output_bytes=_tree_size(target),
        )


def measure_provider(
    runtime: ProviderObservation,
    *,
    include_cold_install: bool = False,
) -> ProviderMeasurement:
    """Measure one exact available runtime without changing the operation catalog."""

    if runtime.provider == "jacobian.lean4":
        cold_start = _lean_probe(reproduction=False)
        reproduction = _lean_probe(reproduction=True)
    else:
        cold_start = _python_probe(runtime, "cold-start")
        reproduction = _python_probe(runtime, "reproduction")
    return ProviderMeasurement(
        provider_runtime=runtime,
        installed_size=_installed_size(runtime),
        cold_install=_measure_cold_install(
            runtime,
            enabled=include_cold_install,
        ),
        cold_start=cold_start,
        reproduction_case=reproduction,
    )
