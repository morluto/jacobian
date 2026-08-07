"""Independent checker for pinned core and mathlib Lean certificates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jacobian.process_policy import (
    ProcessRequest,
    ProcessTermination,
    execute_process,
)
from jacobian.worker_environment import worker_environment

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

LEAN_VERSION = "4.31.0"
LEAN_COMMIT = "68218e876d2a38b1985b8590fff244a83c321783"
LEAN_TOOLCHAIN = f"leanprover/lean4:v{LEAN_VERSION}"
MATHLIB_COMMIT = "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f"
MATHLIB_AXIOMS = frozenset({"Classical.choice", "Quot.sound", "propext"})
_TOOLCHAIN_PROBE_TIMEOUT_SECONDS = 15
_MATHLIB_COMPILE_TIMEOUT_SECONDS = 180
_FORBIDDEN = re.compile(
    r"\b(?:admit|axiom|class|def|elab|end|example|import|instance|lemma|macro|"
    r"namespace|native_decide|opaque|run_tac|section|set_option|sorry|syntax|"
    r"theorem|unsafe)\b|#",
    re.IGNORECASE,
)
_AXIOMS = re.compile(r"'jacobian_theorem' depends on axioms: \[([^\]]*)\]")
_LEAN_ERROR = re.compile(
    r"^[^\r\n]+:(?P<line>\d+):(?P<column>\d+):\s*error:\s*(?P<message>.+)$",
    re.MULTILINE,
)
_QUOTED_LOCAL_PATH = re.compile(
    r"""(?:
        "(?:[A-Za-z]:[\\/]|/|~[\\/]|\\\\)[^"\r\n]*"
        |'(?:[A-Za-z]:[\\/]|/|~[\\/]|\\\\)[^'\r\n]*'
    )""",
    re.VERBOSE,
)
_UNQUOTED_LOCAL_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]|/|~[\\/]|\\\\).*$",
)
_INTERNAL_LABEL = re.compile(
    r"""\b(?:provider|internal[-_ ]?id|request[-_ ]?id)
        \s*(?:=|:)\s*(?:"[^"]*"|'[^']*'|[^\s,;]+)""",
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class _LeanRunResult:
    """Decoded output from one bounded Lean compiler invocation."""

    stdout: str
    stderr: str
    returncode: int


class _LeanSetupError(RuntimeError):
    """A locally actionable Lean setup failure safe to return to the caller."""


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "SYMBOLIC",
        "method": "CHECKED_CERTIFICATE",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _lean_rejection(diagnostics: str) -> str:
    match = _LEAN_ERROR.search(diagnostics)
    if match is None:
        return (
            "Lean rejected the proof. Check the statement and proof body, then retry."
        )
    message = _QUOTED_LOCAL_PATH.sub("<local-path>", match.group("message"))
    message = _UNQUOTED_LOCAL_PATH.sub("<local-path>", message)
    message = " ".join(_INTERNAL_LABEL.sub("", message).split())
    if not message:
        return (
            "Lean rejected the proof. Check the statement and proof body, then retry."
        )
    return (
        f"Lean rejected the proof at line {match.group('line')}, column "
        f"{match.group('column')}: {message[:400]}. Correct the proof body and retry."
    )


def _text(value: object, *, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > limit or "\x00" in value:
        raise ValueError(f"{name} exceeds its accepted source boundary")
    if _FORBIDDEN.search(value):
        raise ValueError(f"{name} contains a forbidden Lean command")
    return value


def _source(statement: str, proof: str, import_name: str | None) -> str:
    if "\n" in statement or "\r" in statement:
        raise ValueError("statement must be one Lean expression")
    if any(marker in statement for marker in ("--", "/-", "-/")):
        raise ValueError(
            "statement comments are outside the single-expression boundary"
        )
    proof_lines = proof.splitlines()
    complete_proof_term = re.match(r"^by(?:\s|$)", proof.lstrip()) is not None
    theorem = (
        f"theorem jacobian_theorem : ({statement}) := {proof}"
        if complete_proof_term
        else "\n".join(
            (
                f"theorem jacobian_theorem : ({statement}) := by",
                "\n".join(f"  {line}" for line in proof_lines),
            )
        )
    )
    lines = [
        *([f"import {import_name}"] if import_name is not None else []),
        *(
            (
                "set_option autoImplicit false",
                "set_option warningAsError true",
                theorem,
                "#print axioms jacobian_theorem",
                "",
            )
        ),
    ]
    return "\n".join(lines)


def _authorized_lean_runtime() -> Path:
    executable = os.environ.get("JACOBIAN_CHECKER_EXECUTABLE")
    expected_digest = os.environ.get("JACOBIAN_CHECKER_RUNTIME_DIGEST")
    if (
        executable is None
        or expected_digest is None
        or _DIGEST.fullmatch(expected_digest) is None
    ):
        raise _LeanSetupError("TOOLCHAIN_RESOLUTION: Lean runtime is not authorized")
    path = Path(executable).resolve(strict=True)
    if str(path) != executable or not path.is_file() or path.is_symlink():
        raise _LeanSetupError("TOOLCHAIN_RESOLUTION: Lean runtime path is not exact")
    actual_digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_digest != expected_digest:
        raise _LeanSetupError("TOOLCHAIN_RESOLUTION: Lean runtime digest changed")
    return path


def _toolchain_sibling(name: str, lean_executable: Path) -> Path:
    """Return ``name`` from the same toolchain bin directory as *lean_executable*."""

    candidate = lean_executable.with_name(
        f"{name}.exe" if lean_executable.suffix.lower() == ".exe" else name
    )
    if not candidate.is_file():
        raise _LeanSetupError(
            f"TOOLCHAIN_RESOLUTION: The pinned Lean {LEAN_VERSION} {name} "
            "launcher is unavailable. Install "
            f"it with `elan toolchain install {LEAN_TOOLCHAIN}`, then retry."
        )
    return candidate


def _lean_command(name: str) -> tuple[str, ...]:
    if os.environ.get("JACOBIAN_CHECKER_EXECUTABLE") is not None:
        lean = _authorized_lean_runtime()
        if name == "lean":
            return (str(lean),)
        return (str(_toolchain_sibling(name, lean)),)
    elan = shutil.which("elan")
    if elan is not None:
        return (elan, "run", LEAN_TOOLCHAIN, name)
    launcher = shutil.which(name)
    if launcher is None:
        raise _LeanSetupError(
            f"TOOLCHAIN_RESOLUTION: The pinned Lean {LEAN_VERSION} {name} "
            "launcher is unavailable. Install "
            f"it with `elan toolchain install {LEAN_TOOLCHAIN}`, then retry."
        )
    return (launcher,)


def _validate_lean(
    command: tuple[str, ...],
    *,
    cwd: Path | None = None,
) -> None:
    elan_home = _elan_home(command)
    overrides: dict[str, str] = {}
    if elan_home is not None:
        overrides["ELAN_HOME"] = elan_home
    probe_environment = worker_environment(overrides=overrides)
    probe_cwd = str(cwd) if cwd is not None else str(Path.cwd())
    try:
        version_result = execute_process(
            ProcessRequest(
                executable=command[0],
                arguments=(*command[1:], "-V"),
                environment=probe_environment,
                cwd=probe_cwd,
                timeout_seconds=float(_TOOLCHAIN_PROBE_TIMEOUT_SECONDS),
                stdin_bytes=b"",
                stdout_limit_bytes=4096,
                stderr_limit_bytes=4096,
            )
        )
        commit_result = execute_process(
            ProcessRequest(
                executable=command[0],
                arguments=(*command[1:], "-g"),
                environment=probe_environment,
                cwd=probe_cwd,
                timeout_seconds=float(_TOOLCHAIN_PROBE_TIMEOUT_SECONDS),
                stdin_bytes=b"",
                stdout_limit_bytes=4096,
                stderr_limit_bytes=4096,
            )
        )
    except OSError as exc:
        raise _LeanSetupError(
            f"TOOLCHAIN_PROBE: The pinned Lean {LEAN_VERSION} toolchain is "
            "unavailable. Install "
            f"it with `elan toolchain install {LEAN_TOOLCHAIN}`, then retry."
        ) from exc
    if (
        version_result.termination is not ProcessTermination.EXITED
        or commit_result.termination is not ProcessTermination.EXITED
        or version_result.returncode != 0
        or commit_result.returncode != 0
    ):
        raise _LeanSetupError(
            f"TOOLCHAIN_PROBE: The pinned Lean {LEAN_VERSION} toolchain is "
            "unavailable. Install "
            f"it with `elan toolchain install {LEAN_TOOLCHAIN}`, then retry."
        )
    version = version_result.stdout.decode("utf-8", errors="replace").strip()
    commit = commit_result.stdout.decode("utf-8", errors="replace").strip()
    if version != LEAN_VERSION or commit != LEAN_COMMIT:
        raise _LeanSetupError(
            f"TOOLCHAIN_PROBE: The installed Lean toolchain does not match "
            "Jacobian's pinned "
            f"{LEAN_VERSION} release. Reinstall it with `elan toolchain install "
            f"{LEAN_TOOLCHAIN}`, then retry."
        )


def _elan_home(command: tuple[str, ...]) -> str | None:
    configured = os.environ.get("ELAN_HOME")
    if configured is not None or len(command) == 1:
        return configured
    original_home = os.environ.get("HOME")
    return str(Path(original_home) / ".elan") if original_home is not None else None


def _validate_package_checkout(
    packages_directory: Path,
    package: dict[str, Any],
) -> None:
    name = package.get("name")
    revision = package.get("rev")
    if (
        package.get("type") != "git"
        or not isinstance(name, str)
        or not name
        or Path(name).name != name
        or not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
    ):
        raise RuntimeError("the mathlib manifest contains an invalid package")
    checkout = packages_directory / name
    git_environment = worker_environment(locale="C")
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("git is unavailable for package checkout validation")
    rev_result = execute_process(
        ProcessRequest(
            executable=git_executable,
            arguments=("-C", str(checkout), "rev-parse", "HEAD"),
            environment=git_environment,
            cwd=str(checkout),
            timeout_seconds=5.0,
            stdin_bytes=b"",
            stdout_limit_bytes=4096,
            stderr_limit_bytes=4096,
        )
    )
    if (
        rev_result.termination is not ProcessTermination.EXITED
        or rev_result.returncode != 0
    ):
        raise RuntimeError(f"the installed {name} commit could not be verified")
    actual_revision = rev_result.stdout.decode("utf-8", errors="replace").strip()
    if actual_revision != revision:
        raise RuntimeError(f"the installed {name} commit is not authorized")
    status_result = execute_process(
        ProcessRequest(
            executable=git_executable,
            arguments=(
                "-C",
                str(checkout),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ),
            environment=git_environment,
            cwd=str(checkout),
            timeout_seconds=5.0,
            stdin_bytes=b"",
            stdout_limit_bytes=4096,
            stderr_limit_bytes=4096,
        )
    )
    if (
        status_result.termination is not ProcessTermination.EXITED
        or status_result.returncode != 0
    ):
        raise RuntimeError(f"the installed {name} source could not be checked")
    tracked_changes = status_result.stdout.decode("utf-8", errors="replace").strip()
    if tracked_changes:
        raise RuntimeError(f"the installed {name} source has tracked modifications")


def _mathlib_runtime() -> Path:
    configured = os.environ.get("JACOBIAN_LEAN_RUNTIME")
    runtime = (
        Path(configured)
        if configured is not None
        else Path(__file__).resolve().parents[2] / "lean"
    )
    manifest_path = runtime / "lake-manifest.json"
    toolchain_path = runtime / "lean-toolchain"
    if not manifest_path.is_file() or not toolchain_path.is_file():
        raise _LeanSetupError(
            "MATHLIB_MANIFEST: the pinned mathlib runtime is unavailable"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _LeanSetupError(
            "MATHLIB_MANIFEST: the pinned mathlib manifest could not be read"
        ) from exc
    packages = manifest.get("packages")
    if manifest.get("packagesDir") != ".lake/packages" or not isinstance(
        packages, list
    ):
        raise _LeanSetupError("MATHLIB_MANIFEST: the mathlib manifest is malformed")
    revisions = {
        package.get("name"): package.get("rev")
        for package in packages
        if isinstance(package, dict)
    }
    if revisions.get("mathlib") != MATHLIB_COMMIT:
        raise _LeanSetupError(
            "MATHLIB_MANIFEST: the installed mathlib commit is not authorized"
        )
    packages_directory = runtime / ".lake" / "packages"
    for package in packages:
        if not isinstance(package, dict):
            raise _LeanSetupError(
                "MATHLIB_MANIFEST: the mathlib manifest contains an invalid package"
            )
        try:
            _validate_package_checkout(packages_directory, package)
        except (OSError, RuntimeError) as exc:
            raise _LeanSetupError(
                "MATHLIB_MANIFEST: a pinned mathlib package checkout failed "
                "integrity validation"
            ) from exc
    if (
        toolchain_path.read_text(encoding="utf-8").strip()
        != f"leanprover/lean4:v{LEAN_VERSION}"
    ):
        raise _LeanSetupError(
            "MATHLIB_MANIFEST: the mathlib runtime requests another Lean toolchain"
        )
    return runtime


def _resolve_elan_toolchain_executable(command: tuple[str, ...]) -> Path:
    """Resolve the pinned toolchain ``lean`` binary, not the elan multi-call proxy.

    ``elan run <toolchain> which lean`` can return ``$ELAN_HOME/bin/lean``, which is
    the elan proxy. That proxy requires a default toolchain and breaks under the
    isolated ``HOME`` used by CORE checkers and declaration/elaboration workers.
    ``lean --print-prefix`` returns the toolchain root that contains the real
    ``bin/lean`` launcher.
    """

    elan_home = _elan_home(command)
    overrides: dict[str, str] = {"ELAN_TOOLCHAIN": LEAN_TOOLCHAIN}
    if elan_home is not None:
        overrides["ELAN_HOME"] = elan_home
    environment = worker_environment(
        extra_variables=("PATH", "HOME"),
        overrides=overrides,
    )
    try:
        prefix_result = execute_process(
            ProcessRequest(
                executable=command[0],
                arguments=(*command[1:], "--print-prefix"),
                environment=environment,
                cwd=str(Path.cwd()),
                timeout_seconds=5.0,
                stdin_bytes=b"",
                stdout_limit_bytes=4096,
                stderr_limit_bytes=4096,
            )
        )
    except OSError as exc:
        raise _LeanSetupError(
            f"The pinned Lean {LEAN_VERSION} executable could not be resolved."
        ) from exc
    if (
        prefix_result.termination is not ProcessTermination.EXITED
        or prefix_result.returncode != 0
    ):
        raise _LeanSetupError(
            f"The pinned Lean {LEAN_VERSION} executable could not be resolved."
        )
    prefix = prefix_result.stdout.decode("utf-8", errors="replace").strip()
    if not prefix or "\n" in prefix or "\x00" in prefix:
        raise _LeanSetupError(
            f"The pinned Lean {LEAN_VERSION} executable could not be resolved."
        )
    executable = Path(prefix) / "bin" / "lean"
    if not executable.is_file():
        raise _LeanSetupError(
            f"The pinned Lean {LEAN_VERSION} executable is unavailable."
        )
    elan = shutil.which("elan")
    if elan is not None:
        try:
            if executable.samefile(elan):
                raise _LeanSetupError(
                    f"The pinned Lean {LEAN_VERSION} executable resolved to the "
                    "elan proxy rather than the toolchain binary."
                )
        except OSError as exc:
            raise _LeanSetupError(
                f"The pinned Lean {LEAN_VERSION} executable could not be resolved."
            ) from exc
    return executable


def inspect_runtime(*, require_mathlib: bool) -> tuple[Path, Path | None]:
    """Validate the pinned runtime and return the exact executable and profile root."""

    command = _lean_command("lean")
    _validate_lean(command)
    if len(command) == 1:
        executable = Path(command[0])
    else:
        executable = _resolve_elan_toolchain_executable(command)
    if not executable.is_file():
        raise _LeanSetupError(
            f"The pinned Lean {LEAN_VERSION} executable is unavailable."
        )
    mathlib_runtime = _mathlib_runtime() if require_mathlib else None
    return executable, mathlib_runtime


def _run_lean(
    source: str,
    *,
    environment_name: str,
) -> _LeanRunResult:
    if environment_name == "CORE":
        command = list(_lean_command("lean"))
        _validate_lean(tuple(command))
        memory_mb = "1024"
        timeout_seconds = 25
        cwd_context = tempfile.TemporaryDirectory(prefix="jacobian-lean-")
        cwd = Path(cwd_context.name)
        runtime_home = cwd_context.name
    elif environment_name == "MATHLIB":
        runtime = _mathlib_runtime()
        lake_command = _lean_command("lake")
        command = [*lake_command, "env", "lean"]
        _validate_lean(tuple(command), cwd=runtime)
        memory_mb = "8192"
        timeout_seconds = _MATHLIB_COMPILE_TIMEOUT_SECONDS
        cwd_context = tempfile.TemporaryDirectory(prefix="jacobian-lean-home-")
        cwd = runtime
        runtime_home = os.environ.get("HOME", cwd_context.name)
    else:
        raise ValueError("unknown Lean environment")
    elan_home = _elan_home(tuple(command))
    process_environment = worker_environment(
        overrides={
            "HOME": runtime_home,
            "PATH": (
                os.environ.get("PATH", str(Path(command[0]).parent))
                if environment_name == "MATHLIB"
                else str(Path(command[0]).parent)
            ),
            **({"ELAN_HOME": elan_home} if elan_home is not None else {}),
        },
    )
    with cwd_context:
        result = execute_process(
            ProcessRequest(
                executable=command[0],
                arguments=(
                    *command[1:],
                    "--stdin",
                    "-t",
                    "0",
                    "-T",
                    "1000000000",
                    "-M",
                    memory_mb,
                    "-j",
                    "1",
                    "--trust=0",
                ),
                environment=process_environment,
                cwd=str(cwd),
                timeout_seconds=float(timeout_seconds),
                stdin_bytes=source.encode("utf-8"),
                stdout_limit_bytes=64 * 1024,
                stderr_limit_bytes=64 * 1024,
            )
        )
    if result.termination is ProcessTermination.TIMED_OUT:
        stage = (
            "LEAN_COMPILE_TIMEOUT"
            if environment_name == "MATHLIB"
            else "LEAN_CORE_TIMEOUT"
        )
        raise _LeanSetupError(
            f"{stage}: Lean exceeded its {timeout_seconds}-second compile "
            "budget; no proof conclusion follows"
        )
    return _LeanRunResult(
        stdout=result.stdout.decode("utf-8", errors="replace"),
        stderr=result.stderr.decode("utf-8", errors="replace"),
        returncode=result.returncode if result.returncode is not None else -1,
    )


def _reported_axioms(diagnostics: str) -> frozenset[str]:
    if "'jacobian_theorem' does not depend on any axioms" in diagnostics:
        return frozenset()
    match = _AXIOMS.search(diagnostics)
    if match is None:
        raise ValueError("Lean did not report the theorem trust base")
    return frozenset(item.strip() for item in match.group(1).split(",") if item.strip())


def _profile(
    environment_name: object,
) -> tuple[str | None, str | None, frozenset[str]]:
    if environment_name == "CORE":
        return None, None, frozenset()
    if environment_name == "MATHLIB":
        return "Mathlib", MATHLIB_COMMIT, MATHLIB_AXIOMS
    raise ValueError("unknown Lean environment")


def check_kernel_certificate(request: dict[str, Any]) -> dict[str, Any]:
    """Compile the exact bound proposition under its authorized trust profile."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        certificate = request["certificate"]["payload"]
        if certificate.get("certificate_type") != "lean4.kernel":
            return _reject("unexpected certificate format")
        if certificate.get("format_version") != "1":
            return _reject("unsupported certificate format version")
        if certificate.get("bindings") != request["expected_bindings"]:
            return _reject("certificate bindings do not match the request")
        payload = certificate["payload"]
        claim = request["claim"]["payload"]
        candidate = request["candidate"]["payload"]
        environment_name = payload.get("environment")
        import_name, mathlib_commit, allowed_axioms = _profile(environment_name)
        statement = _text(payload.get("statement"), name="statement", limit=2_000)
        proof = _text(payload.get("proof"), name="proof", limit=20_000)
        if (
            claim.get("statement") != statement
            or candidate.get("statement") != statement
            or candidate.get("proof") != proof
        ):
            return _reject("claim, candidate, and certificate source differ")
        if (
            claim.get("environment") != environment_name
            or candidate.get("environment") != environment_name
        ):
            return _reject("claim, candidate, and certificate profiles differ")
        expected_axioms = sorted(allowed_axioms)
        if (
            sorted(claim.get("allowed_axioms", [])) != expected_axioms
            or sorted(payload.get("allowed_axioms", [])) != expected_axioms
        ):
            return _reject("certificate requests an unauthorized Lean trust base")
        if payload.get("declaration_name") != "jacobian_theorem":
            return _reject("unexpected Lean declaration name")
        if (
            payload.get("lean_version") != LEAN_VERSION
            or payload.get("lean_commit") != LEAN_COMMIT
            or payload.get("import_name") != import_name
            or payload.get("mathlib_commit") != mathlib_commit
        ):
            return _reject("certificate requests another Lean runtime")
        completed = _run_lean(
            _source(statement, proof, import_name),
            environment_name=environment_name,
        )
        diagnostics = (completed.stdout + completed.stderr).strip()
        if completed.returncode != 0:
            return _reject(_lean_rejection(diagnostics))
        reported_axioms = _reported_axioms(diagnostics)
        if not reported_axioms.issubset(allowed_axioms):
            return _reject("Lean proof has an unapproved trust base")
        trust_base = (
            ", ".join(sorted(reported_axioms)) if reported_axioms else "no axioms"
        )
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "SYMBOLIC",
            "method": "CHECKED_CERTIFICATE",
            "coverage": "NOT_APPLICABLE",
            "detail": (
                f"Lean {LEAN_VERSION} kernel accepted the exact proposition "
                f"under {environment_name} with {trust_base}"
            ),
        }
    except ValueError as exc:
        return _reject(str(exc))
    except (KeyError, TypeError):
        return _reject(
            "The Lean certificate is incomplete or invalid. Recreate it from the "
            "statement and proof body, then retry."
        )
    except _LeanSetupError as exc:
        return _reject(str(exc))
    except OSError:
        return _reject(
            f"Lean {LEAN_VERSION} could not run locally. Confirm the pinned "
            f"toolchain with `elan run {LEAN_TOOLCHAIN} lean -V`, then retry."
        )
