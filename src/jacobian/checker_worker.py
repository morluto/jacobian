"""Minimal subprocess dispatcher for operator-authorized checker entrypoints."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    loads_strict_json,
)
from jacobian.checker_identity import (
    UndeclaredCheckerImportError,
    checker_implementation_digest,
    install_manifest_import_guard,
    require_manifest_material_unchanged,
    require_manifest_unchanged,
)
from jacobian.contracts.checkers import CheckerManifest
from jacobian.contracts.operations import (
    ProviderDigestKind,
    ProviderObservation,
)
from jacobian.provider_runtime import (
    ProviderRuntimeError,
    ProviderRuntimeErrorCode,
    require_provider_runtime_unchanged,
)
from jacobian.verification.checker_protocol import (
    CheckerWorkerErrorCode,
    CheckerWorkerFailure,
    CheckerWorkerSuccess,
)


class _CheckerWorkerFailureError(ValueError):
    """A bounded failure classification for checker-owned runtime parsing."""

    def __init__(self, code: CheckerWorkerErrorCode) -> None:
        self.code = code
        super().__init__(code)


def _resolve(entrypoint: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    module_name, separator, function_name = entrypoint.partition(":")
    if not separator:
        raise ValueError("checker entrypoint must be module:function")
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError("checker entrypoint is not callable")
    return cast(Callable[[dict[str, Any]], dict[str, Any]], function)


def _measure_runtime(
    runtime: ProviderObservation | None,
) -> tuple[ProviderObservation | None, str | None]:
    os.environ.pop("JACOBIAN_CHECKER_EXECUTABLE", None)
    os.environ.pop("JACOBIAN_CHECKER_RUNTIME_DIGEST", None)
    os.environ.pop("JACOBIAN_CHECKER_LAKE_DIGEST", None)
    os.environ.pop("JACOBIAN_CHECKER_LEAN_PROJECT_ROOT", None)
    if runtime is None:
        return None, None
    try:
        require_provider_runtime_unchanged(runtime)
    except ProviderRuntimeError as exc:
        code: CheckerWorkerErrorCode = (
            "MALFORMED_RUNTIME"
            if exc.code is ProviderRuntimeErrorCode.MALFORMED_RUNTIME
            else "EXECUTION_FAILED"
        )
        raise _CheckerWorkerFailureError(code) from exc
    except (OSError, ValueError) as exc:
        raise _CheckerWorkerFailureError("MALFORMED_RUNTIME") from exc
    if runtime.digest_kind is ProviderDigestKind.EXECUTABLE:
        executable = runtime.configuration.get("executable")
        if not isinstance(executable, str):
            raise ValueError("checker executable identity is incomplete")
        path = Path(executable).resolve(strict=True)
        if str(path) != executable:
            raise ValueError("checker runtime path is not exact")
        os.environ["JACOBIAN_CHECKER_EXECUTABLE"] = str(path)
        _bind_lake_launcher(runtime)
    _bind_lean_semantic_environment(runtime)
    if runtime.digest is None:
        raise RuntimeError("runtime digest is unexpectedly None")
    os.environ["JACOBIAN_CHECKER_RUNTIME_DIGEST"] = runtime.digest
    return runtime, runtime.digest


def _bind_lean_semantic_environment(runtime: ProviderObservation) -> None:
    if runtime.provider != "jacobian.lean4":
        return
    semantic_runtime = runtime.configuration.get("semantic_runtime")
    if semantic_runtime is None:
        return
    if not isinstance(semantic_runtime, dict):
        raise _CheckerWorkerFailureError("MALFORMED_RUNTIME")
    lean_runtime = importlib.import_module("jacobian.providers.lean_runtime")
    try:
        lean_runtime.require_lean_semantic_runtime_identity(runtime)
    except lean_runtime.LeanRuntimeIdentityError as exc:
        raise _CheckerWorkerFailureError("EXECUTION_FAILED") from exc
    project = semantic_runtime.get("mathlib_project")
    if isinstance(project, dict) and isinstance(project.get("root"), str):
        os.environ["JACOBIAN_CHECKER_LEAN_PROJECT_ROOT"] = project["root"]


def _bind_lake_launcher(runtime: ProviderObservation) -> None:
    """Bind an optional sibling Lake launcher into the measured runtime identity.

    The Lean runtime carries the Lake launcher path and digest in its
    configuration only when a MATHLIB profile is in scope.  Re-measure the
    on-disk launcher here so a Lake binary swapped after provider inspection
    is rejected before the checker executes, and forward the digest to the
    checker process so ``lake`` is never run as an unauthenticated sibling.
    """
    lake_executable = runtime.configuration.get("lake_executable")
    lake_digest = runtime.configuration.get("lake_digest")
    if lake_executable is None and lake_digest is None:
        return
    if not isinstance(lake_executable, str) or not isinstance(lake_digest, str):
        raise ValueError("checker lake launcher identity is incomplete")
    path = Path(lake_executable).resolve(strict=True)
    if str(path) != lake_executable or not path.is_file() or path.is_symlink():
        raise ValueError("checker lake launcher path is not exact")
    measured = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if measured != lake_digest:
        raise ValueError("checker lake launcher digest changed")
    os.environ["JACOBIAN_CHECKER_LAKE_DIGEST"] = lake_digest


def _execute(manifest_json: str, request_bytes: bytes) -> CheckerWorkerSuccess:
    """Run one manifest-bound checker after validating its complete identity."""

    try:
        manifest = CheckerManifest.model_validate(loads_strict_json(manifest_json))
    except (CanonicalizationError, ValidationError) as exc:
        raise _CheckerWorkerFailureError("MALFORMED_RUNTIME") from exc
    try:
        request = loads_strict_json(request_bytes)
    except CanonicalizationError as exc:
        raise _CheckerWorkerFailureError("INVALID_REQUEST") from exc
    if not isinstance(request, dict):
        raise _CheckerWorkerFailureError("INVALID_REQUEST")
    measured_before = require_manifest_unchanged(manifest)
    if measured_before != checker_implementation_digest(manifest):
        raise _CheckerWorkerFailureError("SOURCE_CHANGED")
    runtime, runtime_digest_before = _measure_runtime(manifest.provider_runtime)
    install_manifest_import_guard(manifest)
    with contextlib.redirect_stdout(sys.stderr):
        checker = _resolve(manifest.entrypoint)
        response = checker(request)
    measured_after = require_manifest_material_unchanged(manifest)
    if measured_after != measured_before:
        raise _CheckerWorkerFailureError("SOURCE_CHANGED")
    _, runtime_digest_after = _measure_runtime(runtime)
    if runtime_digest_after != runtime_digest_before:
        raise _CheckerWorkerFailureError("SOURCE_CHANGED")
    try:
        return CheckerWorkerSuccess.model_validate(
            {
                "decision": response,
                "measured_implementation_digest": measured_after,
                "measured_runtime_digest": runtime_digest_after,
            }
        )
    except ValidationError as exc:
        raise _CheckerWorkerFailureError("RESPONSE_INVALID") from exc


def _write_response(response: CheckerWorkerSuccess | CheckerWorkerFailure) -> int:
    sys.stdout.buffer.write(canonicalize_json(response.model_dump(mode="json")))
    sys.stdout.buffer.write(b"\n")
    return 0 if isinstance(response, CheckerWorkerSuccess) else 1


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: python -m jacobian.checker_worker checker-manifest-json",
            file=sys.stderr,
        )
        return 2
    try:
        return _write_response(_execute(sys.argv[1], sys.stdin.buffer.read()))
    except _CheckerWorkerFailureError as exc:
        return _write_response(CheckerWorkerFailure(error_code=exc.code))
    except UndeclaredCheckerImportError as exc:
        print(str(exc), file=sys.stderr)
        return _write_response(CheckerWorkerFailure(error_code="UNDECLARED_IMPORT"))
    except Exception:  # checker isolation turns all failures into ERROR
        return _write_response(CheckerWorkerFailure(error_code="EXECUTION_FAILED"))


if __name__ == "__main__":
    raise SystemExit(main())
