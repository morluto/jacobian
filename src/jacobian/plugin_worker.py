"""Subprocess dispatcher for operator-installed, untrusted plugin code."""

from __future__ import annotations

import contextlib
import importlib
import sys
from collections.abc import Callable
from typing import Any, cast

from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    loads_strict_json,
)
from jacobian.implementation import (
    install_source_only_importer,
    package_source_digest,
)
from jacobian.plugin_protocol import (
    PluginWorkerContractFailure,
    PluginWorkerFailure,
    PluginWorkerOperationalFailure,
    PluginWorkerSuccess,
)


def _resolve(entrypoint: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    module_name, separator, function_name = entrypoint.partition(":")
    if not separator:
        raise ValueError("plugin entrypoint must be module:function")
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError("plugin entrypoint is not callable")
    return cast(Callable[[dict[str, Any]], dict[str, Any]], function)


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: python -m jacobian.plugin_worker module:function expected-digest",
            file=sys.stderr,
        )
        return 2
    failure: PluginWorkerFailure = PluginWorkerOperationalFailure(
        error_code="EXECUTION_FAILED"
    )
    request_decoded = False
    try:
        request = loads_strict_json(sys.stdin.buffer.read())
        request_decoded = True
        if not isinstance(request, dict):
            failure = PluginWorkerContractFailure(
                error_code="INVALID_REQUEST",
                path="/",
                expected="object",
                actual_type=type(request).__name__,
            )
            raise TypeError("plugin request must be a JSON object")
        measured_before = package_source_digest(sys.argv[1])
        if measured_before != sys.argv[2]:
            failure = PluginWorkerOperationalFailure(error_code="SOURCE_CHANGED")
            raise ValueError("plugin source differs from its resolved digest")
        install_source_only_importer(sys.argv[1])
        with contextlib.redirect_stdout(sys.stderr):
            operation = _resolve(sys.argv[1])
            response = operation(request)
        if not isinstance(response, dict):
            failure = PluginWorkerContractFailure(
                error_code="RESPONSE_INVALID",
                path="/response",
                expected="object",
                actual_type=type(response).__name__,
            )
            raise TypeError("plugin response must be a JSON object")
        measured_after = package_source_digest(sys.argv[1])
        if measured_after != measured_before:
            failure = PluginWorkerOperationalFailure(error_code="SOURCE_CHANGED")
            raise ValueError("plugin source changed during execution")
        success = PluginWorkerSuccess(
            response=response,
            measured_implementation_digest=measured_after,
        )
        sys.stdout.buffer.write(canonicalize_json(success.model_dump(mode="json")))
        sys.stdout.buffer.write(b"\n")
        return 0
    except CanonicalizationError:
        failure = PluginWorkerContractFailure(
            error_code="RESPONSE_INVALID" if request_decoded else "INVALID_REQUEST",
            path="/response" if request_decoded else "/",
            expected="canonical JSON object",
            actual_type="response" if request_decoded else "bytes",
        )
        sys.stdout.buffer.write(canonicalize_json(failure.model_dump(mode="json")))
        sys.stdout.buffer.write(b"\n")
        return 1
    except Exception:  # untrusted failures are operational, never logical
        sys.stdout.buffer.write(canonicalize_json(failure.model_dump(mode="json")))
        sys.stdout.buffer.write(b"\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
