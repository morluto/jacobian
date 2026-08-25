"""Synthetic process-backed server used only by MCP cancellation tests."""

from __future__ import annotations

import os
import sys

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from jacobian.process import run_bounded_process


class ProcessRequest(StrictModel):
    marker: str = Field(min_length=1, max_length=4_096)


class ProcessResult(StrictModel):
    cancelled: bool


def _run_process(request: ProcessRequest) -> ProcessResult:
    completed = run_bounded_process(
        [
            sys.executable,
            "-c",
            (
                "import json, os, pathlib, signal, subprocess, sys, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "child=subprocess.Popen([sys.executable, '-c', "
                "'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)']); "
                "pathlib.Path(sys.argv[1]).write_text(json.dumps([os.getpid(), child.pid])); "
                "time.sleep(30)"
            ),
            request.marker,
        ],
        input_bytes=b"",
        timeout_seconds=20,
        environment=dict(os.environ),
        stdout_limit=4_096,
        stderr_limit=4_096,
    )
    return ProcessResult(cancelled=completed.cancelled)


def main() -> None:
    operation = MathTool(
        operation_id="test.process.wait",
        title="Wait in an owned process tree",
        description="Synthetic bounded process used only by MCP regression tests.",
        request_type=ProcessRequest,
        result_type=ProcessResult,
        run=_run_process,
    )
    _build_server(
        state=AppState(operation_catalog=Catalog((*BUILTIN_TOOLS, operation)))
    ).run("stdio")


if __name__ == "__main__":
    main()
