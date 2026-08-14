"""Thin CLI projection over the compiled mathematical operation catalog."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from typer import _click
from typer.core import TyperGroup

from jacobian.canonical import loads_strict_json
from jacobian.contracts.operations import (
    OperationCatalogSnapshot,
    OperationDescriptor,
    OperationRequest,
)
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.serving_catalog import ServingCatalog


class JacobianGroup(TyperGroup):
    """Translate command failures into one stable JSON error."""

    def invoke(self, ctx: _click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except (_click.ClickException, typer.Abort, typer.Exit):
            raise
        except Exception as exc:
            typer.echo(
                json.dumps(
                    {
                        "error": {
                            "code": type(exc).__name__.upper(),
                            "message": str(exc),
                        }
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                err=True,
            )
            raise typer.Exit(code=1) from None
        finally:
            state = ctx.obj
            if isinstance(state, CliState):
                active_failure = sys.exception()
                try:
                    state.close()
                except BaseException as cleanup_exc:
                    if active_failure is None:
                        raise
                    active_failure.add_note(f"CLI cleanup also failed: {cleanup_exc}")


class CliState:
    def __init__(
        self,
    ) -> None:
        self._runtime: Any | None = None
        self._catalog: ServingCatalog | None = None

    @property
    def runtime(self) -> Any:
        if self._runtime is None:
            from jacobian.runtime.execution import create_inline_serving_runtime

            self._runtime = create_inline_serving_runtime(self.catalog)
        return self._runtime

    @property
    def catalog(self) -> ServingCatalog:
        if self._catalog is None:
            self._catalog = ServingCatalog.open(policy=OperationVisibilityPolicy())
        return self._catalog

    def catalog_snapshot(self) -> OperationCatalogSnapshot:
        return self.catalog.snapshot()

    def inspect(self, operation_id: str) -> OperationDescriptor | None:
        return self.catalog.inspect(operation_id)

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None


def catalog(context: typer.Context) -> None:
    """Print the complete installed operation catalog."""

    value = _state(context).catalog_snapshot()
    _emit(value.model_dump(mode="json"))


def inspect_operation(context: typer.Context, operation_id: str) -> None:
    """Print one exact installed operation declaration."""

    descriptor = _state(context).inspect(operation_id)
    if descriptor is None:
        raise ValueError(f"operation {operation_id!r} is not installed")
    _emit(descriptor.model_dump(mode="json"))


def run_operation(
    context: typer.Context,
    operation_id: str,
    json_payload: Annotated[
        str | None,
        typer.Option("--json", help="Complete operation payload as strict JSON."),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option("--file", help="Path to a strict JSON operation payload."),
    ] = None,
) -> None:
    """Run one installed operation with one parsed JSON payload."""

    if (json_payload is None) == (file is None):
        raise ValueError("pass exactly one of --json or --file")
    if json_payload is not None:
        source = json_payload.encode("utf-8")
    else:
        assert file is not None
        source = file.read_bytes()
    payload = loads_strict_json(source)
    if not isinstance(payload, dict):
        raise ValueError("operation payload must be a JSON object")
    result = _state(context).runtime.operations.invoke(
        OperationRequest(operation_id=operation_id, input=payload)
    )
    _emit(result.model_dump(mode="json"))


def create_cli_app() -> typer.Typer:
    """Create the operator CLI over the immutable operation library."""

    application = typer.Typer(
        name="jacobian",
        cls=JacobianGroup,
        help="Run installed atomic mathematical operations.",
        no_args_is_help=True,
    )

    @application.callback()
    def configure(
        context: typer.Context,
    ) -> None:
        context.obj = CliState()

    application.command("catalog")(catalog)
    application.command("inspect")(inspect_operation)
    application.command("run")(run_operation)
    return application


app = create_cli_app()


def _state(context: typer.Context) -> CliState:
    state = context.obj
    if not isinstance(state, CliState):
        raise RuntimeError("CLI state was not initialized")
    return state


def _emit(payload: Any) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
