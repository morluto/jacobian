"""Thin CLI projection over the compiled mathematical operation catalog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from typer import _click
from typer.core import TyperGroup

from jacobian.canonical import loads_strict_json
from jacobian.serving_catalog import ServingCatalog


class JacobianGroup(TyperGroup):
    """Translate command failures into one stable JSON error."""

    def invoke(self, ctx: _click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except (_click.ClickException, typer.Abort, typer.Exit):
            raise
        except Exception as exc:
            code = (
                "INVALID_ARGUMENT" if isinstance(exc, ValueError) else "COMMAND_FAILED"
            )
            typer.echo(
                json.dumps(
                    {
                        "error": {
                            "code": code,
                            "message": str(exc),
                        }
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                err=True,
            )
            raise typer.Exit(code=1) from None


def catalog() -> None:
    """Print the complete installed operation catalog."""

    value = ServingCatalog.open().snapshot()
    _emit(value.model_dump(mode="json"))


def inspect_operation(operation_id: str) -> None:
    """Print one exact installed operation declaration."""

    descriptor = ServingCatalog.open().inspect(operation_id)
    if descriptor is None:
        raise ValueError(f"operation {operation_id!r} is not installed")
    _emit(descriptor.model_dump(mode="json"))


def run_operation(
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
    elif file is not None:
        source = file.read_bytes()
    else:
        raise ValueError("pass exactly one of --json or --file")
    payload = loads_strict_json(source)
    if not isinstance(payload, dict):
        raise ValueError("operation payload must be a JSON object")
    from jacobian.operation_dispatcher import invoke_operation

    result = invoke_operation(
        operation_id,
        payload,
        ServingCatalog.open(),
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

    application.command("catalog")(catalog)
    application.command("inspect")(inspect_operation)
    application.command("run")(run_operation)
    return application


app = create_cli_app()


def _emit(payload: Any) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
