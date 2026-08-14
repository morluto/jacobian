"""Thin CLI projection over the compiled mathematical operation catalog."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from typer import _click
from typer.core import TyperGroup

from jacobian.canonical import loads_strict_json
from jacobian.contracts.operations import (
    OperationCatalogSnapshot,
    OperationDescriptor,
    OperationRequest,
)
from jacobian.operation_catalog import OperationCatalog
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.operator_lifecycle import (
    CheckerAuthorization,
    active_catalog_revision,
    initialize_state,
    update_state,
)

if TYPE_CHECKING:
    from jacobian.runtime.model import JacobianRuntime

RuntimeOpener = Callable[..., Any]


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
        state_dir: Path,
        *,
        runtime_opener: RuntimeOpener | None = None,
    ) -> None:
        self.state_dir = state_dir
        self._runtime_opener = runtime_opener
        self._runtime: JacobianRuntime | None = None

    @property
    def runtime(self) -> JacobianRuntime:
        if self._runtime is None:
            opener = self._runtime_opener
            if opener is None:
                from jacobian.runtime.execution import create_execution_runtime

                self._runtime = create_execution_runtime(
                    self.state_dir,
                    self.catalog,
                    operation_policy=OperationVisibilityPolicy(),
                )
            else:
                self._runtime = opener(
                    self.state_dir,
                )
        return self._runtime

    @property
    def catalog(self) -> OperationCatalog:
        from jacobian import __version__

        return OperationCatalog(
            self.state_dir / "metadata.sqlite3",
            OperationVisibilityPolicy(),
            expected_package_version=__version__,
        )

    def catalog_snapshot(self) -> OperationCatalogSnapshot:
        if self._runtime_opener is not None:
            return self.runtime.core.operations.snapshot()
        return self.catalog.snapshot()

    def inspect(self, operation_id: str) -> OperationDescriptor | None:
        if self._runtime_opener is not None:
            return self.runtime.core.operations.inspect(operation_id)
        return self.catalog.inspect(operation_id)

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None


def initialize(
    context: typer.Context,
    checker_authorization: Annotated[
        CheckerAuthorization,
        typer.Option("--checker-authorization"),
    ] = CheckerAuthorization.BUNDLED,
) -> None:
    """Initialize storage and report the installed operation count."""

    state = _state(context)
    previous_revision = active_catalog_revision(state.state_dir)
    result = initialize_state(
        state.state_dir,
        checker_authorization=checker_authorization,
    )
    if previous_revision == result.revision:
        typer.echo(f"Jacobian state is already current in {state.state_dir.resolve()}")
        typer.echo(
            f"Catalog contains {result.operation_count} mathematical operations."
        )
    else:
        typer.echo(f"Initialized Jacobian state in {state.state_dir.resolve()}")
        typer.echo(f"Compiled {result.operation_count} mathematical operations.")


def update(
    context: typer.Context,
    checker_authorization: Annotated[
        CheckerAuthorization,
        typer.Option("--checker-authorization"),
    ] = CheckerAuthorization.BUNDLED,
) -> None:
    """Migrate existing state and atomically select a fresh operation catalog."""

    state = _state(context)
    result = update_state(
        state.state_dir,
        checker_authorization=checker_authorization,
    )
    typer.echo(f"Updated Jacobian state in {state.state_dir.resolve()}")
    typer.echo(f"Compiled {result.operation_count} mathematical operations.")
    typer.echo("Restart running Jacobian servers to load the new catalog revision.")


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
    result = _state(context).runtime.core.operations.invoke(
        OperationRequest(operation_id=operation_id, input=payload)
    )
    _emit(result.model_dump(mode="json"))


def create_cli_app(*, runtime_opener: RuntimeOpener | None = None) -> typer.Typer:
    """Create the operator CLI, optionally over a caller-owned runtime opener."""

    application = typer.Typer(
        name="jacobian",
        cls=JacobianGroup,
        help="Run installed atomic mathematical operations.",
        no_args_is_help=True,
    )

    @application.callback()
    def configure(
        context: typer.Context,
        state_dir: Annotated[
            Path,
            typer.Option("--state-dir", help="Local artifact and metadata directory."),
        ] = Path(".jacobian"),
    ) -> None:
        context.obj = CliState(
            state_dir,
            runtime_opener=runtime_opener,
        )

    application.command("init")(initialize)
    application.command("update")(update)
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
