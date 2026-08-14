"""Selected-bundle runtimes for tests that must not assemble the portfolio."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from jacobian.catalog_build_context import create_catalog_build_context
from jacobian.operation_declarations import OperationDeclarations
from jacobian.polytope import PolytopeService
from jacobian.runtime.bootstrap import bootstrap_services
from jacobian.runtime.model import JacobianRuntime
from jacobian.verification.service import VerificationService
from tests.support.catalog_build_options import CheckerAuthorityMode
from tests.support.services import atomic_installation


def create_selected_runtime(
    root: str | Path,
    bundles: Sequence[OperationDeclarations] = (),
    *,
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.NONE,
    **_kwargs: object,
) -> JacobianRuntime:
    """Compose one runtime that binds only the supplied operations."""

    core = bootstrap_services(
        root,
        bind_existing_checkers=(
            checker_authority is CheckerAuthorityMode.HYDRATE_EXISTING
        ),
    )
    try:
        verification = VerificationService(
            core.store,
            core.checkers,
            core.schemas,
            checker_timeout_seconds=105,
        )
        polytope = PolytopeService(core.store, core.schemas)
        installation = create_catalog_build_context(
            core,
            verification,
            authorize_bundled_checkers=(
                checker_authority is CheckerAuthorityMode.INSTALL_BUNDLED
            ),
        )
        if bundles:
            with atomic_installation(core):
                for operations in bundles:
                    bound = installation.binder.bind(operations)
                    for adapter in bound.adapters:
                        installation.register_operation(adapter)
        return JacobianRuntime(core, verification, polytope)
    except BaseException as error:
        cleanup_failures: list[BaseException] = []
        try:
            core.close()
        except BaseException as cleanup_error:
            cleanup_failures.append(cleanup_error)
        if cleanup_failures:
            error.add_note(
                "selected runtime construction cleanup also failed: "
                + "; ".join(str(failure) for failure in cleanup_failures)
            )
        raise


def selected_runtime_opener(*bundles: OperationDeclarations):
    """Return a ``create_catalog_build_runtime``-shaped opener for the supplied bundles."""

    def opener(
        root: str | Path,
        *,
        checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.NONE,
        **kwargs: object,
    ) -> JacobianRuntime:
        return create_selected_runtime(
            root,
            bundles,
            checker_authority=checker_authority,
            **kwargs,
        )

    return opener
