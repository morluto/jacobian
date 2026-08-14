"""Operator-only compilation of the built-in mathematical catalog."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from jacobian import __version__
from jacobian.builtin_operation_modules import load_builtin_operation_modules
from jacobian.catalog_build import build_catalog_operations
from jacobian.catalog_build_context import create_catalog_build_context
from jacobian.contracts.operations import OperationDescriptor
from jacobian.operation_catalog import (
    CatalogBuildResult,
    CompiledCatalogEntry,
    OperationCatalogStore,
    declaration_digest,
    exact_checker_declaration_digest,
    operation_declaration_digest,
    public_operation_descriptor,
)
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.polytope import PolytopeService
from jacobian.registry import CheckerRegistry
from jacobian.runtime.bootstrap import bootstrap_services
from jacobian.runtime.model import JacobianRuntime
from jacobian.verification.service import VerificationService


def compile_operation_catalog(
    state_dir: Path,
    *,
    authorize_bundled_checkers: bool,
) -> CatalogBuildResult:
    """Authorize checkers and atomically compile one catalog revision."""

    core = bootstrap_services(state_dir, operation_policy=OperationVisibilityPolicy())
    runtime: JacobianRuntime | None = None
    resources = None
    try:
        verification = VerificationService(
            core.store,
            core.checkers,
            core.schemas,
            checker_timeout_seconds=105,
        )
        polytope = PolytopeService(core.store, core.schemas)
        runtime = JacobianRuntime(core, verification, polytope)
        context = create_catalog_build_context(
            runtime.core,
            verification,
            authorize_bundled_checkers=authorize_bundled_checkers,
        )
        resources = build_catalog_operations(context, polytope)
        bound_descriptors = tuple(
            sorted(
                core.operations.snapshot().operations,
                key=lambda item: item.operation_id,
            )
        )
        checker_bindings = _checker_bindings(core.checkers, bound_descriptors)
        descriptors = tuple(
            public_operation_descriptor(descriptor) for descriptor in bound_descriptors
        )
        entries = _compiled_entries(core.operations._adapters, descriptors)
        return OperationCatalogStore(state_dir / "metadata.sqlite3").commit(
            package_version=__version__,
            checker_binding_digest=declaration_digest(
                {
                    "bindings": [
                        [operation_id, [list(binding) for binding in bindings]]
                        for operation_id, bindings in sorted(checker_bindings.items())
                    ]
                }
            ),
            entries=entries,
            checker_bindings=checker_bindings,
        )
    finally:
        try:
            if resources is not None:
                resources.close()
        finally:
            if runtime is None:
                core.close()
            else:
                runtime.close()


def _compiled_entries(
    adapters: Mapping[str, object],
    descriptors: tuple[OperationDescriptor, ...],
) -> tuple[CompiledCatalogEntry, ...]:
    loaded_modules = load_builtin_operation_modules()
    declarations = {
        operation.operation_id: (module_name, operation)
        for module_name, operations, _checker_declarations in loaded_modules
        for operation in operations
    }
    exact_verifiers = {
        declaration.verification_operation_id: (module_name, declaration)
        for module_name, _operations, checker_declarations in loaded_modules
        for declaration in checker_declarations
        if declaration.verification_operation_id is not None
    }
    return tuple(
        CompiledCatalogEntry(
            descriptor=descriptor,
            declaration_module=(
                declarations[descriptor.operation_id][0]
                if descriptor.operation_id in declarations
                else exact_verifiers[descriptor.operation_id][0]
                if descriptor.operation_id in exact_verifiers
                else type(adapters[descriptor.operation_id]).__module__
            ),
            declaration_digest=(
                operation_declaration_digest(declarations[descriptor.operation_id][1])
                if descriptor.operation_id in declarations
                else exact_checker_declaration_digest(
                    exact_verifiers[descriptor.operation_id][1],
                    descriptor,
                )
                if descriptor.operation_id in exact_verifiers
                else declaration_digest(
                    {
                        "operation_id": descriptor.operation_id,
                        "version": descriptor.version,
                        "input_schema": descriptor.input_schema,
                        "output_schema": descriptor.output_schema,
                    }
                )
            ),
        )
        for descriptor in descriptors
    )


def _checker_bindings(
    checkers: CheckerRegistry,
    descriptors: tuple[OperationDescriptor, ...],
) -> dict[str, tuple[tuple[str, str], ...]]:
    bindings: dict[str, tuple[tuple[str, str], ...]] = {}
    for descriptor in descriptors:
        runtime = descriptor.provider_runtime
        if runtime is None or not runtime.checker_ids:
            continue
        bindings[descriptor.operation_id] = tuple(
            (
                str(checker_id),
                checkers.require_active(str(checker_id)).implementation_digest,
            )
            for checker_id in runtime.checker_ids
        )
    return bindings


__all__ = ["compile_operation_catalog"]
