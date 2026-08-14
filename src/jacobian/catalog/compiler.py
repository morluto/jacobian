"""Operator-only compilation of the built-in mathematical catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jacobian import __version__
from jacobian.builtin_operation_modules import load_builtin_operation_modules
from jacobian.catalog.build import (
    build_catalog_operations,
    create_catalog_build_context,
)
from jacobian.contracts.operations import OperationDescriptor
from jacobian.operation_catalog import (
    CatalogBuildResult,
    CompiledCatalogEntry,
    OperationCatalogStore,
    declaration_digest,
    exact_checker_declaration_digest,
    operation_declaration_digest,
    operation_declaration_digest_from_descriptor,
    public_operation_descriptor,
)
from jacobian.operation_locators import FamilyLocator, ModuleLocator, encode_locator
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.package_index import load_package_index
from jacobian.polytope import PolytopeService
from jacobian.registry import CheckerRegistry
from jacobian.runtime.bootstrap import bootstrap_services
from jacobian.runtime.model import JacobianRuntime
from jacobian.runtime.selected_families import selected_operation_origin
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
            core,
            verification,
            authorize_bundled_checkers=authorize_bundled_checkers,
        )
        resources = build_catalog_operations(context, polytope)
        operations = core.operations
        if operations is None:
            raise RuntimeError("catalog compilation requires an operation collector")
        bound_descriptors = tuple(
            sorted(
                operations.snapshot().operations,
                key=lambda item: item.operation_id,
            )
        )
        checker_bindings = _checker_bindings(core.checkers, bound_descriptors)
        descriptors = tuple(
            public_operation_descriptor(descriptor) for descriptor in bound_descriptors
        )
        entries = _compiled_entries(descriptors)
        packaged = load_package_index()
        bound_ids = {descriptor.operation_id for descriptor in bound_descriptors}
        omitted_operations = tuple(
            sorted(
                operation_id
                for operation_id in packaged.entries
                if operation_id not in bound_ids
            )
        )
        result = OperationCatalogStore(state_dir / "metadata.sqlite3").commit(
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
            omitted_operations=omitted_operations,
        )
        return CatalogBuildResult(
            revision=result.revision,
            operation_count=len(bound_descriptors),
            omitted_operations=omitted_operations,
            diagnostics=result.diagnostics,
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
    packaged = load_package_index()
    return tuple(
        CompiledCatalogEntry(
            descriptor=descriptor,
            declaration_module=_persisted_locator(
                descriptor.operation_id,
                declarations,
                exact_verifiers,
            ),
            declaration_digest=(
                operation_declaration_digest(declarations[descriptor.operation_id][1])
                if descriptor.operation_id in declarations
                else exact_checker_declaration_digest(
                    exact_verifiers[descriptor.operation_id][1],
                    descriptor,
                )
                if descriptor.operation_id in exact_verifiers
                else operation_declaration_digest_from_descriptor(descriptor)
            ),
        )
        for descriptor in descriptors
        if not packaged.contains(descriptor.operation_id)
    )


def _persisted_locator(
    operation_id: str,
    declarations: dict[str, tuple[str, Any]],
    exact_verifiers: dict[str, tuple[str, Any]],
) -> str:
    if operation_id in declarations:
        return encode_locator(ModuleLocator(module=declarations[operation_id][0]))
    if operation_id in exact_verifiers:
        return encode_locator(ModuleLocator(module=exact_verifiers[operation_id][0]))
    origin = selected_operation_origin(operation_id)
    if origin is None:
        raise ValueError(
            "catalog operation has no declaration or selected family owner: "
            f"{operation_id}"
        )
    return encode_locator(FamilyLocator(family=origin))


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
