"""Installation of built-in declaration-module operations."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.builtin_operation_modules import load_builtin_operation_modules
from jacobian.catalog_build_context import CatalogBuildContext
from jacobian.checker_identity import batch_checker_manifest_measurement
from jacobian.exact_domain_checkers import (
    ExactDomainCheckerInstallation,
    ExactOperationGroup,
)
from jacobian.exact_domain_verification import (
    install_exact_domain_verification,
)


@dataclass(frozen=True, slots=True)
class CatalogOperationBuilder:
    """Build declaration-module operations and exact-domain checkers."""

    context: CatalogBuildContext

    def bind(self) -> None:
        """Load builtin declaration modules and bind exact-domain verification."""

        ctx = self.context
        exact_groups: dict[str, ExactOperationGroup] = {}
        for (
            module_name,
            operations,
            checker_declarations,
        ) in load_builtin_operation_modules():
            bound = ctx.binder.bind(operations)
            for adapter in bound.adapters:
                ctx.register_operation(adapter)
            if checker_declarations:
                exact_groups[module_name] = (
                    operations,
                    bound,
                    checker_declarations,
                )
        self.bind_domain_verification(exact_groups)

    def bind_domain_verification(
        self,
        operation_groups: dict[str, ExactOperationGroup],
    ) -> ExactDomainCheckerInstallation | None:
        ctx = self.context
        if not operation_groups:
            return None
        # Batch identity material across the complete declaration set while the
        # exact-domain installer resolves both legacy and declaration-owned
        # provider runtimes. Nested measurement remains safe for direct callers.
        with batch_checker_manifest_measurement():
            adapters, installation = install_exact_domain_verification(
                ctx.store,
                ctx.schemas,
                ctx.artifacts,
                ctx.values,
                ctx.verification,
                ctx.checkers,
                groups=operation_groups,
                authorize=ctx.authorize_bundled_checkers,
            )
        for adapter in adapters:
            self.context.register_operation(adapter)
        return installation


__all__ = ["CatalogOperationBuilder"]
