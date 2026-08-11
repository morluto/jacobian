"""Installation of operator-authorized references, Lean, and entrypoints."""

from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass
from typing import cast

from jacobian.builtin_capabilities import (
    LeanCheckAdapter,
    LeanDeclarationInspectAdapter,
    LeanDeclarationSearchAdapter,
    LeanDependencyGraphAdapter,
)
from jacobian.capability_service import CapabilityAdapter, CapabilityError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
)
from jacobian.contracts.lean import LeanDependencyGraphArtifact
from jacobian.installation.context import InstallationContext
from jacobian.lean_frontend.declarations import installed_lean_declaration_service
from jacobian.lean_frontend.exploration import install_lean_exploration_capabilities
from jacobian.lean_frontend.proof_axioms import (
    install_lean_proof_axioms_capability,
)
from jacobian.lean_frontend.proof_edit import install_lean_proof_edit_capability
from jacobian.lean_frontend.proof_state_inspect import (
    install_lean_proof_state_inspect_only,
)
from jacobian.lean_frontend.service import LeanService
from jacobian.portfolio.provider_resolution import ProviderAvailabilityResolver
from jacobian.portfolio.result import PortfolioInstallation
from jacobian.provider_runtime import jacobian_provider_runtime
from jacobian.references import REFERENCE_INSTALLATION_DOMAINS
from jacobian.runtime.config import CheckerAuthorityMode
from jacobian.runtime.services import ApplicationServices

_LOGGER = logging.getLogger(__name__)
_ENTRYPOINT_SEPARATOR = ":"
_ENTRYPOINT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
_ENTRYPOINT_ERRORS: tuple[type[BaseException], ...] = (
    AttributeError,
    ImportError,
    TypeError,
)


@dataclass(frozen=True, slots=True)
class ReferenceLeanInstaller:
    """Install references and Lean according to context-owned authority."""

    context: InstallationContext
    provider_resolver: ProviderAvailabilityResolver

    def install(
        self,
        application: ApplicationServices,
        result: PortfolioInstallation,
        *,
        capability_adapter_entrypoints: tuple[str, ...] = (),
    ) -> None:
        ctx = self.context
        core = application.core
        if ctx.authorizes_bundled_checkers or (
            ctx.checker_authority is CheckerAuthorityMode.HYDRATE_EXISTING
            and core.plugins.has_any_domain(REFERENCE_INSTALLATION_DOMAINS)
        ):
            self._install_references(application, result)
        for entrypoint in capability_adapter_entrypoints:
            ctx.register_capability(
                self._load_entrypoint_adapter(entrypoint, application)
            )

    def _install_references(
        self,
        application: ApplicationServices,
        result: PortfolioInstallation,
    ) -> None:
        ctx = self.context
        result.references = application.reference_installer.install_all()
        result.polytope_checkers = (
            application.reference_installer.install_polytope_checkers(
                claim_schema_uri=application.polytope.claim_schema_uri,
                semantics_uri=application.polytope.semantics_uri,
                point_schema_uri=application.polytope.point_schema_uri,
            )
        )
        result.lean_checkers, checker_runtime = (
            application.reference_installer.install_lean_checkers(
                resolve_provider_runtime=lambda profiles: (
                    self.provider_resolver.resolve_lean(
                        profiles=profiles,
                        checker_ids=(),
                    )
                )
            )
        )
        runtime = checker_runtime.model_copy(
            update={
                "checker_ids": tuple(
                    installation.checker_id
                    for _, installation in sorted(
                        result.lean_checkers.items(),
                        key=lambda item: item[0].value,
                    )
                    if installation.checker_id is not None
                )
            }
        )
        result.lean_runtime = runtime
        inspect_adapter = install_lean_proof_state_inspect_only(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            result.lean_checkers,
            jacobian_provider_runtime(
                "jacobian.lean4",
                features=("immutable-proof-state", "read-only-inspection"),
            ),
        )
        ctx.register_capability(inspect_adapter)
        if runtime.availability is not CapabilityProviderAvailability.AVAILABLE:
            _LOGGER.warning("lean.check is not installed: %s", runtime.diagnostic)
            return
        if any(
            installation.checker_id is None
            for installation in result.lean_checkers.values()
        ):
            _LOGGER.warning("lean.check is not installed: no active Lean checker")
            return
        try:
            result.lean_declarations = installed_lean_declaration_service(runtime)
        except (OSError, RuntimeError) as exc:
            _LOGGER.warning("Lean declaration discovery is not installed: %s", exc)
        self._install_lean_declaration_adapters(result, runtime)
        result.lean = LeanService(
            ctx.store,
            ctx.artifacts,
            ctx.verification,
            result.lean_checkers,
        )
        ctx.register_capability(LeanCheckAdapter(result.lean, runtime))
        proof_axioms_adapter, result.lean_proof_axioms = (
            install_lean_proof_axioms_capability(
                ctx.store,
                ctx.schemas,
                ctx.artifacts,
                result.lean_checkers,
                runtime,
            )
        )
        ctx.register_capability(proof_axioms_adapter)
        adapters, result.lean_exploration = install_lean_exploration_capabilities(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            result.lean_checkers,
            runtime,
        )
        for adapter in adapters:
            if adapter.descriptor.capability_id == "lean.proof_state.inspect":
                continue
            ctx.register_capability(adapter)
        proof_edit_adapter, result.lean_proof_edit = install_lean_proof_edit_capability(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            result.lean,
            runtime,
        )
        ctx.register_capability(proof_edit_adapter)

    def _install_lean_declaration_adapters(
        self,
        result: PortfolioInstallation,
        runtime: CapabilityProviderRuntime,
    ) -> None:
        if result.lean_declarations is None:
            return
        ctx = self.context
        ctx.register_capability(
            LeanDeclarationSearchAdapter(result.lean_declarations, runtime)
        )
        ctx.register_capability(
            LeanDependencyGraphAdapter(
                result.lean_declarations,
                runtime,
                ctx.artifacts,
                semantics_uri=ctx.store.register_descriptor(
                    kind="semantics",
                    name="jacobian.lean4-declaration-dependencies",
                    version="1",
                    definition={
                        "description": (
                            "bounded constant dependencies extracted from elaborated "
                            "Lean declaration types and values"
                        ),
                        "provider_digest": runtime.digest,
                        "dependency_api": "Lean.Expr.getUsedConstantsAsSet",
                        "verification": "computed metadata; no theorem verification",
                    },
                ),
                dependency_graph_schema_uri=ctx.schemas.register(
                    name="jacobian.lean4-dependency-graph",
                    version="1",
                    schema=LeanDependencyGraphArtifact.model_json_schema(),
                ),
            )
        )
        ctx.register_capability(
            LeanDeclarationInspectAdapter(result.lean_declarations, runtime)
        )

    def _load_entrypoint_adapter(
        self,
        entrypoint: str,
        application: ApplicationServices,
    ) -> CapabilityAdapter:
        if not _ENTRYPOINT_PATTERN.fullmatch(entrypoint):
            raise CapabilityError("capability adapter entrypoint has an invalid format")
        module_name, attribute_name = entrypoint.split(_ENTRYPOINT_SEPARATOR, 1)
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, attribute_name)
            adapter = factory(application)
            descriptor = adapter.descriptor
            invoke = adapter.invoke
        except _ENTRYPOINT_ERRORS as exc:
            raise CapabilityError(
                f"cannot load capability adapter entrypoint: {entrypoint}"
            ) from exc
        if not isinstance(descriptor, CapabilityDescriptor) or not callable(invoke):
            raise CapabilityError("capability adapter does not implement the protocol")
        return cast(CapabilityAdapter, adapter)
