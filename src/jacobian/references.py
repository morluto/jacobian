"""Operator bootstrap for the bundled reference domains."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jacobian.artifacts import ArtifactService
from jacobian.checker_installation import CheckerInstaller
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.capabilities import (
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.claims import ClaimSpec
from jacobian.contracts.evidence import (
    CertificateEnvelope,
    WitnessEnvelope,
)
from jacobian.contracts.lean import LeanCandidate, LeanClaim, LeanEnvironment
from jacobian.contracts.plugins import PluginManifest
from jacobian.plugins.registry import PluginRegistry
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository

REFERENCE_INSTALLATION_DOMAINS = frozenset(
    {
        "jacobian.graph-paths",
        "jacobian.integer-matrices",
        "jacobian.erdos-straus",
    }
)


@dataclass(frozen=True, slots=True)
class ReferenceInstallation:
    name: str
    domain_id: str
    domain_version: str
    available_capabilities: tuple[str, ...]
    plugin_id: str
    semantics_uri: str
    claim_schema_uri: str
    candidate_schema_uri: str
    witness_schema_uri: str
    certificate_schema_uri: str
    witness_checker_ids: dict[str, str]
    certificate_checker_ids: dict[str, str]
    preservation_checker_ids: dict[str, str]
    transformation_checker_ids: dict[str, str]
    representation_schema_uris: dict[str, str]
    representation_semantics_uris: dict[str, str]


@dataclass(frozen=True, slots=True)
class PolytopeCheckerInstallation:
    witness_checker_id: str
    certificate_checker_id: str


@dataclass(frozen=True, slots=True)
class LeanCheckerInstallation:
    environment: LeanEnvironment
    lean_version: str
    lean_commit: str
    import_name: str | None
    mathlib_commit: str | None
    allowed_axioms: tuple[str, ...]
    checker_timeout_seconds: int
    semantics_uri: str
    claim_schema_uri: str
    candidate_schema_uri: str
    certificate_schema_uri: str
    checker_id: str | None


class ReferenceInstaller:
    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        artifacts: ArtifactService,
        plugins: PluginRegistry,
        checkers: CheckerRegistry,
        *,
        transformation_claim_schema_uri: str,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.artifacts = artifacts
        self.plugins = plugins
        self.checkers = checkers
        self.transformation_claim_schema_uri = transformation_claim_schema_uri
        self.manifest_schema_uri = schemas.register(
            name="jacobian.plugin-manifest",
            version="1",
            schema=model_schema(PluginManifest),
        )
        self.witness_schema_uri = schemas.register(
            name="jacobian.witness-envelope",
            version="1",
            schema=model_schema(WitnessEnvelope),
        )
        self.certificate_schema_uri = schemas.register(
            name="jacobian.certificate-envelope",
            version="1",
            schema=model_schema(CertificateEnvelope),
        )
        self.manifest_semantics_uri = store.register_descriptor(
            kind="semantics",
            name="jacobian.plugin-manifest",
            version="1",
            definition={"description": "untrusted domain capability metadata"},
        )

    def install_all(self) -> dict[str, ReferenceInstallation]:
        graph = self.install_graph_paths()
        matrix = self.install_matrices()
        erdos_straus = self.install_erdos_straus()
        return {
            graph.name: graph,
            matrix.name: matrix,
            erdos_straus.name: erdos_straus,
        }

    def install_polytope_checkers(
        self,
        *,
        claim_schema_uri: str,
        semantics_uri: str,
        point_schema_uri: str,
    ) -> PolytopeCheckerInstallation:
        """Authorize the separately implemented finite-polytope replay code."""

        witness_checker_id = self._authorize_checker(
            name="finite-polytope convex-combination checker",
            entrypoint=("jacobian_checkers.polytope:check_convex_combination"),
            evidence_kind="WITNESS",
            format_id="polytope.convex_combination",
            claim_schema=claim_schema_uri,
            semantics=semantics_uri,
            candidate_schema=point_schema_uri,
        )
        certificate_checker_id = self._authorize_checker(
            name="finite-polytope linear-separator checker",
            entrypoint="jacobian_checkers.polytope:check_linear_separator",
            evidence_kind="CERTIFICATE",
            format_id="polytope.linear_separator",
            claim_schema=claim_schema_uri,
            semantics=semantics_uri,
            candidate_schema=point_schema_uri,
        )
        return PolytopeCheckerInstallation(
            witness_checker_id=witness_checker_id,
            certificate_checker_id=certificate_checker_id,
        )

    def install_lean_checkers(
        self,
        *,
        resolve_provider_runtime: Callable[
            [dict[str, dict[str, Any]]], CapabilityProviderRuntime
        ],
    ) -> tuple[
        dict[LeanEnvironment, LeanCheckerInstallation], CapabilityProviderRuntime
    ]:
        """Authorize Lean checkers bound to their measured provider runtime."""

        mathlib_commit = "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f"
        lean_version = "4.31.0"
        lean_commit = "68218e876d2a38b1985b8590fff244a83c321783"
        claim_schema_uri = self.schemas.register(
            name="jacobian.lean4.claim",
            version="1",
            schema=model_schema(LeanClaim),
        )
        candidate_schema_uri = self.schemas.register(
            name="jacobian.lean4.candidate",
            version="1",
            schema=model_schema(LeanCandidate),
        )
        configurations: dict[
            LeanEnvironment,
            tuple[str | None, str | None, tuple[str, ...], int],
        ] = {
            LeanEnvironment.CORE: (None, None, (), 30),
            LeanEnvironment.MATHLIB: (
                "Mathlib",
                mathlib_commit,
                (
                    "Classical.choice",
                    "Quot.sound",
                    "propext",
                ),
                225,
            ),
        }
        profiles: dict[str, dict[str, Any]] = {}
        for environment, (
            import_name,
            pinned_mathlib,
            allowed_axioms,
            checker_timeout_seconds,
        ) in configurations.items():
            semantics_uri = self.store.register_descriptor(
                kind="semantics",
                name=f"jacobian.lean4-{environment.value.lower()}",
                version="1",
                definition={
                    "description": (
                        "exact Lean proposition checked by the pinned Lean kernel"
                    ),
                    "environment": environment.value,
                    "lean_version": lean_version,
                    "lean_commit": lean_commit,
                    "import_name": import_name,
                    "mathlib_commit": pinned_mathlib,
                    "allowed_axioms": list(allowed_axioms),
                    "checker_timeout_seconds": checker_timeout_seconds,
                    "trust_level": 0,
                },
            )
            profiles[environment.value] = {
                "semantics_uri": semantics_uri,
                "import_name": import_name,
                "mathlib_commit": pinned_mathlib,
                "allowed_axioms": list(allowed_axioms),
                "checker_timeout_seconds": checker_timeout_seconds,
            }
        provider_runtime = resolve_provider_runtime(profiles)

        installations: dict[LeanEnvironment, LeanCheckerInstallation] = {}
        for environment, (
            import_name,
            pinned_mathlib,
            allowed_axioms,
            checker_timeout_seconds,
        ) in configurations.items():
            semantics_uri = profiles[environment.value]["semantics_uri"]
            assert isinstance(semantics_uri, str)
            checker_id = None
            if (
                provider_runtime.availability
                is CapabilityProviderAvailability.AVAILABLE
            ):
                checker_id = self._authorize_checker(
                    name=f"pinned {environment.value} Lean kernel checker",
                    entrypoint="jacobian_checkers.lean4:check_kernel_certificate",
                    evidence_kind="CERTIFICATE",
                    format_id="lean4.kernel",
                    claim_schema=claim_schema_uri,
                    semantics=semantics_uri,
                    candidate_schema=candidate_schema_uri,
                    provider_runtime=provider_runtime,
                )
            installations[environment] = LeanCheckerInstallation(
                environment=environment,
                lean_version=lean_version,
                lean_commit=lean_commit,
                import_name=import_name,
                mathlib_commit=pinned_mathlib,
                allowed_axioms=allowed_axioms,
                checker_timeout_seconds=checker_timeout_seconds,
                semantics_uri=semantics_uri,
                claim_schema_uri=claim_schema_uri,
                candidate_schema_uri=candidate_schema_uri,
                certificate_schema_uri=self.certificate_schema_uri,
                checker_id=checker_id,
            )
        return installations, provider_runtime

    def install_graph_paths(self) -> ReferenceInstallation:
        domain = "jacobian.graph-paths"
        semantics = self.store.register_descriptor(
            kind="semantics",
            name=domain,
            version="1",
            definition={
                "description": (
                    "finite directed graphs, underlying-edge bipartiteness, "
                    "and all simple source-terminal paths"
                ),
                "path_semantics": "all simple paths induced by graph arcs",
                "bipartite_semantics": "underlying undirected graph",
            },
        )
        claim_schema = self.schemas.register(
            name=f"{domain}.claim",
            version="1",
            schema=_claim_schema(
                predicate_parameters={
                    "intended_paths_complete": {
                        "type": "object",
                        "properties": {
                            "simple": {"const": True},
                            "max_path_length": {
                                "type": "integer",
                                "minimum": 1,
                            },
                        },
                        "required": ["simple"],
                        "additionalProperties": False,
                    },
                    "is_bipartite": {
                        "type": "object",
                        "maxProperties": 0,
                    },
                }
            ),
        )
        candidate_schema = self.schemas.register(
            name=f"{domain}.candidate",
            version="1",
            schema=_graph_candidate_schema(),
        )
        capabilities = self._capabilities(
            {
                "Evaluator": ("jacobian.plugins.graph_paths:evaluate_capability"),
                "WitnessOracle": (
                    "jacobian.plugins.graph_paths:find_witness_capability"
                ),
                "Reducer": ("jacobian.plugins.graph_paths:reductions_capability"),
                "SemanticEnumerator": ("jacobian.plugins.graph_paths:materialize"),
                "Canonicalizer": (
                    "jacobian.plugins.graph_paths:canonicalize_capability"
                ),
                "CandidateEnumerator": (
                    "jacobian.plugins.graph_paths:enumerate_candidates_capability"
                ),
            }
        )
        plugin_id = self._install_manifest(
            domain=domain,
            semantics_uri=semantics,
            claim_schema_uri=claim_schema,
            candidate_schema_uri=candidate_schema,
            capabilities=capabilities,
        )
        witness_checkers = {
            "graph.omitted_path": self._authorize_checker(
                name="graph omitted-path witness checker",
                entrypoint=("jacobian_checkers.graph_paths:check_omitted_path"),
                evidence_kind="WITNESS",
                format_id="graph.omitted_path",
                claim_schema=claim_schema,
                semantics=semantics,
                candidate_schema=candidate_schema,
            ),
            "graph.odd_cycle": self._authorize_checker(
                name="graph odd-cycle witness checker",
                entrypoint="jacobian_checkers.graph_paths:check_odd_cycle",
                evidence_kind="WITNESS",
                format_id="graph.odd_cycle",
                claim_schema=claim_schema,
                semantics=semantics,
                candidate_schema=candidate_schema,
            ),
            "graph.2coloring": self._authorize_checker(
                name="graph two-coloring witness checker",
                entrypoint="jacobian_checkers.graph_paths:check_two_coloring",
                evidence_kind="WITNESS",
                format_id="graph.2coloring",
                claim_schema=claim_schema,
                semantics=semantics,
                candidate_schema=candidate_schema,
            ),
        }
        certificate_checkers = {
            "graph.path_enumeration": self._authorize_checker(
                name="graph path-enumeration certificate checker",
                entrypoint=("jacobian_checkers.graph_paths:check_path_enumeration"),
                evidence_kind="CERTIFICATE",
                format_id="graph.path_enumeration",
                claim_schema=claim_schema,
                semantics=semantics,
                candidate_schema=candidate_schema,
            )
        }
        preservation_checkers = {
            "graph.counterexample_preservation": self._authorize_checker(
                name="graph counterexample preservation checker",
                entrypoint=(
                    "jacobian_checkers.graph_paths:check_counterexample_preservation"
                ),
                evidence_kind="PRESERVATION",
                format_id="graph.counterexample_preservation",
                claim_schema=claim_schema,
                semantics=semantics,
                candidate_schema=candidate_schema,
            )
        }
        return ReferenceInstallation(
            name="graph_paths",
            domain_id=domain,
            domain_version="1",
            available_capabilities=tuple(sorted(capabilities)),
            plugin_id=plugin_id,
            semantics_uri=semantics,
            claim_schema_uri=claim_schema,
            candidate_schema_uri=candidate_schema,
            witness_schema_uri=self.witness_schema_uri,
            certificate_schema_uri=self.certificate_schema_uri,
            witness_checker_ids=witness_checkers,
            certificate_checker_ids=certificate_checkers,
            preservation_checker_ids=preservation_checkers,
            transformation_checker_ids={},
            representation_schema_uris={},
            representation_semantics_uris={},
        )

    def install_matrices(self) -> ReferenceInstallation:
        domain = "jacobian.integer-matrices"
        semantics = self.store.register_descriptor(
            kind="semantics",
            name=domain,
            version="1",
            definition={
                "description": (
                    "finite rectangular integer matrices with exact rational "
                    "kernel witnesses and bounded determinant scopes"
                )
            },
        )
        claim_schema = self.schemas.register(
            name=f"{domain}.claim",
            version="1",
            schema=_claim_schema(
                predicate_parameters={
                    "is_nonsingular": {
                        "type": "object",
                        "maxProperties": 0,
                    },
                    "maximize_absolute_determinant": {
                        "type": "object",
                        "properties": {
                            "scope": _matrix_scope_schema(),
                        },
                        "required": ["scope"],
                        "additionalProperties": False,
                    },
                }
            ),
        )
        candidate_schema = self.schemas.register(
            name=f"{domain}.candidate",
            version="1",
            schema=_matrix_candidate_schema(),
        )
        row_major_semantics = self.store.register_descriptor(
            kind="semantics",
            name=f"{domain}.row-major",
            version="1",
            definition={
                "description": (
                    "row-major exact integer encoding of a rectangular matrix"
                )
            },
        )
        row_major_schema = self.schemas.register(
            name=f"{domain}.row-major",
            version="1",
            schema={
                "type": "object",
                "properties": {
                    "rows": {"type": "integer", "minimum": 1},
                    "cols": {"type": "integer", "minimum": 1},
                    "values": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "integer"},
                                {
                                    "type": "string",
                                    "pattern": "^-?(?:0|[1-9][0-9]*)$",
                                },
                            ]
                        },
                    },
                },
                "required": ["rows", "cols", "values"],
                "additionalProperties": False,
            },
        )
        capabilities = self._capabilities(
            {
                "Evaluator": ("jacobian.plugins.matrices:evaluate_capability"),
                "WitnessOracle": ("jacobian.plugins.matrices:find_witness_capability"),
                "Reducer": "jacobian.plugins.matrices:reductions_capability",
                "SemanticEnumerator": ("jacobian.plugins.matrices:materialize"),
                "CandidateEnumerator": (
                    "jacobian.plugins.matrices:enumerate_candidates_capability"
                ),
                "Transformer": (
                    "jacobian.plugins.matrices:transform_row_major_capability"
                ),
            }
        )
        plugin_id = self._install_manifest(
            domain=domain,
            semantics_uri=semantics,
            claim_schema_uri=claim_schema,
            candidate_schema_uri=candidate_schema,
            capabilities=capabilities,
        )
        witness_checkers = {
            "matrix.kernel_vector": self._authorize_checker(
                name="matrix rational-kernel witness checker",
                entrypoint=("jacobian_checkers.matrices:check_kernel_vector"),
                evidence_kind="WITNESS",
                format_id="matrix.kernel_vector",
                claim_schema=claim_schema,
                semantics=semantics,
                candidate_schema=candidate_schema,
            ),
            "matrix.maximizer": self._authorize_checker(
                name="matrix maximum-determinant witness checker",
                entrypoint=("jacobian_checkers.matrices:check_maximizer_witness"),
                evidence_kind="WITNESS",
                format_id="matrix.maximizer",
                claim_schema=claim_schema,
                semantics=semantics,
                candidate_schema=candidate_schema,
            ),
        }
        certificate_checkers = {
            "matrix.maxdet_enumeration": self._authorize_checker(
                name="matrix max-determinant enumeration checker",
                entrypoint=("jacobian_checkers.matrices:check_maxdet_enumeration"),
                evidence_kind="CERTIFICATE",
                format_id="matrix.maxdet_enumeration",
                claim_schema=claim_schema,
                semantics=semantics,
                candidate_schema=candidate_schema,
            )
        }
        preservation_checkers = {
            "matrix.singular_preservation": self._authorize_checker(
                name="matrix singularity preservation checker",
                entrypoint=("jacobian_checkers.matrices:check_singular_preservation"),
                evidence_kind="PRESERVATION",
                format_id="matrix.singular_preservation",
                claim_schema=claim_schema,
                semantics=semantics,
                candidate_schema=candidate_schema,
            )
        }
        transformation_checkers = {
            "matrix.row_major": self._authorize_checker(
                name="matrix row-major transformation checker",
                entrypoint=(
                    "jacobian_checkers.matrices:check_row_major_transformation"
                ),
                evidence_kind="TRANSFORMATION",
                format_id="matrix.row_major",
                claim_schema=self.transformation_claim_schema_uri,
                semantics=semantics,
                candidate_schema=candidate_schema,
                target_schema=row_major_schema,
                target_semantics=row_major_semantics,
            )
        }
        return ReferenceInstallation(
            name="matrices",
            domain_id=domain,
            domain_version="1",
            available_capabilities=tuple(sorted(capabilities)),
            plugin_id=plugin_id,
            semantics_uri=semantics,
            claim_schema_uri=claim_schema,
            candidate_schema_uri=candidate_schema,
            witness_schema_uri=self.witness_schema_uri,
            certificate_schema_uri=self.certificate_schema_uri,
            witness_checker_ids=witness_checkers,
            certificate_checker_ids=certificate_checkers,
            preservation_checker_ids=preservation_checkers,
            transformation_checker_ids=transformation_checkers,
            representation_schema_uris={"row_major": row_major_schema},
            representation_semantics_uris={"row_major": row_major_semantics},
        )

    def install_erdos_straus(self) -> ReferenceInstallation:
        domain = "jacobian.erdos-straus"
        semantics = self.store.register_descriptor(
            kind="semantics",
            name=domain,
            version="1",
            definition={
                "description": (
                    "bounded Erdős-Straus claims over every integer n in one "
                    "closed interval"
                ),
                "equation": "4/n = 1/x + 1/y + 1/z",
                "variables": "positive integers n, x, y, z",
                "scope_rule": (
                    "a verified result covers only the exact finite interval "
                    "bound to the claim and candidate"
                ),
                "checker_identity": "4*x*y*z = n*(x*y + x*z + y*z)",
                "reference_limit": 10_000,
            },
        )
        range_schema = _erdos_straus_range_schema()
        claim_schema = self.schemas.register(
            name=f"{domain}.claim",
            version="1",
            schema=_claim_schema(
                predicate_parameters={"erdos_straus_range": range_schema}
            ),
        )
        candidate_schema = self.schemas.register(
            name=f"{domain}.candidate",
            version="1",
            schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                **range_schema,
            },
        )
        capabilities = self._capabilities(
            {
                "Evaluator": ("jacobian.plugins.erdos_straus:evaluate_capability"),
                "WitnessOracle": (
                    "jacobian.plugins.erdos_straus:find_witness_capability"
                ),
            }
        )
        plugin_id = self._install_manifest(
            domain=domain,
            semantics_uri=semantics,
            claim_schema_uri=claim_schema,
            candidate_schema_uri=candidate_schema,
            capabilities=capabilities,
        )
        witness_checkers = {
            "erdos_straus.decomposition_table": self._authorize_checker(
                name="bounded Erdős-Straus decomposition-table checker",
                entrypoint=("jacobian_checkers.erdos_straus:check_decomposition_table"),
                evidence_kind="WITNESS",
                format_id="erdos_straus.decomposition_table",
                claim_schema=claim_schema,
                semantics=semantics,
                candidate_schema=candidate_schema,
            )
        }
        return ReferenceInstallation(
            name="erdos_straus",
            domain_id=domain,
            domain_version="1",
            available_capabilities=tuple(sorted(capabilities)),
            plugin_id=plugin_id,
            semantics_uri=semantics,
            claim_schema_uri=claim_schema,
            candidate_schema_uri=candidate_schema,
            witness_schema_uri=self.witness_schema_uri,
            certificate_schema_uri=self.certificate_schema_uri,
            witness_checker_ids=witness_checkers,
            certificate_checker_ids={},
            preservation_checker_ids={},
            transformation_checker_ids={},
            representation_schema_uris={},
            representation_semantics_uris={},
        )

    def _capabilities(
        self,
        entrypoints: dict[str, str],
    ) -> dict[str, dict[str, str]]:
        capabilities: dict[str, dict[str, str]] = {}
        for name, entrypoint in entrypoints.items():
            implementation = self.plugins.register_implementation(entrypoint)
            capabilities[name] = {
                "implementation_uri": implementation,
                "entrypoint": entrypoint,
                "version": "1",
            }
        return capabilities

    def _install_manifest(
        self,
        *,
        domain: str,
        semantics_uri: str,
        claim_schema_uri: str,
        candidate_schema_uri: str,
        capabilities: dict[str, dict[str, str]],
    ) -> str:
        manifest = self.artifacts.put(
            schema_uri=self.manifest_schema_uri,
            semantics_uri=self.manifest_semantics_uri,
            payload={
                "plugin_schema_version": "1",
                "domain_id": domain,
                "domain_version": "1",
                "semantics_uri": semantics_uri,
                "claim_schema_uri": claim_schema_uri,
                "candidate_schema_uri": candidate_schema_uri,
                "witness_schema_uris": [self.witness_schema_uri],
                "certificate_schema_uris": [self.certificate_schema_uri],
                "capabilities": capabilities,
            },
            summary=f"reference plugin: {domain}",
        )
        self.plugins.install(manifest.artifact_uri)
        return manifest.artifact_uri

    def _authorize_checker(
        self,
        *,
        name: str,
        entrypoint: str,
        evidence_kind: str,
        format_id: str,
        claim_schema: str,
        semantics: str,
        candidate_schema: str,
        target_schema: str | None = None,
        target_semantics: str | None = None,
        provider_runtime: CapabilityProviderRuntime | None = None,
    ) -> str:
        return (
            CheckerInstaller(self.checkers)
            .install(
                CheckerOperation(
                    name=name,
                    entrypoint=entrypoint,
                    evidence_kind=EvidenceKind(evidence_kind),
                    format_id=format_id,
                    format_version="1",
                    claim_schema_uris=(claim_schema,),
                    semantics_uris=(semantics,),
                    candidate_schema_uris=(candidate_schema,),
                    target_schema_uris=(
                        (target_schema,) if target_schema is not None else ()
                    ),
                    target_semantics_uris=(
                        (target_semantics,) if target_semantics is not None else ()
                    ),
                    provider_runtime=provider_runtime,
                    reason="bundled reference checker",
                ),
                authorize=not self.checkers.bind_existing_when_omitted,
            )
            .require_checker_id()
        )


def reference_catalog(
    references: dict[str, ReferenceInstallation],
    *,
    graph: Any | None = None,
    polytope: Any | None = None,
    polytope_checkers: PolytopeCheckerInstallation | None = None,
    polynomial: Any | None = None,
    universal_algebra: Any | None = None,
    lean: dict[LeanEnvironment, LeanCheckerInstallation] | None = None,
) -> dict[str, Any]:
    """Return stable operator-facing identifiers for installed references."""

    catalog: dict[str, Any] = {
        name: {
            "domain_id": reference.domain_id,
            "domain_version": reference.domain_version,
            "available_capabilities": reference.available_capabilities,
            "plugin_id": reference.plugin_id,
            "semantics_uri": reference.semantics_uri,
            "claim_schema_uri": reference.claim_schema_uri,
            "candidate_schema_uri": reference.candidate_schema_uri,
            "witness_schema_uri": reference.witness_schema_uri,
            "certificate_schema_uri": reference.certificate_schema_uri,
            "witness_checker_ids": reference.witness_checker_ids,
            "certificate_checker_ids": reference.certificate_checker_ids,
            "preservation_checker_ids": reference.preservation_checker_ids,
            "transformation_checker_ids": reference.transformation_checker_ids,
            "representation_schema_uris": reference.representation_schema_uris,
            "representation_semantics_uris": (reference.representation_semantics_uris),
        }
        for name, reference in sorted(references.items())
    }
    if graph is not None:
        catalog["simple_undirected_graphs"] = {
            "domain_id": "jacobian.simple-undirected-graph",
            "domain_version": "1",
            "semantics_uri": graph.semantics_uri,
            "graph_schema_uri": graph.graph_schema_uri,
            "neighborhood_independence_schema_uri": graph.neighborhood_schema_uri,
            "neighborhood_independence_claim_schema_uri": (
                graph.neighborhood_claim_schema_uri
            ),
            "certificate_schema_uri": graph.certificate_schema_uri,
            "neighborhood_independence_certificate_format": (
                "graph.neighborhood_independence"
            ),
            "neighborhood_independence_checker_id": (graph.neighborhood_checker_id),
        }
    if polytope is not None:
        catalog["finite_polytopes"] = {
            "semantics_uri": polytope.semantics_uri,
            "claim_schema_uri": polytope.claim_schema_uri,
            "point_schema_uri": polytope.point_schema_uri,
            "generator_set_schema_uri": polytope.generator_set_schema_uri,
            "witness_checker_id": (
                polytope_checkers.witness_checker_id
                if polytope_checkers is not None
                else None
            ),
            "certificate_checker_id": (
                polytope_checkers.certificate_checker_id
                if polytope_checkers is not None
                else None
            ),
        }
    if polynomial is not None:
        catalog["rational_polynomial_maps"] = {
            "domain_id": "jacobian.rational-polynomial-map",
            "domain_version": "1",
            "semantics_uri": polynomial.semantics_uri,
            "map_schema_uri": polynomial.map_schema_uri,
            "evaluation_schema_uri": polynomial.evaluation_schema_uri,
            "jacobian_schema_uri": polynomial.jacobian_schema_uri,
            "claim_schema_uri": polynomial.claim_schema_uri,
            "jacobian_claim_schema_uri": polynomial.jacobian_claim_schema_uri,
            "keller_claim_schema_uri": polynomial.keller_claim_schema_uri,
            "inverse_collision_claim_schema_uri": (
                polynomial.inverse_collision_claim_schema_uri
            ),
            "witness_schema_uri": polynomial.witness_schema_uri,
            "certificate_schema_uri": polynomial.certificate_schema_uri,
            "witness_format": "polynomial.map_collision",
            "collision_checker_id": polynomial.collision_checker_id,
            "jacobian_certificate_format": "polynomial.jacobian_replay",
            "jacobian_checker_id": polynomial.jacobian_checker_id,
            "keller_certificate_format": ("polynomial.map.keller_condition.replay"),
            "keller_checker_id": polynomial.keller_checker_id,
            "inverse_collision_witness_format": (
                "polynomial.map_collision_refutes_inverse"
            ),
            "inverse_collision_checker_id": (polynomial.inverse_collision_checker_id),
        }
    if universal_algebra is not None:
        catalog["finite_magmas"] = {
            "domain_id": "jacobian.finite-magma-laws",
            "domain_version": "1",
            "semantics_uri": universal_algebra.semantics_uri,
            "problem_schema_uri": universal_algebra.problem_schema_uri,
            "evaluation_schema_uri": universal_algebra.evaluation_schema_uri,
            "countermodel_schema_uri": universal_algebra.countermodel_schema_uri,
            "claim_schema_uri": universal_algebra.claim_schema_uri,
            "certificate_schema_uri": universal_algebra.certificate_schema_uri,
            "certificate_format": "universal_algebra.law_evaluation",
            "evaluation_checker_id": universal_algebra.evaluation_checker_id,
        }
    if lean is not None and LeanEnvironment.CORE in lean:
        core = lean[LeanEnvironment.CORE]
        catalog["lean4"] = {
            "domain_id": "jacobian.lean4",
            "domain_version": "1",
            "lean_version": core.lean_version,
            "lean_commit": core.lean_commit,
            "claim_schema_uri": core.claim_schema_uri,
            "candidate_schema_uri": core.candidate_schema_uri,
            "certificate_schema_uri": core.certificate_schema_uri,
            "certificate_type": "lean4.kernel",
            "profiles": {
                environment.value: {
                    "semantics_uri": installation.semantics_uri,
                    "certificate_checker_id": installation.checker_id,
                    "import_name": installation.import_name,
                    "mathlib_commit": installation.mathlib_commit,
                    "allowed_axioms": installation.allowed_axioms,
                    "checker_timeout_seconds": (installation.checker_timeout_seconds),
                }
                for environment, installation in sorted(
                    lean.items(),
                    key=lambda item: item[0].value,
                )
            },
        }
    return catalog


def _claim_schema(
    *,
    predicate_parameters: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    schema = model_schema(ClaimSpec)
    predicate = schema["$defs"]["PredicateSpec"]
    predicate["properties"]["name"]["enum"] = list(predicate_parameters)
    schema["allOf"] = [
        {
            "if": {
                "properties": {
                    "predicate": {
                        "properties": {"name": {"const": name}},
                        "required": ["name"],
                    }
                },
                "required": ["predicate"],
            },
            "then": {
                "properties": {
                    "predicate": {"properties": {"parameters": parameter_schema}}
                }
            },
        }
        for name, parameter_schema in predicate_parameters.items()
    ]
    return schema


def _matrix_scope_schema() -> dict[str, Any]:
    exact_integer = {
        "oneOf": [
            {"type": "integer"},
            {"type": "string", "pattern": r"^-?(?:0|[1-9][0-9]*)$"},
        ]
    }
    return {
        "type": "object",
        "properties": {
            "rows": {"type": "integer", "minimum": 1, "maximum": 32},
            "cols": {"type": "integer", "minimum": 1, "maximum": 32},
            "entries": {
                "type": "array",
                "minItems": 1,
                "maxItems": 32,
                "uniqueItems": True,
                "items": exact_integer,
            },
        },
        "required": ["rows", "cols", "entries"],
        "additionalProperties": False,
    }


def _erdos_straus_range_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "lower_bound": {"type": "integer", "minimum": 2, "maximum": 10_000},
            "upper_bound": {"type": "integer", "minimum": 2, "maximum": 10_000},
        },
        "required": ["lower_bound", "upper_bound"],
        "additionalProperties": False,
    }


def _graph_candidate_schema() -> dict[str, Any]:
    vertex = {"type": "string", "minLength": 1, "maxLength": 128}
    path = {
        "type": "array",
        "minItems": 2,
        "uniqueItems": True,
        "items": vertex,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "vertices": {
                "type": "array",
                "minItems": 1,
                "maxItems": 128,
                "uniqueItems": True,
                "items": vertex,
            },
            "arcs": {
                "type": "array",
                "maxItems": 4096,
                "uniqueItems": True,
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "prefixItems": [vertex, vertex],
                    "items": False,
                },
            },
            "source": vertex,
            "terminals": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": vertex,
            },
            "intended_paths": {
                "type": "array",
                "uniqueItems": True,
                "items": path,
            },
        },
        "required": ["vertices", "arcs"],
        "additionalProperties": False,
    }


def _matrix_candidate_schema() -> dict[str, Any]:
    exact_integer = {
        "oneOf": [
            {"type": "integer"},
            {"type": "string", "pattern": r"^-?(?:0|[1-9][0-9]*)$"},
        ]
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "rows": {"type": "integer", "minimum": 1, "maximum": 32},
            "cols": {"type": "integer", "minimum": 1, "maximum": 32},
            "entries": {
                "type": "array",
                "minItems": 1,
                "maxItems": 32,
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "items": exact_integer,
                },
            },
        },
        "required": ["rows", "cols", "entries"],
        "additionalProperties": False,
    }
